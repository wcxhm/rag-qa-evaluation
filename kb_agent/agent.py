"""Agent 编排：一个显式状态机（route → retrieve → grade → rewrite* → answer → verify）。

不依赖 LangGraph，但节点/条件边/重试预算/可观测 trace 与图编排同构：
- 检索与取块注册为工具，节点只通过工具访问知识库；
- grade 节点按重排分数决定「直接回答 / 改写重检 / 拒答」；
- rewrite 用伪相关反馈（把命中最好的章节名回填进 query）做查询扩展；
- verify 对每条引用做证据校验，无有效引用则降级为拒答，避免无出处答案。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .generate import Answer, ExtractiveGenerator, parse_citations
from .kb import KnowledgeBase
from .text import normalize, tokenize, unique_tokens

GREETINGS = {"你好", "您好", "hi", "hello", "在吗", "谢谢", "你是谁"}
REFUSAL_TEXT = "根据现有资料无法确定。"


@dataclass
class AgentState:
    query: str
    original_query: str = ""
    route: str = "kb"
    attempts: int = 0
    candidates: list[dict] = field(default_factory=list)
    support: float = 0.0
    oov: list[str] = field(default_factory=list)
    answer: Answer | None = None
    trace: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.original_query:
            self.original_query = self.query


class KBAgent:
    def __init__(
        self,
        kb: KnowledgeBase,
        generator=None,
        support_threshold: float = 0.20,
        rewrite_threshold: float = 0.34,
        min_coverage: float = 0.52,
        max_rewrites: int = 1,
        top_k: int = 6,
    ):
        self.kb = kb
        self.generator = generator or ExtractiveGenerator()
        self.support_threshold = support_threshold
        self.rewrite_threshold = rewrite_threshold
        self.min_coverage = min_coverage
        self.max_rewrites = max_rewrites
        self.top_k = top_k
        self.tools = {"kb_search": self._tool_kb_search, "kb_fetch": self._tool_kb_fetch}
        self.graph = {
            "route": self._route,
            "retrieve": self._retrieve,
            "grade": self._grade,
            "rewrite": self._rewrite,
            "answer": self._answer,
            "verify": self._verify,
            "refuse": self._refuse,
            "smalltalk": self._smalltalk,
        }

    # ---------- 工具 ----------
    def _tool_kb_search(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.kb.retrieve(query, top_k=top_k or self.top_k)

    def _tool_kb_fetch(self, idx: int) -> Chunk:
        return self.kb.chunk(idx)

    # ---------- 节点 ----------
    def _route(self, state: AgentState) -> tuple[AgentState, str]:
        text = normalize(state.query)
        if text in GREETINGS or len(text) < 2:
            return state, "smalltalk"
        return state, "retrieve"

    def _retrieve(self, state: AgentState) -> tuple[AgentState, str]:
        state.candidates = self.tools["kb_search"](state.query, self.top_k)
        return state, "grade"

    def _grade(self, state: AgentState) -> tuple[AgentState, str]:
        if not state.candidates:
            return state, "refuse"
        coverage, oov = self.kb.answerability(state.query, state.candidates)
        state.support = round(coverage, 4)
        state.oov = oov
        if oov or coverage < self.min_coverage:
            return state, "refuse"
        top_score = float(state.candidates[0].get("rerank", 0.0))
        if top_score < self.support_threshold:
            return state, "refuse"
        if top_score < self.rewrite_threshold and state.attempts < self.max_rewrites:
            return state, "rewrite"
        return state, "answer"

    def _rewrite(self, state: AgentState) -> tuple[AgentState, str]:
        state.attempts += 1
        section = self.kb.chunk(state.candidates[0]["idx"]).section
        if section and section not in state.query:
            state.query = f"{state.query} {section}"
        return state, "retrieve"

    def _answer(self, state: AgentState) -> tuple[AgentState, str]:
        state.answer = self.generator.generate(state.query, state.candidates, self.kb.chunks)
        return state, "verify"

    def _verify(self, state: AgentState) -> tuple[AgentState, str]:
        answer = state.answer
        assert answer is not None
        if answer.refused:
            return state, "end"
        answer_tokens = unique_tokens(answer.text)
        valid: list[int] = []
        for citation in answer.citations:
            if not 1 <= citation <= len(state.candidates):
                continue
            chunk = self.kb.chunk(state.candidates[citation - 1]["idx"])
            evidence_tokens = set(tokenize(chunk.text))
            overlap = len(answer_tokens & evidence_tokens) / max(1, len(answer_tokens))
            if overlap >= 0.2:
                valid.append(citation)
        answer.citations = valid
        if not valid:
            state.answer = Answer(REFUSAL_TEXT, [], refused=True, mode=answer.mode)
        return state, "end"

    def _refuse(self, state: AgentState) -> tuple[AgentState, str]:
        state.answer = Answer(REFUSAL_TEXT, [], refused=True, mode=self.generator.mode)
        return state, "end"

    def _smalltalk(self, state: AgentState) -> tuple[AgentState, str]:
        state.answer = Answer(
            "你好，我是企业知识库助手，可以回答监控、告警、发布、权限、值班等制度与产品问题。",
            [],
            refused=False,
            mode="smalltalk",
        )
        return state, "end"

    # ---------- 主循环 ----------
    def run(self, query: str) -> dict:
        started = time.perf_counter()
        state = AgentState(query=query)
        node: str | None = "route"
        while node and node != "end":
            state.trace.append({"node": node})
            state, node = self.graph[node](state)
        elapsed_ms = (time.perf_counter() - started) * 1000

        answer = state.answer or Answer(REFUSAL_TEXT, [], refused=True)
        citations = []
        for number in answer.citations:
            chunk = self.kb.chunk(state.candidates[number - 1]["idx"])
            citations.append(
                {
                    "n": number,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "section": chunk.section,
                }
            )
        return {
            "query": query,
            "rewritten_query": state.query,
            "answer": answer.text,
            "citations": citations,
            "refused": answer.refused,
            "support": state.support,
            "oov": state.oov,
            "mode": answer.mode,
            "retrieved": [
                {
                    "chunk_id": self.kb.chunk(c["idx"]).chunk_id,
                    "doc_id": self.kb.chunk(c["idx"]).doc_id,
                    "section": self.kb.chunk(c["idx"]).section,
                    "score": round(float(c.get("rerank", c.get("rrf", 0.0))), 4),
                }
                for c in state.candidates
            ],
            "trace": state.trace,
            "latency_ms": round(elapsed_ms, 2),
        }
