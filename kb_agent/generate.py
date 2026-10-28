"""生成层：可插拔 LLM（OpenAI 兼容接口）+ 零依赖抽取式兜底。

环境无 GPU、也无法访问公网模型服务时，默认走抽取式生成：从召回的
证据里选出与问题最相关的句子并按 [n] 标注出处。这样检索、引用校验、
评测链路都能离线复现；配置 DEEPSEEK_API_KEY / OPENAI_API_KEY 后自动
切换为真实 LLM 生成。
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .chunking import Chunk
from .text import split_sentences, tokenize

_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "你是企业知识库问答助手。只能依据给定的资料片段回答，"
    "每个结论后面用 [序号] 标注来源；资料不足时直接回答“根据现有资料无法确定”，不要编造。"
)


@dataclass
class Answer:
    text: str
    citations: list[int] = field(default_factory=list)
    refused: bool = False
    mode: str = "extractive"
    raw: dict | None = None


def parse_citations(text: str) -> list[int]:
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


class ExtractiveGenerator:
    mode = "extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, query: str, candidates: list[dict], chunks: list[Chunk]) -> Answer:
        if not candidates:
            return Answer("根据现有资料无法确定。", [], refused=True, mode=self.mode)

        query_tokens = set(tokenize(query))
        picked: list[tuple[float, int, str]] = []
        for citation, cand in enumerate(candidates[:3], start=1):
            chunk = chunks[cand["idx"]]
            for sent in split_sentences(chunk.text):
                sent_tokens = set(tokenize(sent))
                if not sent_tokens:
                    continue
                overlap = len(query_tokens & sent_tokens) / max(1, len(query_tokens))
                picked.append((overlap + 0.15 * (1.0 / citation), citation, sent))

        if not picked:
            return Answer("根据现有资料无法确定。", [], refused=True, mode=self.mode)

        picked.sort(key=lambda item: -item[0])
        best_score = picked[0][0]
        chosen = [item for item in picked if item[0] >= 0.75 * best_score][: self.max_sentences]
        chosen.sort(key=lambda item: item[1])
        chosen.sort(key=lambda item: item[1])
        body = "".join(f"{sent}[{citation}]" for _, citation, sent in chosen)
        if not body.endswith(("。", "！", "？")):
            body += "。"
        return Answer(body, parse_citations(body), refused=False, mode=self.mode)


class LLMGenerator:
    """OpenAI 兼容的 chat/completions 调用（DeepSeek / 通义 / vLLM 均可）。"""

    mode = "llm"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or ("https://api.deepseek.com/v1" if os.getenv("DEEPSEEK_API_KEY") else "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or "deepseek-chat"
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def build_prompt(self, query: str, candidates: list[dict], chunks: list[Chunk]) -> str:
        blocks = []
        for i, cand in enumerate(candidates[:5], start=1):
            chunk = chunks[cand["idx"]]
            blocks.append(f"[{i}] 《{chunk.doc_title}》/ {chunk.section}\n{chunk.text}")
        return "资料：\n" + "\n\n".join(blocks) + f"\n\n问题：{query}\n请给出答案并标注引用。"

    def generate(self, query: str, candidates: list[dict], chunks: list[Chunk]) -> Answer:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_prompt(query, candidates, chunks)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
        refused = "无法确定" in text and not parse_citations(text)
        return Answer(text, parse_citations(text), refused=refused, mode=self.mode, raw=data.get("usage"))


def build_generator(force_offline: bool = True):
    """默认离线；显式 force_offline=False 且存在 API key 时启用 LLM。"""
    if not force_offline:
        llm = LLMGenerator()
        if llm.available:
            return llm
    return ExtractiveGenerator()
