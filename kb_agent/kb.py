"""知识库加载与索引构建。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from collections import Counter

from .chunking import Chunk, chunk_markdown
from .index import HybridIndex
from .rerank import FeatureReranker
from .text import STOPWORDS, segment, tokenize

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
QA_PATH = ROOT / "data" / "qa" / "eval_set.jsonl"
RESULTS_DIR = ROOT / "results"


@dataclass
class KnowledgeBase:
    chunks: list[Chunk]
    index: HybridIndex
    reranker: FeatureReranker | None = None
    df: Counter = field(default_factory=Counter)
    vocab2: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.df:
            for chunk in self.chunks:
                self.df.update(set(tokenize(chunk.text)))
            self.vocab2 = {t for t in self.df if len(t) == 2 and self.df[t] >= 2}
        if self.reranker is None:
            self.reranker = FeatureReranker(df=self.df, n_docs=len(self.chunks))

    def answerability(self, query: str, candidates: list[dict]) -> tuple[float, list[str]]:
        """内容词覆盖率 + 未登录实体检测，作为「资料是否足够回答」的门控。

        返回 (最佳覆盖率, 语料中不存在的实体词)。出现未登录实体时，说明
        知识库根本没有这条信息，即使检索分数很高也应该拒答。
        """
        terms = segment(query, self.vocab2)
        focus = [t for t in terms if t not in STOPWORDS and self.df.get(t, 0) > 0]
        # 只对「词表级」未登录词报 OOV：ASCII 实体（ipv6 / java / sdk）或
        # 语料中从未出现的双字词。单字连写会在「哪些」「接手」这类口语上
        # 误报，因此不做拼接。
        oov = [
            t
            for t in terms
            if t not in STOPWORDS
            and self.df.get(t, 0) == 0
            and (len(t) >= 2 or t.isascii())
        ]
        best = 0.0
        for cand in candidates:
            chunk_tokens = set(tokenize(self.chunks[cand["idx"]].text))
            hit = sum(1 for t in focus if t in chunk_tokens)
            best = max(best, hit / max(1, len(focus)))
        return best, oov

    @classmethod
    def build(cls, docs_dir: Path = DOCS_DIR, max_chars: int = 320, overlap: int = 60) -> "KnowledgeBase":
        chunks: list[Chunk] = []
        for path in sorted(docs_dir.glob("*.md")):
            chunks.extend(
                chunk_markdown(
                    path.read_text(encoding="utf-8"),
                    doc_id=path.stem,
                    max_chars=max_chars,
                    overlap_chars=overlap,
                )
            )
        index = HybridIndex([c.text for c in chunks])
        return cls(chunks=chunks, index=index)

    def retrieve(self, query: str, top_k: int = 8, rerank: bool = True) -> list[dict]:
        candidates = self.index.search(query, k=max(top_k * 3, 20), pool=30)
        if rerank:
            candidates = self.reranker.rerank(
                query, candidates, self.chunks, top_k=top_k, vocab2=self.vocab2
            )
        return candidates[:top_k]

    def chunk(self, idx: int) -> Chunk:
        return self.chunks[idx]


def load_qa(path: Path = QA_PATH) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
