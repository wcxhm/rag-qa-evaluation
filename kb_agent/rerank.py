"""重排层：无监督特征融合重排（可插拔为 cross-encoder）。

为什么不用 bge-reranker：当前环境无法安装 torch/GPU 依赖；这里把重排
拆成「IDF 加权词项覆盖 + 章节标题命中 + 稠密相似度 + 稀疏分数 + 答案
类型匹配」的线性融合，后续替换成 cross-encoder 只需实现 rerank()。

权重在 42 条可答问题上是网格搜索得到的（dev 集调参，test 集同分布），
其中章节标题命中是最强信号：问题问「回滚」，标题含「回滚操作」的块
应当排在只是正文提到「回滚」的块前面。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk
from .text import STOPWORDS, segment, tokenize

_NUMERIC_HINTS = ("多少", "几", "多久", "多长时间", "什么时候", "百分比")


@dataclass
class RerankWeights:
    coverage: float = 0.10
    heading: float = 0.50
    dense: float = 0.20
    sparse: float = 0.10
    numeric: float = 0.10


class FeatureReranker:
    def __init__(self, df: Counter | None = None, n_docs: int = 0, weights: RerankWeights | None = None):
        self.weights = weights or RerankWeights()
        self.df = df or Counter()
        self.n_docs = n_docs
        self.idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1 for term, count in self.df.items()
        }

    def _weight(self, term: str) -> float:
        return self.idf.get(term, 1.0)

    def features(self, query: str, chunk: Chunk, candidate: dict, vocab2: set[str] | None = None) -> dict:
        terms = [t for t in segment(query, vocab2 or set()) if t not in STOPWORDS]
        total = sum(self._weight(t) for t in terms) or 1.0
        body = set(tokenize(chunk.text))
        heading = set(tokenize(f"{chunk.doc_title} {chunk.section}"))

        coverage = sum(self._weight(t) for t in terms if t in body) / total
        heading_cov = sum(self._weight(t) for t in terms if t in heading) / total
        asks_number = any(hint in query for hint in _NUMERIC_HINTS)
        numeric = 1.0 if asks_number and any(ch.isdigit() for ch in chunk.text) else 0.0
        return {
            "coverage": round(coverage, 4),
            "heading": round(heading_cov, 4),
            "dense": round(max(0.0, float(candidate.get("dense", 0.0))), 4),
            "sparse": round(float(candidate.get("bm25_norm", 0.0)), 4),
            "numeric": numeric,
        }

    def score(self, query: str, chunk: Chunk, candidate: dict, vocab2: set[str] | None = None) -> tuple[float, dict]:
        feat = self.features(query, chunk, candidate, vocab2)
        w = self.weights
        value = (
            w.coverage * feat["coverage"]
            + w.heading * feat["heading"]
            + w.dense * feat["dense"]
            + w.sparse * feat["sparse"]
            + w.numeric * feat["numeric"]
        )
        return value, feat

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        chunks: list[Chunk],
        top_k: int = 10,
        vocab2: set[str] | None = None,
    ) -> list[dict]:
        scored = []
        for cand in candidates:
            value, feat = self.score(query, chunks[cand["idx"]], cand, vocab2)
            enriched = dict(cand)
            enriched["rerank"] = round(value, 6)
            enriched["features"] = feat
            scored.append(enriched)
        scored.sort(key=lambda d: -d["rerank"])
        return scored[:top_k]
