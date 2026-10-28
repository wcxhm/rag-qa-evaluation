"""检索层：BM25 稀疏检索 + TF-IDF/SVD 稠密检索 + RRF 融合。

设计成可替换的 Embedder 接口：默认 LsaEmbedder 零第三方依赖、可离线复现；
接入 BGE / text-embedding-3 时，只需实现 encode() 并返回同形状归一化向量。
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if matrix.ndim == 1:
        norm = float(np.linalg.norm(matrix))
        return matrix / max(norm, eps)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


class BM25:
    """Okapi BM25（k1=1.5, b=0.75），对中文单字+双字词项效果稳定。"""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_corpus)
        self.tfs = [Counter(tokens) for tokens in tokenized_corpus]
        self.doc_len = np.array([len(tokens) for tokens in tokenized_corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self.tfs:
            doc_freq.update(tf.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        result = np.zeros(self.n_docs, dtype=float)
        if not self.n_docs:
            return result
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                result[i] += idf * freq * (self.k1 + 1) / denom
        return result


class Embedder:
    """嵌入器接口，便于把 LSA 换成 BGE / OpenAI embedding。"""

    dim: int

    def fit(self, texts: list[str]) -> "Embedder":  # pragma: no cover - interface
        raise NotImplementedError

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class LsaEmbedder(Embedder):
    """TF-IDF 加权后做截断 SVD（LSA），得到可余弦检索的稠密向量。"""

    def __init__(self, dim: int = 64, max_features: int = 4000):
        self.dim = dim
        self.max_features = max_features

    def _project(self, counts: Counter[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=float)
        for term, freq in counts.items():
            j = self.vocab.get(term)
            if j is None:
                continue
            vec[j] = (1 + math.log(freq)) * self.idf[j]
        return vec

    def fit(self, texts: list[str]) -> "LsaEmbedder":
        counts = [Counter(tokenize(t)) for t in texts]
        doc_freq: Counter[str] = Counter()
        for c in counts:
            doc_freq.update(c.keys())
        terms = [t for t, _ in sorted(doc_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
        self.vocab = {t: i for i, t in enumerate(terms[: self.max_features])}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + doc_freq[t])) + 1 for t in self.vocab], dtype=float
        )
        matrix = np.vstack([self._project(c) for c in counts]) if n else np.zeros((0, len(self.vocab)))
        matrix = _l2_normalize(matrix)
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        dim = max(2, min(self.dim, len(s)))
        self.components = vt[:dim]  # (dim, vocab)
        self.doc_vectors = _l2_normalize(u[:, :dim] * s[:dim])
        self.dim = dim
        return self

    def encode(self, text: str) -> np.ndarray:
        if not self.vocab:
            return np.zeros(self.dim, dtype=float)
        vec = self._project(Counter(tokenize(text))) @ self.components.T
        return _l2_normalize(vec)


class HybridIndex:
    """BM25 + 稠密向量，用 RRF（k=60）做排名融合。"""

    def __init__(self, texts: list[str], embedder: Embedder | None = None):
        self.texts = texts
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.embedder = embedder or LsaEmbedder()
        self.embedder.fit(texts)
        self.doc_vectors = getattr(self.embedder, "doc_vectors", None)

    def _dense_scores(self, query: str) -> np.ndarray:
        if self.doc_vectors is None:
            return np.zeros(len(self.texts))
        return self.doc_vectors @ self.embedder.encode(query)

    def sparse_scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(tokenize(query))

    def dense_scores(self, query: str) -> np.ndarray:
        return self._dense_scores(query)

    def search(
        self,
        query: str,
        k: int = 10,
        pool: int = 30,
        rrf_k: int = 60,
    ) -> list[dict]:
        """返回融合后的候选，附带各路子分数供重排使用。"""
        n = len(self.texts)
        if n == 0:
            return []
        sparse = self.sparse_scores(query)
        dense = self._dense_scores(query)

        sparse_rank = np.argsort(-sparse)[: min(pool, n)]
        dense_rank = np.argsort(-dense)[: min(pool, n)]

        fused: dict[int, dict] = {}
        for rank, idx in enumerate(sparse_rank, start=1):
            fused[int(idx)] = {
                "idx": int(idx),
                "rrf": 1.0 / (rrf_k + rank),
                "bm25": float(sparse[idx]),
                "dense": float(dense[idx]),
            }
        for rank, idx in enumerate(dense_rank, start=1):
            item = fused.setdefault(
                int(idx),
                {"idx": int(idx), "rrf": 0.0, "bm25": float(sparse[idx]), "dense": float(dense[idx])},
            )
            item["rrf"] += 1.0 / (rrf_k + rank)

        ordered = sorted(fused.values(), key=lambda d: -d["rrf"])[:k]
        max_bm25 = max((d["bm25"] for d in ordered), default=1.0) or 1.0
        max_rrf = max((d["rrf"] for d in ordered), default=1.0) or 1.0
        for item in ordered:
            item["bm25_norm"] = item["bm25"] / max_bm25
            item["rrf_norm"] = item["rrf"] / max_rrf
            item["dense"] = max(0.0, item["dense"])
        return ordered
