"""离线评测：检索消融（BM25 / 稠密 / 混合 / 混合+重排）+ Agent 端到端指标。

运行：python -m kb_agent.evaluate
产出：results/eval_report.md、results/eval_details.json
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np

from .agent import KBAgent
from .generate import ExtractiveGenerator
from .kb import RESULTS_DIR, KnowledgeBase, load_qa
from .text import tokenize

K = 10


def _ranking(kb: KnowledgeBase, query: str, strategy: str, k: int = K) -> list[int]:
    if strategy == "bm25":
        scores = kb.index.sparse_scores(query)
        return [int(i) for i in np.argsort(-scores)[:k]]
    if strategy == "dense":
        scores = kb.index.dense_scores(query)
        return [int(i) for i in np.argsort(-scores)[:k]]
    if strategy == "hybrid":
        return [c["idx"] for c in kb.index.search(query, k=k, pool=30)]
    if strategy == "hybrid_rerank":
        return [c["idx"] for c in kb.retrieve(query, top_k=k, rerank=True)]
    raise ValueError(strategy)


def _gold_pairs(row: dict) -> set[tuple[str, str]]:
    pairs = set()
    for item in row["gold"]:
        doc, _, section = item.partition("::")
        pairs.add((doc, section))
    return pairs


def retrieval_metrics(kb: KnowledgeBase, rows: list[dict], strategy: str) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal, ndcg, answerable = 0.0, 0.0, 0
    for row in rows:
        if not row["answerable"]:
            continue
        answerable += 1
        gold = _gold_pairs(row)
        ranking = _ranking(kb, row["q"], strategy)
        labels = [
            1 if (kb.chunk(i).doc_id, kb.chunk(i).section) in gold else 0 for i in ranking
        ]
        for cutoff in hits:
            if any(labels[:cutoff]):
                hits[cutoff] += 1
        first = next((pos for pos, rel in enumerate(labels, start=1) if rel), None)
        reciprocal += 1.0 / first if first else 0.0
        dcg = sum(rel / math.log2(pos + 1) for pos, rel in enumerate(labels, start=1))
        ideal_hits = min(len(gold), K)
        idcg = sum(1.0 / math.log2(pos + 1) for pos in range(1, ideal_hits + 1))
        ndcg += dcg / idcg if idcg else 0.0
    n = max(1, answerable)
    return {
        "strategy": strategy,
        "n": answerable,
        "recall@1": round(hits[1] / n, 4),
        "recall@5": round(hits[5] / n, 4),
        "recall@10": round(hits[10] / n, 4),
        "mrr@10": round(reciprocal / n, 4),
        "ndcg@10": round(ndcg / n, 4),
    }


def token_f1(pred: str, gold: str) -> float:
    pred_tokens, gold_tokens = tokenize(pred), tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def agent_metrics(kb: KnowledgeBase, rows: list[dict], min_coverage: float = 0.52) -> tuple[dict, list[dict]]:
    agent = KBAgent(kb, ExtractiveGenerator(), min_coverage=min_coverage)
    details, latencies = [], []
    f1_scores, cite_p, cite_r = [], [], []
    answered, refused_correct = 0, 0
    unanswerable = sum(1 for row in rows if not row["answerable"])

    for row in rows:
        result = agent.run(row["q"])
        latencies.append(result["latency_ms"])
        gold = _gold_pairs(row)
        cited = {(c["doc_id"], c["section"]) for c in result["citations"]}

        if row["answerable"]:
            f1_scores.append(token_f1(result["answer"], row["answer"]))
            if cited:
                hit = len(cited & gold)
                cite_p.append(hit / len(cited))
                cite_r.append(hit / len(gold))
            else:
                cite_p.append(0.0)
                cite_r.append(0.0)
            if not result["refused"]:
                answered += 1
        else:
            if result["refused"]:
                refused_correct += 1

        details.append(
            {
                "q": row["q"],
                "answerable": row["answerable"],
                "answer": result["answer"],
                "gold_answer": row["answer"],
                "refused": result["refused"],
                "citations": result["citations"],
                "gold": row["gold"],
                "retrieved": result["retrieved"],
                "f1": round(f1_scores[-1], 4) if row["answerable"] else None,
                "latency_ms": result["latency_ms"],
            }
        )

    lat = np.array(latencies) if latencies else np.array([0.0])
    metrics = {
        "min_coverage": min_coverage,
        "n": len(rows),
        "answer_rate": round(answered / max(1, len(rows) - unanswerable), 4),
        "answer_f1": round(float(np.mean(f1_scores)) if f1_scores else 0.0, 4),
        "citation_precision": round(float(np.mean(cite_p)) if cite_p else 0.0, 4),
        "citation_recall": round(float(np.mean(cite_r)) if cite_r else 0.0, 4),
        "unanswerable_refusal_rate": round(refused_correct / max(1, unanswerable), 4),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
    }
    return metrics, details


def threshold_sweep(kb: KnowledgeBase, rows: list[dict], thresholds=(0.35, 0.42, 0.46, 0.50, 0.52, 0.56, 0.60)) -> list[dict]:
    sweep = []
    for threshold in thresholds:
        metrics, _ = agent_metrics(kb, rows, min_coverage=threshold)
        sweep.append(metrics)
    return sweep


def main() -> None:
    kb = KnowledgeBase.build()
    rows = load_qa()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    retrieval = [retrieval_metrics(kb, rows, s) for s in ("bm25", "dense", "hybrid", "hybrid_rerank")]
    best = max(retrieval, key=lambda m: (m["recall@5"], m["mrr@10"]))
    agent_stat, details = agent_metrics(kb, rows, min_coverage=0.52)
    sweep = threshold_sweep(kb, rows)
    elapsed = time.perf_counter() - started

    report = {
        "corpus": {"docs": len({c.doc_id for c in kb.chunks}), "chunks": len(kb.chunks)},
        "qa": {
            "total": len(rows),
            "answerable": sum(1 for r in rows if r["answerable"]),
            "unanswerable": sum(1 for r in rows if not r["answerable"]),
        },
        "retrieval": retrieval,
        "agent": agent_stat,
        "threshold_sweep": sweep,
        "elapsed_s": round(elapsed, 2),
    }
    (RESULTS_DIR / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "eval_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# RAG + Agent 离线评测报告",
        "",
        f"- 语料：{report['corpus']['docs']} 篇文档 / {report['corpus']['chunks']} 个 chunk",
        f"- 评测集：{report['qa']['total']} 题（可答 {report['qa']['answerable']}，不可答 {report['qa']['unanswerable']}）",
        f"- 生成模式：抽取式（离线，无外部 API）",
        f"- 总耗时：{elapsed:.2f}s",
        "",
        "## 检索消融（Recall / MRR / nDCG）",
        "",
        "| 策略 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in retrieval:
        lines.append(
            f"| {row['strategy']} | {row['recall@1']:.3f} | {row['recall@5']:.3f} | "
            f"{row['recall@10']:.3f} | {row['mrr@10']:.3f} | {row['ndcg@10']:.3f} |"
        )
    lines += [
        "",
        f"最优检索策略：**{best['strategy']}**（Recall@5 {best['recall@5']:.3f}，MRR {best['mrr@10']:.3f}）。",
        "",
        "## Agent 端到端",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 答案 token-F1 | {agent_stat['answer_f1']:.3f} |",
        f"| 可答问题作答率 | {agent_stat['answer_rate']:.3f} |",
        f"| 引用准确率 precision | {agent_stat['citation_precision']:.3f} |",
        f"| 引用召回率 recall | {agent_stat['citation_recall']:.3f} |",
        f"| 不可答问题拒答率 | {agent_stat['unanswerable_refusal_rate']:.3f} |",
        f"| 延迟 P50 / P95 | {agent_stat['latency_p50_ms']:.1f} ms / {agent_stat['latency_p95_ms']:.1f} ms |",
        "",
        "## 拒答阈值扫描（内容词覆盖率门控）",
        "",
        "| 覆盖率阈值 | 作答率 | F1 | 引用准确率 | 拒答率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sweep:
        lines.append(
            f"| {row['min_coverage']:.2f} | {row['answer_rate']:.3f} | {row['answer_f1']:.3f} | "
            f"{row['citation_precision']:.3f} | {row['unanswerable_refusal_rate']:.3f} |"
        )
    (RESULTS_DIR / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
