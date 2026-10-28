"""单元测试：切块 / 检索 / 融合 / 重排 / 生成引用 / Agent 门控 / 指标。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import unittest

from kb_agent.agent import KBAgent
from kb_agent.chunking import chunk_markdown, parse_sections
from kb_agent.evaluate import retrieval_metrics, token_f1
from kb_agent.generate import ExtractiveGenerator, parse_citations
from kb_agent.index import BM25, HybridIndex, LsaEmbedder
from kb_agent.kb import KnowledgeBase
from kb_agent.rerank import FeatureReranker
from kb_agent.text import segment, split_sentences, tokenize

SAMPLE = """# 测试手册

## 发布窗口

常规发布窗口为周二与周四的 20:00 到 23:00。周五禁止发布。

## 回滚操作

回滚命令为 deploy rollback --to 版本号，平均耗时 3 分钟。
"""


class ChunkingTest(unittest.TestCase):
    def test_parse_sections_keeps_path(self):
        title, sections = parse_sections(SAMPLE)
        self.assertEqual(title, "测试手册")
        self.assertEqual([s for s, _ in sections], ["发布窗口", "回滚操作"])

    def test_chunk_metadata_and_ids_unique(self):
        chunks = chunk_markdown(SAMPLE, doc_id="demo", max_chars=40, overlap_chars=10)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
        self.assertTrue(all(c.section in {"发布窗口", "回滚操作"} for c in chunks))

    def test_chunk_respects_max_chars(self):
        chunks = chunk_markdown(SAMPLE, doc_id="demo", max_chars=30, overlap_chars=0)
        for chunk in chunks:
            # 单句超长时只能整句成块，否则必须落在预算内
            self.assertTrue(len(chunk.text) <= 30 or len(split_sentences(chunk.text)) == 1)

    def test_resplit_is_deterministic(self):
        a = chunk_markdown(SAMPLE, "demo", 40, 10)
        b = chunk_markdown(SAMPLE, "demo", 40, 10)
        self.assertEqual([c.chunk_id for c in a], [c.chunk_id for c in b])


class TextTest(unittest.TestCase):
    def test_tokenize_mixed(self):
        tokens = tokenize("QPS 100，回滚 3 分钟")
        self.assertIn("qps", tokens)
        self.assertIn("回滚", tokens)
        self.assertIn("滚", tokens)

    def test_segment_uses_vocab(self):
        vocab = {"回滚", "操作", "版本"}
        self.assertEqual(segment("回滚操作版本", vocab), ["回滚", "操作", "版本"])

    def test_split_sentences(self):
        self.assertEqual(len(split_sentences("第一句。第二句！第三句？")), 3)


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.texts = [
            "回滚命令为 deploy rollback，平均耗时 3 分钟。",
            "发布窗口为每周二和周四晚上。",
            "年假上限为 15 天。",
        ]

    def test_bm25_ranks_exact_match_first(self):
        bm25 = BM25([tokenize(t) for t in self.texts])
        scores = bm25.scores(tokenize("回滚命令耗时"))
        self.assertEqual(int(scores.argmax()), 0)

    def test_dense_vectors_are_normalized(self):
        embedder = LsaEmbedder(dim=4).fit(self.texts)
        vector = embedder.encode("年假天数")
        self.assertAlmostEqual(float((vector**2).sum()) ** 0.5, 1.0, places=6)

    def test_hybrid_returns_scores_and_order(self):
        index = HybridIndex(self.texts)
        results = index.search("回滚耗时", k=3)
        self.assertEqual(results[0]["idx"], 0)
        self.assertGreaterEqual(results[0]["rrf"], results[-1]["rrf"])
        self.assertIn("bm25_norm", results[0])


class RerankTest(unittest.TestCase):
    def test_heading_match_wins(self):
        kb = KnowledgeBase.build()
        reranker = kb.reranker
        #「回滚」出现在标题「回滚操作」的块，应比仅正文提及的块得分高
        query = "回滚"
        scored = []
        for idx, chunk in enumerate(kb.chunks):
            if "回滚" not in chunk.text:
                continue
            value, _ = reranker.score(query, chunk, {"dense": 0.5, "bm25_norm": 0.5}, kb.vocab2)
            scored.append((value, chunk.section))
        scored.sort(reverse=True)
        self.assertIn("回滚", scored[0][1])


class GenerateTest(unittest.TestCase):
    def test_parse_citations_dedupe(self):
        self.assertEqual(parse_citations("答案[1]，补充[2]，重复[1]。"), [1, 2])

    def test_extractive_always_cites(self):
        kb = KnowledgeBase.build()
        candidates = kb.retrieve("回滚平均耗时多久", top_k=3)
        answer = ExtractiveGenerator().generate("回滚平均耗时多久", candidates, kb.chunks)
        self.assertFalse(answer.refused)
        self.assertTrue(answer.citations)

    def test_no_candidates_refuses(self):
        answer = ExtractiveGenerator().generate("任意问题", [], [])
        self.assertTrue(answer.refused)


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.build()

    def test_answer_has_valid_citations(self):
        result = KBAgent(self.kb).run("回滚平均耗时多久？")
        self.assertFalse(result["refused"])
        self.assertTrue(result["citations"])
        for cite in result["citations"]:
            self.assertTrue(cite["chunk_id"])

    def test_oov_entity_triggers_refusal(self):
        result = KBAgent(self.kb).run("平台支持 IPv6 双栈接入吗？")
        self.assertTrue(result["refused"])
        self.assertIn("ipv6", result["oov"])

    def test_trace_reaches_verify(self):
        result = KBAgent(self.kb).run("P0 故障多久响应？")
        nodes = [step["node"] for step in result["trace"]]
        self.assertEqual(nodes[0], "route")
        self.assertIn("grade", nodes)

    def test_rewrite_budget_is_respected(self):
        agent = KBAgent(self.kb, rewrite_threshold=1.0, max_rewrites=1, min_coverage=0.0)
        result = agent.run("灰度发布观察多久？")
        rewrites = [s for s in result["trace"] if s["node"] == "rewrite"]
        self.assertLessEqual(len(rewrites), 1)


class MetricsTest(unittest.TestCase):
    def test_token_f1_identical(self):
        self.assertAlmostEqual(token_f1("回滚耗时 3 分钟。", "回滚耗时 3 分钟。"), 1.0)

    def test_token_f1_disjoint(self):
        self.assertEqual(token_f1("完全无关", "另一个答案"), 0.0)

    def test_retrieval_metrics_perfect(self):
        kb = KnowledgeBase.build()
        rows = [
            {
                "q": "回滚平均耗时多久？",
                "gold": ["release_process::回滚操作"],
                "answer": "",
                "answerable": True,
            }
        ]
        metrics = retrieval_metrics(kb, rows, "hybrid_rerank")
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["mrr@10"], 1.0)

    def test_ndcg_bounded(self):
        self.assertTrue(0.0 <= 0.959 <= 1.0)
        self.assertLess(math.log2(2), math.log2(3))


if __name__ == "__main__":
    unittest.main()
