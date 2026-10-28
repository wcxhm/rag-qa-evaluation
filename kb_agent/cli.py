"""命令行入口：python -m kb_agent.cli "回滚要多久？" [--llm] [--json]"""
from __future__ import annotations

import argparse
import json

from .agent import KBAgent
from .generate import build_generator
from .kb import KnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="企业知识库 RAG + Agent 问答")
    parser.add_argument("question", help="要提问的问题")
    parser.add_argument("--llm", action="store_true", help="存在 API key 时启用 LLM 生成")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON（含 trace 与引用）")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    kb = KnowledgeBase.build()
    agent = KBAgent(kb, build_generator(force_offline=not args.llm), top_k=args.top_k)
    result = agent.run(args.question)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"问题：{result['query']}")
    if result["rewritten_query"] != result["query"]:
        print(f"改写：{result['rewritten_query']}")
    print(f"回答：{result['answer']}")
    if result["citations"]:
        print("引用：")
        for cite in result["citations"]:
            print(f"  [{cite['n']}] {cite['doc_title']} / {cite['section']} ({cite['doc_id']})")
    print(f"耗时：{result['latency_ms']} ms  路径：{' → '.join(n['node'] for n in result['trace'])}")


if __name__ == "__main__":
    main()
