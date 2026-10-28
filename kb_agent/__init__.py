"""kb_agent —— 零依赖的企业知识库 RAG + Agent 问答参考实现。"""
from .agent import KBAgent
from .kb import KnowledgeBase, load_qa

__all__ = ["KBAgent", "KnowledgeBase", "load_qa"]
__version__ = "0.1.0"
