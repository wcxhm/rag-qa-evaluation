"""结构感知切块：先按 Markdown 标题还原章节路径，再在章节内按句窗口切分。

相比定长滑窗，章节路径会作为元数据写进 chunk，检索命中后可以给出
「文档 > 章节」级别的引用，而不是只给一段无出处的文本。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict

from .text import split_sentences

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_id(*parts: str, length: int = 10) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:length]


def parse_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """返回 (文档标题, [(章节路径, 该章节正文), ...])。

    第一个一级标题作为文档标题，不计入章节路径。
    """
    doc_title = ""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    current_path = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer and "".join(buffer).strip():
            sections.append((current_path, list(buffer)))
        buffer.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if level == 1 and not doc_title:
                doc_title = title
                stack = [(1, title)]
                current_path = ""
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path_parts = [t for lvl, t in stack if lvl > 1]
            current_path = " > ".join(path_parts)
            continue
        buffer.append(line)
    flush()
    return doc_title, sections


def chunk_markdown(
    markdown: str,
    doc_id: str,
    max_chars: int = 320,
    overlap_chars: int = 60,
) -> list[Chunk]:
    """把一篇 Markdown 切成带章节元数据的 chunk。"""
    doc_title, sections = parse_sections(markdown)
    chunks: list[Chunk] = []

    for section, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        sentences = split_sentences(body)
        buf: list[str] = []
        buf_len = 0

        def emit() -> None:
            nonlocal buf, buf_len
            text = "".join(buf).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=_stable_id(doc_id, section, text),
                        doc_id=doc_id,
                        doc_title=doc_title,
                        section=section,
                        text=text,
                    )
                )
            tail = ""
            for sent in reversed(buf):
                if len(tail) + len(sent) > overlap_chars:
                    break
                tail = sent + tail
            buf = [tail] if tail else []
            buf_len = len(tail)

        for sent in sentences:
            if buf_len + len(sent) > max_chars and buf:
                emit()
            buf.append(sent)
            buf_len += len(sent)
        if buf_len:
            emit()

    # 同一章节内去重（overlap 可能产生完全重复的短块）
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        key = chunk.text
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique
