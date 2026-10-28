"""中文/英文混合文本的归一化、切句与分词。

不依赖 jieba：中文按「单字 + 相邻双字」切分，既保留召回，又让 BM25 对
「告警规则」「限流阈值」这类术语敏感；英文/数字按词切分。
"""
from __future__ import annotations

import re
import unicodedata

_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_SENT_END = re.compile(r"(?<=[。！？!?；;])|\n+")
_WS = re.compile(r"[ \t\u3000]+")


def normalize(text: str) -> str:
    """全角转半角、统一大小写，压缩空白。"""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


def tokenize(text: str) -> list[str]:
    """返回用于检索的词项列表（含重复，供 BM25 统计词频）。"""
    text = normalize(text)
    tokens: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_.\-]*|[\u4e00-\u9fff]+", text):
        seg = match.group(0)
        if seg[0].isascii():
            tokens.append(seg)
        else:
            chars = list(seg)
            tokens.extend(chars)
            tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


STOPWORDS = set(
    "的 了 是 在 和 与 及 或 有 无 对 为 以 被 把 从 到 中 上 下 个 各 该 其 这 那 "
    "什么 怎么 如何 多少 多久 哪些 哪个 是否 能否 可以 需要 应该 请 问 平台 目前 "
    "现在 一下 进行 以及 还是 时候 一般 大约 具体 相关 情况 吗 呢 会 要 能 就 都 "
    "也 我们 你们 之后 之前 内 时 按 由 于 呢 啊".split()
)


def segment(text: str, vocab2: set[str]) -> list[str]:
    """基于语料双字词表的贪婪最长匹配分词（无第三方词典依赖）。

    vocab2 由知识库统计得到（df>=2 的双字组合），用于把「默认采集周期」
    切成「默认 / 采集 / 周期」，避免单字+双字混切产生的伪词项。
    """
    text = normalize(text)
    out: list[str] = []
    for match in re.finditer(r"[a-z0-9][a-z0-9_.\-]*|[\u4e00-\u9fff]+", text):
        seg = match.group(0)
        if seg[0].isascii():
            out.append(seg)
            continue
        i = 0
        while i < len(seg):
            if i + 1 < len(seg) and seg[i : i + 2] in vocab2:
                out.append(seg[i : i + 2])
                i += 2
            else:
                out.append(seg[i])
                i += 1
    return out


def split_sentences(text: str) -> list[str]:
    """按中文标点/换行切句，保留标点，过滤空句。"""
    parts = [p.strip() for p in _SENT_END.split(text)]
    return [p for p in parts if p]


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
