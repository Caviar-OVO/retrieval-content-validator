"""Tokenization, sentence segmentation, and lexical overlap utilities.

Zero third-party dependencies. Handles mixed Chinese / English text; CJK is
tokenized as character bigrams so that no external segmenter is required.
"""

from __future__ import annotations

import re
from typing import Iterable

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

STOPWORDS = {
    # English function words
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "as", "with", "by", "from", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "not", "no", "yes", "so", "than", "then", "they", "them",
    "their", "he", "she", "his", "her", "we", "our", "you", "your", "i", "my",
    "me", "us", "one", "also", "more", "most", "very", "about", "into", "over",
    # Chinese function words (single characters)
    "的", "了", "是", "在", "和", "与", "及", "或", "而", "但", "就", "都",
    "也", "很", "对", "把", "被", "给", "为", "以", "于", "之", "其", "这",
    "那", "他", "她", "它", "我", "你", "们", "个", "上", "下", "中", "里",
    "外", "会", "能", "要", "将", "已", "还", "又", "并", "则", "如", "若",
}

_LATIN = re.compile(r"[a-z][a-z0-9\-]{1,}|\d+(?:[.,]\d+)*%?")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")


def is_cjk(char: str) -> bool:
    return bool(_CJK.match(char))


def tokenize(text: str | None) -> list[str]:
    """Return discriminative tokens (Latin words/numbers + CJK bigrams)."""
    if not text:
        return []
    out: list[str] = []
    lower = text.lower()

    for word in _LATIN.findall(lower):
        if word not in STOPWORDS:
            out.append(word)

    for run in _CJK_RUN.findall(lower):
        chars = [c for c in run if c not in STOPWORDS]
        if len(chars) == 1:
            out.append(chars[0])
            continue
        for i in range(len(chars) - 1):
            out.append(chars[i] + chars[i + 1])
    return out


def split_claims(text: str | None) -> list[str]:
    """Split raw text into candidate claims, stripping markdown scaffolding."""
    if not text:
        return []
    cleaned = text.replace("\r", "")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)

    sentences: list[str] = []
    for line in re.split(r"\n+", cleaned):
        for piece in _SENT_SPLIT.split(line):
            s = piece.strip()
            if s:
                sentences.append(s)
    return [s for s in sentences if len(s) >= 4 and re.search(r"[^\W_]", s, flags=re.UNICODE)]


def overlap(needle_tokens: Iterable[str], haystack_tokens: Iterable[str]) -> float:
    """Fraction of the needle's vocabulary that appears in the haystack."""
    needle = set(needle_tokens)
    if not needle:
        return 0.0
    hay = set(haystack_tokens)
    return len(needle & hay) / len(needle)


def shared_token_count(needle_tokens: Iterable[str], haystack_tokens: Iterable[str]) -> int:
    return len(set(needle_tokens) & set(haystack_tokens))