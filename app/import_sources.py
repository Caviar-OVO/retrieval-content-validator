"""Import "retrieved sources" from pasted text or an uploaded file.

The validator expects a list of sources shaped like::

    {"id": "S1", "text": "...", "title": "...", "url": "...", "tag": "...", "traffic": 0.9}

Real retrieval tools and hand-copied search results name those fields
differently (``link`` / ``href``, ``snippet`` / ``content`` / ``page_content``,
``score``, a nested ``metadata`` block, ...). This module normalizes the common
shapes - JSON, JSONL, CSV/TSV, a URL per line, Markdown links, and plain
"title + URL + snippet" text blocks - into that schema.

It is a UI-side convenience only: nothing here changes validation logic, and
every record is still validated through the real ``Source`` model before it is
handed to the console.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from rva.models import Source

__all__ = ["ImportResult", "parse_sources"]

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")

# Field aliases, lowercase with `-`/space folded to `_`.
_ID_KEYS = ("id", "source_id", "doc_id", "key", "ref", "reference")
_TEXT_KEYS = (
    "text", "snippet", "content", "body", "description", "summary", "chunk",
    "page_content", "passage", "excerpt", "abstract", "raw_text", "answer",
)
_TITLE_KEYS = ("title", "name", "heading", "headline", "label")
_URL_KEYS = ("url", "link", "href", "uri", "source_url", "permalink")
_TAG_KEYS = ("tag", "domain", "site", "host", "publisher", "source_name")
_TRAFFIC_KEYS = ("traffic", "score", "weight", "relevance")
_DATE_KEYS = ("published", "date", "published_date", "pub_date", "timestamp")
_META_KEYS = ("metadata", "meta", "attributes", "extra", "info")
_LIST_KEYS = (
    "sources", "results", "items", "data", "documents", "organic_results",
    "hits", "pages", "entries", "citations", "context",
)
_LABEL_KEYS = ("url", "link", "title", "tag", "source", "date", "published")


@dataclass
class ImportResult:
    """Parsed + normalized records, ready to drop into the sources table."""

    records: list[dict] = field(default_factory=list)
    format: str = "unknown"
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# value coercion
# --------------------------------------------------------------------------
def _norm_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _flatten(record: dict) -> dict:
    """Lift a nested metadata/meta block up so aliases are found either way."""
    flat = dict(record)
    for mk in _META_KEYS:
        meta = record.get(mk)
        if isinstance(meta, dict):
            for k, v in meta.items():
                flat.setdefault(k, v)
    return flat


def _lookup(flat: dict, keys: tuple[str, ...]) -> Optional[Any]:
    norm: dict[str, Any] = {}
    for k, v in flat.items():
        nk = _norm_key(k)
        if nk not in norm or norm[nk] in (None, ""):
            norm[nk] = v
    for k in keys:
        val = norm.get(k)
        if val not in (None, "", [], {}):
            return val
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _as_traffic(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= num <= 1.0:
        return round(num, 4)
    if 1.0 < num <= 100.0:  # a 0-100 score / percent
        return round(num / 100.0, 4)
    return None


def _domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


# --------------------------------------------------------------------------
# per-format parsers (each returns None when it does not recognise the input)
# --------------------------------------------------------------------------
def _records_from_json(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            val = data.get(key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
        return [data]
    return []


def _parse_json(raw: str) -> Optional[list[dict]]:
    text = raw.strip()
    if not text.startswith(("[", "{")):
        return None
    try:
        data = json.loads(text)
    except Exception:
        return None
    records = _records_from_json(data)
    return records or None


def _parse_jsonl(raw: str) -> Optional[list[dict]]:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2 or not lines[0].lstrip().startswith("{"):
        return None
    records: list[dict] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records or None


def _parse_url_lines(raw: str) -> Optional[list[dict]]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return None
    hits = sum(1 for ln in lines if _URL_RE.match(ln) or _MD_LINK_RE.search(ln))
    if hits == 0 or hits / len(lines) < 0.6:
        return None
    records: list[dict] = []
    for ln in lines:
        md = _MD_LINK_RE.search(ln)
        if md:
            rec = {"url": md.group(2)}
            if md.group(1).strip() and md.group(1).strip() != md.group(2):
                rec["title"] = md.group(1).strip()
            records.append(rec)
            continue
        if _URL_RE.match(ln):
            records.append({"url": ln})
    return records or None


def _parse_csv(raw: str, filename: str = "") -> Optional[list[dict]]:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    header = lines[0]
    delim = "\t" if (filename.lower().endswith(".tsv") or
                     (header.count("\t") > header.count(",") and "\t" in header)) else ","
    if delim not in header:
        return None
    known = set(_ID_KEYS + _TEXT_KEYS + _TITLE_KEYS + _URL_KEYS + _TAG_KEYS +
                _TRAFFIC_KEYS + _DATE_KEYS + ("source",))
    if not (set(_norm_key(c) for c in header.split(delim)) & known):
        return None
    reader = csv.DictReader(io.StringIO(raw.strip().lstrip("\ufeff")), delimiter=delim)
    return [dict(row) for row in reader]


def _parse_blocks(raw: str) -> list[dict]:
    """Blank-line separated blocks: a title/URL/snippet, however it was pasted."""
    records: list[dict] = []
    for block in re.split(r"\n[ \t]*\n", raw.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        rec: dict[str, Any] = {}
        rest: list[str] = []
        url_at: Optional[int] = None
        for i, line in enumerate(lines):
            md = _MD_LINK_RE.search(line)
            if md and not rec.get("url"):
                rec["url"] = md.group(2)
                if md.group(1).strip() and md.group(1).strip() != md.group(2):
                    rec.setdefault("title", md.group(1).strip())
                url_at = i
                continue
            if _URL_RE.match(line) and not rec.get("url"):
                rec["url"] = line
                url_at = i
                continue
            key, sep, val = line.partition(":")
            if sep and _norm_key(key) in _LABEL_KEYS and val.strip():
                kk = _norm_key(key)
                rec.setdefault("url" if kk in ("link", "source") else kk, val.strip())
                continue
            rest.append(line)
        # a short line right above the URL is the title
        if url_at is not None and url_at > 0:
            prev = lines[url_at - 1]
            if prev in rest and "title" not in rec and len(prev) <= 160:
                rec["title"] = prev
                rest.remove(prev)
        if not rec.get("url") and len(rest) >= 2 and len(rest[0]) <= 120:
            rec["title"] = rest[0]
            rest = rest[1:]
        if rest:
            rec["text"] = "\n".join(rest)
        if rec.get("url") or rec.get("text"):
            records.append(rec)
    return records


# --------------------------------------------------------------------------
# normalize + validate
# --------------------------------------------------------------------------
def _build(records: list[dict], warnings: list[str]) -> list[dict]:
    normalized: list[dict] = []
    for rec in records:
        flat = _flatten(rec)
        url = _as_text(_lookup(flat, _URL_KEYS))
        raw_source = _as_text(_lookup(flat, ("source",)))
        if not url and raw_source.lower().startswith("http"):
            url = raw_source
        tag = _as_text(_lookup(flat, _TAG_KEYS))
        if not tag:
            tag = _domain_of(url)
        item: dict[str, Any] = {
            "id": _as_text(_lookup(flat, _ID_KEYS)),
            "text": _as_text(_lookup(flat, _TEXT_KEYS)),
        }
        title = _as_text(_lookup(flat, _TITLE_KEYS))
        if title:
            item["title"] = title
        if url:
            item["url"] = url
        if tag:
            item["tag"] = tag
        traffic = _as_traffic(_lookup(flat, _TRAFFIC_KEYS))
        if traffic is not None:
            item["traffic"] = traffic
        published = _as_text(_lookup(flat, _DATE_KEYS))
        if published:
            item["published"] = published
        if not item["text"] and not url:
            continue  # nothing usable
        normalized.append(item)

    # drop duplicates (same URL, or same text) while keeping the first
    kept: list[dict] = []
    seen: set[str] = set()
    for item in normalized:
        key = (item.get("url") or "").lower() or ("text:" + item["text"][:200])
        if key in seen:
            warnings.append(f"跳过重复来源：{item.get('url') or item['text'][:40]}")
            continue
        seen.add(key)
        kept.append(item)

    # fill in missing / clashing ids
    used: set[str] = set()
    for n, item in enumerate(kept, 1):
        sid = item["id"]
        if not sid:
            sid = f"S{n}"
        base, i = sid, 1
        while sid in used:
            i += 1
            sid = f"{base}_{i}"
        used.add(sid)
        item["id"] = sid

    valid: list[dict] = []
    for item in kept:
        try:
            valid.append(Source(**item).model_dump(exclude_none=True))
        except Exception as exc:  # surface, never silently drop
            warnings.append(f"跳过校验不通过的来源 {item.get('id')}：{exc}")
    return valid


def parse_sources(raw: str, filename: str = "") -> ImportResult:
    """Parse pasted text / an uploaded file into normalized source records."""
    warnings: list[str] = []
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ImportResult([], "empty", ["输入为空：请粘贴来源内容或选择一个文件。"])

    ext = filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    candidates: list[tuple[str, Any]] = []
    if ext == "json":
        candidates = [("json", lambda t: _parse_json(t))]
    elif ext in ("jsonl", "ndjson"):
        candidates = [("jsonl", lambda t: _parse_jsonl(t))]
    elif ext in ("csv", "tsv"):
        candidates = [(ext, lambda t: _parse_csv(t, filename))]
    elif ext in ("txt", "md", "markdown", "text", ""):
        candidates = [
            ("json", lambda t: _parse_json(t)),
            ("jsonl", lambda t: _parse_jsonl(t)),
            ("url per line", lambda t: _parse_url_lines(t)),
            ("csv/tsv", lambda t: _parse_csv(t, filename)),
            ("text blocks", lambda t: _parse_blocks(t) or None),
        ]
        if ext in ("txt", "md", "markdown", "text"):
            # a bare .txt/.md is far more likely to be links or blocks than JSON
            candidates.insert(2, candidates.pop(2))
    else:
        candidates = [
            ("json", lambda t: _parse_json(t)),
            ("jsonl", lambda t: _parse_jsonl(t)),
            ("csv/tsv", lambda t: _parse_csv(t, filename)),
            ("url per line", lambda t: _parse_url_lines(t)),
            ("text blocks", lambda t: _parse_blocks(t) or None),
        ]

    for fmt, parser in candidates:
        try:
            records = parser(text)
        except Exception as exc:
            warnings.append(f"{fmt} 解析失败：{exc}")
            continue
        if records:
            return ImportResult(_build(records, warnings), fmt, warnings)

    return ImportResult([], "unrecognized", warnings + [
        "无法识别输入格式。支持 JSON / JSONL / CSV / TSV、每行一个 URL、"
        "Markdown 链接，以及「标题 + URL + 摘要」的文本块。"
    ])
