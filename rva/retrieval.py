"""Vector retrieval over the retrieved sources.

Modelled on Google ADK's RAG engine. ADK's `VertexAiRagRetrieval`
(google/adk-python, src/google/adk/tools/retrieval/vertex_ai_rag_retrieval.py)
wraps Vertex AI RAG with a small, stable contract:

    rag.retrieval_query(text=query, similarity_top_k=..., vector_distance_threshold=...)

and returns the matched contexts (`response.contexts.contexts`).

We keep exactly that contract -- index a corpus, then query for the top-k
contexts under a distance threshold -- but make the embedding backend
pluggable so the pipeline still runs offline:

  * `HashingEmbedder`  - deterministic, dependency-free signed char-ngram
                         vectors with optional corpus IDF weighting (default,
                         used when no API key is present).
  * `OpenAIEmbedder`   - real embeddings via any OpenAI-compatible endpoint
                         (OPENAI_API_KEY / OPENAI_BASE_URL / RVA_EMBED_MODEL).

Sources are chunked on sentence boundaries so a claim can match the relevant
span of a long document, mirroring how a RAG corpus is chunked before indexing.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .text import tokenize

# --- embedders -------------------------------------------------------------

_DIM = 2048


def _stable_hash(token: str, salt: bytes = b"") -> int:
    return int.from_bytes(hashlib.md5(salt + token.encode("utf-8")).digest()[:8], "big")


def _sign(token: str) -> float:
    """Signed hashing: two tokens colliding in one bucket may still cancel out."""
    return 1.0 if (_stable_hash(token, b"sign") & 1) else -1.0


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class HashingEmbedder:
    """Deterministic, offline embedding: signed hashing of discriminative tokens.

    Not a semantic model, but a real vector-space representation: texts become
    L2-normalised vectors compared by cosine similarity. `fit` learns corpus IDF
    so common tokens stop dominating the match.
    """

    name = "hashing-char-ngram"

    def __init__(self, dim: int = _DIM):
        self.dim = dim
        self.idf: dict[str, float] = {}
        self._n_docs = 0

    def fit(self, corpus: list[str]) -> "HashingEmbedder":
        df: dict[str, int] = {}
        for doc in corpus:
            for token in set(tokenize(doc)):
                df[token] = df.get(token, 0) + 1
        self._n_docs = len(corpus)
        self.idf = {
            token: math.log((1 + self._n_docs) / (1 + count)) + 1.0
            for token, count in df.items()
        }
        return self

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in tokenize(text):
                idx = _stable_hash(token) % self.dim
                vec[idx] += _sign(token) * self.idf.get(token, 1.0)
            out.append(_normalize(vec))
        return out


class OpenAIEmbedder:
    """Real embeddings via an OpenAI-compatible endpoint (network required)."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, batch_size: int = 64):
        from .config import get_api_key, get_base_url, get_embed_model

        self.model = model or get_embed_model()
        self.api_key = api_key or get_api_key()
        self.base_url = base_url or get_base_url()
        self.batch_size = batch_size
        self.name = f"openai-embeddings:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if (t and t.strip()) else " " for t in texts[start:start + self.batch_size]]
            resp = client.embeddings.create(model=self.model, input=batch)
            for item in resp.data:
                out.append(_normalize([float(x) for x in item.embedding]))
        return out


def build_embedder(backend: str = "hash", *, llm_ok: bool = False):
    """Return an embedder for `backend` in {hash, openai, auto}."""
    backend = (backend or "hash").lower()
    if backend == "auto":
        backend = "openai" if llm_ok else "hash"
    if backend == "openai":
        return OpenAIEmbedder()
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, clamped to [0.0, 1.0].

    The raw cosine of two L2-normalised vectors can be negative (anti-correlated
    token distributions). Downstream this value is a *similarity*: it becomes
    `EvidenceRef.similarity` / `coverage` (both `ge=0`) and drives the distance
    threshold. A negative score therefore means "less related than unrelated"
    and floors at 0 instead of leaking a negative into the report models.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(0.0, sum(x * y for x, y in zip(a, b)))


# --- chunking --------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;.])\s*")


def chunk_text(text: str, max_chars: int = 320) -> list[str]:
    """Split into sentence-packed chunks so long sources can be matched in parts."""
    text = (text or "").strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current}{sentence}" if current else sentence
    if current:
        chunks.append(current)
    return chunks or [text]


# --- retriever -------------------------------------------------------------

@dataclass
class RetrievedChunk:
    source_id: str
    text: str
    similarity: float
    distance: float


class VectorRetriever:
    """Index sources once, then query for top-k contexts under a distance threshold.

    The public knobs deliberately echo ADK's `VertexAiRagRetrieval`:
    `similarity_top_k` and `vector_distance_threshold` (a cosine distance, so a
    chunk is kept when `similarity >= 1 - threshold`).
    """

    # With real embeddings a cosine distance of ~0.45 already means "clearly
    # related"; the deterministic offline embedder produces much sparser
    # vectors, so its default is looser. Callers can override either.
    DEFAULT_DISTANCE = {"openai": 0.45, "hash": 0.80}

    def __init__(self, sources=None, *, embedder=None, backend: str = "hash",
                 llm_ok: bool = False, similarity_top_k: int = 5,
                 vector_distance_threshold: float | None = None, max_chunk_chars: int = 320):
        self.sources = list(sources or [])
        self.similarity_top_k = similarity_top_k
        self.max_chunk_chars = max_chunk_chars

        self._chunks: list[tuple[str, str]] = []
        for source in self.sources:
            for chunk in chunk_text(source.text, max_chunk_chars):
                self._chunks.append((source.id, chunk))

        texts = [c[1] for c in self._chunks]
        self._embedder, self._vectors, self.resolved_backend = self._index(
            embedder or build_embedder(backend, llm_ok=llm_ok), texts
        )

        if vector_distance_threshold is None:
            family = "openai" if "openai" in self.resolved_backend else "hash"
            vector_distance_threshold = self.DEFAULT_DISTANCE[family]
        self.vector_distance_threshold = vector_distance_threshold

    def _index(self, embedder, texts: list[str]):
        candidates = [embedder, HashingEmbedder()]
        for candidate in candidates:
            try:
                if hasattr(candidate, "fit"):
                    candidate.fit(texts)
                vectors = candidate.embed(texts) if texts else []
                return candidate, vectors, getattr(candidate, "name", "unknown")
            except Exception:
                continue  # fall back to the offline embedder
        return HashingEmbedder(), [], "hashing-char-ngram"

    def retrieve(self, query: str, *, similarity_top_k: int | None = None,
                 vector_distance_threshold: float | None = None) -> list[RetrievedChunk]:
        """Return the most similar chunks, best chunk per source, above threshold."""
        if not self._vectors or not query:
            return []
        top_k = similarity_top_k or self.similarity_top_k
        threshold = (self.vector_distance_threshold if vector_distance_threshold is None
                     else vector_distance_threshold)

        query_vec = self._embedder.embed([query])[0]
        scored = [
            RetrievedChunk(sid, text, round(cosine(query_vec, vec), 4),
                           round(1.0 - cosine(query_vec, vec), 4))
            for (sid, text), vec in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda c: c.similarity, reverse=True)

        out: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in scored:
            if chunk.distance > threshold:
                break
            if chunk.source_id in seen:
                continue
            seen.add(chunk.source_id)
            out.append(chunk)
            if len(out) >= top_k:
                break
        return out

    def similarity_map(self, query: str) -> dict[str, float]:
        """Best similarity per source across all chunks (for transparency)."""
        if not self._vectors or not query:
            return {}
        query_vec = self._embedder.embed([query])[0]
        best: dict[str, float] = {}
        for (sid, _text), vec in zip(self._chunks, self._vectors):
            sim = cosine(query_vec, vec)
            if sim > best.get(sid, -1.0):
                best[sid] = sim
        return best