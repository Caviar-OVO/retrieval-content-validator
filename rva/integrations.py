"""Optional LLM-judge integrations: Ragas and TruLens *grounding* backends.

These frameworks answer one question - "is this claim entailed by the supplied
contexts?" - which is ADK-style **internal faithfulness (grounding)**, NOT the
external fact validity that RVA's five-level verdict measures. So they are wired
to the **grounding** hook and reported separately:

    from rva.integrations import build_judge
    from rva.evaluate import evaluate_grounding

    result = evaluate_grounding(claims, verifications, batch_judge=build_judge("ragas", sources))

A judge returns `{claim_id: record}`, where `record` is a positional
`(grounded, score, rationale)` sequence, an equivalent mapping, or an object
with those attributes. It can never change a factuality verdict; an
unrecognisable record raises `ValueError` instead of looking like "no verdict".

These are heavyweight, network-dependent dependencies, so they are imported
lazily and kept out of the core pipeline (`evaluate.py` is fully offline). If a
backend or API key is unavailable, adapters return None so the caller can report
"no grounding verdict" instead of crashing.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

from .config import get_base_url, get_judge_model
from .models import Claim, Verification


# --- helpers ---------------------------------------------------------------

def _contexts_for(v: Verification, sources: dict[str, str] | None = None) -> list[str]:
    """Raw text of the evidence offered for a claim (supports + partials + contradicts)."""
    refs = v.evidence.supports + v.evidence.partials + v.evidence.contradicts
    out = []
    for r in refs:
        text = (sources or {}).get(r.source_id)
        if text:
            out.append(text)
    return out


def _finite(value) -> Optional[float]:
    """Return the value as a finite float in [0, 1], or None if unusable.

    Ragas/TruLens can emit NaN when a judge response fails to parse; such scores
    must be treated as 'no verdict' so the caller falls back to the local check
    instead of poisoning the report with NaN.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return max(0.0, min(1.0, f))


#: A judge score at or above this means the claim is grounded in its contexts.
#: NOTE this is a *grounding* cut-off, not a factuality verdict: RVA's five-level
#: verdict is decided by source reliability in `evaluate.py`, never by this score.
GROUNDED_THRESHOLD = 0.75


def is_grounded(score: Optional[float]) -> Optional[bool]:
    """Classify a grounding score; None in, None out (no verdict)."""
    if score is None:
        return None
    return score >= GROUNDED_THRESHOLD


def _source_texts(sources) -> dict[str, str]:
    return {s.id: (s.text or "") for s in (sources or [])}


def availability() -> dict:
    """Report which judge backends can actually be used right now."""
    from .config import get_api_key

    info: dict = {"openai_api_key": bool(get_api_key())}
    from .config import get_embed_model, get_nli_backend, get_retriever_backend

    info["retriever_backend"] = get_retriever_backend()
    info["nli_backend"] = get_nli_backend()
    info["embed_model"] = get_embed_model()
    for name in ("openai", "numpy", "ragas", "trulens", "trulens.providers.openai", "langchain_openai", "datasets"):
        try:
            __import__(name)
            info[name] = True
        except Exception as exc:  # pragma: no cover - environment dependent
            info[name] = f"unavailable: {type(exc).__name__}"
    return info


# --- Ragas -----------------------------------------------------------------

def _ragas_llm(llm=None):
    if llm is not None:
        return llm
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    kwargs = {"model": get_judge_model(), "temperature": 0.0}
    base_url = get_base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return LangchainLLMWrapper(ChatOpenAI(**kwargs))


def ragas_batch_judge(sources=None, llm=None) -> Callable:
    """Batch grounding judge backed by Ragas' Faithfulness metric.

    Builds one EvaluationDataset over all knowledge claims (response = claim,
    retrieved_contexts = its evidence) and runs Ragas once. Returns
    {claim_id: (grounded: bool, score: float, rationale)} - a *grounding* verdict,
    never a factuality one. Claims with no context are reported ungrounded
    without an LLM call.
    """
    source_map = _source_texts(sources)

    def batch(claims: list[Claim], verifications: list[Verification]) -> Optional[dict]:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import faithfulness

        from .evaluate import is_verifiable

        out: dict = {}
        samples = []
        pending: list[str] = []

        for v in verifications:
            if not is_verifiable(v.claim_type):
                continue  # knowledge claims only: never spend a judge call on an opinion
            contexts = _contexts_for(v, source_map)
            if not contexts:
                out[v.claim_id] = (False, None, "no retrieved context to ground the claim")
                continue
            samples.append(
                SingleTurnSample(user_input=v.claim_text, retrieved_contexts=contexts, response=v.claim_text)
            )
            pending.append(v.claim_id)

        if not samples:
            return out or None

        try:
            wrapped = _ragas_llm(llm)
            result = evaluate(EvaluationDataset(samples=samples), metrics=[faithfulness], llm=wrapped)
            df = result.to_pandas()
            scores = [float(x) for x in df["faithfulness"].tolist()]
        except Exception:  # pragma: no cover - depends on live API/key
            return None  # no grounding verdict available

        for claim_id, score in zip(pending, scores):
            finite = _finite(score)
            if finite is None:
                continue  # leave this claim without a grounding verdict
            out[claim_id] = (is_grounded(finite), finite, f"ragas faithfulness={finite:.2f}")
        return out or None

    return batch


def ragas_judge(sources=None, llm=None) -> Callable:
    """Single-claim Ragas grounding judge (evaluates one claim at a time)."""
    batch = ragas_batch_judge(sources=sources, llm=llm)

    def judge(claim: Claim, verification: Verification):
        result = batch([claim], [verification])
        if not result:
            return None
        return result.get(claim.claim_id)

    return judge


# --- TruLens ---------------------------------------------------------------

def trulens_batch_judge(sources=None, llm=None) -> Callable:
    """Batch grounding judge backed by TruLens' groundedness feedback.

    TruLens' groundedness_measure_with_cot_reasons(source, statement) returns a
    score on a 0..3 scale plus reasons; we normalize it to 0..1 and report it as
    grounding. Returns {claim_id: (grounded, score, rationale)}, never factuality.
    """
    source_map = _source_texts(sources)

    def batch(claims: list[Claim], verifications: list[Verification]) -> dict:
        try:
            if llm is not None:
                provider = llm
            else:
                from trulens.providers.openai import OpenAI as TruOpenAI

                base_url = get_base_url()
                provider = TruOpenAI(model_engine=get_judge_model(), base_url=base_url) if base_url else TruOpenAI(model_engine=get_judge_model())
        except Exception:  # pragma: no cover
            return None

        from .evaluate import is_verifiable

        out: dict = {}
        for v in verifications:
            if not is_verifiable(v.claim_type):
                continue  # knowledge claims only: never spend a judge call on an opinion
            contexts = _contexts_for(v, source_map)
            if not contexts:
                out[v.claim_id] = (False, None, "no retrieved context to ground the claim")
                continue
            try:
                # TruLens expects a single source string; join the retrieved blocks.
                source_text = "\n\n".join(contexts)
                raw_score, reasons = provider.groundedness_measure_with_cot_reasons(
                    source_text, v.claim_text
                )
                score = _finite(float(raw_score) / 3.0)
            except Exception:  # pragma: no cover - depends on live API/key
                return None  # no grounding verdict available
            if score is None:
                continue
            out[v.claim_id] = (is_grounded(score), score, f"trulens groundedness={score:.2f}")
        return out or None

    return batch


def trulens_judge(llm=None) -> Callable:
    """Single-claim TruLens grounding judge."""
    batch = trulens_batch_judge(llm=llm)

    def judge(claim: Claim, verification: Verification):
        result = batch([claim], [verification])
        if not result:
            return None
        return result.get(claim.claim_id)

    return judge


# --- factory ---------------------------------------------------------------

def build_judge(name: str, sources=None):
    """Return a *grounding* batch-judge callable for `name`, or None for no judge."""
    name = (name or "local").lower()
    if name in ("local", "none", ""):
        return None
    if name == "ragas":
        return ragas_batch_judge(sources=sources)
    if name == "trulens":
        return trulens_batch_judge()
    raise ValueError(f"unknown judge backend: {name!r} (choose local, ragas, or trulens)")