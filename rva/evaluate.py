"""Factuality - five-level, *source-anchored* claim verification.

Each claim in the answer is judged against **external reliable sources**, not
against the LLM's own context and not against its opinion of itself:

    Accurate       - reliable sources confirm the claim
    Inaccurate     - reliable sources contradict the claim
    Disputed       - reliable sources conflict with each other
    Unsupported    - no reliable source substantiates the claim
    Not Applicable - opinion / belief; needs no external verification

Score (source-anchored accuracy):

    score = (n_accurate + n_not_applicable) / n_total_claims

Relation to ADK (deliberate, and NOT an equivalence)
----------------------------------------------------
We borrow the *label names* and the *shape* of the score from ADK's
`hallucinations_v1`, but the **anchor is different and stricter**:

* ADK anchors on the supplied **context / tool outputs** and measures internal
  grounding ("is the answer faithful to what it was given?"). A claim can be
  `supported` there and still be factually false about the world.
* RVA anchors on **external reliable sources** and measures factual validity
  ("is the answer true?"). Reliability tiers are our addition; ADK has no such
  notion, and its `not_applicable` is broader (anything not checkable against
  context) than our narrow "subjective opinion".

Consequences to keep in mind when reading the score:

* An `Unsupported` claim may still be *true* - the reliable source simply was
  not in the retrieved set. This is a coverage limit, not a falsity claim.
* We credit `Not Applicable` as correct, matching ADK's formula; but because our
  exempt set is narrower (opinions only), the two scores are not comparable
  across systems without re-mapping.

In short: same taxonomy, same score shape, **different anchor** - external fact
validity here, internal context grounding in ADK.

This module runs a deterministic, offline, source-anchored check. The five-level
verdict is anchored on **external reliable sources** only and is never delegated
to an LLM judge. Two rules matter:

* **Anchor on sources, not on the model.** The verdict follows the reliability of
  the evidence: a claim is Accurate only when a source above the reliability
  threshold supports it, Inaccurate only when such a source refutes it, and
  Disputed only when both sides clear the threshold.
* **Type filtering.** Only knowledge claims (facts, inferences, and logical
  arguments) are judged. Opinions are resolved to Not Applicable without any
  external call.

Grounding is a *separate* report, not a verdict
------------------------------------------------
Ragas / TruLens measure whether a claim is entailed by the **supplied contexts** -
the ADK-style internal-faithfulness anchor. That is a different question from
factual validity, and the two genuinely disagree: a claim can be perfectly
grounded in a narrow or biased context and still be false about the world. So
`evaluate_grounding` returns its own `GroundingResult` (grounded / ungrounded
plus the raw judge score), reported *alongside* factuality and never folded into
the five-level verdict. The judge adapters therefore live behind the grounding
hook, not behind `evaluate_factuality`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Callable, Optional

from .models import (
    Claim,
    ClaimGrounding,
    ClaimVerdict,
    ClaimType,
    FactualityResult,
    GroundingResult,
    Verification,
    Verdict,
)
from .stages.output import RELIABILITY_THRESHOLD

#: Claim types judged against external sources. The LLM Auditor treats a claim as
#: "a statement of fact about the world OR a logical argument", so reasoning is
#: judged too; only subjective opinion / belief is exempt (Not Applicable).
VERIFIABLE_TYPES = (ClaimType.FACT, ClaimType.INFERENCE, ClaimType.REASONING)

#: Verdict for claims that need no external verification (subjective opinion /
#: belief / fiction). Not Applicable counts as *correct* in the ADK accuracy score,
#: so its score is 1.0: the claim is not wrong, it is simply not a factual
#: assertion.
NOT_APPLICABLE_VERDICT = (
    Verdict.NOT_APPLICABLE,
    1.0,
    "subjective opinion / personal belief: needs no external verification, "
    "so the external judge is not invoked",
)

#: Backwards-compatible alias (the earlier name for the filtered verdict).
FILTERED_VERDICT = NOT_APPLICABLE_VERDICT


#: Canonical score for each verdict: a claim is either correct (confirmed, or
#: exempt as non-factual) or it is not. Kept separate from the judge's raw
#: confidence, which belongs in the rationale.
_VERDICT_SCORES = {
    Verdict.ACCURATE: 1.0,
    Verdict.NOT_APPLICABLE: 1.0,
    Verdict.INACCURATE: 0.0,
    Verdict.DISPUTED: 0.0,
    Verdict.UNSUPPORTED: 0.0,
}


def verdict_score(verdict: Verdict) -> float:
    """Canonical 0/1 score for a verdict, independent of any judge confidence."""
    return _VERDICT_SCORES.get(verdict, 0.0)


def is_verifiable(claim_type: ClaimType) -> bool:
    """True for the claim types that are judged against external sources."""
    return claim_type in VERIFIABLE_TYPES


def _verdict_for(v: Verification) -> tuple[Verdict, float, str]:
    """Offline, source-anchored five-level verdict for one claim."""
    if not is_verifiable(v.claim_type):
        return NOT_APPLICABLE_VERDICT

    evidence = v.evidence
    strong_sup = [e for e in evidence.supports if e.reliability >= RELIABILITY_THRESHOLD]
    strong_con = [e for e in evidence.contradicts if e.reliability >= RELIABILITY_THRESHOLD]

    if strong_sup and strong_con:
        domains = sorted({e.domain for e in strong_sup + strong_con if e.domain})
        return (
            Verdict.DISPUTED,
            0.0,
            "reliable sources conflict on this claim (" + ", ".join(domains) + ")",
        )

    if strong_con:
        domains = sorted({e.domain for e in strong_con if e.domain})
        return (
            Verdict.INACCURATE,
            0.0,
            "contradicted by reliable source(s): " + ", ".join(domains),
        )

    if strong_sup:
        note = "confirmed by reliable source(s)"
        if evidence.contradicts:
            note += " (a lower-reliability source disagrees)"
        if v.popularity_bias.biased:
            note += " but the support is popularity-driven"
        return Verdict.ACCURATE, 1.0, note

    if evidence.supports or evidence.partials:
        return (
            Verdict.UNSUPPORTED,
            0.0,
            "related sources exist but none is reliable enough to substantiate the claim",
        )
    return Verdict.UNSUPPORTED, 0.0, "no reliable source found to substantiate this claim"


def evaluate_factuality(
    claims: list[Claim],
    verifications: list[Verification],
) -> FactualityResult:
    """Five-level, source-anchored verdicts. No LLM judge is involved.

    Every verdict follows the reliability of the retrieved sources (see
    `_verdict_for`); opinions resolve to Not Applicable. The metric can therefore
    only be as external as the evidence pipeline - a claim with no reliable
    source in the retrieved set is Unsupported, which is a coverage limit, not a
    claim of falsity.

    Grounding (is the claim entailed by the supplied contexts?) is a different
    question and is reported separately by `evaluate_grounding`; it is never
    mixed into this verdict.
    """
    by_id = {c.id: c for c in claims}
    verdicts: list[ClaimVerdict] = []
    judged = 0

    for v in verifications:
        claim = by_id.get(v.claim_id)
        if claim is None:
            continue

        if not is_verifiable(v.claim_type):
            verdict, score, rationale = NOT_APPLICABLE_VERDICT
        else:
            verdict, score, rationale = _verdict_for(v)
            judged += 1

        verdicts.append(
            ClaimVerdict(
                claim_id=v.claim_id,
                claim_text=v.claim_text,
                claim_type=v.claim_type,
                verdict=verdict,
                score=round(score, 2),
                evidence_domains=sorted({e.domain for e in v.evidence.supports if e.domain}),
                rationale=rationale,
            )
        )

    total = len(verdicts)
    accurate = sum(1 for x in verdicts if x.verdict is Verdict.ACCURATE)
    inaccurate = sum(1 for x in verdicts if x.verdict is Verdict.INACCURATE)
    disputed = sum(1 for x in verdicts if x.verdict is Verdict.DISPUTED)
    unsupported = sum(1 for x in verdicts if x.verdict is Verdict.UNSUPPORTED)
    not_applicable = sum(1 for x in verdicts if x.verdict is Verdict.NOT_APPLICABLE)

    # ADK Accuracy Score: verified or legitimately exempt, over all claims.
    score = round((accurate + not_applicable) / total, 2) if total else 0.0

    return FactualityResult(
        score=score,
        accurate=accurate,
        inaccurate=inaccurate,
        disputed=disputed,
        unsupported=unsupported,
        not_applicable=not_applicable,
        total_claims=total,
        verifiable_claims=judged,
        applicable_claims=total - not_applicable,
        verdicts=verdicts,
        method="local_source_anchored",
    )


def evaluate_grounding(
    claims: list[Claim],
    verifications: list[Verification],
    *,
    judge: Optional[Callable[[Claim, Verification], Optional[tuple]]] = None,
    batch_judge: Optional[Callable[[list[Claim], list[Verification]], Optional[dict]]] = None,
) -> GroundingResult:
    """Grounding-only report from an external judge (Ragas / TruLens).

    Asks "is the claim entailed by the *supplied contexts*?" - the ADK-style
    internal-faithfulness anchor. Kept strictly separate from factuality:

    * It never reads or writes `FactualityResult`, so a judge score cannot change
      a five-level verdict.
    * Only knowledge claims (fact / inference / reasoning) are judged; opinions
      are skipped without an API call, and any judge output for a non-knowledge
      claim is discarded.

    A judge returns `{claim_id: record}`. Each record may be either a positional
    `(grounded, score, rationale)` sequence, a mapping with those keys, or an
    object exposing them as attributes; the three are equivalent, so a custom
    judge is not tied to one framework's shape. A record in any *other* shape
    raises `ValueError` rather than decaying into a silent "no verdict". This is
    the *only* hook for the Ragas / TruLens adapters.
    """
    by_id = {c.id: c for c in claims}
    targets = [v for v in verifications if is_verifiable(v.claim_type) and v.claim_id in by_id]

    external: dict = {}
    if batch_judge is not None and targets:
        try:
            result = batch_judge([by_id[v.claim_id] for v in targets], targets)
        except Exception:  # pragma: no cover - depends on live backend
            result = None
        if result is not None and not isinstance(result, Mapping):
            raise ValueError(
                "grounding batch judge must return {claim_id: record}, "
                f"got {type(result).__name__}"
            )
        if result:
            external = result

    items: list[ClaimGrounding] = []
    for v in verifications:
        claim = by_id.get(v.claim_id)
        if claim is None:
            continue

        if not is_verifiable(v.claim_type):
            # Never spend a judge call on an opinion; ignore any verdict if present.
            items.append(
                ClaimGrounding(
                    claim_id=v.claim_id, claim_text=v.claim_text, claim_type=v.claim_type,
                    grounded=None, score=None,
                    rationale="non-knowledge claim: grounding judge not invoked",
                )
            )
            continue

        raw = external.get(v.claim_id)
        if raw is None and judge is not None:
            try:
                raw = judge(claim, v)
            except Exception:  # pragma: no cover - depends on live backend
                raw = None

        grounded = None
        score = None
        rationale = "no grounding verdict returned by the judge"
        if raw is not None:
            try:
                grounded, score, rationale = _normalise_grounding(raw)
            except ValueError as exc:
                raise ValueError(
                    f"grounding judge returned an unrecognisable record for "
                    f"claim {v.claim_id!r}: {exc}"
                ) from None

        items.append(
            ClaimGrounding(
                claim_id=v.claim_id, claim_text=v.claim_text, claim_type=v.claim_type,
                grounded=grounded, score=score, rationale=rationale,
            )
        )

    scored = [it.score for it in items if it.score is not None]
    judged = [it for it in items if it.grounded is not None]

    return GroundingResult(
        method="external_judge",
        judged_claims=len(judged),
        grounded_count=sum(1 for it in judged if it.grounded is True),
        ungrounded_count=sum(1 for it in judged if it.grounded is False),
        skipped_claims=sum(1 for it in items if not is_verifiable(it.claim_type)),
        total_claims=len(items),
        mean_score=round(sum(scored) / len(scored), 2) if scored else None,
        items=items,
    )


def _normalise_grounding(raw) -> tuple[Optional[bool], Optional[float], str]:
    """Coerce one judge record into (grounded, score, rationale).

    Three shapes are accepted, and they mean the same thing:

      * a mapping with (optional) ``grounded`` / ``score`` / ``rationale`` keys;
      * an object exposing those as attributes; and
      * a positional sequence ``(grounded, score, rationale)``.

    Anything else - a bare string, a number, an empty record - raises
    ``ValueError``. Silently returning "no verdict" here is the worse failure:
    it looks exactly like a judge that legitimately had nothing to say, so a
    broken adapter would go unnoticed.
    """
    if isinstance(raw, Mapping):
        grounded = raw.get("grounded")
        score = raw.get("score")
        rationale = raw.get("rationale", "")
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        grounded = raw[0] if len(raw) > 0 else None
        score = raw[1] if len(raw) > 1 else None
        rationale = raw[2] if len(raw) > 2 else ""
    elif hasattr(raw, "grounded") or hasattr(raw, "score"):
        grounded = getattr(raw, "grounded", None)
        score = getattr(raw, "score", None)
        rationale = getattr(raw, "rationale", "") or ""
    else:
        raise ValueError(
            "expected a mapping, an object with grounded/score, or a "
            f"(grounded, score, rationale) sequence; got {type(raw).__name__}"
        )

    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        else:
            if not math.isfinite(score):
                score = None
            else:
                score = max(0.0, min(1.0, score))

    if grounded is not None:
        grounded = bool(grounded)

    return grounded, score, str(rationale or "")


def format_text(f: FactualityResult) -> str:
    lines = [
        "=== Factuality: five-level verdicts (source-anchored) ===",
        f"factuality {f.score} | accurate {f.accurate} | inaccurate {f.inaccurate} | "
        f"disputed {f.disputed} | unsupported {f.unsupported} | not_applicable {f.not_applicable}",
        f"claims {f.total_claims} | judged against sources {f.verifiable_claims} | "
        f"exempt {f.not_applicable}",
        f"method: {f.method} (external judges do NOT affect these verdicts)",
        "",
    ]
    for v in f.verdicts:
        lines.append(f"  [{v.claim_id}] {v.verdict.value} ({v.score}) - {v.rationale}")
        lines.append(f"        {v.claim_text}")
    return "\n".join(lines)


# Backwards-compatible alias: the metric was previously called "faithfulness".
evaluate_faithfulness = evaluate_factuality
