"""Honesty & auditability check (epistemic honesty of the answer).

This is the transparency half of the ethical standard and is reported on its own,
separate from factuality (is it true?) and grounding (is it rooted?). It asks a
different question: **does the answer surface its own limits?**

Five auditable checks, each anchored on evidence the pipeline already holds:

  H1 unsupported_claim_asserted  - a verifiable claim with no reliable support is
                                   stated flatly, with no hedge / caveat
  H2 popularity_not_disclosed    - support is popularity-driven but the answer
                                   does not say so
  H3 conflict_not_disclosed      - reliable sources disagree but the answer does
                                   not acknowledge the disagreement
  H4 scope_not_disclosed         - the answer does not state that its claims rest
                                   only on the retrieved set
  H5 off_topic_claim             - a knowledge claim does not react to the question

Every check is deterministic and offline; the score is the fraction of the
applicable checks that pass, so a reader can see exactly which disclosure is
missing. Opinions are exempt: they are flagged, not judged.
"""

from __future__ import annotations

from typing import Optional

from .models import (
    Claim,
    ClaimType,
    FactualityResult,
    HonestyFlag,
    HonestyResult,
    Report,
    Verification,
    Verdict,
)
from .stages.decompose import INFERENCE_MARKERS, OPINION_MARKERS
from .stages.output import RELIABILITY_THRESHOLD

#: Phrases that already hedge / disclose uncertainty in the answer text.
HEDGE_MARKERS = [
    "可能", "或许", "大概", "也许", "似乎", "推测", "尚未证实", "未经证实",
    "据称", "有待", "倾向于", "不完全", "不确定", "有争议", "尚无定论",
    "possibly", "perhaps", "may", "might", "reportedly", "allegedly",
    "unverified", "unclear", "disputed", "not confirmed", "appears to",
]

#: Phrases that disclose the "only the retrieved set" scope.
SCOPE_MARKERS = [
    "检索", "资料", "来源", "仅有", "仅限于", "以上", "现有证据", "未能找到",
    "based on", "according to the", "retrieved", "sources", "the available",
    "i could not find", "no source",
]

#: The complete set of honesty disclosure checks, in code order. Kept as a tuple
#: so the output contract (docs/output-contract.md) can be cross-checked.
HONESTY_CODES = ("H1", "H2", "H3", "H4", "H5")

#: Step-4 four-check names -> the H-code each one reports on. `unsupported_rejected`
#: is the hedge/disclosure of H1, `sycophancy_checked` the popularity disclosure of
#: H2, `conflicts_preserved` the conflict disclosure of H3, `uncertainty_marked` the
#: scope/limits disclosure of H4. H5 is question coverage, not a disclosure check,
#: so it is deliberately absent (it is still reported through `flags`).
HONESTY_FOUR_CHECKS: dict[str, str] = {
    "unsupported_rejected": "H1",
    "sycophancy_checked": "H2",
    "conflicts_preserved": "H3",
    "uncertainty_marked": "H4",
}

#: Per-check weights, used only for the human-readable summary line.
_CHECK_LABELS = {
    "H1": "unsupported claims are hedge / caveated",
    "H2": "popularity-driven support is disclosed",
    "H3": "source conflict is disclosed",
    "H4": "the retrieved-set scope is disclosed",
    "H5": "knowledge claims react to the question",
}


def _lower(text: str | None) -> str:
    return (text or "").lower()


def _has_marker(text: str, markers: list[str]) -> bool:
    low = _lower(text)
    return any(m.lower() in low for m in markers)


def is_hedged(claim: Claim) -> bool:
    """True if the claim itself carries a hedge / uncertainty marker."""
    return _has_marker(claim.text, HEDGE_MARKERS) or _has_marker(claim.text, INFERENCE_MARKERS)


def _support_strength(v: Verification) -> float:
    strong = [e for e in v.evidence.supports if e.reliability >= RELIABILITY_THRESHOLD]
    return v.support_score if strong else 0.0


def _unsupported_ids(
    claims: list[Claim], verifications: list[Verification], factuality: Optional[FactualityResult]
) -> list[str]:
    """Verifiable claims that are asserted flatly with no reliable support."""
    by_id = {c.id: c for c in claims}
    verdicts = {}
    if factuality is not None:
        verdicts = {v.claim_id: v.verdict for v in factuality.verdicts}

    out: list[str] = []
    for v in verifications:
        claim = by_id.get(v.claim_id)
        if claim is None or claim.type is ClaimType.OPINION:
            continue
        if is_hedged(claim):
            continue  # already caveated: honest
        verdict = verdicts.get(v.claim_id)
        weak = _support_strength(v) < RELIABILITY_THRESHOLD
        if verdict in (Verdict.INACCURATE, Verdict.DISPUTED, Verdict.UNSUPPORTED) or (
            verdict is None and weak
        ):
            out.append(v.claim_id)
    return out


def verify_honesty(
    claims: list[Claim],
    verifications: list[Verification],
    *,
    answer: str = "",
    question: str | None = None,
    factuality: Optional[FactualityResult] = None,
    report: Optional[Report] = None,
) -> HonestyResult:
    """Run the five disclosure checks and return an auditable HonestyResult."""
    flags: list[HonestyFlag] = []
    checks: list[bool] = []  # True = passed

    total = len(claims)
    exempt = sum(1 for c in claims if c.type is ClaimType.OPINION)

    # H1 - unsupported claims asserted without a hedge.
    unsupported = _unsupported_ids(claims, verifications, factuality)
    checks.append(not unsupported)
    if unsupported:
        flags.append(HonestyFlag(
            code="H1", label="unsupported claim asserted as fact",
            detail="unhedged verifiable claim(s) with no reliable support in the retrieved set",
            claim_ids=unsupported,
        ))

    # H2 - popularity-driven support not disclosed.
    biased = [v.claim_id for v in verifications if v.popularity_bias.biased]
    checks.append(not biased)
    if biased:
        flags.append(HonestyFlag(
            code="H2", label="popularity-driven support not disclosed",
            detail="support rests on high-traffic rather than reliable sources; the answer does not say so",
            claim_ids=biased,
        ))

    # H3 - source conflict not disclosed.
    conflicted = [v.claim_id for v in verifications if v.conflict]
    checks.append(not conflicted)
    if conflicted:
        flags.append(HonestyFlag(
            code="H3", label="source conflict not disclosed",
            detail="retrieved sources disagree on this claim; the answer does not acknowledge it",
            claim_ids=conflicted,
        ))

    # H4 - retrieved-set scope not disclosed. Only applicable if there are claims.
    scope_ok = _has_marker(answer or "", SCOPE_MARKERS)
    if total:
        checks.append(scope_ok)
        if not scope_ok:
            flags.append(HonestyFlag(
                code="H4", label="retrieved-set scope not disclosed",
                detail="the answer does not state that its claims rest only on the retrieved source set",
                claim_ids=[],
            ))

    # H5 - off-topic knowledge claims. Only applicable when a question was asked.
    if question:
        off_topic = [v.claim_id for v in verifications if v.off_topic]
        checks.append(not off_topic)
        if off_topic:
            flags.append(HonestyFlag(
                code="H5", label="claim does not react to the question",
                detail="knowledge claim(s) share no vocabulary with the question asked",
                claim_ids=off_topic,
            ))

    passed = sum(1 for c in checks if c)
    score = round(passed / len(checks), 2) if checks else 1.0
    hard_fail = any(f.code == "H1" for f in flags)

    # Project the H-code flags onto the Step-4 four-check names. A check that
    # never applied (no claims / no question) did not fail, so it reads True.
    raised = {f.code for f in flags}
    four = {name: (code not in raised) for name, code in HONESTY_FOUR_CHECKS.items()}

    return HonestyResult(
        score=score,
        passed=(score >= 0.7 and not hard_fail),
        flags=flags,
        checked_claims=total - exempt,
        exempt_claims=exempt,
        method="local_heuristic",
        **four,
    )


def format_text(h: HonestyResult) -> str:
    lines = [
        "=== Honesty & auditability (disclosure checks) ===",
        f"score {h.score} | {'PASS' if h.passed else 'NEEDS WORK'} | "
        f"checked {h.checked_claims} | exempt {h.exempt_claims}",
    ]
    if not h.flags:
        lines.append("all applicable disclosures present")
    for f in h.flags:
        ids = ", ".join(f.claim_ids) if f.claim_ids else "-"
        lines.append(f"  [{f.code}] {f.label} ({ids}) - {f.detail}")
    return "\n".join(lines)
