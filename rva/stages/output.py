"""Stage 3 - transparent, layered output.

Turns verifications into a report with credibility scores, explicit
keep/demote/drop/contextualize criteria, boundary conditions, and a
contestable three-layer evidence view.
"""

from __future__ import annotations

from ..models import (
    Claim,
    ClaimType,
    Criterion,
    Decision,
    Layers,
    Report,
    ReportItem,
    ReportSummary,
    Verification,
)

RELIABILITY_THRESHOLD = 0.7

CRITERIA: dict[str, list[Criterion]] = {
    "keep": [
        Criterion(id="K1", rule="claim is supported by >=1 source with reliability >= 0.7"),
        Criterion(id="K2", rule="support is not dominated by a high-traffic / low-reliability signal"),
        Criterion(id="K3", rule="no contradicting source of equal or higher reliability"),
        Criterion(id="K4", rule="reasoning: premises are individually supported by reliable evidence"),
    ],
    "demote": [
        Criterion(id="D1", rule="claim is an opinion or unsupported inference"),
        Criterion(id="D2", rule="only weak/partial coverage from low-reliability sources"),
        Criterion(id="D3", rule="support rests on popularity rather than reliability"),
        Criterion(id="D4", rule="reasoning: at least one premise lacks reliable support"),
        Criterion(id="D5", rule="claim does not react to the question asked (off-topic relative to the question)"),
    ],
    "drop": [
        Criterion(id="X1", rule="contradicted by a higher-reliability source"),
        Criterion(id="X2", rule="no supporting evidence at all for a verifiable factual claim"),
    ],
    "contextualize": [
        Criterion(id="N1", rule="subjective/opinion claim: not asserted as an established fact"),
        Criterion(id="N2", rule="opinion is presented to the user with its subjective status made explicit"),
    ],
}


def _decide(v: Verification) -> tuple[Decision, list[str], str]:
    evidence = v.evidence
    strong = [e for e in evidence.supports if e.reliability >= RELIABILITY_THRESHOLD]
    strong_contra = [e for e in evidence.contradicts if e.reliability >= RELIABILITY_THRESHOLD]

    if v.claim_type is ClaimType.OPINION:
        return Decision.CONTEXTUALIZE, ["N1", "N2"], "subjective opinion, flagged as non-factual"

    if v.off_topic:
        # Anchored on the question: a claim that ignores the question is not
        # evidence for the answer, however well-sourced it may be.
        return Decision.DEMOTE, ["D5"], "off-topic: does not react to the question asked"

    if strong_contra:
        return Decision.DROP, ["X1"], "contradicted by higher-reliability source"

    if v.claim_type is ClaimType.REASONING:
        if strong and not v.popularity_bias.biased:
            return Decision.KEEP, ["K4", "K2"], "reasoning with grounded premises"
        return (Decision.DEMOTE, ["D3"] if v.popularity_bias.biased else ["D4"],
                "reasoning with unverified premises")

    if not evidence.supports and v.claim_type is ClaimType.FACT:
        if v.popularity_bias.biased or evidence.partials:
            return Decision.DEMOTE, ["D3"], "factual claim without reliable support"
        return Decision.DROP, ["X2"], "no supporting evidence"

    if v.popularity_bias.biased:
        reasons = ["D3"] + (["D1"] if v.claim_type is ClaimType.INFERENCE else [])
        return Decision.DEMOTE, reasons, "kept with caution: popularity-driven support"

    if strong:
        reasons = []
        if not v.conflict:
            reasons.append("K1")
        if not v.popularity_bias.biased:
            reasons.append("K2")
        if not v.conflict:
            reasons.append("K3")
        return Decision.KEEP, reasons, "well supported"

    if evidence.partials or evidence.supports:
        reasons = ["D2"] + (["D1"] if v.claim_type is ClaimType.INFERENCE else [])
        return Decision.DEMOTE, reasons, "weak / partial support"

    return Decision.DROP, ["X2"], "no reliable evidence"


def _credibility(v: Verification) -> float:
    score = v.support_score
    score += min(0.15, 0.05 * v.independent_domains)
    score -= 0.25 * len(v.evidence.contradicts)
    if v.popularity_bias.biased:
        score -= 0.15
    if v.off_topic:
        score -= 0.2
    if v.claim_type is ClaimType.OPINION:
        score = min(score, 0.15)
    if v.claim_type is ClaimType.INFERENCE:
        score *= 0.85
    if v.claim_type is ClaimType.REASONING:
        score *= 0.9
    if not v.evidence.supports:
        score = min(score, 0.1)
    return max(0.0, min(1.0, round(score, 2)))


def _boundary_conditions(v: Verification) -> list[str]:
    out: list[str] = []
    evidence = v.evidence
    if v.claim_type is ClaimType.OPINION:
        out.append("This is a subjective statement; it is not a verifiable fact and should not be treated as knowledge.")
    if v.claim_type is ClaimType.INFERENCE:
        out.append("This is an inference, not a verified fact; it holds only if its premises hold.")
    if v.claim_type is ClaimType.REASONING:
        out.append("Logical validity does not guarantee true premises; conclusion depends on the premises.")
    if not evidence.supports and v.claim_type is not ClaimType.OPINION:
        out.append("No source in the retrieved set supports this claim.")
    if evidence.partials:
        out.append("Some sources only partially cover the claim; scope may be narrower than stated.")
    if evidence.contradicts:
        out.append("At least one source conflicts with this claim.")
    if v.independent_domains < 2 and evidence.supports:
        out.append("Corroboration comes from a single domain; independence is limited.")
    if v.popularity_bias.biased:
        out.append("Support correlates with popularity (traffic), not necessarily accuracy.")
    if v.off_topic:
        out.append("This claim does not react to the question asked; it may answer a different question.")
    out.append("Credibility reflects only the retrieved evidence set, not the entire literature.")
    return out


def _lookup(ids: list[str]) -> list[Criterion]:
    all_criteria = [c for group in CRITERIA.values() for c in group]
    by_id = {c.id: c for c in all_criteria}
    return [by_id.get(i, Criterion(id=i, rule=i)) for i in ids]


def _efficacy_notes(v: Verification) -> list[str]:
    """Human-readable notes from the per-claim evidence-efficacy checks."""
    notes: list[str] = []
    if v.source_quality.grade.value == "self_media":
        notes.append("best bound source is self-media / low-credibility")
    if v.freshness.dynamic:
        notes.append("stale: " + v.freshness.note if v.freshness.stale else "freshness: " + v.freshness.note)
    if not v.consistency.consistent:
        notes.append("consistency: " + v.consistency.note)
    if v.popularity_suspect_source_ids:
        notes.append("traffic bias suspect: " + ", ".join(v.popularity_suspect_source_ids))
    return notes


def render(claims: list[Claim], verifications: list[Verification],
           supplemented_text: str | None = None) -> Report:
    by_id = {c.id: c for c in claims}
    items: list[ReportItem] = []

    for v in verifications:
        claim = by_id.get(v.claim_id)
        decision, reasons, label = _decide(v)
        items.append(
            ReportItem(
                claim_id=v.claim_id,
                text=v.claim_text,
                type=v.claim_type,
                credibility=_credibility(v),
                decision=decision,
                decision_label=label,
                criteria=_lookup(reasons),
                boundary_conditions=_boundary_conditions(v),
                layers=Layers(
                    l1_primary=[e for e in v.evidence.supports if e.reliability >= RELIABILITY_THRESHOLD],
                    l2_secondary=list(v.evidence.partials),
                    l3_conflicting=list(v.evidence.contradicts),
                ),
                popularity_bias=v.popularity_bias,
                decomposition={
                    "cues": [c.model_dump() for c in claim.cues] if claim else [],
                    "score_breakdown": claim.score_breakdown if claim else {},
                    "classification_confidence": claim.confidence if claim else None,
                    "question_relevance": v.question_relevance,
                    "off_topic": v.off_topic,
                    "source_quality": v.source_quality.grade.value,
                    "freshness_stale": v.freshness.stale,
                    "support_score": v.support_score,
                    "consistency_ok": v.consistency.consistent,
                    "popularity_suspects": v.popularity_suspect_source_ids,
                    "efficacy_notes": _efficacy_notes(v),
                },
            )
        )

    summary = ReportSummary(
        total=len(items),
        kept=sum(1 for i in items if i.decision is Decision.KEEP),
        demoted=sum(1 for i in items if i.decision is Decision.DEMOTE),
        dropped=sum(1 for i in items if i.decision is Decision.DROP),
        contextualized=sum(1 for i in items if i.decision is Decision.CONTEXTUALIZE),
        avg_credibility=round(sum(i.credibility for i in items) / len(items), 2) if items else 0.0,
    )
    return Report(summary=summary, items=items, criteria=CRITERIA,
                  supplemented_text=supplemented_text)


def format_text(report: Report) -> str:
    s = report.summary
    lines = ["=== Response Validation Report ==="]
    lines.append(
        f"Claims: {s.total} | keep {s.kept} | demote {s.demoted} | "
        f"drop {s.dropped} | contextualize {s.contextualized} | avg credibility {s.avg_credibility}"
    )
    lines.append("")
    for it in report.items:
        lines.append(
            f"[{it.claim_id}] ({it.type.value}, cred {it.credibility}) "
            f"{it.decision.value.upper()} - {it.decision_label}"
        )
        lines.append(f"    {it.text}")
        if it.layers.l1_primary:
            lines.append("    L1 primary evidence: " + ", ".join(e.domain or e.title or e.source_id for e in it.layers.l1_primary))
        if it.layers.l2_secondary:
            lines.append("    L2 partial: " + ", ".join(e.domain or e.title or e.source_id for e in it.layers.l2_secondary))
        if it.layers.l3_conflicting:
            lines.append("    L3 conflicting: " + ", ".join(e.domain or e.title or e.source_id for e in it.layers.l3_conflicting))
        lines.append(
            f"    efficacy: quality {it.decomposition.get('source_quality', '-')} | "
            f"freshness {'stale' if it.decomposition.get('freshness_stale') else 'ok'} | "
            f"support {it.decomposition.get('support_score', '-')}"
        )
        for c in it.decomposition.get("efficacy_notes", []):
            lines.append(f"    efficacy note: {c}")
        lines.append("    criteria: " + ", ".join(c.id for c in it.criteria))
        for b in it.boundary_conditions:
            lines.append(f"    boundary: {b}")
        lines.append("")
    return "\n".join(lines)