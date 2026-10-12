"""Per-claim evidence-efficacy checks (Stage 2 extension).

For every claim, four checks run against its bound candidate sources:

  1. **source authority rating** (`assess_source_quality`) - grade each bound
     source: official/authoritative, professional media, self-media/low-
     credibility, or unverifiable (来源权威性评级).
  2. **freshness** (`assess_freshness`) - is the material current? A claim about a
     dynamically-changing fact (policy, price, event progress) records the
     check time and is flagged `stale` when no recent date can be established.
  3. **consistency** (`assess_consistency`) - compare the claim's numbers and
     qualifiers against its supporting sources, to catch tampering, exaggeration,
     or cherry-picking (数字被篡改 / 限定词被夸大 / 断章取义).
  4. **traffic vs validity** (`detect_popularity_bias`, in verify.py) - a
     high-traffic, low-reliability document ranked into the support set is
     flagged as a popularity-bias suspect and never credited on popularity.

The three per-claim outputs are `source_quality`, `freshness`, and
`support_score` (authority-weighted support, discounted by inconsistency and by
popularity-driven support). All checks are offline and deterministic.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from .models import (
    AuthorityGrade,
    Consistency,
    Evidence,
    Freshness,
    Source,
    SourceGrade,
    SourceQuality,
)

#: How many years old a date may be before a dynamic claim counts as stale.
STALE_YEARS = 2

#: tier (from DEFAULT_REGISTRY) -> authority grade.
TIER_TO_GRADE: dict[str, AuthorityGrade] = {
    "primary-official": AuthorityGrade.OFFICIAL,
    "peer-reviewed": AuthorityGrade.PROFESSIONAL,
    "academic": AuthorityGrade.PROFESSIONAL,
    "wire": AuthorityGrade.PROFESSIONAL,
    "preprint": AuthorityGrade.SELF_MEDIA,
    "encyclopedic": AuthorityGrade.SELF_MEDIA,
    "blog": AuthorityGrade.SELF_MEDIA,
    "ugc": AuthorityGrade.SELF_MEDIA,
    "unknown": AuthorityGrade.UNVERIFIABLE,
}

#: Higher = more authoritative; used to pick the best grade for a claim.
GRADE_RANK: dict[AuthorityGrade, int] = {
    AuthorityGrade.OFFICIAL: 3,
    AuthorityGrade.PROFESSIONAL: 2,
    AuthorityGrade.SELF_MEDIA: 1,
    AuthorityGrade.UNVERIFIABLE: 0,
}

#: Topics/facts that change over time; a claim touching one needs a check time.
DYNAMIC_MARKERS = [
    "政策", "法规", "法律", "价格", "汇率", "利率", "通胀", "补贴", "关税",
    "选举", "事件", "进展", "最新", "目前", "当前", "今年", "本月", "本周", "近日", "现在",
    "policy", "price", "rate", "latest", "current", "currently", "this year",
    "today", "breaking", "update", "progress", "ongoing", "as of",
]

_YEAR = re.compile(r"(1[5-9]\d{2}|20\d{2})")
_NUM = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|percent(?:age)?|亿|万|million|billion|倍)?", re.I)

#: Quantifier strength, strongest first. A claim that strengthens a qualifier
#: relative to its evidence has "overreached" (夸大).
_QUALIFIERS: list[tuple[str, int]] = [
    ("全部", 4), ("所有", 4), ("一律", 4), ("一定", 4), ("必然", 4), ("肯定", 4), ("all", 4), ("every", 4), ("always", 4),
    ("大多数", 3), ("多数", 3), ("绝大部分", 3), ("都", 3), ("总是", 3), ("most", 3),
    ("通常", 2), ("一般", 2), ("部分", 2), ("有些", 2), ("少数", 2), ("往往", 2), ("usually", 2), ("some", 2),
    ("可能", 1), ("或许", 1), ("也许", 1), ("据称", 1), ("疑似", 1), ("推测", 1), ("大概", 1),
    ("may", 1), ("might", 1), ("reportedly", 1), ("perhaps", 1),
]


def grade_for_tier(tier: str) -> AuthorityGrade:
    """Map a reliability tier onto one of the four authority grades."""
    return TIER_TO_GRADE.get(tier or "", AuthorityGrade.UNVERIFIABLE)


def _norm_number(token: str) -> str:
    return re.sub(r"\s+", "", token).lower()


def _numbers(text: str) -> set[str]:
    return {_norm_number(m) for m in _NUM.findall(text or "") if m.strip()}


def _qualifier_strength(text: str) -> int:
    low = (text or "").lower()
    return max((strength for marker, strength in _QUALIFIERS if marker in low), default=0)


def assess_source_quality(evidence: Evidence, sources: list[Source]) -> SourceQuality:
    """Grade every bound candidate source and pick the claim's best grade."""
    bound = evidence.supports + evidence.partials + evidence.contradicts
    best_by_id: dict[str, object] = {}
    for ref in bound + evidence.screened:
        prev = best_by_id.get(ref.source_id)
        if prev is None or ref.reliability > prev.reliability:
            best_by_id[ref.source_id] = ref

    graded = [
        SourceGrade(
            source_id=ref.source_id, grade=grade_for_tier(ref.tier),
            reliability=ref.reliability, tier=ref.tier,
            note=f"{ref.domain or ref.source_id} ({ref.tier})",
        )
        for ref in best_by_id.values()
    ]
    graded.sort(key=lambda g: (-GRADE_RANK[g.grade], -g.reliability))

    counts: dict[str, int] = {}
    for g in graded:
        counts[g.grade.value] = counts.get(g.grade.value, 0) + 1

    # Overall grade = best grade among the sources actually arguing about the
    # claim (not ones screened out for irrelevance).
    overall = AuthorityGrade.UNVERIFIABLE
    for ref in bound:
        grade = grade_for_tier(ref.tier)
        if GRADE_RANK[grade] > GRADE_RANK[overall]:
            overall = grade

    if not graded:
        note = "no candidate source is bound to this claim"
    elif overall is AuthorityGrade.UNVERIFIABLE:
        note = "bound sources have no recognized authority tier"
    elif overall is AuthorityGrade.SELF_MEDIA:
        note = "best bound source is self-media / low-credibility"
    else:
        note = f"best bound source rated {overall.value}"
    return SourceQuality(grade=overall, counts=counts, sources=graded, note=note)


def assess_freshness(
    claim_text: str,
    evidence: Evidence,
    sources: list[Source],
    *,
    now: Optional[datetime] = None,
) -> Freshness:
    """Flag a dynamic claim whose material has no recent date (see module doc)."""
    now = now or datetime.now(timezone.utc)
    checked_at = now.isoformat()

    bound_ids = {r.source_id for r in evidence.supports + evidence.partials}
    texts = [claim_text or ""]
    for source in sources or []:
        if source.id in bound_ids:
            texts.append(source.text or "")
            if source.published:
                texts.append(source.published)

    dynamic = any(marker in (claim_text or "").lower() for marker in DYNAMIC_MARKERS) \
        or bool(_YEAR.search(claim_text or ""))

    years = [int(y) for y in _YEAR.findall("\n".join(texts))]
    newest = max(years) if years else None
    has_date = newest is not None
    stale = dynamic and (not has_date or newest < now.year - STALE_YEARS)

    if not dynamic:
        note = "not a time-sensitive claim; freshness not enforced"
    elif stale and not has_date:
        note = f"dynamic fact with no dated material; check time recorded ({checked_at})"
    elif stale:
        note = f"material dated {newest} looks outdated for a dynamic fact; check time {checked_at}"
    else:
        note = f"dynamic fact; material dated {newest}; check time {checked_at}"

    return Freshness(
        checked_at=checked_at, dynamic=dynamic, has_date=has_date,
        newest_date=str(newest) if newest else None, stale=stale, note=note,
    )


def assess_consistency(
    claim_text: str, evidence: Evidence, sources: list[Source]
) -> Consistency:
    """Compare the claim's numbers / qualifiers against its supporting sources."""
    support_ids = {r.source_id for r in evidence.supports}
    support_text = "\n".join(
        (s.text or "") for s in (sources or []) if s.id in support_ids
    )

    if not support_ids:
        return Consistency(
            consistent=True, claim_numbers=sorted(_numbers(claim_text)),
            note="no supporting source to compare against",
        )

    claim_nums = _numbers(claim_text)
    source_nums = _numbers(support_text)
    # A number in the claim that appears in none of its supporting sources is a
    # possible fabrication / substitution (数字被篡改).
    unmatched = sorted(claim_nums - source_nums)
    number_mismatch = bool(unmatched)

    claim_q = _qualifier_strength(claim_text)
    source_q = _qualifier_strength(support_text)
    # A stronger claim than its evidence supports (e.g. source "part", claim
    # "all") is an overreach (限定词被夸大 / 断章取义).
    qualifier_overreach = claim_q >= 3 and claim_q > source_q

    if number_mismatch and qualifier_overreach:
        note = f"number(s) {unmatched} absent from sources; qualifier strengthened vs sources"
    elif number_mismatch:
        note = f"number(s) {unmatched} absent from the supporting sources"
    elif qualifier_overreach:
        note = f"claim qualifier (strength {claim_q}) stronger than sources (strength {source_q})"
    else:
        note = "numbers and qualifiers consistent with the supporting sources"

    return Consistency(
        consistent=not (number_mismatch or qualifier_overreach),
        number_mismatch=number_mismatch, qualifier_overreach=qualifier_overreach,
        claim_numbers=sorted(claim_nums), source_numbers=sorted(source_nums), note=note,
    )


def score_support(
    evidence: Evidence, consistency: Consistency, popularity_biased: bool
) -> float:
    """Authority-weighted support, discounted for inconsistency / popularity.

    Base = mean reliability of the supporting sources (as before), so a claim
    with no inconsistency and no popularity bias scores exactly as it used to.
    """
    weights = [e.reliability for e in evidence.supports]
    base = sum(weights) / len(weights) if weights else 0.0
    factor = 1.0
    if consistency.number_mismatch or consistency.qualifier_overreach:
        factor *= 0.7
    if popularity_biased:
        factor *= 0.85
    return round(max(0.0, min(1.0, base * factor)), 2)


def suspicious_traffic_ids(evidence: Evidence) -> list[str]:
    """Sources in the support set that are high-traffic but low-reliability."""
    return sorted({
        r.source_id for r in evidence.supports
        if r.traffic is not None and r.traffic > 0.6 and r.reliability < 0.7
    })
