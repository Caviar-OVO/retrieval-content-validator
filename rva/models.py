"""Response Validation Agent - structured models (Pydantic v2).

Every intermediate and final artifact in the pipeline is a typed model so that
reasoning is inspectable, serializable, and contestable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# --- Enumerations ----------------------------------------------------------

class ClaimType(str, Enum):
    FACT = "fact"
    REASONING = "reasoning"
    OPINION = "opinion"
    INFERENCE = "inference"


class Decision(str, Enum):
    KEEP = "keep"
    DEMOTE = "demote"
    DROP = "drop"
    CONTEXTUALIZE = "contextualize"


class Verdict(str, Enum):
    """Five-level claim verdict, anchored on external reliable sources.

    Label names follow the ADK `hallucinations_v1` sentence labels (supported /
    unsupported / contradictory / disputed / not_applicable) and the Google LLM
    Auditor verdicts, renamed as requested. Note the *anchor* is not ADK's: ADK
    judges grounding against supplied context, while here every verdict is
    decided by external reliable sources (factual validity, not faithfulness):

        ACCURATE      - reliable sources confirm the claim
        INACCURATE    - reliable sources contradict the claim
        DISPUTED      - reliable sources conflict with each other
        UNSUPPORTED   - no reliable source substantiates the claim
        NOT_APPLICABLE- opinion / belief / fiction; needs no external verification
    """

    ACCURATE = "accurate"
    INACCURATE = "inaccurate"
    DISPUTED = "disputed"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


# --- Inputs ----------------------------------------------------------------

class Source(BaseModel):
    """A retrieved document (search result) offered to the validator."""

    id: str
    text: str = ""
    title: Optional[str] = None
    url: Optional[str] = None
    tag: Optional[str] = None
    domain: Optional[str] = None
    traffic: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    #: Optional publication date (free text) used by the freshness check.
    published: Optional[str] = None


# --- Stage 1: cognitive decomposition -------------------------------------

class Cue(BaseModel):
    """A single auditable signal that influenced a claim's classification."""

    category: ClaimType
    kind: str
    value: Any


class QuestionAnchor(BaseModel):
    """How a claim relates to the user's question.

    The critic extracts claims from a (question, answer) pair, so every claim is
    anchored to the question it is supposed to answer. Relevance is measured by
    lexical overlap with the question. `reactant` is False only when the claim
    shares no question vocabulary at all, which is the dangerous case: the claim
    addresses something else.
    """

    question: str = ""
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    reactant: bool = True
    note: str = ""


class Claim(BaseModel):
    id: str
    text: str
    type: ClaimType
    confidence: float = Field(ge=0.0, le=1.0)
    cues: List[Cue] = Field(default_factory=list)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    note: Optional[str] = None
    verifiable: bool = False
    question_anchor: Optional[QuestionAnchor] = None


class DecompositionStats(BaseModel):
    total: int = 0
    fact: int = 0
    reasoning: int = 0
    opinion: int = 0
    inference: int = 0


# --- Stage 2: evidence & source verification ------------------------------

class ReliabilityProfile(BaseModel):
    tier: str
    score: float = Field(ge=0.0, le=1.0)
    note: str
    matched: str
    domain: str


class EvidenceRef(BaseModel):
    """A retrieved source as it relates to one specific claim."""

    source_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    domain: str = ""
    reliability: float = Field(ge=0.0, le=1.0)
    tier: str = "unknown"
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    shared_tokens: int = 0
    traffic: Optional[float] = None
    negation: Optional[str] = None
    stance: Optional[str] = None

    @field_validator("reliability", "similarity", "coverage", mode="before")
    @classmethod
    def _clamp_unit_interval(cls, value):
        """Clamp [0,1] scores instead of raising on an out-of-range float.

        These fields carry a *graded* similarity/reliability signal; a stray
        negative (anti-correlated embeddings) or >1 micro-float must not abort
        the whole report. Clamping keeps the invariant while letting the
        pipeline stay total. Non-numeric input is left to Pydantic to reject.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
        return value


class Evidence(BaseModel):
    supports: List[EvidenceRef] = Field(default_factory=list)
    partials: List[EvidenceRef] = Field(default_factory=list)
    contradicts: List[EvidenceRef] = Field(default_factory=list)
    # Candidates the vector search ranked below the distance threshold, kept so
    # the screening decision is visible and contestable.
    screened: List[EvidenceRef] = Field(default_factory=list)
    claim_negation: Optional[str] = None


class AuthorityGrade(str, Enum):
    """Source authority rating (来源权威性评级)."""

    OFFICIAL = "official"            # 官方 / 权威机构
    PROFESSIONAL = "professional"    # 专业媒体 / 同行评审
    SELF_MEDIA = "self_media"        # 自媒体 / 低可信
    UNVERIFIABLE = "unverifiable"    # 不可考


class SourceGrade(BaseModel):
    """One bound candidate source, graded on authority."""

    source_id: str
    grade: AuthorityGrade
    reliability: float = Field(ge=0.0, le=1.0)
    tier: str = "unknown"
    note: str = ""


class SourceQuality(BaseModel):
    """Per-claim source authority rating over all bound candidate sources."""

    grade: AuthorityGrade = AuthorityGrade.UNVERIFIABLE
    counts: Dict[str, int] = Field(default_factory=dict)
    sources: List[SourceGrade] = Field(default_factory=list)
    note: str = ""


class Freshness(BaseModel):
    """Per-claim freshness check (flagged stale for undated dynamic facts)."""

    checked_at: str = ""
    dynamic: bool = False
    has_date: bool = False
    newest_date: Optional[str] = None
    stale: bool = False
    note: str = ""


class Consistency(BaseModel):
    """Per-claim consistency check of numbers and qualifiers vs the sources."""

    consistent: bool = True
    number_mismatch: bool = False
    qualifier_overreach: bool = False
    claim_numbers: List[str] = Field(default_factory=list)
    source_numbers: List[str] = Field(default_factory=list)
    note: str = ""


class PopularityBias(BaseModel):
    biased: bool = False
    ratio: float = 0.0
    note: str = ""
    avg_traffic: Optional[float] = None


class Verification(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: ClaimType
    evidence: Evidence
    support_score: float = 0.0
    independent_domains: int = 0
    conflict: bool = False
    #: Evidence-efficacy checks (per claim): authority rating, freshness, and a
    #: consistency check of numbers / qualifiers against the sources.
    source_quality: SourceQuality = Field(default_factory=SourceQuality)
    freshness: Freshness = Field(default_factory=Freshness)
    consistency: Consistency = Field(default_factory=Consistency)
    #: Sources in the support set that are high-traffic but low-reliability.
    popularity_suspect_source_ids: List[str] = Field(default_factory=list)
    popularity_bias: PopularityBias = Field(default_factory=PopularityBias)
    retrieval_backend: str = ""
    nli_backend: str = ""
    screened_count: int = 0
    question_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    off_topic: bool = False


# --- Stage 3: transparent, layered output ---------------------------------

class Criterion(BaseModel):
    id: str
    rule: str


class Layers(BaseModel):
    l1_primary: List[EvidenceRef] = Field(default_factory=list)
    l2_secondary: List[EvidenceRef] = Field(default_factory=list)
    l3_conflicting: List[EvidenceRef] = Field(default_factory=list)


class ReportItem(BaseModel):
    claim_id: str
    text: str
    type: ClaimType
    credibility: float = Field(ge=0.0, le=1.0)
    decision: Decision
    decision_label: str
    criteria: List[Criterion] = Field(default_factory=list)
    boundary_conditions: List[str] = Field(default_factory=list)
    layers: Layers
    popularity_bias: PopularityBias
    decomposition: Dict[str, Any] = Field(default_factory=dict)
    challengeable: bool = True


class ReportSummary(BaseModel):
    total: int = 0
    kept: int = 0
    demoted: int = 0
    dropped: int = 0
    contextualized: int = 0
    avg_credibility: float = 0.0


class Report(BaseModel):
    summary: ReportSummary
    items: List[ReportItem]
    criteria: Dict[str, List[Criterion]]
    #: Optional reviser output (the rewritten "supplemented" text). None unless
    #: a reviser backend was requested; the original answer is never mutated.
    supplemented_text: Optional[str] = None


# --- Evaluation: Factuality (source-anchored) -----------------------------

class ClaimVerdict(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: ClaimType
    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0)
    evidence_domains: List[str] = Field(default_factory=list)
    rationale: str = ""


class FactualityResult(BaseModel):
    """Five-level claim verdicts plus a source-anchored accuracy score.

    Each judged claim (fact / inference / reasoning) gets one of accurate /
    inaccurate / disputed / unsupported, anchored on **external reliable
    sources**. Opinions are NOT_APPLICABLE and need no external verification.

    score = (n_accurate + n_not_applicable) / n_total_claims

    The label names and score shape follow ADK's `hallucinations_v1`, but the
    anchor is different: ADK grounds the answer in the supplied context / tool
    outputs (internal grounding), whereas this score is anchored on external
    reliable sources (factual validity). Therefore `unsupported` here can still
    be true about the world - it just was not substantiated by the retrieved
    reliable sources. The two scores are not directly comparable.
    """

    score: float = Field(ge=0.0, le=1.0)
    accurate: int = 0
    inaccurate: int = 0
    disputed: int = 0
    unsupported: int = 0
    not_applicable: int = 0
    total_claims: int = 0
    verifiable_claims: int = 0
    applicable_claims: int = 0
    verdicts: List[ClaimVerdict] = Field(default_factory=list)
    method: str = "local_source_anchored"


# Backwards-compatible alias: the metric was previously called "faithfulness".
FaithfulnessResult = FactualityResult

# --- Evaluation: Grounding (grounding-only, NOT factuality) ----------------

class ClaimGrounding(BaseModel):
    """How well one knowledge claim is grounded in the *supplied contexts*.

    This is the ADK-style internal-faithfulness signal, kept deliberately
    separate from `ClaimVerdict` (which is anchored on external reliable
    sources). `grounded=None` means the judge returned no usable verdict for
    this claim, so nothing is claimed about it.
    """

    claim_id: str
    claim_text: str
    claim_type: ClaimType
    grounded: Optional[bool] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rationale: str = ""


class GroundingResult(BaseModel):
    """Grounding-only report produced by an external judge (Ragas / TruLens).

    The question answered here is "is the claim entailed by the retrieved
    context?" - internal faithfulness, ADK-style. It is reported *alongside*
    Factuality and never mixed into the five-level verdict, because grounding
    and factual validity can disagree: a claim can be perfectly grounded in a
    narrow or biased context and still be false about the world.

    Only knowledge claims (fact / inference / reasoning) are judged; opinions
    are skipped without an API call.
    """

    method: str = "external_judge"
    judged_claims: int = 0
    grounded_count: int = 0
    ungrounded_count: int = 0
    skipped_claims: int = 0
    total_claims: int = 0
    mean_score: Optional[float] = None
    items: List[ClaimGrounding] = Field(default_factory=list)

    @property
    def text(self) -> str:
        lines = [
            "=== Grounding (grounding-only, NOT factuality) ===",
            f"method: {self.method}",
            f"judged {self.judged_claims}/{self.total_claims} | grounded {self.grounded_count} | "
            f"ungrounded {self.ungrounded_count} | skipped {self.skipped_claims}",
            f"mean grounding score: {self.mean_score}",
            "",
        ]
        for item in self.items:
            label = {True: "grounded", False: "ungrounded", None: "no verdict"}[item.grounded]
            lines.append(f"  [{item.claim_id}] {label} ({item.score}) - {item.rationale}")
            lines.append(f"        {item.claim_text}")
        return "\n".join(lines)




# --- Evaluation: Honesty (epistemic-honesty / transparency) ---------------

class HonestyFlag(BaseModel):
    """One auditable reason the answer is not fully honest about its footing."""

    code: str
    label: str
    detail: str
    claim_ids: List[str] = Field(default_factory=list)


class HonestyResult(BaseModel):
    """Claims-and-auditability check: does the answer surface its own limits?

    This is the transparency / "log-evidence" half of the ethical standard,
    distinct from both factuality (is it true?) and grounding (is it rooted?):

      * confidence calibration - verifiable claims are hedged when unsupported
      * unsupported-claim disclosure - unsupported assertions are not asserted flatly
      * popularity disclosure - popularity-driven support is admitted
      * conflict disclosure - conflicting sources are acknowledged
      * scope disclosure - the "only the retrieved set" boundary is stated
      * question coverage - claimed facts react to the question asked (H5)

    The disclosure logic is also exposed under the Step-4 four-check names, so a
    reader can grep the report for the check instead of decoding H-codes. Each is
    a pass/fail (True = honest on that axis) and maps 1:1 to an H-code:

      * ``unsupported_rejected``  - no flat assertion of an unsupported claim (H1)
      * ``sycophancy_checked``    - popularity-driven support is not passed off
                                    as reliability (H2)
      * ``conflicts_preserved``   - source conflict is surfaced, not suppressed (H3)
      * ``uncertainty_marked``    - the retrieved-set scope / limits are stated (H4)

    H5 (question coverage) is a relevance check, not a disclosure check, so it
    stays outside the four-name set and is reported through ``flags`` only.

    Non-factual (opinion) claims are exempt and never lower the score.
    """

    score: float = Field(ge=0.0, le=1.0)
    passed: bool = False
    flags: List[HonestyFlag] = Field(default_factory=list)
    checked_claims: int = 0
    exempt_claims: int = 0
    method: str = "local_heuristic"
    # Step-4 four-check crosswalk (True = honest on that axis); see the class docstring.
    conflicts_preserved: bool = True   # H3
    uncertainty_marked: bool = True    # H4
    unsupported_rejected: bool = True  # H1
    sycophancy_checked: bool = True    # H2


# --- Stage 4: revision (Reviser role, optional) ----------------------------

class RevisionItem(BaseModel):
    """The revision applied to one claim, with the audit trace that drove it."""

    claim_id: str
    action: str  # keep | annotate | qualify | drop | answer | add_caveat
    original_text: str
    revised_text: Optional[str] = None
    reason: str = ""
    criteria: List[str] = Field(default_factory=list)


class RecheckRound(BaseModel):
    """One revision->recheck cycle: a candidate text and how it re-scored.

    Produced by the revision->recheck gate. After a Reviser pass the revised text
    is treated as *newly generated text* and re-run through the source-anchored
    factuality pipeline; a round records what the re-check found.
    """

    round: int
    revised_text: str
    factuality_score: float = Field(ge=0.0, le=1.0)
    inaccurate: int = 0
    #: Claims NOT substantiated in this round (inaccurate / disputed /
    #: unsupported) that were fine in the input verdicts - i.e. the revision
    #: either edited a previously-fine claim into a contradiction or introduced a
    #: brand-new unsubstantiated claim (a smuggled hallucination usually lands in
    #: `unsupported`, since it has no source at all).
    new_inaccurate: List[str] = Field(default_factory=list)
    changed: int = 0
    method: str = "local_rewriter"
    passed: bool = False


class RecheckResult(BaseModel):
    """The revision->recheck gate: second-pass factuality over the revised text.

    An LLM revision is newly generated text, so it can smuggle in new
    hallucinations (a source-faithful citation can still be wrong). This gate
    re-runs the five-level, source-anchored factuality check on the revised text
    and forces another minimal revision (up to `max_rounds`) whenever the
    revision introduced new *not-substantiated* claims (inaccurate / disputed /
    unsupported) *and* the score fell below the threshold. The best-scoring
    candidate is kept, so the gate can never make the output worse than the first
    revision.
    """

    ran: bool = False
    passed: bool = True
    threshold: float = 0.7
    initial_score: float = 0.0
    final_score: float = 0.0
    max_rounds: int = 2
    rounds: List[RecheckRound] = Field(default_factory=list)
    #: New-inaccurate claims that survived every permitted round.
    leftover_inaccurate: List[str] = Field(default_factory=list)
    method: str = "factuality_recheck"

    @property
    def text(self) -> str:
        lines = [
            "=== Revision->recheck gate (second-pass factuality) ===",
            f"ran {self.ran} | {'PASS' if self.passed else 'STILL REGRESSING'} | "
            f"threshold {self.threshold} | rounds {len(self.rounds)}/{self.max_rounds}",
            f"score {self.initial_score} -> {self.final_score}",
        ]
        for r in self.rounds:
            flag = "pass" if r.passed else "re-edit"
            lines.append(
                f"  round {r.round}: score {r.factuality_score} | inaccurate {r.inaccurate} | "
                f"new {r.new_inaccurate or '-'} | changed {r.changed} | {r.method} [{flag}]"
            )
        if self.leftover_inaccurate:
            lines.append(f"  leftover new-inaccurate: {', '.join(self.leftover_inaccurate)}")
        return "\n".join(lines)


class RevisionResult(BaseModel):
    """Optional rewrite produced by the Reviser role.

    The original answer is never mutated; `revised_text` carries claim markers
    (`[C1]`) so every rewrite stays traceable to its verdict, and
    `revision_log` records one entry per claim. When no reviser backend is
    configured this is None and the pipeline is audit-only.
    """

    revised_text: str = ""
    supplemented: bool = False
    changed: int = 0
    revision_log: List[RevisionItem] = Field(default_factory=list)
    method: str = "local_rewriter"
    #: Audit trail, kept identical across both tracks (deterministic / LLM):
    #:   * diff              - unified diff, original answer -> revised text
    #:   * input_verdicts    - the verdicts the Reviser was asked to act on
    #:   * end_marker        - the edit-terminator the LLM track must emit
    #:   * end_marker_present- whether the LLM response actually carried it
    #:   * fallback          - LLM backend failed and the local rewriter ran
    diff: Optional[str] = None
    input_verdicts: List[str] = Field(default_factory=list)
    end_marker: str = "---END-OF-EDIT---"
    end_marker_present: bool = False
    fallback: bool = False
    #: The revision->recheck gate report (None unless the gate ran).
    recheck: Optional["RecheckResult"] = None

    @property
    def text(self) -> str:
        lines = [
            "=== Revised answer (supplemented; markers link back to [C#] claims) ===",
            f"method: {self.method} | changed {self.changed}",
            "",
            self.revised_text,
        ]
        if self.revision_log:
            lines.append("")
            lines.append("--- revision log ---")
            for it in self.revision_log:
                lines.append(f"  [{it.claim_id}] {it.action} ({', '.join(it.criteria)}) - {it.reason}")
        if self.input_verdicts:
            lines.append("")
            lines.append("--- input verdicts (what the Reviser acted on) ---")
            lines.extend(f"  {v}" for v in self.input_verdicts)
        if self.fallback:
            lines.append("")
            lines.append("  [note] LLM reviser failed; fell back to the deterministic rewriter.")
        lines.append("")
        lines.append(f"--- edit terminator: {self.end_marker} "
                     f"({'present' if self.end_marker_present else 'not emitted / local track'}) ---")
        if self.diff:
            lines.append("")
            lines.append("--- diff (original -> revised) ---")
            lines.append(self.diff.rstrip("\n"))
        if self.recheck is not None:
            lines.append("")
            lines.append(self.recheck.text)
        return "\n".join(lines)


# --- Top-level pipeline result --------------------------------------------

class ValidationSummary(BaseModel):
    """The overall assessment, restated - never a single fused verdict.

    This is deliberately **not** a scalar "overall confidence score" that a
    reader could mistake for the five-level verdict. It only restates what the
    separate reports already found, so the five-level `factuality.verdicts`
    remain the sole judgement:

      * `factuality_distribution` - the five-level claim counts (accurate /
        inaccurate / disputed / unsupported / not_applicable)
      * `factuality_pass_rate` - fraction of claims that passed, i.e.
        `factuality.score`; a distribution ratio, not a verdict
      * `grounding_score` - the grounding judge's score on its own 0-100 scale,
        present only when a grounding judge ran (`None` otherwise). Kept apart
        because grounding and factual validity can disagree.
      * `honesty_score` - the disclosure-check pass ratio

    Merging these into one number would re-introduce exactly the failure the
    five-level vocabulary exists to prevent, so no such field is offered.
    """

    factuality_distribution: Dict[str, int] = Field(default_factory=dict)
    factuality_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    honesty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    note: str = (
        "report-level aggregate only; the five-level factuality verdict is the "
        "judgement, this summary is not a score"
    )

    @classmethod
    def from_reports(
        cls,
        factuality: "FactualityResult",
        grounding: Optional["GroundingResult"] = None,
        honesty: Optional["HonestyResult"] = None,
    ) -> "ValidationSummary":
        """Restate the separate reports as one aggregate (never a fused score)."""
        return cls(
            factuality_distribution={
                "accurate": factuality.accurate,
                "inaccurate": factuality.inaccurate,
                "disputed": factuality.disputed,
                "unsupported": factuality.unsupported,
                "not_applicable": factuality.not_applicable,
            },
            factuality_pass_rate=factuality.score,
            grounding_score=(
                None if grounding is None or grounding.mean_score is None
                else round(grounding.mean_score * 100, 1)
            ),
            honesty_score=(honesty.score if honesty is not None else 0.0),
        )

    @property
    def text(self) -> str:
        order = ("accurate", "inaccurate", "disputed", "unsupported", "not_applicable")
        dist = " ".join(f"{k}={self.factuality_distribution.get(k, 0)}" for k in order)
        grounding = "n/a (no judge)" if self.grounding_score is None else f"{self.grounding_score}/100"
        return "\n".join([
            "=== Summary (report-level aggregate, NOT a single score) ===",
            f"factuality verdicts: {dist}",
            f"factuality pass rate: {self.factuality_pass_rate}",
            f"grounding score: {grounding}",
            f"honesty disclosure score: {self.honesty_score}",
            f"note: {self.note}",
        ])


class ValidationResult(BaseModel):
    answer: str
    question: Optional[str] = None
    domain: str = "general"
    #: Which critic backend produced the claims + verdicts, for traceability:
    #: "deterministic" (offline rules) or "llm" (LLM critic). Mirrors the
    #: engine actually used (falls back to "deterministic" when the LLM critic
    #: failed), so a reader can tell whether the five-level verdicts came from
    #: `local_source_anchored` or `llm_critic`.
    critic_backend: str = "deterministic"
    source_count: int = 0
    stage1_stats: DecompositionStats
    claims: List[Claim]
    verifications: List[Verification]
    report: Report
    factuality: FactualityResult
    grounding: Optional[GroundingResult] = None
    honesty: Optional[HonestyResult] = None
    revision: Optional[RevisionResult] = None
    #: Optional report-level aggregate (see `ValidationSummary`). Never a single
    #: fused score: it restates the five-level distribution and the grounding /
    #: honesty ratios, and the five-level verdict stays the judgement.
    summary: Optional[ValidationSummary] = None

    @property
    def faithfulness(self) -> FactualityResult:
        """Backwards-compatible alias for `factuality` (the metric was renamed)."""
        return self.factuality