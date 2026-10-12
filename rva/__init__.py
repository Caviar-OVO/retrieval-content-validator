"""Retrieval-content-validator (code name: RVA, Response Validation Agent).

Audits an LLM answer produced over web-search results. `validate()` takes a
(question, answer) pair plus the retrieved sources and runs:

  1. cognitive decomposition   (facts / reasoning / opinions / inferences), each
     claim anchored to the question it should answer
  2. evidence & source verification (vector retrieval + three-way NLI)
  3. transparent, layered output (criteria, layers, boundary conditions)
  4. revision (optional Reviser rewrite; the original answer is never mutated)

The two stages are cast as named roles with one shared ethical standard
(see `rva/roles.py`): Stage 1 the **Critic** (investigative journalist: decompose,
verify, judge) and Stage 2 the **Reviser** (editor: minimal, additive rewrite).
Both answer to 能力要可证 / 路径要可审 / 态度要诚实 / 结果要有据.

Four reports are produced, deliberately kept apart:

  * **Factuality** - five-level verdicts (`accurate` / `inaccurate` / `disputed` /
    `unsupported` / `not_applicable`) anchored on **external reliable sources**
    (factual validity). Fully local and deterministic; no LLM judge is involved.
  * **Honesty** - disclosure checks (`H1`-`H5`): unsupported claims, popularity,
    conflict, retrieved-set scope, question relevance.
  * **Grounding** - an *optional* external-judge report (Ragas / TruLens) answering
    "is the claim entailed by the supplied contexts?" (ADK-style internal
    faithfulness). Never mixed into the five-level verdict.
  * **Revision** - the optional Reviser output, traceable via `[C#]` markers.

The older name "faithfulness" is kept as an alias for Factuality.
"""

from __future__ import annotations

from .evaluate import (
    VERIFIABLE_TYPES,
    evaluate_factuality,
    evaluate_faithfulness,
    evaluate_grounding,
    is_verifiable,
    verdict_score,
    format_text as format_factuality,
    format_text as format_faithfulness,
)
from .honesty import format_text as format_honesty
from .honesty import verify_honesty
from .models import (
    AuthorityGrade,
    Claim,
    ClaimGrounding,
    Consistency,
    Freshness,
    SourceGrade,
    SourceQuality,
    ClaimType,
    Decision,
    FactualityResult,
    FaithfulnessResult,
    GroundingResult,
    HonestyResult,
    Report,
    RevisionResult,
    Source,
    ValidationResult,
    ValidationSummary,
    Verdict,
)
from .efficacy import (
    assess_consistency,
    assess_freshness,
    assess_source_quality,
    grade_for_tier,
    score_support,
    suspicious_traffic_ids,
)
from .critic import (
    DETERMINISTIC_METHOD,
    LLM_METHOD,
    CriticOutput,
    DeterministicCritic,
    LLMCritic,
    build_critic,
)
from .revise import (
    END_OF_EDIT_MARKER,
    RECHECK_MAX_ROUNDS,
    RECHECK_THRESHOLD,
    LLMReviser,
    build_recheck_fn,
    build_revision,
    build_reviser,
    run_recheck_gate,
    strip_validator_artifacts,
)
from .roles import (
    CRITIC,
    ETHICAL_IDS,
    ETHICAL_STANDARD,
    REVISER,
    ROLE_ASSIGNMENT,
    ROLE_SEQUENCE,
    ROLES,
    Role,
)
from .stages.decompose import decompose
from .stages.output import format_text as format_report
from .stages.output import render
from .nli import build_nli_judge, parse_label
from .retrieval import VectorRetriever, chunk_text
from .stages.verify import DEFAULT_REGISTRY, score_source, verify



def format_summary(summary: "ValidationSummary") -> str:
    """Render the optional report-level aggregate (never a single score)."""
    return summary.text


__all__ = [
    "validate",
    "decompose",
    "verify",
    "render",
    "evaluate_factuality",
    "evaluate_faithfulness",
    "evaluate_grounding",
    "is_verifiable",
    "VERIFIABLE_TYPES",
    "verdict_score",
    "format_report",
    "format_factuality",
    "format_faithfulness",
    "format_grounding",
    "format_honesty",
    "verify_honesty",
    "build_critic",
    "assess_source_quality",
    "assess_freshness",
    "assess_consistency",
    "score_support",
    "suspicious_traffic_ids",
    "grade_for_tier",
    "DeterministicCritic",
    "LLMCritic",
    "CriticOutput",
    "DETERMINISTIC_METHOD",
    "LLM_METHOD",
    "build_revision",
    "build_reviser",
    "LLMReviser",
    "build_recheck_fn",
    "run_recheck_gate",
    "strip_validator_artifacts",
    "END_OF_EDIT_MARKER",
    "RECHECK_THRESHOLD",
    "RECHECK_MAX_ROUNDS",
    "CRITIC",
    "REVISER",
    "ROLES",
    "Role",
    "ROLE_SEQUENCE",
    "ROLE_ASSIGNMENT",
    "ETHICAL_STANDARD",
    "ETHICAL_IDS",
    "score_source",
    "VectorRetriever",
    "chunk_text",
    "build_nli_judge",
    "parse_label",
    "DEFAULT_REGISTRY",
    "Source",
    "Claim",
    "AuthorityGrade",
    "SourceGrade",
    "SourceQuality",
    "Freshness",
    "Consistency",
    "ClaimType",
    "Decision",
    "Verdict",
    "Report",
    "FactualityResult",
    "FaithfulnessResult",
    "GroundingResult",
    "ClaimGrounding",
    "HonestyResult",
    "ValidationSummary",
    "format_summary",
    "RevisionResult",
    "ValidationResult",
]

__version__ = "0.9.0"


def format_grounding(g: GroundingResult) -> str:
    """Render a grounding-only report (separate from the factuality verdicts)."""
    return g.text


def validate(
    answer: str,
    sources: list[Source] | None = None,
    domain: str = "general",
    registry: dict | None = None,
    judge: str = "local",
    retriever: str | None = None,
    nli: str | None = None,
    question: str | None = None,
    critic_backend: str | None = None,
    critic=None,
    revise: bool = False,
    reviser=None,
    reviser_backend: str | None = None,
    recheck: bool | None = None,
    recheck_threshold: float | None = None,
    recheck_max_rounds: int | None = None,
    summary: bool = False,
) -> ValidationResult:
    """Run the full pipeline on an LLM answer and its retrieved sources.

    `question` is the user's question. Following the LLM Auditor's
    (question, answer) contract, every extracted claim is anchored to it: the
    claim records its question relevance, and a knowledge claim that shares no
    vocabulary with the question is flagged off-topic (`D5`). Without a question
    the claims are unanchored and those two checks are skipped.

    Stage 2 matching uses vector retrieval + three-way NLI. `retriever` picks the
    embedding backend ("hash" offline, "openai" real embeddings, "auto"), and
    `nli` picks the stance judge ("heuristic" offline, "llm", "auto"). Both
    default to the values in `.env` (RVA_RETRIEVER / RVA_NLI) and degrade to the
    offline backends if no API key or network is available.

    `judge` selects the **grounding** backend: "local" (off, no external call),
    "ragas", or "trulens". Its output is reported in `result.grounding` and never
    changes `result.factuality`, whose five-level verdicts stay anchored on
    source reliability.

    The Critic runs on **two tracks**. By default (`critic_backend="deterministic"`
    or unset) it is the offline rule-based critic (decompose -> verify ->
    source-anchored factuality). `critic_backend="llm"` (or `critic=` with a
    custom backend / `LLMCritic`) injects the Critic persona
    (`rva.roles.CRITIC.prompt`) and lets the model do the extraction + five-level
    verdict in one pass (`factuality.method="llm_critic"`); it falls back to the
    deterministic critic on any failure. Both tracks return the identical schema
    (`Claim` with `cues` / `note`, a `FactualityResult` whose verdicts carry
    `rationale` + `evidence_domains`).

    `revise=True` runs the Reviser stage and fills `result.revision` (the
    marked-up `revised_text` plus a per-claim `revision_log`).

    The Reviser runs on **two tracks**. By default (`reviser_backend="local"` or
    unset) the deterministic, offline rewriter is used - the auditable,
    zero-cost baseline. `reviser_backend="llm"` (or `reviser=` with a custom
    callable / `LLMReviser`) switches to the LLM "editor" persona
    (`rva.roles.REVISER.prompt`); its output must end with `---END-OF-EDIT---`,
    and if it fails the deterministic rewriter takes over (`fallback=True`).

    Whenever revision runs, the **revision->recheck gate** re-runs the
    source-anchored factuality check on the revised text (an LLM revision is
    newly generated text and can introduce fresh hallucinations). If the revision
    adds new Inaccurate claims and the score falls below `recheck_threshold`, the
    gate forces another edit, up to `recheck_max_rounds`; the best-scoring
    candidate is kept. Both tracks record a before/after diff and the input
    verdict list, so the evidence loop is closed regardless of backend.
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("validate() requires a non-empty `answer` string")

    from .config import (
        _load_env_file,
        get_api_key,
        get_critic_backend,
        get_nli_backend,
        get_recheck_enabled,
        get_recheck_max_rounds,
        get_recheck_threshold,
        get_retriever_backend,
        get_reviser_backend,
    )

    _load_env_file()  # pick up OPENAI_API_KEY etc. from the .env file
    llm_ok = bool(get_api_key())

    srcs = sources or []

    from .nli import build_nli_judge
    from .retrieval import VectorRetriever

    retriever_backend = (retriever or get_retriever_backend()).lower()
    nli_backend = (nli or get_nli_backend()).lower()
    active_retriever = VectorRetriever(srcs, backend=retriever_backend, llm_ok=llm_ok)
    active_nli = build_nli_judge(nli_backend, llm_ok=llm_ok)

    # Critic (Stage 1-3 + the source-anchored verdicts), two tracks. The default
    # deterministic track still gets its verdicts from evaluate_factuality, which
    # keeps having no judge hook; only critic_backend="llm" authors them via LLM.
    active_critic = critic
    if active_critic is None:
        if critic_backend is None:
            critic_backend = get_critic_backend()
        active_critic = build_critic(critic_backend or "deterministic", llm_ok=llm_ok,
                                     has_sources=bool(srcs))

    outcome = active_critic.critique(
        answer, srcs, question=question, domain=domain, registry=registry,
        retriever=active_retriever, nli=active_nli,
    )
    claims = outcome.claims
    verifications = outcome.verifications
    stats = outcome.stats
    factuality = outcome.factuality

    report = render(claims, verifications)

    # Grounding is a *separate* report, produced only when a judge is requested.
    grounding = None
    if judge and judge.lower() not in ("local", "none", ""):
        from .integrations import build_judge

        batch_judge = build_judge(judge, sources=srcs)
        grounding = evaluate_grounding(claims, verifications, batch_judge=batch_judge)

    # Honesty & auditability: its own disclosure checks, grounded in the above.
    honesty = verify_honesty(
        claims, verifications, answer=answer, question=question,
        factuality=factuality, report=report,
    )

    # Optional revision (Reviser role); the original answer is never mutated.
    revision = None
    if revise:
        active_reviser = reviser
        if active_reviser is None and reviser_backend is None:
            reviser_backend = get_reviser_backend()
        if active_reviser is None:
            active_reviser = build_reviser(reviser_backend or "local", llm_ok=llm_ok)

        do_recheck = get_recheck_enabled() if recheck is None else recheck
        threshold = get_recheck_threshold() if recheck_threshold is None else recheck_threshold
        max_rounds = get_recheck_max_rounds() if recheck_max_rounds is None else recheck_max_rounds
        recheck_fn = None
        if do_recheck:
            recheck_fn = build_recheck_fn(
                srcs, question=question, domain=domain, registry=registry,
                retriever=active_retriever, nli=active_nli,
            )

        revision = build_revision(
            claims, verifications, report,
            reviser=active_reviser, answer=answer, factuality=factuality,
            recheck=do_recheck, recheck_fn=recheck_fn,
            max_rounds=max_rounds, threshold=threshold,
        )
        report = render(claims, verifications, supplemented_text=revision.revised_text)

    # Traceability: record the critic engine that actually produced the verdicts
    # (fallback-aware), not merely what was requested.
    effective_critic = "llm" if outcome.method == LLM_METHOD else "deterministic"

    # Optional report-level aggregate. It restates the separate reports and
    # carries no fused score: the five-level factuality verdict stays the
    # judgement. Grounding keeps its own 0-100 scale, reported only when a
    # grounding judge actually ran.
    aggregate = None
    if summary:
        aggregate = ValidationSummary.from_reports(factuality, grounding, honesty)

    return ValidationResult(
        answer=answer,
        question=question,
        domain=domain,
        critic_backend=effective_critic,
        source_count=len(srcs),
        stage1_stats=stats,
        claims=claims,
        verifications=verifications,
        report=report,
        factuality=factuality,
        grounding=grounding,
        honesty=honesty,
        revision=revision,
        summary=aggregate,
    )
