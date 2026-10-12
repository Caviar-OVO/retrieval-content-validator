"""Stage 4 - revision (the Reviser role), two-track + revision->recheck gate.

The auditor only judges; the reviser rewrites. Revision is optional and additive:
the original answer is never mutated.

Two backends share one contract (`reviser(text, report) -> str | None`):

  * **local** (default) - the deterministic, offline rewriter below. Offline,
    auditable, zero-cost: the audit baseline, so standard 4 (log & evidence)
    never depends on an external API.
  * **llm** (optional) - an LLM "editor" that rewrites with the Reviser persona
    (`rva.roles.REVISER.prompt`). It must terminate its output with
    `---END-OF-EDIT---`; the marker is stripped and its presence recorded.

The revision->recheck gate
--------------------------
An LLM revision is *newly generated text*, so it can smuggle in new
hallucinations (source-faithful citations can still be wrong - Wallat 2025).
Whenever revision runs, the gate re-runs the source-anchored factuality check on
the revised text. If the revision introduced new *not-substantiated* claims
(inaccurate / disputed / unsupported) *and* the score fell below the threshold,
another minimal edit is forced (up to `max_rounds`). The best-scoring candidate is
kept, so the gate can never leave the output worse than the first revision.

Auditability (both tracks)
--------------------------
Every result carries: the per-claim `revision_log`, a unified before/after
`diff`, the `input_verdicts` the Reviser acted on, the `---END-OF-EDIT---`
terminator + its presence, and (for the LLM track) a `fallback` flag when the
external backend failed and the deterministic rewriter took over. Switching
backends never breaks the evidence loop.

Rewrite kinds (deterministic), all offline:

  * **annotate** (opinion/`contextualize`) - keep the text, prefix its `[C#]`
    marker, and append an explicit subjective marker;
  * **qualify** (demote / off-topic) - keep the text, prefix its `[C#]` marker,
    and append the caveat that explains the lowered credibility;
  * **drop** - remove from the body and move to a `[Validator supplement]` block
    that preserves the `[C#]` marker and the reason;
  * **keep** - left untouched.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from .models import (
    Claim,
    Decision,
    FactualityResult,
    Report,
    RecheckResult,
    RecheckRound,
    ReportItem,
    RevisionItem,
    RevisionResult,
    Verification,
    Verdict,
)
from .text import overlap, tokenize


#: The complete set of revision actions, in escalation order. Kept as a tuple so
#: the output contract (docs/output-contract.md) can be cross-checked.
REVISION_ACTIONS = ("keep", "annotate", "qualify", "drop")

#: The terminator an LLM reviser must emit so the edit boundary is unambiguous.
END_OF_EDIT_MARKER = "---END-OF-EDIT---"

#: Default score below which the recheck gate forces another edit.
RECHECK_THRESHOLD = 0.7

#: Default maximum number of revision->recheck rounds.
RECHECK_MAX_ROUNDS = 2

#: Verdicts that count as "not substantiated by sources". A revised text that
#: turns a fine claim into one of these - or adds a new one - is a regression the
#: gate must catch. (Unsupported is included because a smuggled hallucination
#: usually has no source at all, so it lands here rather than in Inaccurate.)
_BAD_VERDICTS = (Verdict.INACCURATE, Verdict.DISPUTED, Verdict.UNSUPPORTED)


def _caveat(v: Verification, decision: Decision | None = None) -> str:
    if decision is Decision.CONTEXTUALIZE:
        return "[subjective opinion; not an established fact]"
    if v.off_topic:
        return "[off-topic relative to the question]"
    if v.conflict:
        return "[sources conflict; treat as disputed]"
    if v.popularity_bias.biased:
        return "[support is popularity-driven, not necessarily accurate]"
    return "[lower credibility; see validator report]"


def _input_verdict_lines(
    verifications: list[Verification],
    report: Report,
    factuality: FactualityResult | None,
) -> list[str]:
    """The verdicts the Reviser is asked to act on, one line per claim."""
    items_by_id = {i.claim_id: i for i in report.items}
    verdict_by_id = {}
    if factuality is not None:
        verdict_by_id = {v.claim_id: v for v in factuality.verdicts}

    lines: list[str] = []
    for v in verifications:
        item = items_by_id.get(v.claim_id)
        if item is None:
            continue
        criteria = ",".join(c.id for c in item.criteria) or "-"
        cv = verdict_by_id.get(v.claim_id)
        if cv is not None:
            lines.append(
                f"[{v.claim_id}] {cv.verdict.value} (score {cv.score}) -> "
                f"{item.decision.value} [{criteria}] - {cv.rationale}"
            )
        else:
            lines.append(
                f"[{v.claim_id}] {item.decision.value} [{criteria}] - {item.decision_label}"
            )
    return lines


def _diff(before: str, after: str) -> str:
    """A unified before/after diff of the original vs the revised answer."""
    if (before or "") == (after or ""):
        return ""
    lines = difflib.unified_diff(
        (before or "").splitlines(),
        (after or "").splitlines(),
        fromfile="original",
        tofile="revised",
        lineterm="",
    )
    return "\n".join(lines)


#: The bracketed caveats the rewriter appends; stripped before a re-check so the
#: gate judges the answer's content, not the validator's own labels.
_CAVEAT_PREFIXES = (
    "[subjective opinion",
    "[off-topic",
    "[sources conflict",
    "[support is popularity-driven",
    "[lower credibility",
)


def strip_validator_artifacts(text: str) -> str:
    """Remove the rewriter's own labels so a re-check sees only answer content.

    The deterministic rewrite prefixes changed claims with `[C#]`, appends
    bracketed caveats, and moves dropped claims into a `[Validator supplement]`
    block. Re-decomposing that raw text would treat our own labels as new claims
    (and flag them), so they are stripped first. The original answer is never
    affected - this only shapes what the re-check reads.
    """
    body = (text or "").split("[Validator supplement]")[0]
    out_lines = []
    for line in body.splitlines():
        line = re.sub(r"^\s*\[C\d+\]\s*", "", line)  # leading [C#] marker
        stripped = line.rstrip()
        for prefix in _CAVEAT_PREFIXES:  # trailing caveat bracket
            if stripped.endswith("]") and prefix in stripped:
                idx = stripped.rfind(prefix)
                stripped = stripped[:idx].rstrip()
                break
        out_lines.append(stripped)
    return "\n".join(l for l in out_lines if l).strip()


def _deterministic_rewrite(
    claims: list[Claim], verifications: list[Verification], report: Report
) -> tuple[str, list[RevisionItem], int]:
    """The offline, deterministic rewrite. Returns (text, log, changed)."""
    by_id = {c.id: c for c in claims}
    items_by_id = {i.claim_id: i for i in report.items}

    log: list[RevisionItem] = []
    body_lines: list[str] = []
    supplement_lines: list[str] = []
    changed = 0

    for v in verifications:
        item = items_by_id.get(v.claim_id)
        if item is None or by_id.get(v.claim_id) is None:
            continue
        criteria = [c.id for c in item.criteria]
        reason = item.decision_label
        revised: Optional[str]
        action: str

        if item.decision is Decision.KEEP:
            action, revised = "keep", v.claim_text
            body_lines.append(v.claim_text)
        elif item.decision is Decision.DROP:
            action, revised = "drop", None
            reason = "does not react to the question asked" if v.off_topic else item.decision_label
            supplement_lines.append(f"- [{v.claim_id}] removed: {v.claim_text} ({reason})")
        else:
            # contextualize (opinion) and demote both keep the text with a caveat.
            action = "annotate" if item.decision is Decision.CONTEXTUALIZE else "qualify"
            # Changed claims keep their [C#] marker so the rewrite is traceable.
            revised = f"[{v.claim_id}] {v.claim_text} {_caveat(v, item.decision)}".strip()
            reason = "subjective opinion: flagged as non-factual" if action == "annotate" else reason
            body_lines.append(revised)

        if action != "keep":
            changed += 1
        log.append(RevisionItem(
            claim_id=v.claim_id, action=action, original_text=v.claim_text,
            revised_text=revised, reason=reason, criteria=criteria,
        ))

    revised_text = "\n".join(l for l in body_lines if l).strip()
    if supplement_lines:
        block = "[Validator supplement]\n" + "\n".join(supplement_lines)
        revised_text = f"{revised_text}\n\n{block}".strip() if revised_text else block
    return revised_text, log, changed


def _is_new_inaccurate(text: str, baseline: list[str]) -> bool:
    """True if `text` is an inaccurate claim the input verdicts did not already flag."""
    if not baseline:
        return True
    tokens = tokenize(text)
    for base in baseline:
        if overlap(tokens, tokenize(base)) >= 0.5 or text.strip() == base.strip():
            return False
    return True


def run_recheck_gate(
    first_text: str,
    *,
    recheck_fn: Callable[[str], tuple[list[Claim], list[Verification], FactualityResult]],
    baseline_inaccurate: list[str],
    threshold: float = RECHECK_THRESHOLD,
    max_rounds: int = RECHECK_MAX_ROUNDS,
    method: str = "local_rewriter",
    re_edit: Optional[Callable[[str], Optional[tuple[str, int]]]] = None,
) -> tuple[RecheckResult, str]:
    """Re-run factuality on the revised text; force another edit on regression.

    A claim counts as a regression when it is newly *not substantiated* by the
    sources (inaccurate / disputed / unsupported) - broader than "inaccurate",
    because a smuggled hallucination usually has no source at all and so lands in
    `unsupported`. Returns the gate report and the best-scoring candidate text.
    `re_edit` (used only for the LLM track) takes a text and returns
    `(new_text, changed)` or None.
    """
    rounds: list[RecheckRound] = []
    candidates: list[tuple[float, str]] = []
    text = first_text
    changed_this_round = 0

    for i in range(1, max(1, max_rounds) + 1):
        _, _, fact = recheck_fn(text)
        bad = [cv.claim_text for cv in fact.verdicts if cv.verdict in _BAD_VERDICTS]
        new_bad = [t for t in bad if _is_new_inaccurate(t, baseline_inaccurate)]
        # A new Inaccurate claim is a regression the gate must not accept silently.
        passed = (fact.score >= threshold) or not new_bad
        rounds.append(RecheckRound(
            round=i, revised_text=text, factuality_score=fact.score,
            inaccurate=len(bad), new_inaccurate=new_bad,
            changed=changed_this_round, method=method, passed=passed,
        ))
        candidates.append((fact.score, text))
        if passed or i >= max_rounds or re_edit is None:
            break
        candidate = re_edit(text)
        if not candidate or candidate[0] == text:
            break  # no progress possible; stop rather than loop
        text, changed_this_round = candidate

    best_score, best_text = max(candidates, key=lambda kv: kv[0])
    last = rounds[-1]
    return (
        RecheckResult(
            ran=True,
            passed=last.passed,
            threshold=threshold,
            initial_score=rounds[0].factuality_score,
            final_score=best_score,
            max_rounds=max_rounds,
            rounds=rounds,
            leftover_inaccurate=[] if last.passed else last.new_inaccurate,
            method="factuality_recheck",
        ),
        best_text,
    )


def build_recheck_fn(
    sources: list,
    *,
    question: str | None = None,
    domain: str = "general",
    registry: dict | None = None,
    retriever=None,
    nli=None,
):
    """Build the recheck callable: revise text -> (claims, verifications, factuality)."""
    from .evaluate import evaluate_factuality
    from .stages.decompose import decompose
    from .stages.verify import verify

    def recheck(text: str):
        clean = strip_validator_artifacts(text)
        claims, _ = decompose(clean, question=question)
        verifications = verify(claims, sources, registry=registry, domain=domain,
                               retriever=retriever, nli=nli)
        return claims, verifications, evaluate_factuality(claims, verifications)

    return recheck


def build_revision(
    claims: list[Claim],
    verifications: list[Verification],
    report: Report,
    *,
    reviser: Optional[Callable[..., Optional[str]]] = None,
    answer: str | None = None,
    factuality: FactualityResult | None = None,
    recheck: bool = False,
    recheck_fn: Optional[Callable[[str], tuple[list[Claim], list[Verification], FactualityResult]]] = None,
    max_rounds: int = RECHECK_MAX_ROUNDS,
    threshold: float = RECHECK_THRESHOLD,
) -> RevisionResult:
    """Rewrite the answer from its verdicts; never mutate the original.

    `reviser` selects the backend: None -> the deterministic rewriter (default),
    or a callable (the LLM reviser) that returns the rewrite. The audit trail -
    diff, input verdicts, end marker - is recorded for both tracks.

    When `recheck` is set and a `recheck_fn` is supplied, the revision->recheck
    gate re-runs factuality on the revised text and forces another edit while the
    revision introduces new Inaccurate claims and stays below `threshold`.
    """
    revised_text, log, changed = _deterministic_rewrite(claims, verifications, report)
    method = "local_rewriter"
    supplemented = False
    fallback = False
    end_marker_present = False

    if reviser is not None:
        try:
            replacement = reviser(revised_text, report)
        except Exception:  # pragma: no cover - depends on live backend
            replacement = None
        if replacement:
            revised_text = replacement
            supplemented = True
            fallback = False
            end_marker_present = bool(getattr(reviser, "last_end_marker_present", False))
            method = getattr(reviser, "method", "llm_reviser")
        else:
            # The external backend failed or returned nothing: keep the
            # deterministic rewrite and record that we fell back.
            fallback = True

    original = answer if answer is not None else "\n".join(v.claim_text for v in verifications)

    recheck_result: Optional[RecheckResult] = None
    if recheck and recheck_fn is not None:
        baseline_inaccurate = [
            cv.claim_text for cv in (factuality.verdicts if factuality else [])
            if cv.verdict in _BAD_VERDICTS
        ]

        re_edit = None
        if reviser is not None and getattr(reviser, "accepts_critique", False):
            critique = "Fix only the newly inaccurate claims flagged by the recheck; change nothing else."

            def re_edit(text: str) -> Optional[tuple[str, int]]:  # noqa: F811
                try:
                    candidate = reviser(text, report, critique)
                except TypeError:
                    candidate = reviser(text, report)
                except Exception:  # pragma: no cover - depends on live backend
                    return None
                return (candidate, 0) if candidate else None

        recheck_result, best_text = run_recheck_gate(
            revised_text, recheck_fn=recheck_fn, baseline_inaccurate=baseline_inaccurate,
            threshold=threshold, max_rounds=max_rounds, method=method, re_edit=re_edit,
        )
        revised_text = best_text

    return RevisionResult(
        revised_text=revised_text,
        supplemented=supplemented,
        changed=changed,
        revision_log=log,
        method=method,
        diff=_diff(original, revised_text),
        input_verdicts=_input_verdict_lines(verifications, report, factuality),
        end_marker=END_OF_EDIT_MARKER,
        end_marker_present=end_marker_present,
        fallback=fallback,
        recheck=recheck_result,
    )


# --- LLM reviser backend ---------------------------------------------------

class LLMReviser:
    """Optional LLM "editor" backend for the Reviser role.

    Uses `rva.roles.REVISER.prompt` as the system prompt (editor persona + the
    shared ethical standard), hands the model the deterministic draft plus the
    input verdict list, and expects the output to terminate with
    `---END-OF-EDIT---`. The rewrite is still non-destructive: the caller keeps
    the original answer and records the marker presence and any fallback.
    """

    #: Signals to `build_revision` that this backend can take a recheck critique.
    accepts_critique = True

    def __init__(self, llm=None, model: str | None = None, temperature: float = 0.0):
        self._client = llm
        self._model = model
        self._temperature = temperature
        self.calls = 0
        self.failures = 0
        self.last_end_marker_present = False
        self.method = "llm_reviser"

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        from .config import get_api_key, get_base_url, get_judge_model

        kwargs = {"api_key": get_api_key()}
        base_url = get_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        if not self._model:
            self._model = get_judge_model()
        return self._client

    @staticmethod
    def _content(resp) -> str:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            value = resp.get("content", "")
        else:
            value = getattr(resp, "content", "")
        return value if isinstance(value, str) else str(value or "")

    @staticmethod
    def _system_prompt() -> str:
        from .roles import REVISER

        few_shot = (
            "\n\n示例（few-shot）：\n"
            "输入草稿：\n"
            "[C2] 因此，污染越严重的城市，居民的健康风险就越高。 [lower credibility; see validator report]\n"
            "要求：C2 前提未获支持，降级但保留原句。\n"
            "输出：\n"
            "[C2] 因此，污染越严重的城市，居民的健康风险就越高。 [lower credibility; see validator report]\n"
            "---END-OF-EDIT---"
        )
        return REVISER.prompt + few_shot

    def _user_prompt(self, text: str, report: Report, critique: str | None) -> str:
        verdicts = [
            f"- [{i.claim_id}] {i.decision.value} ({', '.join(c.id for c in i.criteria)}): {i.decision_label}"
            for i in report.items
        ]
        parts = [
            "这是待修订的回答草稿（已由确定性改写器给出初稿，含 [C#] 标记）：",
            text,
            "",
            "输入 verdict 清单（必须逐条遵循，不得引入新事实）：",
            *verdicts,
        ]
        if critique:
            parts += ["", f"复查意见（只需处理这些，其余不动）：{critique}"]
        parts += ["", f"请只做最小修订，并以单独一行 {END_OF_EDIT_MARKER} 结束输出。"]
        return "\n".join(parts)

    def __call__(self, text: str, report: Report, critique: str | None = None) -> Optional[str]:
        if not (text or "").strip():
            return None
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._user_prompt(text, report, critique)},
                ],
            )
        except Exception:  # pragma: no cover - depends on live API/key
            self.failures += 1
            self.last_end_marker_present = False
            return None

        self.calls += 1
        raw = self._content(resp.choices[0].message)
        if END_OF_EDIT_MARKER in raw:
            self.last_end_marker_present = True
            body = raw.split(END_OF_EDIT_MARKER)[0]
        else:
            self.last_end_marker_present = False
            body = raw
        body = body.strip()
        return body or None


def build_reviser(name: str = "local", *, llm=None, llm_ok: bool = False):
    """Return a reviser backend for `name` in {local, llm, auto}.

    * ``local`` (default) -> None: use the deterministic rewriter inside
      `build_revision` (offline, auditable, zero-cost).
    * ``llm`` -> an `LLMReviser` (editor persona; needs a key + network).
    * ``auto`` -> LLM when a key is available, else the deterministic rewriter.
    """
    name = (name or "local").lower()
    if name == "auto":
        name = "llm" if llm_ok else "local"
    if name in ("llm", "openai", "editor"):
        return LLMReviser(llm=llm)
    return None


def format_text(r: RevisionResult) -> str:
    return r.text
