"""Two-track Reviser backend + the revision->recheck gate.

Covers:
  * backend selection (local default, llm, auto);
  * the LLM editor track: ---END-OF-EDIT--- handling and fallback;
  * the audit trail for both tracks (diff, input verdicts, end marker);
  * the gate: re-runs factuality, forces another edit on a new-inaccurate
    regression, and never keeps a worse candidate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import (  # noqa: E402
    END_OF_EDIT_MARKER,
    LLMReviser,
    build_recheck_fn,
    build_revision,
    build_reviser,
    decompose,
    render,
    run_recheck_gate,
    validate,
)
from rva.models import (  # noqa: E402
    ClaimType,
    ClaimVerdict,
    FactualityResult,
    Source,
    Verdict,
)
from rva.stages.verify import verify  # noqa: E402

WHO = Source(id="A", url="https://www.who.int/fact",
             text="室外空气污染导致全球每年约420万人过早死亡。")
QUESTION = "空气污染每年导致多少人过早死亡？"


def _fact(pairs) -> FactualityResult:
    """Build a FactualityResult from (claim_text, verdict) pairs."""
    verdicts = [
        ClaimVerdict(claim_id=f"C{i + 1}", claim_text=t, claim_type=ClaimType.FACT,
                     verdict=v, score=(1.0 if v in (Verdict.ACCURATE, Verdict.NOT_APPLICABLE) else 0.0),
                     rationale="test")
        for i, (t, v) in enumerate(pairs)
    ]
    total = len(verdicts)
    accurate = sum(1 for x in verdicts if x.verdict is Verdict.ACCURATE)
    inaccurate = sum(1 for x in verdicts if x.verdict is Verdict.INACCURATE)
    disputed = sum(1 for x in verdicts if x.verdict is Verdict.DISPUTED)
    unsupported = sum(1 for x in verdicts if x.verdict is Verdict.UNSUPPORTED)
    not_applicable = sum(1 for x in verdicts if x.verdict is Verdict.NOT_APPLICABLE)
    score = round((accurate + not_applicable) / total, 2) if total else 0.0
    return FactualityResult(
        score=score, accurate=accurate, inaccurate=inaccurate, disputed=disputed,
        unsupported=unsupported, not_applicable=not_applicable, total_claims=total,
        verifiable_claims=total, applicable_claims=total - not_applicable, verdicts=verdicts,
    )


# --- fake LLM clients (same shape as tests/test_retrieval_nli.py) ----------

class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        message = type("M", (), {"content": self._content})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


class _FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


class _BoomClient:
    @property
    def chat(self):
        raise RuntimeError("no api key")


class TestBackendSelection(unittest.TestCase):
    def test_local_is_default_and_returns_none(self):
        self.assertIsNone(build_reviser("local"))
        self.assertIsNone(build_reviser(""))

    def test_llm_returns_editor_backend(self):
        r = build_reviser("llm")
        self.assertIsInstance(r, LLMReviser)
        self.assertEqual(r.method, "llm_reviser")

    def test_auto_follows_key_availability(self):
        self.assertIsNone(build_reviser("auto", llm_ok=False))
        self.assertIsInstance(build_reviser("auto", llm_ok=True), LLMReviser)


class TestLLMReviser(unittest.TestCase):
    def _report(self, answer):
        claims, _ = decompose(answer, question=QUESTION)
        return claims, verify(claims, [WHO])

    def test_strips_end_marker_and_records_presence(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        claims, verifications = self._report(answer)
        report = render(claims, verifications)
        client = _FakeClient(f"REWRITTEN BY EDITOR\n{END_OF_EDIT_MARKER}")
        reviser = LLMReviser(llm=client)
        out = reviser("draft", report)
        self.assertEqual(out, "REWRITTEN BY EDITOR")
        self.assertTrue(reviser.last_end_marker_present)
        self.assertEqual(reviser.calls, 1)

    def test_missing_marker_is_recorded(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        claims, verifications = self._report(answer)
        report = render(claims, verifications)
        reviser = LLMReviser(llm=_FakeClient("REWRITTEN WITHOUT A MARKER"))
        out = reviser("draft", report)
        self.assertEqual(out, "REWRITTEN WITHOUT A MARKER")
        self.assertFalse(reviser.last_end_marker_present)

    def test_failure_returns_none(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        claims, verifications = self._report(answer)
        report = render(claims, verifications)
        reviser = LLMReviser(llm=_BoomClient())
        self.assertIsNone(reviser("draft", report))
        self.assertEqual(reviser.failures, 1)


class TestAuditTrail(unittest.TestCase):
    def _parts(self, answer):
        claims, _ = decompose(answer, question=QUESTION)
        verifications = verify(claims, [WHO])
        report = render(claims, verifications)
        return claims, verifications, report

    def test_input_verdicts_and_diff_are_recorded(self):
        answer = "空气污染每年导致约420万人过早死亡。我认为这是最糟糕的问题。"
        claims, verifications, report = self._parts(answer)
        fact = _fact([("空气污染每年导致约420万人过早死亡。", Verdict.ACCURATE)])
        r = build_revision(claims, verifications, report, answer=answer, factuality=fact)
        self.assertTrue(r.input_verdicts)
        self.assertTrue(any("->" in line for line in r.input_verdicts))
        self.assertTrue(r.diff)
        self.assertIn("--- original", r.diff)
        self.assertIn("+++ revised", r.diff)
        self.assertEqual(r.end_marker, END_OF_EDIT_MARKER)
        self.assertFalse(r.end_marker_present)  # local track emits no marker
        self.assertFalse(r.fallback)

    def test_llm_track_marks_and_audits(self):
        answer = "空气污染每年导致约420万人过早死亡。我认为这是最糟糕的问题。"
        claims, verifications, report = self._parts(answer)
        client = _FakeClient(f"[C1] rewritten\n{END_OF_EDIT_MARKER}")
        r = build_revision(claims, verifications, report, answer=answer,
                           reviser=LLMReviser(llm=client))
        self.assertTrue(r.supplemented)
        self.assertEqual(r.method, "llm_reviser")
        self.assertTrue(r.end_marker_present)
        self.assertFalse(r.fallback)

    def test_llm_failure_falls_back_to_local(self):
        answer = "空气污染每年导致约420万人过早死亡。我认为这是最糟糕的问题。"
        claims, verifications, report = self._parts(answer)
        local_only = build_revision(claims, verifications, report, answer=answer)
        r = build_revision(claims, verifications, report, answer=answer,
                           reviser=LLMReviser(llm=_BoomClient()))
        self.assertTrue(r.fallback)
        self.assertFalse(r.supplemented)
        self.assertEqual(r.method, "local_rewriter")
        self.assertEqual(r.revised_text, local_only.revised_text)


class TestArtifactStripper(unittest.TestCase):
    def test_strips_markers_caveats_and_supplement(self):
        from rva import strip_validator_artifacts

        raw = (
            "[C2] 因此，污染越严重的城市，居民的健康风险就越高。 [lower credibility; see validator report]\n"
            "有研究可能表明PM2.5与死亡率相关。\n\n"
            "[Validator supplement]\n- [C5] removed: 总之，空气污染可能没有官方说的那么严重。"
        )
        clean = strip_validator_artifacts(raw)
        self.assertNotIn("[C2]", clean)
        self.assertNotIn("[lower credibility", clean)
        self.assertNotIn("[Validator supplement]", clean)
        self.assertNotIn("removed:", clean)
        # the answer's own content survives
        self.assertIn("污染越严重的城市", clean)
        self.assertIn("PM2.5", clean)

    def test_gate_does_not_flag_its_own_labels(self):
        # Regression: the recheck must not treat the rewriter's [C#] / caveat
        # brackets as new unsubstantiated claims.
        answer = "空气污染每年导致约420万人过早死亡。我认为这是最糟糕的问题。"
        out = validate(question=QUESTION, answer=answer, sources=[WHO], revise=True)
        self.assertTrue(out.revision.recheck.rounds[0].new_inaccurate == [])


class TestRecheckGate(unittest.TestCase):
    def test_no_regression_passes_first_round(self):
        scripted = [(0.4, []), (0.4, [])]

        def recheck_fn(text):
            score, bad = scripted.pop(0)
            return [], [], _fact([(t, Verdict.INACCURATE) for t in bad] or [("ok", Verdict.ACCURATE)])

        res, best = run_recheck_gate("draft", recheck_fn=lambda t: recheck_fn(t),
                                     baseline_inaccurate=[], max_rounds=2)
        self.assertTrue(res.ran)
        self.assertEqual(len(res.rounds), 1)  # no new-inaccurate -> stop immediately
        self.assertTrue(res.passed)

    def test_new_inaccurate_forces_re_edit_then_passes(self):
        calls = {"n": 0}

        def recheck_fn(text):
            calls["n"] += 1
            if text == "draft":
                return [], [], _fact([("brand new bad claim", Verdict.INACCURATE),
                                      ("fine", Verdict.ACCURATE)])
            return [], [], _fact([("fine", Verdict.ACCURATE), ("also fine", Verdict.ACCURATE)])

        edits = []

        def re_edit(text):
            edits.append(text)
            return ("fixed", 1)

        res, best = run_recheck_gate(
            "draft", recheck_fn=recheck_fn, baseline_inaccurate=[],
            max_rounds=2, re_edit=re_edit,
        )
        self.assertEqual(edits, ["draft"])
        self.assertEqual(len(res.rounds), 2)
        self.assertTrue(res.passed)
        self.assertEqual(best, "fixed")
        self.assertEqual(res.final_score, 1.0)

    def test_pre_existing_inaccurate_is_not_a_regression(self):
        # The same inaccurate claim was already flagged in the input verdicts.
        def recheck_fn(text):
            return [], [], _fact([("known bad", Verdict.INACCURATE)])

        res, _ = run_recheck_gate("draft", recheck_fn=recheck_fn,
                                  baseline_inaccurate=["known bad"], max_rounds=2)
        self.assertTrue(res.passed)
        self.assertEqual(res.leftover_inaccurate, [])

    def test_no_progress_keeps_best_and_reports_leftover(self):
        def recheck_fn(text):
            return [], [], _fact([("bad", Verdict.INACCURATE)])

        res, best = run_recheck_gate("draft", recheck_fn=recheck_fn,
                                     baseline_inaccurate=[], max_rounds=2,
                                     re_edit=lambda t: None)
        self.assertFalse(res.passed)
        self.assertEqual(res.leftover_inaccurate, ["bad"])
        self.assertEqual(best, "draft")

    def test_unsupported_regression_is_caught(self):
        # A smuggled hallucination usually lands in `unsupported` (no source at
        # all), so the gate must treat that as a regression too - not only
        # `inaccurate`.
        def recheck_fn(text):
            return [], [], _fact([("brand new unsupported claim", Verdict.UNSUPPORTED),
                                  ("fine", Verdict.ACCURATE)])

        res, _ = run_recheck_gate("draft", recheck_fn=recheck_fn,
                                  baseline_inaccurate=[], max_rounds=2,
                                  re_edit=lambda t: None)
        self.assertFalse(res.passed)
        self.assertIn("brand new unsupported claim", res.leftover_inaccurate)

    def test_disputed_regression_is_caught(self):
        def recheck_fn(text):
            return [], [], _fact([("now disputed", Verdict.DISPUTED), ("fine", Verdict.ACCURATE)])

        res, _ = run_recheck_gate("draft", recheck_fn=recheck_fn,
                                  baseline_inaccurate=[], max_rounds=2,
                                  re_edit=lambda t: None)
        self.assertFalse(res.passed)

    def test_unsupported_already_in_baseline_is_not_a_regression(self):
        def recheck_fn(text):
            return [], [], _fact([("known weak", Verdict.UNSUPPORTED)])

        res, _ = run_recheck_gate("draft", recheck_fn=recheck_fn,
                                  baseline_inaccurate=["known weak"], max_rounds=2)
        self.assertTrue(res.passed)
        self.assertEqual(res.leftover_inaccurate, [])

    def test_best_candidate_is_kept_not_the_last(self):
        seen = {"n": 0}

        def recheck_fn(text):
            seen["n"] += 1
            if text == "draft":
                return [], [], _fact([("bad", Verdict.INACCURATE), ("x", Verdict.ACCURATE)])
            # the second attempt is *worse* than the first: must not be kept
            return [], [], _fact([("bad", Verdict.INACCURATE), ("y", Verdict.INACCURATE)])

        res, best = run_recheck_gate(
            "draft", recheck_fn=recheck_fn, baseline_inaccurate=[],
            max_rounds=2, re_edit=lambda t: ("worse", 1),
        )
        self.assertEqual(best, "draft")  # 0.5 kept over 0.0
        self.assertGreater(res.final_score, 0.0)


class TestValidateWiring(unittest.TestCase):
    def test_recheck_runs_by_default_and_gate_present(self):
        out = validate(question=QUESTION,
                       answer="空气污染每年导致约420万人过早死亡。我认为这最糟糕。",
                       sources=[WHO], revise=True)
        self.assertIsNotNone(out.revision)
        self.assertIsNotNone(out.revision.recheck)
        self.assertTrue(out.revision.recheck.ran)
        self.assertEqual(out.revision.recheck.max_rounds, 2)
        self.assertEqual(out.revision.recheck.method, "factuality_recheck")

    def test_recheck_can_be_disabled(self):
        out = validate(question=QUESTION,
                       answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], revise=True, recheck=False)
        self.assertIsNone(out.revision.recheck)

    def test_original_answer_never_mutated(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        out = validate(question=QUESTION, answer=answer, sources=[WHO], revise=True)
        self.assertEqual(out.answer, answer)

    def test_end_to_end_llm_track(self):
        from unittest import mock

        answer = "空气污染每年导致约420万人过早死亡。"
        fake = LLMReviser(llm=_FakeClient(f"[C1] edited by the editor persona\n{END_OF_EDIT_MARKER}"))
        with mock.patch("rva.revise.LLMReviser", return_value=fake):
            out = validate(question=QUESTION, answer=answer, sources=[WHO],
                           revise=True, reviser_backend="llm")
        self.assertTrue(out.revision.supplemented)
        self.assertEqual(out.revision.method, "llm_reviser")
        self.assertTrue(out.revision.end_marker_present)
        self.assertIn("diff", out.revision.model_dump())
        self.assertIsNotNone(out.revision.recheck)

    def test_recheck_fn_is_reusable(self):
        fn = build_recheck_fn([WHO], question=QUESTION)
        claims, verifications, fact = fn("空气污染每年导致约420万人过早死亡。")
        self.assertTrue(claims)
        self.assertTrue(verifications)
        self.assertEqual(fact.method, "local_source_anchored")


class _SmugglingReviser:
    """Editor stub that smuggles a hallucination, then removes it on critique."""

    accepts_critique = True
    method = "llm_reviser"

    def __call__(self, text, report, critique=None):
        if critique is None:
            return text + "\n月球背面存在一座巨型城市。"
        return text.replace("\n月球背面存在一座巨型城市。", "")


class TestGateEndToEnd(unittest.TestCase):
    def test_gate_forces_a_second_round_and_removes_the_smuggled_claim(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        out = validate(question=QUESTION, answer=answer, sources=[WHO],
                       revise=True, reviser=_SmugglingReviser())
        gate = out.revision.recheck
        self.assertEqual(len(gate.rounds), 2)          # first round failed -> re-edit
        self.assertFalse(gate.rounds[0].passed)
        self.assertTrue(gate.rounds[-1].passed)
        self.assertGreater(gate.final_score, gate.initial_score)
        self.assertNotIn("月球背面", out.revision.revised_text)
        self.assertEqual(out.answer, answer)           # original never mutated

    def test_gate_keeps_the_original_when_nothing_improves(self):
        class NoProgress:
            accepts_critique = True
            method = "llm_reviser"

            def __call__(self, text, report, critique=None):
                return text + ("\n月球背面存在一座巨型城市。" if critique is None else "")

        answer = "空气污染每年导致约420万人过早死亡。"
        # Critique is a no-op here, so the smuggled claim cannot be removed; the
        # gate must stop and report the leftover rather than loop forever.
        class Stuck:
            accepts_critique = True
            method = "llm_reviser"

            def __call__(self, text, report, critique=None):
                return text + "\n月球背面存在一座巨型城市。"

        out = validate(question=QUESTION, answer=answer, sources=[WHO],
                       revise=True, reviser=Stuck())
        gate = out.revision.recheck
        self.assertFalse(gate.passed)
        self.assertTrue(gate.leftover_inaccurate)


if __name__ == "__main__":
    unittest.main()
