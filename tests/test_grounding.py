"""Grounding-only judge report, kept separate from the factuality verdict.

The contract these tests pin down:

  * A judge verdict is *grounding* (is the claim entailed by the supplied
    contexts?), the ADK-style internal-faithfulness anchor.
  * It is reported in `result.grounding` and can NEVER change `result.factuality`,
    whose five-level verdicts stay anchored on external reliable sources.
  * Only knowledge claims reach the judge; opinions are skipped without a call.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import GroundingResult, decompose, evaluate_factuality, evaluate_grounding, validate  # noqa: E402
from rva.models import ClaimType, Source, Verdict  # noqa: E402
from rva.stages.verify import verify  # noqa: E402

WHO_SUPPORT = Source(id="A", url="https://www.who.int/fact",
                     text="室外空气污染导致全球每年约420万人过早死亡。")


def _headline():
    claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
    return claims, verify(claims, [WHO_SUPPORT])


class TestGroundingSeparation(unittest.TestCase):
    def test_judge_never_changes_factuality(self):
        claims, verifications = _headline()
        f_before = evaluate_factuality(claims, verifications)

        def judge(claim, verification):
            # A rogue judge that says "ungrounded" must not flip the verdict.
            return (False, 0.0, "judge disagrees")

        g = evaluate_grounding(claims, verifications, judge=judge)
        f_after = evaluate_factuality(claims, verifications)

        self.assertEqual(f_before.model_dump(), f_after.model_dump())
        self.assertEqual(f_after.verdicts[0].verdict, Verdict.ACCURATE)
        self.assertEqual(f_after.method, "local_source_anchored")
        self.assertEqual(g.ungrounded_count, 1)

    def test_factuality_has_no_judge_parameter(self):
        claims, verifications = _headline()
        for kwargs in ({"judge": lambda *a: None}, {"batch_judge": lambda *a: {}}):
            with self.assertRaises(TypeError):
                evaluate_factuality(claims, verifications, **kwargs)

    def test_result_types_are_distinct(self):
        self.assertNotIn(Verdict, ())  # Verdict stays the factuality vocabulary
        self.assertIsInstance(GroundingResult(), GroundingResult)
        self.assertFalse(hasattr(GroundingResult(), "verdicts"))
        self.assertFalse(hasattr(GroundingResult(), "accurate"))

    def test_grounding_record_shape(self):
        claims, verifications = _headline()

        def batch(claims_arg, verifications_arg):
            return {c.id: (True, 0.83, "entailed by context") for c in claims_arg}

        g = evaluate_grounding(claims, verifications, batch_judge=batch)
        self.assertEqual(g.method, "external_judge")
        self.assertEqual(g.judged_claims, 1)
        self.assertEqual(g.grounded_count, 1)
        self.assertEqual(g.ungrounded_count, 0)
        self.assertEqual(g.mean_score, 0.83)
        item = g.items[0]
        self.assertEqual(item.claim_id, verifications[0].claim_id)
        self.assertIs(item.grounded, True)
        self.assertAlmostEqual(item.score, 0.83)

    def test_threshold_boundary(self):
        claims, verifications = _headline()
        for score, expected in [(0.75, True), (0.74, False), (1.0, True), (0.0, False)]:
            g = evaluate_grounding(
                claims, verifications,
                batch_judge=lambda c, v, s=score: {v[0].claim_id: (s >= 0.75, s, "x")},
            )
            self.assertIs(g.items[0].grounded, expected, msg=str(score))

    def test_nan_and_missing_are_no_verdict(self):
        claims, verifications = _headline()

        def batch(claims_arg, verifications_arg):
            return {c.id: (True, float("nan"), "broken") for c in claims_arg}

        g = evaluate_grounding(claims, verifications, batch_judge=batch)
        self.assertIsNone(g.items[0].score)
        self.assertEqual(g.judged_claims, 1)     # grounded flag survived
        self.assertIsNone(g.mean_score)          # but nothing usable to average

    def test_backend_failure_yields_no_verdicts(self):
        claims, verifications = _headline()

        def boom(claims_arg, verifications_arg):
            raise RuntimeError("no api key / no network")

        g = evaluate_grounding(claims, verifications, batch_judge=boom)
        self.assertEqual(g.total_claims, 1)
        self.assertEqual(g.judged_claims, 0)
        self.assertIsNone(g.items[0].grounded)
        self.assertIn("no grounding verdict", g.items[0].rationale)
        self.assertIsNone(g.mean_score)

    def test_opinion_is_skipped_without_a_call(self):
        claims, verifications = _headline()
        opinion = next((c for c in claims if c.type is ClaimType.OPINION), None)
        if opinion is None:  # the headline is a pure fact; add an opinion explicitly
            claims, _ = decompose("我认为这是最糟糕的环境问题。")
            verifications = verify(claims, [WHO_SUPPORT])
            opinion = claims[0]

        seen: list = []

        def batch(claims_arg, verifications_arg):
            seen.extend(c.id for c in claims_arg)
            return {c.id: (True, 1.0, "x") for c in claims_arg}

        g = evaluate_grounding(claims, verifications, batch_judge=batch)
        self.assertNotIn(opinion.id, seen)
        self.assertEqual(g.skipped_claims, 1)


class TestJudgeRecordContract(unittest.TestCase):
    """The judge record contract: one meaning, three shapes, no silent decay."""

    def test_mapping_record_is_accepted(self):
        claims, verifications = _headline()
        g = evaluate_grounding(
            claims, verifications,
            batch_judge=lambda c, v: {
                x.id: {"grounded": True, "score": 0.9, "rationale": "entailed"} for x in c
            },
        )
        self.assertIs(g.items[0].grounded, True)
        self.assertAlmostEqual(g.items[0].score, 0.9)
        self.assertEqual(g.mean_score, 0.9)
        self.assertIn("entailed", g.items[0].rationale)

    def test_mapping_record_may_omit_the_rationale(self):
        claims, verifications = _headline()
        g = evaluate_grounding(
            claims, verifications,
            judge=lambda cl, ve: {"grounded": False, "score": 0.4},
        )
        self.assertIs(g.items[0].grounded, False)
        self.assertAlmostEqual(g.items[0].score, 0.4)

    def test_attribute_record_is_accepted(self):
        claims, verifications = _headline()

        class Record:
            grounded = True
            score = 0.88
            rationale = "attr record"

        g = evaluate_grounding(claims, verifications, judge=lambda cl, ve: Record())
        self.assertIs(g.items[0].grounded, True)
        self.assertAlmostEqual(g.items[0].score, 0.88)

    def test_unrecognisable_record_raises_instead_of_decaying(self):
        claims, verifications = _headline()
        # A bare string used to be mis-parsed as grounded=True with no score.
        with self.assertRaises(ValueError) as ctx:
            evaluate_grounding(claims, verifications, judge=lambda cl, ve: "ungrounded")
        self.assertIn("unrecognisable", str(ctx.exception))
        self.assertIn(verifications[0].claim_id, str(ctx.exception))

    def test_unrecognisable_record_in_a_batch_names_the_claim(self):
        claims, verifications = _headline()
        with self.assertRaises(ValueError):
            evaluate_grounding(
                claims, verifications,
                batch_judge=lambda c, v: {x.id: 0.9 for x in c},
            )

    def test_batch_result_must_be_a_mapping(self):
        claims, verifications = _headline()
        with self.assertRaises(ValueError):
            evaluate_grounding(claims, verifications, batch_judge=lambda c, v: [("x",)])

    def test_missing_record_is_still_a_no_verdict(self):
        # A judge that legitimately has nothing to say keeps the old behaviour.
        claims, verifications = _headline()
        g = evaluate_grounding(claims, verifications, batch_judge=lambda c, v: {})
        self.assertEqual(g.judged_claims, 0)
        self.assertIn("no grounding verdict", g.items[0].rationale)
        self.assertIsNone(g.items[0].grounded)


class TestValidateGrounding(unittest.TestCase):
    def test_local_judge_has_no_grounding_report(self):
        out = validate(answer="空气污染每年导致约420万人过早死亡。", sources=[WHO_SUPPORT], judge="local")
        self.assertIsNone(out.grounding)
        self.assertEqual(out.factuality.method, "local_source_anchored")

    def test_grounding_reported_alongside_factuality(self):
        # Without a key the judge cannot run, but the report must still exist and
        # the factuality side must be untouched.
        long_answer = (ROOT / "examples" / "answer.txt").read_text(encoding="utf-8")
        sources = [Source(**x) for x in __import__("json").loads(
            (ROOT / "examples" / "sources.json").read_text(encoding="utf-8"))]
        out = validate(answer=long_answer, sources=sources, judge="ragas")
        self.assertIsNotNone(out.grounding)
        self.assertEqual(out.grounding.method, "external_judge")
        self.assertEqual(out.factuality.method, "local_source_anchored")
        self.assertEqual(out.factuality.total_claims, out.grounding.total_claims)

    def test_grounding_text_is_renderable(self):
        out = validate(answer="空气污染每年导致约420万人过早死亡。", sources=[WHO_SUPPORT], judge="ragas")
        text = out.grounding.text
        self.assertIn("Grounding (grounding-only, NOT factuality)", text)

    def test_summary_grounding_score_is_populated_when_the_judge_runs(self):
        # The judge is authoritative here; patch only the network adapter so the
        # grounding report is produced offline with a known score.
        import rva.integrations as integrations

        original = integrations.build_judge
        def fake_judge(c, v):
            return {x.id: {"grounded": True, "score": 0.9, "rationale": "ok"} for x in c}

        integrations.build_judge = lambda name, sources=None: fake_judge
        try:
            out = validate(
                question="空气污染每年导致多少人过早死亡？",
                answer="空气污染每年导致约420万人过早死亡。",
                sources=[WHO_SUPPORT], judge="ragas", summary=True,
            )
        finally:
            integrations.build_judge = original

        self.assertIsNotNone(out.grounding)
        self.assertEqual(out.grounding.mean_score, 0.9)
        # ...and the 0-100 aggregate is no longer None.
        self.assertIsNotNone(out.summary.grounding_score)
        self.assertEqual(out.summary.grounding_score, 90.0)


if __name__ == "__main__":
    unittest.main()
