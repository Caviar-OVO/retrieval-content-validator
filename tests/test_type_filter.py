"""Claim-type filtering in front of external judges.

The contract: only knowledge claims (fact / inference / reasoning) reach a judge,
and any judge output for an opinion is discarded. The judges here are *grounding*
judges (`evaluate_grounding`), which are separate from - and can never change -
the source-anchored factuality verdicts. These tests use fake judges, so they run
offline.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import (  # noqa: E402
    decompose,
    evaluate_factuality,
    evaluate_grounding,
    is_verifiable,
    validate,
)
from rva.evaluate import (  # noqa: E402
    FILTERED_VERDICT,
    NOT_APPLICABLE_VERDICT,
    VERIFIABLE_TYPES,
)
from rva.models import ClaimType, Source, Verdict  # noqa: E402
from rva.stages.verify import verify  # noqa: E402


def _claims_and_verifications():
    answer = "空气污染每年导致约420万人过早死亡。我认为这是最糟糕的环境问题。"
    sources = [Source(id="A", url="https://www.who.int/fact",
                      text="室外空气污染导致全球每年约420万人过早死亡。")]
    claims, _ = decompose(answer)
    return claims, verify(claims, sources)


class TestIsVerifiable(unittest.TestCase):
    def test_knowledge_claims_are_verifiable(self):
        self.assertTrue(is_verifiable(ClaimType.FACT))
        self.assertTrue(is_verifiable(ClaimType.INFERENCE))

    def test_only_opinion_is_not_verifiable(self):
        self.assertFalse(is_verifiable(ClaimType.OPINION))
        # Logical arguments are claims too (LLM Auditor): they are judged.
        self.assertTrue(is_verifiable(ClaimType.REASONING))

    def test_constant_matches(self):
        self.assertEqual(set(VERIFIABLE_TYPES), {ClaimType.FACT, ClaimType.INFERENCE, ClaimType.REASONING})


class TestBatchJudgeFiltering(unittest.TestCase):
    def test_opinion_is_not_sent_to_batch_judge(self):
        claims, verifications = _claims_and_verifications()
        received: dict = {}

        def batch(claims_arg, verifications_arg):
            received["claim_ids"] = [c.id for c in claims_arg]
            received["verification_ids"] = [v.claim_id for v in verifications_arg]
            return {c.id: (True, 0.9, "forced") for c in claims_arg}

        evaluate_grounding(claims, verifications, batch_judge=batch)

        opinion = next(c for c in claims if c.type is ClaimType.OPINION)
        self.assertNotIn(opinion.id, received["claim_ids"])
        self.assertEqual(received["claim_ids"], received["verification_ids"])
        self.assertEqual(received["claim_ids"], [c.id for c in claims if is_verifiable(c.type)])

    def test_judge_output_for_opinion_is_discarded(self):
        claims, verifications = _claims_and_verifications()
        opinion_id = next(c.id for c in claims if c.type is ClaimType.OPINION)

        # A naive judge returns a grounding verdict for the opinion too.
        def batch(claims_arg, verifications_arg):
            return {c.id: (True, 1.0, "forced") for c in claims_arg} | {
                opinion_id: (True, 1.0, "forced")
            }

        g = evaluate_grounding(claims, verifications, batch_judge=batch)
        opinion = next(it for it in g.items if it.claim_id == opinion_id)

        self.assertIsNone(opinion.grounded)
        self.assertIn("non-knowledge", opinion.rationale)
        self.assertEqual(g.skipped_claims, 1)
        self.assertEqual(g.judged_claims, 1)

        # The factuality verdict for the opinion stays Not Applicable regardless.
        f = evaluate_factuality(claims, verifications)
        self.assertEqual(next(v for v in f.verdicts if v.claim_id == opinion_id).verdict,
                         Verdict.NOT_APPLICABLE)
        self.assertEqual(f.not_applicable, 1)

    def test_single_judge_hook_skips_opinion(self):
        claims, verifications = _claims_and_verifications()
        seen: list = []

        def judge(claim, verification):
            seen.append(claim.id)
            return (True, 1.0, "forced")

        evaluate_grounding(claims, verifications, judge=judge)

        opinion_id = next(c.id for c in claims if c.type is ClaimType.OPINION)
        self.assertNotIn(opinion_id, seen)
        self.assertEqual(seen, [c.id for c in claims if is_verifiable(c.type)])

    def test_batch_judge_not_called_without_knowledge_claims(self):
        claims, _ = decompose("我认为这是最糟糕的环境问题。")
        verifications = verify(claims, [])

        def batch(claims_arg, verifications_arg):
            raise AssertionError("batch judge must not be called for opinions only")

        g = evaluate_grounding(claims, verifications, batch_judge=batch)
        self.assertEqual(g.skipped_claims, 1)
        self.assertEqual(g.judged_claims, 0)
        self.assertIsNone(g.mean_score)

        # Factuality is local and unaffected: an opinion-only answer is exempt.
        f = evaluate_factuality(claims, verifications)
        self.assertEqual(f.verifiable_claims, 0)
        self.assertEqual(f.verdicts[0].verdict, Verdict.NOT_APPLICABLE)
        self.assertEqual(f.score, 1.0)
        self.assertEqual(f.method, "local_source_anchored")

    def test_factuality_has_no_judge_hook(self):
        # The five-level verdict must not be delegatable to an external judge.
        claims, verifications = _claims_and_verifications()
        with self.assertRaises(TypeError):
            evaluate_factuality(claims, verifications, batch_judge=lambda *a: {})

    def test_filtered_verdict_constant(self):
        verdict, score, rationale = FILTERED_VERDICT
        self.assertIs(verdict, Verdict.NOT_APPLICABLE)
        self.assertEqual(score, 1.0)
        self.assertIn("needs no external verification", rationale)
        self.assertIs(NOT_APPLICABLE_VERDICT, FILTERED_VERDICT)


class TestFilteringEndToEnd(unittest.TestCase):
    def test_ragas_backend_sees_only_knowledge_claims(self):
        from unittest import mock
        try:
            import ragas  # noqa: F401
        except Exception:
            self.skipTest("ragas not installed")
        import pandas as pd
        from rva.integrations import ragas_batch_judge

        answer = (ROOT / "examples" / "answer.txt").read_text(encoding="utf-8")
        sources = [Source(**x) for x in __import__("json").loads(
            (ROOT / "examples" / "sources.json").read_text(encoding="utf-8"))]
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)

        captured = {}

        class FakeResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.9] * captured["n"]})

        def fake_evaluate(dataset, metrics=None, llm=None, **kwargs):
            captured["n"] = len(dataset.samples)
            captured["responses"] = [s.response for s in dataset.samples]
            return FakeResult()

        with mock.patch("ragas.evaluate", side_effect=fake_evaluate):
            ragas_batch_judge(sources=sources, llm=object())(claims, verifications)

        opinion_texts = {c.text for c in claims if c.type is ClaimType.OPINION}
        self.assertFalse(opinion_texts & set(captured["responses"]))
        # Every sample handed to Ragas must be a knowledge claim.
        verifiable_texts = {c.text for c in claims if is_verifiable(c.type)}
        self.assertTrue(set(captured["responses"]).issubset(verifiable_texts))

    def test_validate_reports_grounding_separately_from_factuality(self):
        from unittest import mock
        try:
            import ragas  # noqa: F401
        except Exception:
            self.skipTest("ragas not installed")
        import pandas as pd

        sources = [Source(id="A", url="https://www.who.int/fact",
                          text="室外空气污染导致全球每年约420万人过早死亡。")]
        answer = "空气污染每年导致约420万人过早死亡。"

        class FakeResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.2]})

        with mock.patch("ragas.evaluate", return_value=FakeResult()):
            out = validate(answer=answer, sources=sources, judge="ragas")

        # An ungrounded judge score (0.2) never touches the five-level verdict...
        self.assertEqual(out.factuality.accurate, 1)
        self.assertEqual(out.factuality.method, "local_source_anchored")
        # ...and shows up only in the separate grounding report.
        self.assertIsNotNone(out.grounding)
        self.assertEqual(out.grounding.ungrounded_count, 1)
        self.assertEqual(out.grounding.items[0].grounded, False)


if __name__ == "__main__":
    unittest.main()
