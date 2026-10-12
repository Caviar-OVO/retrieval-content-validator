"""Five-level, source-anchored verdicts (ADK hallucinations_v1 taxonomy).

  Accurate / Inaccurate / Disputed / Unsupported / Not Applicable

The verdict must follow external *reliable sources*, not the model's own say-so,
so these tests pin down the reliability-threshold behaviour and the ADK accuracy
score. All offline.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import evaluate_factuality, validate  # noqa: E402
from rva.evaluate import verdict_score  # noqa: E402
from rva.models import Source, Verdict  # noqa: E402


def _score(answer, sources):
    return validate(answer=answer, sources=sources).factuality


WHO_SUPPORT = Source(id="S1", url="https://www.who.int/fact-sheet",
                     text="室外空气污染导致全球每年约420万人过早死亡。")
WHO_REFUTE = Source(id="S2", url="https://www.who.int/correction",
                    text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")
NATURE_SUPPORT = Source(id="S3", url="https://www.nature.com/articles/x",
                        text="室外空气污染导致全球每年约420万人过早死亡。")
NATURE_REFUTE = Source(id="S4", url="https://www.nature.com/articles/y",
                       text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")
BLOG_REFUTE = Source(id="S5", url="https://blog.example.com/x", tag="blog",
                     text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")


class TestFiveLevelVerdicts(unittest.TestCase):
    def test_accurate_from_reliable_source(self):
        f = _score("空气污染每年导致约420万人过早死亡。", [WHO_SUPPORT])
        self.assertEqual(f.accurate, 1)
        self.assertEqual(f.verdicts[0].verdict, Verdict.ACCURATE)
        self.assertEqual(f.score, 1.0)

    def test_inaccurate_from_reliable_refutation(self):
        f = _score("空气污染每年导致约420万人过早死亡。", [WHO_REFUTE])
        self.assertEqual(f.inaccurate, 1)
        self.assertEqual(f.verdicts[0].verdict, Verdict.INACCURATE)
        self.assertEqual(f.score, 0.0)

    def test_disputed_when_reliable_sources_conflict(self):
        f = _score("空气污染每年导致约420万人过早死亡。", [WHO_SUPPORT, NATURE_REFUTE])
        self.assertEqual(f.disputed, 1)
        self.assertEqual(f.verdicts[0].verdict, Verdict.DISPUTED)
        self.assertEqual(f.score, 0.0)

    def test_unsupported_when_nothing_reliable(self):
        f = _score("月球背面存在一座巨型城市。", [Source(id="S", url="https://example.com/x", text="")])
        self.assertEqual(f.unsupported, 1)
        self.assertEqual(f.verdicts[0].verdict, Verdict.UNSUPPORTED)
        self.assertEqual(f.score, 0.0)

    def test_not_applicable_for_opinion(self):
        f = _score("我认为这是最糟糕的环境问题。", [])
        self.assertEqual(f.not_applicable, 1)
        self.assertEqual(f.verdicts[0].verdict, Verdict.NOT_APPLICABLE)
        self.assertEqual(f.score, 1.0)


class TestReliabilityAnchor(unittest.TestCase):
    def test_blog_refutation_does_not_make_it_inaccurate(self):
        f = _score("空气污染每年导致约420万人过早死亡。", [BLOG_REFUTE])
        self.assertEqual(f.inaccurate, 0)
        self.assertEqual(f.unsupported, 1)

    def test_blog_refutation_does_not_dispute_reliable_support(self):
        f = _score("空气污染每年导致约420万人过早死亡。", [WHO_SUPPORT, BLOG_REFUTE])
        self.assertEqual(f.accurate, 1)
        self.assertEqual(f.disputed, 0)
        self.assertEqual(f.inaccurate, 0)
        self.assertEqual(f.score, 1.0)

    def test_two_reliable_sources_agree_is_accurate(self):
        f = _score("空气污染每年导致约420万人过早死亡。", [WHO_SUPPORT, NATURE_SUPPORT])
        self.assertEqual(f.accurate, 1)
        self.assertEqual(f.score, 1.0)


class TestAccuracyScore(unittest.TestCase):
    def test_mixed_answer(self):
        answer = (
            "空气污染每年导致约420万人过早死亡。"          # accurate
            "月球背面存在一座巨型城市。"                    # unsupported
            "我认为这是最糟糕的环境问题。"                  # not applicable
        )
        f = _score(answer, [WHO_SUPPORT])
        self.assertEqual(f.total_claims, 3)
        self.assertEqual(f.verifiable_claims, 2)
        self.assertEqual(f.accurate, 1)
        self.assertEqual(f.unsupported, 1)
        self.assertEqual(f.not_applicable, 1)
        # (1 accurate + 1 not_applicable) / 3
        self.assertAlmostEqual(f.score, 0.67, places=2)

    def test_verdict_values_are_stable_strings(self):
        self.assertEqual(Verdict.ACCURATE.value, "accurate")
        self.assertEqual(Verdict.INACCURATE.value, "inaccurate")
        self.assertEqual(Verdict.DISPUTED.value, "disputed")
        self.assertEqual(Verdict.UNSUPPORTED.value, "unsupported")
        self.assertEqual(Verdict.NOT_APPLICABLE.value, "not_applicable")


class TestVerdictScore(unittest.TestCase):
    def test_canonical_scores(self):
        self.assertEqual(verdict_score(Verdict.ACCURATE), 1.0)
        self.assertEqual(verdict_score(Verdict.NOT_APPLICABLE), 1.0)
        self.assertEqual(verdict_score(Verdict.INACCURATE), 0.0)
        self.assertEqual(verdict_score(Verdict.DISPUTED), 0.0)
        self.assertEqual(verdict_score(Verdict.UNSUPPORTED), 0.0)

    def test_judge_score_stays_out_of_factuality(self):
        from unittest import mock
        from rva import decompose
        from rva.integrations import ragas_batch_judge
        try:
            import ragas  # noqa: F401
        except Exception:
            self.skipTest("ragas not installed")
        import pandas as pd

        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        sources = [Source(id="A", url="https://www.who.int/fact", text="室外空气污染导致全球每年约420万人过早死亡。")]
        from rva.stages.verify import verify
        verifications = verify(claims, sources)

        class FakeResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.5]})

        with mock.patch("ragas.evaluate", return_value=FakeResult()):
            judged = ragas_batch_judge(sources=sources, llm=object())(claims, verifications)

        grounded, score, rationale = judged[verifications[0].claim_id]
        # 0.5 is below the grounding threshold: grounding-only, raw score kept.
        self.assertIs(grounded, False)
        self.assertEqual(score, 0.5)
        self.assertIn("0.50", rationale)

        # The five-level factuality verdict is untouched by that judge score.
        f = evaluate_factuality(claims, verifications)
        self.assertEqual(f.verdicts[0].verdict, Verdict.ACCURATE)
        self.assertEqual(f.score, 1.0)
        self.assertEqual(f.method, "local_source_anchored")


class TestFactualityAliases(unittest.TestCase):
    """The older `faithfulness` names must keep working for existing callers."""

    def test_faithfulness_function_alias(self):
        from rva import evaluate_faithfulness, evaluate_factuality
        self.assertIs(evaluate_faithfulness, evaluate_factuality)

    def test_result_alias(self):
        from rva import FactualityResult, FaithfulnessResult
        self.assertIs(FaithfulnessResult, FactualityResult)

    def test_validation_result_property_alias(self):
        sources = [Source(id="A", url="https://www.who.int/fact",
                          text="室外空气污染导致全球每年约420万人过早死亡。")]
        out = validate(answer="空气污染每年导致约420万人过早死亡。", sources=sources)
        self.assertIs(out.faithfulness, out.factuality)

    def test_cli_accepts_both_flags(self):
        import io
        import contextlib
        from rva.cli import main
        args = ["--answer-file", str(ROOT / "examples" / "answer.txt"),
                "--sources", str(ROOT / "examples" / "sources.json")]
        for flag in ("--factuality", "--faithfulness"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(args + [flag])
            self.assertEqual(code, 0)
            self.assertIn("Factuality: five-level verdicts", buf.getvalue())


if __name__ == "__main__":
    unittest.main()