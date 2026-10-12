"""Behavior tests for the three-stage pipeline and factuality metric.

Runs with the standard library only:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import decompose, evaluate_factuality, evaluate_grounding, render, validate  # noqa: E402
from rva.models import ClaimType, Decision, Source, Verdict  # noqa: E402
from rva.stages.decompose import classify  # noqa: E402
from rva.stages.output import RELIABILITY_THRESHOLD  # noqa: E402
from rva.stages.verify import score_source, verify  # noqa: E402
from rva.text import split_claims  # noqa: E402


class TestText(unittest.TestCase):
    def test_split_claims_strips_markers(self):
        claims = split_claims("- 第一句。第二句！\n3. 第三句？")
        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0], "第一句。")


class TestDecompose(unittest.TestCase):
    def test_classifies_four_types(self):
        answer = (
            "空气污染每年导致约420万人过早死亡。"      # fact
            "因此，污染越严重的城市风险越高。"           # reasoning
            "我认为这个说法被夸大了。"                   # opinion
            "有研究可能表明两者相关。"                   # inference
        )
        claims, stats = decompose(answer)
        types = {c.type for c in claims}
        self.assertIn(ClaimType.FACT, types)
        self.assertIn(ClaimType.REASONING, types)
        self.assertIn(ClaimType.OPINION, types)
        self.assertIn(ClaimType.INFERENCE, types)
        self.assertEqual(stats.total, len(claims))


class TestVerify(unittest.TestCase):
    def test_who_is_high_reliability(self):
        profile = score_source(Source(id="s", url="https://www.who.int/news/x"))
        self.assertEqual(profile.tier, "primary-official")
        self.assertGreaterEqual(profile.score, 0.9)

    def test_blog_is_low_reliability(self):
        profile = score_source(Source(id="s", url="https://blog.example.com/post", tag="blog"))
        self.assertLessEqual(profile.score, 0.4)

    def test_detects_assertion_level_contradiction(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        sources = [
            Source(id="A", url="https://www.who.int/fact",
                   text="室外空气污染导致全球每年约420万人过早死亡。"),
            Source(id="B", url="https://blog.example.com/x", tag="blog",
                   text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。"),
        ]
        results = verify(claims, sources)
        self.assertGreaterEqual(len(results[0].evidence.supports), 1)
        self.assertTrue(results[0].conflict)

    def test_weak_procedural_negation_is_not_a_contradiction(self):
        claims, _ = decompose("有研究可能表明PM2.5与死亡率相关。")
        sources = [Source(id="P", url="https://arxiv.org/abs/1",
                          text="初步结果表明污染暴露与死亡率之间存在关联，但尚未经过同行评审。")]
        results = verify(claims, sources)
        self.assertEqual(len(results[0].evidence.contradicts), 0)

    def test_popularity_bias_flagged(self):
        claims, _ = decompose("空气污染可能没有官方说的那么严重。")
        sources = [Source(id="B", url="https://blog.example.com/x", tag="blog", traffic=0.9,
                          text="空气污染没有官方说的那么严重，危害被夸大了。")]
        results = verify(claims, sources)
        self.assertTrue(results[0].popularity_bias.biased)


class TestOutput(unittest.TestCase):
    def test_opinion_is_contextualized(self):
        claims, _ = decompose("我认为空气污染被夸大了。")
        report = render(claims, verify(claims, []))
        self.assertEqual(report.items[0].decision, Decision.CONTEXTUALIZE)
        self.assertTrue(any("subjective" in b.lower() for b in report.items[0].boundary_conditions))


class TestFactuality(unittest.TestCase):
    def test_score_bounds_and_denominator(self):
        answer = "空气污染每年导致约420万人过早死亡。我认为这是最严重的环境问题。"
        sources = [Source(id="A", url="https://www.who.int/fact",
                          text="室外空气污染导致全球每年约420万人过早死亡。")]
        result = validate(answer=answer, sources=sources)
        f = result.factuality
        self.assertGreaterEqual(f.score, 0.0)
        self.assertLessEqual(f.score, 1.0)
        # the opinion is exempt from verification, so only the fact is judged
        self.assertEqual(f.verifiable_claims, 1)
        self.assertEqual(f.not_applicable, 1)
        self.assertEqual(f.total_claims, 2)

    def test_unsupported_claim_lowers_score(self):
        answer = "空气污染每年导致约420万人过早死亡。月球上发现了大规模城市。"
        sources = [Source(id="A", url="https://www.who.int/fact",
                          text="室外空气污染导致全球每年约420万人过早死亡。")]
        f = validate(answer=answer, sources=sources).factuality
        self.assertLess(f.score, 1.0)
        self.assertGreaterEqual(f.unsupported, 1)

    def test_low_reliability_refutation_is_not_inaccurate(self):
        # A single low-reliability blog cannot make a fact Inaccurate: it is
        # UNSUPPORTED at worst, because the verdict is anchored on sources above
        # the reliability threshold.
        answer = "空气污染每年导致约420万人过早死亡。"
        sources = [Source(id="B", url="https://blog.example.com/x", tag="blog",
                          text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")]
        f = validate(answer=answer, sources=sources).factuality
        self.assertEqual(f.inaccurate, 0)
        self.assertEqual(f.unsupported, 1)
        self.assertEqual(f.score, 0.0)

    def test_reliable_refutation_is_inaccurate(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        sources = [Source(id="W", url="https://www.who.int/fact-sheet",
                          text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")]
        f = validate(answer=answer, sources=sources).factuality
        self.assertEqual(f.inaccurate, 1)
        self.assertEqual(f.score, 0.0)

    def test_conflicting_reliable_sources_are_disputed(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        sources = [
            Source(id="A", url="https://www.who.int/fact",
                   text="室外空气污染导致全球每年约420万人过早死亡。"),
            Source(id="N", url="https://www.nature.com/articles/x",
                   text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。"),
        ]
        f = validate(answer=answer, sources=sources).factuality
        self.assertEqual(f.disputed, 1)
        self.assertEqual(f.score, 0.0)

    def test_low_reliability_contradiction_does_not_sink_supported_fact(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        sources = [
            Source(id="A", url="https://www.who.int/fact",
                   text="室外空气污染导致全球每年约420万人过早死亡。"),
            Source(id="B", url="https://blog.example.com/x", tag="blog",
                   text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。"),
        ]
        f = validate(answer=answer, sources=sources).factuality
        self.assertEqual(f.inaccurate, 0)
        self.assertEqual(f.disputed, 0)
        self.assertEqual(f.accurate, 1)
        self.assertEqual(f.score, 1.0)

    def test_opinion_is_not_applicable(self):
        answer = "我认为这是最严重的环境问题。"
        f = validate(answer=answer, sources=[]).factuality
        self.assertEqual(f.verifiable_claims, 0)
        self.assertEqual(f.not_applicable, 1)
        self.assertEqual(f.total_claims, 1)
        # Not Applicable is "correct by exemption" in the ADK accuracy score.
        self.assertEqual(f.score, 1.0)
        self.assertEqual(f.verdicts[0].verdict, Verdict.NOT_APPLICABLE)
        self.assertIn("needs no external verification", f.verdicts[0].rationale)

    def test_grounding_judge_does_not_change_factuality(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        sources = [Source(id="A", url="https://www.who.int/fact",
                          text="室外空气污染导致全球每年约420万人过早死亡。")]
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)
        calls = {"n": 0}

        def judge(claim, verification):
            calls["n"] += 1
            return (False, 0.0, "judge says ungrounded")

        f = evaluate_factuality(claims, verifications)
        g = evaluate_grounding(claims, verifications, judge=judge)

        # The judge is consulted only for grounding, never for the verdict.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(f.method, "local_source_anchored")
        self.assertEqual(f.verdicts[0].verdict, Verdict.ACCURATE)
        self.assertEqual(f.score, 1.0)
        self.assertEqual(g.ungrounded_count, 1)
        self.assertEqual(g.items[0].grounded, False)


class TestPipeline(unittest.TestCase):
    def test_bundled_example_end_to_end(self):
        answer = (ROOT / "examples" / "answer.txt").read_text(encoding="utf-8")
        sources = [Source(**item) for item in json.loads((ROOT / "examples" / "sources.json").read_text(encoding="utf-8"))]
        out = validate(answer=answer, sources=sources, domain="public-health")

        self.assertEqual(len(out.report.items), len(out.claims))
        for item in out.report.items:
            self.assertTrue(item.challengeable)
            self.assertTrue(item.boundary_conditions)
            self.assertTrue(0.0 <= item.credibility <= 1.0)
        self.assertTrue(0.0 <= out.factuality.score <= 1.0)

    def test_empty_answer_rejected(self):
        with self.assertRaises(ValueError):
            validate(answer="   ")


if __name__ == "__main__":
    unittest.main()