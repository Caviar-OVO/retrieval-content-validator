"""Per-claim evidence-efficacy checks: authority, freshness, consistency, traffic.

All offline. Both critic tracks share these because both route through verify().
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import (  # noqa: E402
    LLMCritic,
    decompose,
    validate,
)
from rva.efficacy import (  # noqa: E402
    assess_consistency,
    assess_freshness,
    assess_source_quality,
    grade_for_tier,
    score_support,
    suspicious_traffic_ids,
)
from rva.models import AuthorityGrade, Source  # noqa: E402
from rva.stages.verify import verify  # noqa: E402

WHO = Source(id="S1", url="https://www.who.int/fact",
             text="室外空气污染导致全球每年约420万人过早死亡。")
BLOG = Source(id="S2", url="https://blog.example.com/x", tag="blog", traffic=0.95,
              text="空气污染导致数百万人死亡的说法其实被夸大了。")
QUESTION = "空气污染每年导致多少人过早死亡？"


def _v(answer="空气污染每年导致约420万人过早死亡。", sources=None, question=QUESTION):
    claims, _ = decompose(answer, question=question)
    return verify(claims, sources or [WHO])[0]


class TestAuthorityRating(unittest.TestCase):
    def test_grade_mapping(self):
        self.assertIs(grade_for_tier("primary-official"), AuthorityGrade.OFFICIAL)
        self.assertIs(grade_for_tier("peer-reviewed"), AuthorityGrade.PROFESSIONAL)
        self.assertIs(grade_for_tier("blog"), AuthorityGrade.SELF_MEDIA)
        self.assertIs(grade_for_tier("unknown"), AuthorityGrade.UNVERIFIABLE)

    def test_official_source_scores_official(self):
        v = _v(sources=[WHO])
        self.assertIs(v.source_quality.grade, AuthorityGrade.OFFICIAL)
        self.assertEqual(v.source_quality.counts.get("official"), 1)

    def test_self_media_only_is_graded_self_media(self):
        v = _v(sources=[BLOG])
        self.assertIs(v.source_quality.grade, AuthorityGrade.SELF_MEDIA)

    def test_best_bound_source_wins(self):
        v = _v(sources=[WHO, BLOG])
        self.assertIs(v.source_quality.grade, AuthorityGrade.OFFICIAL)
        self.assertEqual(v.source_quality.counts.get("official"), 1)
        self.assertEqual(v.source_quality.counts.get("self_media"), 1)

    def test_no_bound_source_is_unverifiable(self):
        claims, _ = decompose("今天天气不错，适合散步。")
        v = verify(claims, [])[0]
        self.assertIs(v.source_quality.grade, AuthorityGrade.UNVERIFIABLE)


class TestFreshness(unittest.TestCase):
    def test_static_claim_is_not_enforced(self):
        v = _v()
        self.assertFalse(v.freshness.dynamic)
        self.assertFalse(v.freshness.stale)

    def test_dynamic_claim_undated_is_stale_with_check_time(self):
        v = _v(answer="当前的空气污染政策要求降低排放。", question="当前的空气污染政策是什么？")
        self.assertTrue(v.freshness.dynamic)
        self.assertFalse(v.freshness.has_date)
        self.assertTrue(v.freshness.stale)
        self.assertIn("check time", v.freshness.note)
        # checked_at is an ISO timestamp
        datetime.fromisoformat(v.freshness.checked_at)

    def test_dynamic_claim_with_recent_date_is_fresh(self):
        year = datetime.now(timezone.utc).year
        src = Source(id="S9", url="https://www.gov.cn/policy",
                     text=f"{year}年的空气污染政策要求降低排放。")
        v = _v(answer=f"{year}年的空气污染政策要求降低排放。",
               question="当前的空气污染政策是什么？", sources=[src])
        self.assertTrue(v.freshness.dynamic)
        self.assertTrue(v.freshness.has_date)
        self.assertFalse(v.freshness.stale)

    def test_old_dated_dynamic_claim_is_stale(self):
        src = Source(id="S8", url="https://www.gov.cn/policy",
                     text="2010年的空气污染政策要求降低排放。")
        v = _v(answer="2010年的空气污染政策要求降低排放。",
               question="当前的空气污染政策是什么？", sources=[src])
        self.assertTrue(v.freshness.stale)

    def test_supplied_now_is_respected(self):
        v = _v(answer="当前的价格是多少？当前价格是5元。", question="当前价格是多少？",
               sources=[Source(id="S7", url="https://x.com/p", text="当前价格是5元。")])
        self.assertTrue(v.freshness.dynamic)
        # deterministic given a fixed clock
        f = assess_freshness("当前价格是5元。", v.evidence, [Source(id="S7", url="https://x.com/p", text="当前价格是5元。")],
                             now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertTrue(f.checked_at.startswith("2026-01-01"))


class TestConsistency(unittest.TestCase):
    def test_numbers_matching_is_consistent(self):
        v = _v(sources=[WHO])
        self.assertTrue(v.consistency.consistent)
        self.assertFalse(v.consistency.number_mismatch)

    def test_number_absent_from_sources_flags_mismatch(self):
        src = Source(id="S3", url="https://www.who.int/other",
                     text="室外空气污染导致全球每年约700万人过早死亡。")
        v = _v(answer="空气污染每年导致约420万人过早死亡。", sources=[src])
        self.assertFalse(v.consistency.consistent)
        self.assertTrue(v.consistency.number_mismatch)

    def test_qualifier_overreach_is_flagged(self):
        src = Source(id="S4", url="https://www.who.int/p",
                     text="部分研究表明空气污染与部分人群的过早死亡相关。")
        v = _v(answer="空气污染必然导致所有人过早死亡。",
               question="空气污染是否导致所有人过早死亡？", sources=[src])
        self.assertTrue(v.consistency.qualifier_overreach)

    def test_no_support_is_consistent_by_default(self):
        claims, _ = decompose("月球背面存在一座巨型城市。")
        v = verify(claims, [WHO])[0]
        self.assertTrue(v.consistency.consistent)

    def test_assess_consistency_signature_is_usable(self):
        v = _v(sources=[WHO])
        c = assess_consistency("空气污染每年导致约420万人过早死亡。", v.evidence, [WHO])
        self.assertTrue(c.consistent)


class TestTrafficVsValidity(unittest.TestCase):
    def test_high_traffic_low_reliability_support_is_flagged(self):
        src = Source(id="S2", url="https://blog.example.com/x", tag="blog", traffic=0.95,
                     text="室外空气污染每年导致全球约420万人过早死亡，这很重要。")
        v = _v(answer="空气污染每年导致约420万人过早死亡。", sources=[src])
        self.assertIn("S2", v.popularity_suspect_source_ids)
        self.assertTrue(v.popularity_bias.biased)

    def test_official_high_traffic_is_not_flagged(self):
        src = Source(id="S1", url="https://www.who.int/f", traffic=0.99,
                     text="室外空气污染导致全球每年约420万人过早死亡。")
        v = _v(sources=[src])
        self.assertEqual(v.popularity_suspect_source_ids, [])

    def test_suspicious_helper(self):
        src = Source(id="S2", url="https://blog.example.com/x", tag="blog", traffic=0.9, text="x")
        v = _v(answer="空气污染每年导致约420万人过早死亡。", sources=[src])
        self.assertEqual(suspicious_traffic_ids(v.evidence), v.popularity_suspect_source_ids)


class TestSupportScore(unittest.TestCase):
    def test_clean_case_matches_reliability_mean(self):
        v = _v(sources=[WHO])
        self.assertEqual(v.support_score, 0.95)  # unchanged when no violations

    def test_traffic_driven_support_is_discounted(self):
        src = Source(id="S2", url="https://blog.example.com/x", tag="blog", traffic=0.95,
                     text="室外空气污染每年导致全球约420万人过早死亡，这很重要。")
        v = _v(sources=[src])
        # 0.4 (blog reliability) * 0.85 (popularity) = 0.34
        self.assertLess(v.support_score, 0.4)

    def test_inconsistency_discounts(self):
        src = Source(id="S3", url="https://www.who.int/other",
                     text="室外空气污染导致全球每年约700万人过早死亡。")
        v = _v(answer="空气污染每年导致约420万人过早死亡。", sources=[src])
        # 0.95 * 0.7 (number mismatch) = 0.665 -> rounds to 0.66
        self.assertAlmostEqual(v.support_score, 0.66, places=2)


class TestBothCriticTracks(unittest.TestCase):
    def test_deterministic_track_exposes_the_checks(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic_backend="deterministic")
        v = out.verifications[0]
        self.assertIs(v.source_quality.grade, AuthorityGrade.OFFICIAL)
        self.assertIsNotNone(v.freshness)
        self.assertIsNotNone(v.consistency)

    def test_llm_track_exposes_the_same_checks(self):
        import json
        payload = json.dumps({"claims": [{
            "text": "空气污染每年导致约420万人过早死亡。", "type": "fact", "confidence": 0.9,
            "verdict": "accurate", "justification": "S1 supports it.", "evidence": ["S1"]}]})

        def client(c):
            msg = type("M", (), {"content": c})()
            ch = type("C", (), {"message": msg})()
            comp = type("X", (), {"create": lambda self, **k: type("R", (), {"choices": [ch]})()})()
            return type("Cl", (), {"chat": type("C2", (), {"completions": comp})()})()

        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic=LLMCritic(llm=client(payload)))
        v = out.verifications[0]
        self.assertIs(v.source_quality.grade, AuthorityGrade.OFFICIAL)
        self.assertIsNotNone(v.freshness)
        self.assertIsNotNone(v.consistency)
        self.assertAlmostEqual(v.support_score, 0.95)


if __name__ == "__main__":
    unittest.main()
