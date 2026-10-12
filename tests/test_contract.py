"""The public output contract, pinned so the documented table cannot drift.

These tests assert the *structure* the README / SKILL.md promise (top-level
fields, the four reports, the five-level vocabulary, the honesty codes, the
revision actions) - not any generated wording.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import validate  # noqa: E402
from rva.models import (  # noqa: E402
    ClaimGrounding,
    GroundingResult,
    HonestyFlag,
    HonestyResult,
    RevisionItem,
    RevisionResult,
    Source,
    ValidationSummary,
    Verdict,
)

WHO = Source(id="A", url="https://www.who.int/fact",
             text="室外空气污染导致全球每年约420万人过早死亡。")
QUESTION = "空气污染每年导致多少人过早死亡？"


class TestTopLevelContract(unittest.TestCase):
    def test_validation_result_fields(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], revise=True)
        self.assertEqual(
            set(out.model_dump().keys()),
            {
                "answer", "question", "domain", "critic_backend",
                "source_count", "stage1_stats",
                "claims", "verifications", "report",
                "factuality", "grounding", "honesty", "revision", "summary",
            },
        )
        # The four reports are siblings; none is nested inside another.
        self.assertIsNotNone(out.factuality)
        self.assertIsNotNone(out.honesty)
        self.assertIsNotNone(out.revision)
        # Report-level traceability: which critic engine produced the verdicts.
        self.assertEqual(out.critic_backend, "deterministic")

    def test_grounding_is_optional_and_off_by_default(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertIsNone(out.grounding)
        self.assertIsNone(out.revision)
        self.assertIsNone(out.report.supplemented_text)

    def test_question_is_a_first_class_field(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertEqual(out.question, QUESTION)
        self.assertTrue(all(c.question_anchor is not None for c in out.claims))


class TestFactualityContract(unittest.TestCase):
    def test_five_level_vocabulary(self):
        self.assertEqual(
            {v.value for v in Verdict},
            {"accurate", "inaccurate", "disputed", "unsupported", "not_applicable"},
        )

    def test_factuality_result_fields(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertEqual(
            set(out.factuality.model_dump().keys()),
            {
                "score", "accurate", "inaccurate", "disputed", "unsupported",
                "not_applicable", "total_claims", "verifiable_claims",
                "applicable_claims", "verdicts", "method",
            },
        )
        self.assertEqual(out.factuality.method, "local_source_anchored")
        self.assertEqual(
            set(out.factuality.verdicts[0].model_dump().keys()),
            {"claim_id", "claim_text", "claim_type", "verdict", "score",
             "evidence_domains", "rationale"},
        )


class TestGroundingContract(unittest.TestCase):
    def test_grounding_is_reported_separately_from_factuality(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], judge="ragas")
        self.assertIsNotNone(out.grounding)          # a report exists...
        self.assertEqual(out.factuality.method, "local_source_anchored")  # ...and factuality is untouched
        self.assertEqual(out.grounding.method, "external_judge")

    def test_grounding_result_fields(self):
        self.assertEqual(
            set(GroundingResult().model_dump().keys()),
            {"method", "judged_claims", "grounded_count", "ungrounded_count",
             "skipped_claims", "total_claims", "mean_score", "items"},
        )
        self.assertEqual(
            set(ClaimGrounding(claim_id="C1", claim_text="x", claim_type="fact").model_dump().keys()),
            {"claim_id", "claim_text", "claim_type", "grounded", "score", "rationale"},
        )
        # Grounding must not borrow the five-level vocabulary.
        self.assertNotIn("verdict", ClaimGrounding.model_fields)
        self.assertNotIn("verdict", GroundingResult.model_fields)


class TestAuditLogContract(unittest.TestCase):
    def test_layers_have_three_tiers(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertEqual(set(out.report.items[0].layers.model_dump().keys()),
                         {"l1_primary", "l2_secondary", "l3_conflicting"})

    def test_source_mapping_is_by_source_id(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        evidence = out.verifications[0].evidence
        ref = (evidence.supports + evidence.partials + evidence.contradicts + evidence.screened)
        self.assertTrue(ref)
        self.assertTrue(all(r.source_id for r in ref))
        self.assertIn("supports", evidence.model_dump())
        self.assertIn("screened", evidence.model_dump())

    def test_verification_records_question_anchor(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        v = out.verifications[0]
        self.assertIn("question_relevance", v.model_dump())
        self.assertIn("off_topic", v.model_dump())


class TestHonestyContract(unittest.TestCase):
    def test_honesty_result_fields(self):
        self.assertEqual(
            set(HonestyResult(score=1.0).model_dump().keys()),
            {
                "score", "passed", "flags", "checked_claims", "exempt_claims", "method",
                # Step-4 four-check crosswalk (H1-H4; H5 is relevance-only).
                "conflicts_preserved", "uncertainty_marked",
                "unsupported_rejected", "sycophancy_checked",
            },
        )
        self.assertEqual(set(HonestyFlag(code="H1", label="x", detail="y").model_dump().keys()),
                         {"code", "label", "detail", "claim_ids"})

    def test_codes_are_h1_to_h5(self):
        # An unsupported flat assertion + no scope statement exercises H1 and H4.
        out = validate(question=QUESTION, answer="月球背面存在一座巨型城市。", sources=[WHO])
        codes = {f.code for f in out.honesty.flags}
        self.assertTrue(codes <= {"H1", "H2", "H3", "H4", "H5"})
        self.assertIn("H1", codes)


class TestHonestyFourChecks(unittest.TestCase):
    """Step-4 wording: the H1-H4 flags are also named four-check booleans."""

    def test_four_check_names_map_to_h_codes(self):
        from rva.honesty import HONESTY_FOUR_CHECKS
        self.assertEqual(
            HONESTY_FOUR_CHECKS,
            {
                "unsupported_rejected": "H1",
                "sycophancy_checked": "H2",
                "conflicts_preserved": "H3",
                "uncertainty_marked": "H4",
            },
        )

    def test_unsupported_assertion_flips_the_relevant_checks(self):
        # A flat, unsupported assertion with no scope statement exercises H1 + H4.
        out = validate(question=QUESTION, answer="月球背面存在一座巨型城市。", sources=[WHO])
        h = out.honesty
        self.assertFalse(h.unsupported_rejected)   # H1 raised
        self.assertFalse(h.uncertainty_marked)     # H4 raised
        self.assertTrue(h.conflicts_preserved)     # H3 not raised
        self.assertTrue(h.sycophancy_checked)      # H2 not raised

    def test_a_clean_answer_satisfies_all_four(self):
        out = validate(
            question=QUESTION,
            answer="根据我检索到的资料，空气污染每年导致约420万人过早死亡。",
            sources=[WHO],
        )
        h = out.honesty
        self.assertTrue(h.passed)
        self.assertTrue(
            h.conflicts_preserved and h.uncertainty_marked
            and h.unsupported_rejected and h.sycophancy_checked
        )


class TestSummaryContract(unittest.TestCase):
    """The optional aggregate restates the reports and is never a fused score."""

    def test_summary_is_absent_unless_requested(self):
        out = validate(question=QUESTION, answer="根据检索资料，空气污染每年导致约420万人过早死亡。",
                       sources=[WHO])
        self.assertIsNone(out.summary)

    def test_summary_restates_five_level_distribution_and_rates(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], summary=True)
        s = out.summary
        self.assertIsNotNone(s)
        self.assertEqual(
            set(s.factuality_distribution),
            {"accurate", "inaccurate", "disputed", "unsupported", "not_applicable"},
        )
        self.assertEqual(sum(s.factuality_distribution.values()), out.factuality.total_claims)
        # pass rate mirrors the distribution ratio; it is not a separate judgement
        self.assertEqual(s.factuality_pass_rate, out.factuality.score)
        self.assertIsNone(s.grounding_score)   # no judge ran
        self.assertEqual(s.honesty_score, out.honesty.score)

    def test_summary_exposes_no_single_fused_score(self):
        fields = set(ValidationSummary.model_fields)
        self.assertEqual(
            fields,
            {
                "factuality_distribution", "factuality_pass_rate",
                "grounding_score", "honesty_score", "note",
            },
        )
        for banned in ("overall", "score", "total_score", "confidence", "grade"):
            self.assertNotIn(banned, fields)

    def test_grounding_score_uses_its_own_0_to_100_scale(self):
        from rva import evaluate_grounding
        from rva.stages.decompose import decompose
        from rva.stages.verify import verify

        claims, _ = decompose("空气污染每年导致约420万人过早死亡。", question=QUESTION)
        verifications = verify(claims, [WHO])
        grounding = evaluate_grounding(
            claims, verifications,
            batch_judge=lambda cl, vs: {c.id: (True, 0.9, "entailed") for c in cl},
        )
        self.assertEqual(grounding.mean_score, 0.9)
        s = ValidationSummary.from_reports(
            validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                     sources=[WHO]).factuality,
            grounding,
        )
        # the summary carries the judge's own 0-100 scale, not a 0-1 copy
        self.assertEqual(s.grounding_score, 90.0)


class TestRevisionContract(unittest.TestCase):
    def test_revision_result_fields(self):
        self.assertEqual(
            set(RevisionResult().model_dump().keys()),
            {
                "revised_text", "supplemented", "changed", "revision_log", "method",
                "diff", "input_verdicts", "end_marker", "end_marker_present",
                "fallback", "recheck",
            },
        )
        item = RevisionItem(claim_id="C1", action="keep", original_text="x")
        self.assertEqual(set(item.model_dump().keys()),
                         {"claim_id", "action", "original_text", "revised_text", "reason", "criteria"})

    def test_revision_is_traceable_and_non_destructive(self):
        answer = ("空气污染每年导致约420万人过早死亡。"
                  "我认为这是最糟糕的环境问题。"
                  "月球背面存在一座巨型城市。")
        out = validate(question=QUESTION, answer=answer, sources=[WHO], revise=True)
        self.assertEqual(out.answer, answer)                              # original intact
        self.assertEqual(out.report.supplemented_text, out.revision.revised_text)
        # every log entry maps to a real claim and carries the driving criteria
        claim_ids = {c.id for c in out.claims}
        self.assertEqual({i.claim_id for i in out.revision.revision_log}, claim_ids)
        self.assertTrue(all(isinstance(i.criteria, list) for i in out.revision.revision_log))
        self.assertIn("[C", out.revision.revised_text)


if __name__ == "__main__":
    unittest.main()
