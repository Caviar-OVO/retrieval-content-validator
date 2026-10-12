"""Question anchoring, honesty checks, and the Reviser stage.

Three additions to the auditor contract, all offline:

  * question anchoring - a claim that shares no vocabulary with the question is
    off-topic and demoted (`D5`);
  * honesty - the answer is scored on whether it discloses its own limits
    (`H1`-`H5`), kept separate from factuality and grounding;
  * revision - the Reviser rewrites the answer from its verdicts and never
    mutates the original.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import (  # noqa: E402
    build_revision,
    decompose,
    evaluate_factuality,
    validate,
    verify_honesty,
)
from rva.models import ClaimType, Decision, Source  # noqa: E402
from rva.stages.verify import verify  # noqa: E402

WHO = Source(id="A", url="https://www.who.int/fact",
             text="室外空气污染导致全球每年约420万人过早死亡。")
QUESTION = "空气污染每年导致多少人过早死亡？"


class TestQuestionAnchoring(unittest.TestCase):
    def test_claim_records_relevance_to_question(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。", question=QUESTION)
        anchor = claims[0].question_anchor
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.question, QUESTION)
        self.assertGreater(anchor.relevance, 0.0)
        self.assertTrue(anchor.reactant)

    def test_unanchored_without_question(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        anchor = claims[0].question_anchor
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.relevance, 0.0)
        self.assertTrue(anchor.reactant)  # unknown, not asserted off-topic
        self.assertIn("no question", anchor.note)

    def test_off_topic_claim_is_flagged_and_demoted(self):
        answer = "空气污染每年导致约420万人过早死亡。月球背面存在一座巨型城市。"
        out = validate(question=QUESTION, answer=answer, sources=[WHO])
        off = [v for v in out.verifications if v.off_topic]
        self.assertEqual([v.claim_id for v in off], ["C2"])
        item = next(i for i in out.report.items if i.claim_id == "C2")
        self.assertEqual(item.decision, Decision.DEMOTE)
        self.assertIn("D5", [c.id for c in item.criteria])

    def test_off_topic_check_skipped_without_question(self):
        answer = "空气污染每年导致约420万人过早死亡。月球背面存在一座巨型城市。"
        out = validate(answer=answer, sources=[WHO])
        self.assertFalse(any(v.off_topic for v in out.verifications))

    def test_opinion_is_never_off_topic(self):
        out = validate(question=QUESTION, answer="我认为这是最糟糕的环境问题。", sources=[WHO])
        self.assertFalse(any(v.off_topic for v in out.verifications))


class TestHonesty(unittest.TestCase):
    def _pipeline(self, answer, question=None, sources=None):
        sources = sources if sources is not None else [WHO]
        claims, _ = decompose(answer, question=question)
        verifications = verify(claims, sources)
        factuality = evaluate_factuality(claims, verifications)
        h = verify_honesty(claims, verifications, answer=answer, question=question,
                           factuality=factuality)
        return h, claims

    def test_unsupported_assertion_is_flagged(self):
        h, _ = self._pipeline("月球背面存在一座巨型城市。")
        self.assertIn("H1", [f.code for f in h.flags])
        self.assertFalse(h.passed)

    def test_hedged_unsupported_claim_is_not_flagged(self):
        # An explicit hedge is exactly the disclosure H1 asks for.
        h, _ = self._pipeline("据说月球背面可能存在一座巨型城市。")
        self.assertNotIn("H1", [f.code for f in h.flags])

    def test_supported_answer_with_scope_disclosure_passes(self):
        answer = "根据我检索到的资料，空气污染每年导致约420万人过早死亡。"
        h, _ = self._pipeline(answer)
        self.assertTrue(h.passed, msg=[f.code for f in h.flags])
        self.assertEqual(h.flags, [])

    def test_scope_not_disclosed_is_flagged(self):
        h, _ = self._pipeline("空气污染每年导致约420万人过早死亡。")
        self.assertIn("H4", [f.code for f in h.flags])

    def test_opinion_is_exempt(self):
        h, _ = self._pipeline("我认为这是最糟糕的环境问题。")
        self.assertEqual(h.exempt_claims, 1)
        self.assertEqual(h.checked_claims, 0)
        # Only the (inapplicable-here) scope check remains; nothing to be dishonest about.
        self.assertNotIn("H1", [f.code for f in h.flags])

    def test_conflict_not_disclosed_is_flagged(self):
        sources = [
            Source(id="A", url="https://www.who.int/fact",
                   text="室外空气污染导致全球每年约420万人过早死亡。"),
            Source(id="N", url="https://www.nature.com/articles/x",
                   text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。"),
        ]
        h, _ = self._pipeline("空气污染每年导致约420万人过早死亡。", sources=sources)
        self.assertIn("H3", [f.code for f in h.flags])

    def test_honesty_is_separate_from_factuality(self):
        out = validate(question=QUESTION, answer="月球背面存在一座巨型城市。", sources=[WHO])
        # Factuality still reports its five-level verdict...
        self.assertEqual(out.factuality.unsupported, 1)
        # ...and honesty reports its own, independent disclosure flags.
        self.assertIsNotNone(out.honesty)
        self.assertIn("H1", [f.code for f in out.honesty.flags])


class TestRevision(unittest.TestCase):
    def test_revision_drops_off_topic_and_annotates_opinion(self):
        answer = (
            "空气污染每年导致约420万人过早死亡。"
            "我认为这是最糟糕的环境问题。"
            "月球背面存在一座巨型城市。"
        )
        out = validate(question=QUESTION, answer=answer, sources=[WHO], revise=True)
        r = out.revision
        self.assertIsNotNone(r)
        actions = {i.claim_id: i.action for i in r.revision_log}
        self.assertEqual(actions["C1"], "keep")
        self.assertEqual(actions["C2"], "annotate")
        self.assertEqual(actions["C3"], "qualify")  # off-topic -> demote
        # The dropped/qualified claims are marked in the body, not silently deleted.
        self.assertIn("[subjective opinion", r.revised_text)
        self.assertIn("off-topic", r.revised_text)

    def test_original_answer_never_mutated(self):
        answer = "空气污染每年导致约420万人过早死亡。"
        out = validate(question=QUESTION, answer=answer, sources=[WHO], revise=True)
        self.assertEqual(out.answer, answer)

    def test_no_revision_unless_requested(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertIsNone(out.revision)
        self.assertIsNone(out.report.supplemented_text)

    def test_pluggable_reviser_backend(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。", question=QUESTION)
        verifications = verify(claims, [WHO])
        report = __import__("rva").render(claims, verifications)
        r = build_revision(claims, verifications, report, reviser=lambda text, rep: "REWRITTEN")
        self.assertEqual(r.revised_text, "REWRITTEN")
        self.assertTrue(r.supplemented)
        self.assertEqual(r.method, "llm_reviser")

    def test_sample_prompts_are_now_decomposable(self):
        # Regression: the README's prompt-shaped examples used to collapse into
        # one claim. They must now split into separate claims.
        answer = (
            "Question: What color are blueberries?\n"
            "Answer: Blueberries are blue because they have pigments on their skin."
        )
        claims, _ = decompose(answer, question="What color are blueberries?")
        self.assertGreaterEqual(len(claims), 2)


if __name__ == "__main__":
    unittest.main()
