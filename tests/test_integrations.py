"""Tests for the optional Ragas / TruLens *grounding* judge adapters.

These backends measure grounding (is the claim entailed by the supplied
contexts?), which is reported separately from - and can never change - the
source-anchored factuality verdicts. The adapters are exercised with fake
backends so the wiring (dataset building, grounding classification, fallback
behaviour) is verified without any network or API key. Real backends require
`pip install ragas trulens trulens-providers-openai` and OPENAI_API_KEY; those
cases are skipped when unavailable.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import decompose, evaluate_grounding, is_verifiable, validate  # noqa: E402
from rva.models import Source, Verdict  # noqa: E402
from rva.stages.verify import verify  # noqa: E402

try:
    import ragas  # noqa: F401
    HAVE_RAGAS = True
except Exception:
    HAVE_RAGAS = False

try:
    import trulens  # noqa: F401
    HAVE_TRULENS = True
except Exception:
    HAVE_TRULENS = False


def _example():
    answer = (ROOT / "examples" / "answer.txt").read_text(encoding="utf-8")
    sources = [Source(**x) for x in json.loads((ROOT / "examples" / "sources.json").read_text(encoding="utf-8"))]
    return answer, sources


class TestBuildJudge(unittest.TestCase):
    def test_local_returns_none(self):
        from rva.integrations import build_judge
        self.assertIsNone(build_judge("local"))
        self.assertIsNone(build_judge(""))

    def test_unknown_backend_raises(self):
        from rva.integrations import build_judge
        with self.assertRaises(ValueError):
            build_judge("bogus")

    def test_availability_reports_keys(self):
        from rva.integrations import availability
        info = availability()
        self.assertIn("openai_api_key", info)
        self.assertIn("ragas", info)


class TestRagasAdapter(unittest.TestCase):
    def test_score_mapping_and_dataset(self):
        if not HAVE_RAGAS:
            self.skipTest("ragas not installed")
        import pandas as pd
        from unittest import mock
        from rva.integrations import ragas_batch_judge

        answer, sources = _example()
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)

        # find a verification that actually has evidence, to ensure a sample is built
        with_ev = next(v for v in verifications if v.evidence.supports or v.evidence.partials)

        captured = {}

        class FakeResult:
            def __init__(self, scores):
                self._scores = scores

            def to_pandas(self):
                return pd.DataFrame({"faithfulness": self._scores})

        def fake_evaluate(dataset, metrics=None, llm=None, **kwargs):
            captured["n"] = len(dataset.samples)
            return FakeResult([0.9] * len(dataset.samples))

        with mock.patch("ragas.evaluate", side_effect=fake_evaluate):
            judge = ragas_batch_judge(sources=sources, llm=object())
            out = judge(claims, verifications)

        # one sample per *knowledge* claim that has evidence (opinions are filtered out)
        expected_samples = sum(
            1 for v in verifications
            if is_verifiable(v.claim_type)
            and (v.evidence.supports or v.evidence.partials or v.evidence.contradicts)
        )
        self.assertEqual(captured["n"], expected_samples)
        grounded, score, rationale = out[with_ev.claim_id]
        # 0.9 is at/above the grounding threshold; the raw judge score is kept.
        self.assertIs(grounded, True)
        self.assertAlmostEqual(score, 0.9)
        self.assertIn("0.90", rationale)

    def test_no_evidence_claim_is_unsupported_without_llm_call(self):
        if not HAVE_RAGAS:
            self.skipTest("ragas not installed")
        from unittest import mock
        from rva.integrations import ragas_batch_judge

        claims, _ = decompose("这是一条完全没有证据支撑的断言。")
        verifications = verify(claims, [])  # no sources -> no contexts

        with mock.patch("ragas.evaluate", side_effect=AssertionError("should not be called")):
            out = ragas_batch_judge(sources=[])(claims, verifications)

        self.assertEqual(out[verifications[0].claim_id], (False, None, "no retrieved context to ground the claim"))

    def test_backend_failure_returns_none_for_fallback(self):
        if not HAVE_RAGAS:
            self.skipTest("ragas not installed")
        from unittest import mock
        from rva.integrations import ragas_batch_judge

        answer, sources = _example()
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)

        with mock.patch("ragas.evaluate", side_effect=RuntimeError("no api key")):
            out = ragas_batch_judge(sources=sources, llm=object())(claims, verifications)
        self.assertIsNone(out)


class TestNanHandling(unittest.TestCase):
    def test_finite_helper(self):
        from rva.integrations import _finite
        self.assertIsNone(_finite(float("nan")))
        self.assertIsNone(_finite(float("inf")))
        self.assertIsNone(_finite("not a number"))
        self.assertEqual(_finite(1.7), 1.0)   # clamped
        self.assertEqual(_finite(-0.2), 0.0)  # clamped
        self.assertAlmostEqual(_finite(0.5), 0.5)

    def test_ragas_nan_scores_are_omitted(self):
        if not HAVE_RAGAS:
            self.skipTest("ragas not installed")
        import pandas as pd
        from unittest import mock
        from rva.integrations import ragas_batch_judge

        answer, sources = _example()
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)

        # only knowledge claims are ever scored; opinions are filtered before the call
        with_ev = [
            v for v in verifications
            if is_verifiable(v.claim_type)
            and (v.evidence.supports or v.evidence.partials or v.evidence.contradicts)
        ]
        no_ev = [
            v for v in verifications
            if is_verifiable(v.claim_type)
            and not (v.evidence.supports or v.evidence.partials or v.evidence.contradicts)
        ]

        class FakeResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [float("nan")] * len(with_ev)})

        with mock.patch("ragas.evaluate", return_value=FakeResult()):
            out = ragas_batch_judge(sources=sources, llm=object())(claims, verifications)

        # Evidence-backed NaN scores are omitted (so no grounding verdict is claimed);
        # claims with no evidence are reported ungrounded deterministically.
        for v in with_ev:
            self.assertNotIn(v.claim_id, out or {})
        for v in no_ev:
            self.assertIs(out[v.claim_id][0], False)

    def test_ragas_partial_nan_keeps_valid_scores(self):
        if not HAVE_RAGAS:
            self.skipTest("ragas not installed")
        import pandas as pd
        from unittest import mock
        from rva.integrations import ragas_batch_judge

        _, sources = _example()
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。有研究可能表明两者相关。")
        verifications = verify(claims, sources)
        with_ev = [
            v for v in verifications
            if is_verifiable(v.claim_type)
            and (v.evidence.supports or v.evidence.partials or v.evidence.contradicts)
        ]

        class FakeResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.9] + [float("nan")] * (len(with_ev) - 1)})

        with mock.patch("ragas.evaluate", return_value=FakeResult()):
            out = ragas_batch_judge(sources=sources, llm=object())(claims, verifications)
        self.assertIsNotNone(out)
        self.assertIn(with_ev[0].claim_id, out)
        self.assertAlmostEqual(out[with_ev[0].claim_id][1], 0.9)
        self.assertIs(out[with_ev[0].claim_id][0], True)


class TestTruLensAdapter(unittest.TestCase):
    def test_three_point_scale_is_normalized(self):
        if not HAVE_TRULENS:
            self.skipTest("trulens not installed")
        from rva.integrations import trulens_batch_judge

        answer, sources = _example()
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)

        class FakeProvider:
            def groundedness_measure_with_cot_reasons(self, source, statement, **kwargs):
                return 3, {"reason": "fully grounded"}

        out = trulens_batch_judge(sources=sources, llm=FakeProvider())(claims, verifications)
        with_ev = next(
            v for v in verifications
            if is_verifiable(v.claim_type) and (v.evidence.supports or v.evidence.partials)
        )
        grounded, score, rationale = out[with_ev.claim_id]
        self.assertIs(grounded, True)
        self.assertAlmostEqual(score, 1.0)
        self.assertIn("trulens", rationale)

    def test_backend_failure_returns_none(self):
        if not HAVE_TRULENS:
            self.skipTest("trulens not installed")
        from rva.integrations import trulens_batch_judge

        answer, sources = _example()
        claims, _ = decompose(answer)
        verifications = verify(claims, sources)

        class Boom:
            def groundedness_measure_with_cot_reasons(self, source, statement, **kwargs):
                raise RuntimeError("no api key")

        self.assertIsNone(trulens_batch_judge(sources=sources, llm=Boom())(claims, verifications))


class TestPipelineFallback(unittest.TestCase):
    def test_factuality_stays_local_when_judge_unavailable(self):
        # Without OPENAI_API_KEY the grounding judge must degrade gracefully.
        # Factuality is local regardless, so it never depends on the backend.
        _, sources = _example()
        result = validate(
            answer="空气污染每年导致约420万人过早死亡。",
            sources=sources,
            judge="ragas",
        )
        self.assertEqual(result.factuality.verifiable_claims, 1)
        self.assertEqual(result.factuality.method, "local_source_anchored")
        # A grounding report exists but may carry no verdicts if the API failed.
        self.assertIsNotNone(result.grounding)
        self.assertEqual(result.grounding.method, "external_judge")


if __name__ == "__main__":
    unittest.main()