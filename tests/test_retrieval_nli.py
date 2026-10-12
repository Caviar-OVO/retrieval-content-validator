"""Tests for Stage 2's vector retrieval and three-way NLI.

These run fully offline: the default embedder is deterministic and the default
NLI judge is cue-based. The LLM NLI judge is exercised with a fake client so its
wiring (parsing, three-way mapping, neutral fallback) is covered without a key.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import decompose, validate  # noqa: E402
from rva.models import Source  # noqa: E402
from rva.nli import (  # noqa: E402
    HeuristicNLI,
    Label,
    LLMNLIJudge,
    build_nli_judge,
    parse_label,
)
from rva.retrieval import (  # noqa: E402
    HashingEmbedder,
    VectorRetriever,
    chunk_text,
    cosine,
)
from rva.stages.verify import verify  # noqa: E402


class TestChunking(unittest.TestCase):
    def test_splits_on_sentence_boundaries(self):
        chunks = chunk_text("第一句。第二句！第三句？", max_chars=6)
        self.assertEqual(chunks, ["第一句。", "第二句！", "第三句？"])

    def test_packs_short_sentences(self):
        chunks = chunk_text("一。二。三。", max_chars=100)
        self.assertEqual(chunks, ["一。二。三。"])

    def test_empty(self):
        self.assertEqual(chunk_text("   "), [])


class TestHashingEmbedder(unittest.TestCase):
    def test_vectors_are_normalized(self):
        emb = HashingEmbedder().fit(["空气污染与健康", "城市交通"])
        (vec,) = emb.embed(["空气污染"])
        norm = sum(v * v for v in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similar_text_scores_higher_than_unrelated(self):
        emb = HashingEmbedder().fit(["空气污染导致死亡", "今天股市上涨"])
        a, b, c = emb.embed(["空气污染导致死亡", "空气污染死亡人数", "股市行情"])
        self.assertGreater(cosine(a, b), cosine(a, c))

    def test_deterministic(self):
        one = HashingEmbedder().fit(["x y z"]).embed(["x y"])
        two = HashingEmbedder().fit(["x y z"]).embed(["x y"])
        self.assertEqual(one, two)


class TestVectorRetriever(unittest.TestCase):
    def setUp(self):
        self.sources = [
            Source(id="A", url="https://www.who.int/x", text="空气污染每年导致约420万人过早死亡。"),
            Source(id="B", url="https://blog.example.com/x", tag="blog", text="今天天气不错，适合跑步。"),
        ]

    def test_top_k_and_threshold(self):
        retriever = VectorRetriever(self.sources, similarity_top_k=5, vector_distance_threshold=0.5)
        hits = retriever.retrieve("空气污染导致每年420万人死亡")
        self.assertEqual(hits[0].source_id, "A")
        self.assertTrue(all(h.distance <= 0.5 for h in hits))
        self.assertEqual(len({h.source_id for h in hits}), len(hits))
        self.assertEqual(retriever.resolved_backend, "hashing-char-ngram")

    def test_similarity_map_covers_every_source(self):
        retriever = VectorRetriever(self.sources)
        sims = retriever.similarity_map("空气污染导致死亡")
        self.assertEqual(set(sims), {"A", "B"})
        self.assertGreater(sims["A"], sims["B"])

    def test_empty_sources(self):
        self.assertEqual(VectorRetriever([]).retrieve("anything"), [])


class TestParseLabel(unittest.TestCase):
    def test_maps_aliases(self):
        self.assertIs(parse_label("SUPPORT"), Label.SUPPORT)
        self.assertIs(parse_label("entailment"), Label.SUPPORT)
        self.assertIs(parse_label("refute"), Label.REFUTE)
        self.assertIs(parse_label({"label": "contradiction"}), Label.REFUTE)
        self.assertIs(parse_label("neutral"), Label.NEUTRAL)
        self.assertIs(parse_label("garbage"), Label.NEUTRAL)
        self.assertIs(parse_label(None), Label.NEUTRAL)


class TestHeuristicNLI(unittest.TestCase):
    def setUp(self):
        self.nli = HeuristicNLI()

    def test_support_for_aligned_text(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, _ = self.nli.judge(claims[0], "室外空气污染导致全球每年约420万人过早死亡。")
        self.assertIs(label, Label.SUPPORT)

    def test_refute_for_assertion_level_denial(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, _ = self.nli.judge(
            claims[0], "空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")
        self.assertIs(label, Label.REFUTE)

    def test_neutral_for_unrelated(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, _ = self.nli.judge(claims[0], "今天天气晴朗，适合外出散步。")
        self.assertIs(label, Label.NEUTRAL)

    def test_empty_evidence_is_neutral(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, _ = self.nli.judge(claims[0], "")
        self.assertIs(label, Label.NEUTRAL)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


class TestLLMNLIJudge(unittest.TestCase):
    def test_parses_support(self):
        judge = LLMNLIJudge(llm=_FakeClient("support"))
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, reason = judge.judge(claims[0], "室外空气污染导致全球每年约420万人过早死亡。")
        self.assertIs(label, Label.SUPPORT)
        self.assertEqual(judge.calls, 1)
        self.assertIn("llm-nli", reason)

    def test_parses_refute(self):
        judge = LLMNLIJudge(llm=_FakeClient("refute"))
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, _ = judge.judge(claims[0], "该说法被夸大了。")
        self.assertIs(label, Label.REFUTE)

    def test_unavailable_client_falls_back_to_neutral(self):
        judge = LLMNLIJudge(llm=_BoomClient())
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        label, reason = judge.judge(claims[0], "任何证据")
        self.assertIs(label, Label.NEUTRAL)
        self.assertIn("unavailable", reason)
        self.assertEqual(judge.failures, 1)


class _BoomClient:
    @property
    def chat(self):
        raise RuntimeError("no api key")


class TestBuildNLIJudge(unittest.TestCase):
    def test_defaults_and_auto(self):
        self.assertIsInstance(build_nli_judge("heuristic"), HeuristicNLI)
        self.assertIsInstance(build_nli_judge("auto"), HeuristicNLI)
        self.assertIsInstance(build_nli_judge("llm"), LLMNLIJudge)
        self.assertIsInstance(build_nli_judge("auto", llm_ok=True), LLMNLIJudge)


class TestVerifyUsesRetrievalAndNLI(unittest.TestCase):
    def test_stance_and_backend_recorded(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        sources = [
            Source(id="A", url="https://www.who.int/fact", text="室外空气污染导致全球每年约420万人过早死亡。"),
            Source(id="B", url="https://blog.example.com/x", tag="blog", text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。"),
        ]
        results = verify(claims, sources)
        v = results[0]
        self.assertEqual(v.retrieval_backend, "hashing-char-ngram")
        self.assertEqual(v.nli_backend, "heuristic-negation")
        self.assertTrue(v.evidence.supports)
        self.assertTrue(v.evidence.contradicts)
        self.assertEqual({e.stance for e in v.evidence.supports}, {"support"})
        self.assertEqual({e.stance for e in v.evidence.contradicts}, {"refute"})

    def test_screened_sources_are_reported(self):
        claims, _ = decompose("空气污染每年导致约420万人过早死亡。")
        sources = [
            Source(id="A", url="https://www.who.int/fact", text="室外空气污染导致全球每年约420万人过早死亡。"),
            Source(id="Z", url="https://forum.example.org/x", tag="forum", text="今天天气不错，适合跑步。"),
        ]
        v = verify(claims, sources)[0]
        self.assertEqual(v.screened_count, 1)
        self.assertEqual(v.evidence.screened[0].source_id, "Z")
        self.assertEqual(v.evidence.screened[0].stance, "screened")


class TestValidateWiring(unittest.TestCase):
    def test_offline_backends_run(self):
        sources = [Source(id="A", url="https://www.who.int/fact", text="室外空气污染导致全球每年约420万人过早死亡。")]
        result = validate(answer="空气污染每年导致约420万人过早死亡。", sources=sources,
                          retriever="hash", nli="heuristic")
        self.assertEqual(result.verifications[0].retrieval_backend, "hashing-char-ngram")
        self.assertEqual(result.verifications[0].nli_backend, "heuristic-negation")
        self.assertTrue(result.verifications[0].evidence.supports)

    def test_auto_without_key_stays_offline(self):
        from unittest import mock
        sources = [Source(id="A", url="https://www.who.int/fact", text="室外空气污染导致全球每年约420万人过早死亡。")]
        with mock.patch("rva.config.get_api_key", return_value=None):
            result = validate(answer="空气污染每年导致约420万人过早死亡。", sources=sources,
                              retriever="auto", nli="auto")
        self.assertEqual(result.verifications[0].retrieval_backend, "hashing-char-ngram")
        self.assertEqual(result.verifications[0].nli_backend, "heuristic-negation")


if __name__ == "__main__":
    unittest.main()