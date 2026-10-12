"""The Critic on two tracks: deterministic (default) and LLM (optional).

Both tracks must return the **same schema**: `Claim` records (with `cues`),
`Verification` evidence layers, and a `FactualityResult` whose verdicts carry
`rationale` (justification) + `evidence_domains`. The LLM track is exercised with
a fake client, so this runs offline.
"""

from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import (  # noqa: E402
    CRITIC,
    DETERMINISTIC_METHOD,
    LLM_METHOD,
    LLMCritic,
    build_critic,
    validate,
)
from rva.models import ClaimType, Source, Verdict  # noqa: E402

WHO = Source(id="S1", url="https://www.who.int/fact",
             text="室外空气污染导致全球每年约420万人过早死亡。")
BLOG = Source(id="S2", url="https://blog.example.com/x", tag="blog",
              text="空气污染导致数百万人死亡的说法其实被夸大了，并不支持这一结论。")
QUESTION = "空气污染每年导致多少人过早死亡？"


# --- fake OpenAI-shaped client --------------------------------------------

def _client(content):
    message = type("M", (), {"content": content})()
    choice = type("C", (), {"message": message})()
    completions = type("Comp", (), {"create": lambda self, **kw: type("R", (), {"choices": [choice]})()})()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def _capture_critic_warnings():
    """Attach a handler to the rva.critic logger; returns (records, restore)."""
    lg = logging.getLogger("rva.critic")
    records: list[logging.LogRecord] = []
    handler = type("H", (logging.Handler,),
                   {"emit": lambda self, r: records.append(r)})()
    old_level, old_propagate = lg.level, lg.propagate
    lg.addHandler(handler)
    lg.setLevel(logging.WARNING)
    lg.propagate = False

    def restore():
        lg.removeHandler(handler)
        lg.setLevel(old_level)
        lg.propagate = old_propagate

    return records, restore


class _RecordingClient:
    """An OpenAI-shaped client that records the kwargs of the last request."""

    def __init__(self, content):
        self._content = content
        self.last_kwargs = None
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        self.last_kwargs = kw
        message = type("M", (), {"content": self._content})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _BoomClient:
    @property
    def chat(self):
        raise RuntimeError("no api key")


def _payload(verdict="accurate", evidence=None, justification="S1 says so"):
    return json.dumps({
        "claims": [{
            "text": "空气污染每年导致约420万人过早死亡。",
            "type": "fact", "confidence": 0.9, "verdict": verdict,
            "justification": justification,
            "evidence": ["S1"] if evidence is None else evidence,
            "cues": [{"category": "fact", "kind": "quantitative", "value": "420万"}],
        }]
    })


class TestBackendSelection(unittest.TestCase):
    def test_deterministic_is_default(self):
        self.assertEqual(build_critic("deterministic").name, "deterministic")
        self.assertEqual(build_critic("").name, "deterministic")

    def test_llm_backend(self):
        self.assertEqual(build_critic("llm").name, "llm")

    def test_auto_follows_key(self):
        # auto needs both a key and sources to pick llm; missing either -> deterministic
        self.assertEqual(build_critic("auto", llm_ok=False, has_sources=True).name, "deterministic")
        self.assertEqual(build_critic("auto", llm_ok=True, has_sources=True).name, "llm")
        self.assertEqual(build_critic("auto", llm_ok=True, has_sources=False).name, "deterministic")


class TestDeterministicCritic(unittest.TestCase):
    def test_shape(self):
        out = build_critic("deterministic").critique(
            "空气污染每年导致约420万人过早死亡。我认为这是最糟糕的问题。",
            [WHO], question=QUESTION)
        self.assertEqual(out.method, DETERMINISTIC_METHOD)
        self.assertFalse(out.fallback)
        self.assertEqual(out.stats.total, len(out.claims))
        self.assertTrue(all(c.cues for c in out.claims))
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)


class TestLLMCritic(unittest.TestCase):
    def test_produces_the_same_schema(self):
        out = LLMCritic(llm=_client(_payload())).critique(
            "空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        self.assertEqual(out.method, LLM_METHOD)
        self.assertFalse(out.fallback)
        # claims keep the audit fields (cues / note = justification)
        c = out.claims[0]
        self.assertEqual(c.type, ClaimType.FACT)
        self.assertTrue(c.cues)
        self.assertEqual(c.note, "S1 says so")
        self.assertTrue(c.verifiable)
        # verdicts keep justification + evidence domains
        v = out.factuality.verdicts[0]
        self.assertEqual(v.verdict, Verdict.ACCURATE)
        self.assertEqual(v.rationale, "S1 says so")
        self.assertEqual(v.evidence_domains, ["who.int"])
        self.assertEqual(out.factuality.method, LLM_METHOD)

    def test_schema_matches_the_rule_track_field_for_field(self):
        det = build_critic("deterministic").critique("空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        llm = LLMCritic(llm=_client(_payload())).critique("空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        self.assertEqual(set(det.claims[0].model_dump()), set(llm.claims[0].model_dump()))
        self.assertEqual(set(det.factuality.model_dump()), set(llm.factuality.model_dump()))
        self.assertEqual(set(det.factuality.verdicts[0].model_dump()),
                         set(llm.factuality.verdicts[0].model_dump()))

    def test_missing_justification_falls_back(self):
        payload = json.dumps({"claims": [{"text": "空气污染每年导致约420万人过早死亡。",
                                          "type": "fact", "verdict": "accurate", "evidence": ["S1"]}]})
        critic = LLMCritic(llm=_client(payload))
        out = critic.critique("空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        self.assertTrue(out.fallback)
        self.assertEqual(critic.failures, 1)
        self.assertEqual(out.method, DETERMINISTIC_METHOD)

    def test_substantiated_verdict_without_evidence_falls_back(self):
        critic = LLMCritic(llm=_client(_payload(evidence=[])))
        out = critic.critique("空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        self.assertTrue(out.fallback)
        self.assertEqual(out.method, DETERMINISTIC_METHOD)

    def test_transport_failure_falls_back(self):
        critic = LLMCritic(llm=_BoomClient())
        out = critic.critique("空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        self.assertTrue(out.fallback)
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)

    def test_opinion_is_forced_not_applicable(self):
        payload = json.dumps({"claims": [{
            "text": "我认为这是最糟糕的环境问题。", "type": "opinion",
            "verdict": "accurate", "justification": "personal view", "evidence": ["S1"],
        }]})
        out = LLMCritic(llm=_client(payload)).critique("我认为这是最糟糕的环境问题。", [WHO])
        self.assertEqual(out.factuality.verdicts[0].verdict, Verdict.NOT_APPLICABLE)
        self.assertEqual(out.factuality.verifiable_claims, 0)

    def test_unsupported_verdict_needs_no_evidence(self):
        out = LLMCritic(llm=_client(_payload(verdict="unsupported", evidence=[]))).critique(
            "月球背面存在一座巨型城市。", [WHO], question=QUESTION)
        self.assertFalse(out.fallback)
        self.assertEqual(out.factuality.verdicts[0].verdict, Verdict.UNSUPPORTED)


class TestReportTraceability(unittest.TestCase):
    def test_deterministic_records_the_backend(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertEqual(out.critic_backend, "deterministic")
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)

    def test_llm_records_the_backend(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic=LLMCritic(llm=_client(_payload())))
        self.assertEqual(out.critic_backend, "llm")
        self.assertEqual(out.factuality.method, LLM_METHOD)

    def test_fallback_records_the_engine_actually_used(self):
        # Requested "llm" but the backend failed: the report must say what
        # actually produced the verdicts, not what was asked for.
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic=LLMCritic(llm=_BoomClient()))
        self.assertEqual(out.critic_backend, "deterministic")
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)

    def test_backend_choice_via_string(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic_backend="deterministic")
        self.assertEqual(out.critic_backend, "deterministic")


class TestValidateWiring(unittest.TestCase):
    def test_default_is_deterministic(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。", sources=[WHO])
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)

    def test_llm_critic_end_to_end(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic=LLMCritic(llm=_client(_payload())))
        self.assertEqual(out.factuality.method, LLM_METHOD)
        self.assertEqual(out.factuality.accurate, 1)
        self.assertEqual(out.claims[0].note, "S1 says so")

    def test_llm_critic_reaches_later_stages(self):
        # render() / honesty must consume LLM-authored claims + verdicts unchanged.
        out = validate(question=QUESTION,
                       answer="空气污染每年导致约420万人过早死亡。我认为这最糟糕。",
                       sources=[WHO], critic=LLMCritic(llm=_client(_payload())),
                       revise=True)
        self.assertEqual(len(out.report.items), len(out.claims))
        self.assertIsNotNone(out.honesty)
        self.assertIsNotNone(out.revision)


class TestCriticPromptContract(unittest.TestCase):
    """The LLM track injects CRITIC.prompt and enforces a strict-JSON contract."""

    def test_system_prompt_injects_the_critic_persona(self):
        prompt = LLMCritic._system_prompt()
        self.assertTrue(prompt.startswith(CRITIC.prompt),
                        msg="the Critic persona must be injected verbatim, up front")

    def test_few_shot_pins_the_output_contract_schema(self):
        prompt = LLMCritic._system_prompt()
        # The JSON skeleton names every field the reply must carry, so the model
        # is steered to the same schema both critic tracks return.
        for key in ('"claims"', '"text"', '"type"', '"confidence"',
                    '"verdict"', '"justification"', '"evidence"', '"cues"'):
            self.assertIn(key, prompt, msg=f"few-shot schema is missing {key}")
        self.assertIn("JSON", prompt)

    def test_request_carries_persona_as_the_system_message(self):
        client = _RecordingClient(_payload())
        LLMCritic(llm=client).critique("空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        messages = client.last_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue(messages[0]["content"].startswith(CRITIC.prompt))
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(QUESTION, messages[1]["content"])
        self.assertIn("[S1]", messages[1]["content"])


class TestCriticClaimsFeedTheReviser(unittest.TestCase):
    """The LLM critic's claims/verdicts are handed to the Reviser unchanged."""

    def test_reviser_receives_the_llm_critic_claims(self):
        critic = LLMCritic(llm=_client(_payload(justification="S1 states the number")))
        out = validate(question=QUESTION,
                       answer="空气污染每年导致约420万人过早死亡。我认为这最糟糕。",
                       sources=[WHO], critic=critic, revise=True)
        # every claim the critic produced is rendered and logged for the reviser
        self.assertEqual({c.id for c in out.claims},
                         {i.claim_id for i in out.report.items})
        self.assertEqual({c.id for c in out.claims},
                         {i.claim_id for i in out.revision.revision_log})
        # the rationale the LLM critic authored is what the Reviser acts on
        self.assertTrue(any("S1 states the number" in line
                            for line in out.revision.input_verdicts))


class TestClassifyFallback(unittest.TestCase):
    """llm fallback must be safe: honest metadata, no silent claim loss, logged."""

    def _fallback(self, client):
        lg = logging.getLogger("rva.critic")
        records = []
        handler = type("H", (logging.Handler,), {"emit": lambda self, r: records.append(r)})()
        old_level = lg.level
        lg.addHandler(handler)
        lg.setLevel(logging.WARNING)
        try:
            out = LLMCritic(llm=client).critique(
                "空气污染每年导致约420万人过早死亡。", [WHO], question=QUESTION)
        finally:
            lg.removeHandler(handler)
            lg.setLevel(old_level)
        return out, records

    def test_missing_justification_falls_back_to_deterministic(self):
        # ① missing justification / evidence -> fall back to the deterministic critic
        payload = json.dumps({"claims": [{"text": "空气污染每年导致约420万人过早死亡。",
                                          "type": "fact", "verdict": "accurate", "evidence": ["S1"]}]})
        out, _ = self._fallback(_client(payload))
        self.assertTrue(out.fallback)
        self.assertEqual(out.method, DETERMINISTIC_METHOD)
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)

    def test_backend_is_reported_as_deterministic_after_fallback(self):
        # ② after fallback, critic_backend honestly reads "deterministic" (no lying)
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[WHO], critic=LLMCritic(llm=_BoomClient()))
        self.assertEqual(out.critic_backend, "deterministic")
        self.assertNotEqual(out.critic_backend, "llm")

    def test_fallback_keeps_claim_level_output(self):
        # ③ fallback must not drop claim-level output
        out, _ = self._fallback(_BoomClient())
        self.assertTrue(out.claims)
        self.assertTrue(out.verifications)
        self.assertEqual(len(out.factuality.verdicts), len(out.claims))
        self.assertTrue(all(c.cues for c in out.claims))

    def test_fallback_emits_a_warning_not_silent(self):
        # ④ downgrade leaves a warning log - not silent
        out, records = self._fallback(_BoomClient())
        self.assertTrue(out.fallback)
        self.assertTrue(records, msg="a fallback must log a warning, not fail silently")
        self.assertTrue(any(r.levelno >= logging.WARNING for r in records))
        self.assertTrue(any("deterministic" in r.getMessage() for r in records))


class TestAutoSelection(unittest.TestCase):
    """`auto` picks a backend safely and reports the choice honestly."""

    def test_explicit_backend_beats_auto(self):
        # ① explicit llm / deterministic wins over auto
        records, restore = _capture_critic_warnings()
        try:
            self.assertEqual(build_critic("deterministic", llm_ok=True, has_sources=True).name,
                             "deterministic")
            # explicit llm wins over auto even without a key or sources
            self.assertEqual(build_critic("llm", llm_ok=False, has_sources=True).name, "llm")
        finally:
            restore()

    def test_explicit_llm_is_allowed_without_sources_but_logged(self):
        # explicit llm with no sources is honoured (caller was explicit) + WARNING
        records, restore = _capture_critic_warnings()
        try:
            critic = build_critic("llm", llm_ok=False, has_sources=False)
        finally:
            restore()
        self.assertEqual(critic.name, "llm")
        self.assertTrue(records, msg="explicit-llm-without-sources must be logged")
        self.assertTrue(any(r.levelno >= logging.WARNING for r in records))

    def test_auto_without_key_falls_back_to_deterministic(self):
        # ② auto with no API key -> safe deterministic, no error
        self.assertEqual(build_critic("auto", llm_ok=False, has_sources=True).name, "deterministic")

    def test_auto_without_sources_falls_back_and_warns(self):
        # ② auto with no sources -> safe deterministic, no error, and a warning log
        records, restore = _capture_critic_warnings()
        try:
            critic = build_critic("auto", llm_ok=True, has_sources=False)
        finally:
            restore()
        self.assertEqual(critic.name, "deterministic")
        self.assertTrue(records, msg="auto-without-sources must leave a warning, not silent")
        self.assertTrue(any(r.levelno >= logging.WARNING for r in records))
        self.assertTrue(any("no sources" in r.getMessage() for r in records))

    def test_auto_with_key_and_sources_picks_llm(self):
        self.assertEqual(build_critic("auto", llm_ok=True, has_sources=True).name, "llm")

    def test_validate_auto_without_sources_selects_deterministic(self):
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[], critic_backend="auto")
        self.assertEqual(out.critic_backend, "deterministic")
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)

    def test_auto_choice_is_recorded_in_metadata(self):
        # ③ the backend auto finally chose is written to the report metadata.
        # No sources -> auto resolves to deterministic without touching the network,
        # and that resolved choice is what the report records.
        out = validate(question=QUESTION, answer="空气污染每年导致约420万人过早死亡。",
                       sources=[], critic_backend="auto")
        self.assertEqual(out.critic_backend, "deterministic")
        self.assertEqual(out.factuality.method, DETERMINISTIC_METHOD)


if __name__ == "__main__":
    unittest.main()
