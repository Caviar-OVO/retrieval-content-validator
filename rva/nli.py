"""Three-way NLI: does a piece of evidence support, refute, or stay neutral?

This is the second half of the Stage-2 upgrade: vector retrieval decides *which*
chunks are relevant to a claim, and an entailment judge decides the *stance* of
each chunk towards the claim.

Two backends share one contract (`judge(claim, evidence) -> Label`):

  * `LLMNLIJudge`   - a natural-language-inference judge built on the same
                      LLM-judge wiring as the Ragas/TruLens adapters. It asks
                      for a strict {support, refute, neutral} label and falls
                      back to neutral on any parse/transport failure.
  * `HeuristicNLI`  - an offline cue-based judge used when no key is present.

Labels follow the standard NLI convention:
    support  == entailment
    refute   == contradiction
    neutral  == neither entailed nor contradicted
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

from .models import Claim
from .stages.verify import relevant_negation


class Label(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    NEUTRAL = "neutral"


_LABELS = {
    "support": Label.SUPPORT, "entailment": Label.SUPPORT, "entails": Label.SUPPORT,
    "refute": Label.REFUTE, "contradiction": Label.REFUTE, "contradicts": Label.REFUTE,
    "neutral": Label.NEUTRAL, "neither": Label.NEUTRAL, "unknown": Label.NEUTRAL,
}


def parse_label(value) -> Label:
    """Map a judge response (string or {'label': ...}) onto a Label."""
    if isinstance(value, Label):
        return value
    if isinstance(value, dict):
        value = value.get("label", value.get("verdict", ""))
    text = str(value or "").strip().lower()
    if text in _LABELS:
        return _LABELS[text]
    for key, label in _LABELS.items():
        if key in text:
            return label
    return Label.NEUTRAL


def label_to_score(label: Label) -> float:
    return {Label.SUPPORT: 1.0, Label.REFUTE: 0.0, Label.NEUTRAL: 0.5}[label]


PROMPT = (
    "You are a strict natural-language-inference judge. Given a CLAIM and a "
    "piece of EVIDENCE, decide the evidence's stance towards the claim.\n"
    "Answer with exactly one word:\n"
    "  support  - the evidence entails the claim's substance\n"
    "  refute   - the evidence contradicts the claim's substance\n"
    "  neutral  - the evidence is related but neither entails nor contradicts it\n"
    "Judge the substance, not the wording. Process/meta remarks such as "
    "'not yet peer reviewed' are neutral, not refutations."
)


def _heuristic(claim: Claim, evidence: str) -> tuple[Label, str]:
    from .text import shared_token_count, tokenize

    if not (evidence or "").strip():
        return Label.NEUTRAL, "empty evidence"

    claim_tokens = tokenize(claim.text)
    unique = set(claim_tokens)
    shared = shared_token_count(claim_tokens, tokenize(evidence))
    ratio = shared / max(1, len(unique))

    # Longest run of claim tokens appearing consecutively in the evidence is a
    # cheap proxy for how aligned (entailed) the two texts are.
    lower = evidence.lower()
    best_run = 0
    run = 0
    for token in claim_tokens:
        if token in lower:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0

    # Refutation needs an assertion-level denial about *the same thing*; a
    # negation in an unrelated sentence is not a contradiction.
    src_neg = relevant_negation(claim_tokens, evidence)
    claim_neg = relevant_negation(claim_tokens, claim.text)
    if (src_neg and src_neg["strength"] == "strong" and not claim_neg
            and (shared >= 3 or best_run >= 2)):
        return Label.REFUTE, f"assertion-level negation '{src_neg['marker']}' near shared content"

    if best_run >= 4 or shared >= 6 or (shared >= 5 and ratio >= 0.30):
        return Label.SUPPORT, f"{shared} shared content tokens"
    return Label.NEUTRAL, f"only {shared} shared content tokens"


class HeuristicNLI:
    """Deterministic offline three-way stance classifier (no network)."""

    name = "heuristic-negation"

    def judge(self, claim: Claim, evidence: str) -> tuple[Label, str]:
        return _heuristic(claim, evidence)


class LLMNLIJudge:
    """LLM-backed three-way stance classifier built on the judge wiring."""

    def __init__(self, llm=None, model: str | None = None, temperature: float = 0.0):
        self._client = llm
        self._model = model
        self._temperature = temperature
        self.calls = 0
        self.failures = 0
        self.name = "llm-nli"

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        from .config import get_api_key, get_base_url, get_judge_model

        kwargs = {"api_key": get_api_key()}
        base_url = get_base_url()
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        if not self._model:
            self._model = get_judge_model()
        return self._client

    @staticmethod
    def _content(resp) -> str:
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            value = resp.get("content", "")
        else:
            value = getattr(resp, "content", "")
        return value if isinstance(value, str) else str(value or "")

    def judge(self, claim: Claim, evidence: str) -> tuple[Label, str]:
        if not (evidence or "").strip():
            return Label.NEUTRAL, "empty evidence"
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content":
                        f"CLAIM:\n{claim.text}\n\nEVIDENCE:\n{evidence}\n\n"
                        'Respond as JSON: {"label": "support|refute|neutral"}'},
                ],
            )
        except Exception as exc:  # pragma: no cover - depends on live API/key
            self.failures += 1
            return Label.NEUTRAL, f"nli judge unavailable ({type(exc).__name__}); treated as neutral"

        self.calls += 1
        raw = self._content(resp.choices[0].message)
        label = parse_label(raw)
        return label, f"llm-nli: {raw.strip()[:60] or 'no response'}"


def build_nli_judge(name: str = "heuristic", *, llm=None, llm_ok: bool = False):
    """Return an NLI judge for `name` in {heuristic, llm, auto}."""
    name = (name or "heuristic").lower()
    if name == "auto":
        name = "llm" if llm_ok else "heuristic"
    if name in ("llm", "openai", "nli"):
        return LLMNLIJudge(llm=llm)
    return HeuristicNLI()