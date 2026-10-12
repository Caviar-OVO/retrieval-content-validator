"""Stage 1 - cognitive decomposition.

Extracts statements from an LLM answer and classifies each as a
fact, a logical reasoning step, a subjective opinion, or an inference.
Rule-based and fully auditable: every classification returns its cues.

`decompose` accepts an optional `question`. The LLM Auditor reads a
(question, answer) pair, so claims are anchored to the question they are meant
to answer; without it a claim has no anchor and cannot be judged for relevance.
When a question is supplied, each claim records its question relevance.
"""

from __future__ import annotations

import re

from ..models import Claim, ClaimType, Cue, DecompositionStats, QuestionAnchor
from ..text import overlap, split_claims, tokenize

OPINION_MARKERS = [
    "我认为", "我觉得", "在我看来", "个人认为", "感觉", "也许", "大概", "或许",
    "似乎", "恐怕", "最好", "糟糕", "很棒", "优秀", "可惜", "遗憾", "令人担忧",
    "i think", "i believe", "in my opinion", "i feel", "probably", "arguably",
    "seems", "maybe", "unfortunately", "great", "terrible", "amazing", "best",
    "worst", "better", "worse", "should",
]

# Strong deductive connectives: the sentence is explicitly drawing a conclusion.
REASONING_STRONG = [
    "因此", "所以", "从而", "由此可见", "这意味着", "这表明", "据此", "故而",
    "这说明", "这证明", "therefore", "thus", "hence", "consequently", "implies",
    "means that", "it follows", "as a result",
]

# Weak causal verbs: add causal flavour but often inside plain factual claims.
REASONING_WEAK = [
    "因为", "由于", "鉴于", "说明", "证明", "推断出", "导致", "之所以",
    "because", "since", "given that", "leads to", "results in", "points to",
]

INFERENCE_MARKERS = [
    "可能", "很可能", "看起来", "推测", "猜测", "预计", "大体上", "大致",
    "倾向于", "probably", "likely", "presumably", "apparently", "suggests",
    "appears", "may indicate", "might", "we can infer", "indicates", "estimate",
]

_QUANT = re.compile(r"(\d+(?:[.,]\d+)?\s*%|\d+(?:[.,]\d+)?\s*(?:倍|万|亿|million|billion|percent|percentage))", re.I)
_YEAR = re.compile(r"(1[5-9]\d{2}|20\d{2})\s*年?")
_ATTR_ZH = re.compile(r"(?:据|根据|按照|来自|来源|报告称|研究显示|数据显示|统计|发布|报道|宣布)")
_ATTR_EN = re.compile(r"(according to|reported by|study|research|data from|published|announced|survey)", re.I)
_PROPER = re.compile(r"^[A-Z][a-zA-Z]{2,}")


def _collect(text: str, markers: list[str]) -> list[str]:
    lower = text.lower()
    return [m for m in markers if m.lower() in lower]


def score_claim(text: str) -> tuple[dict[str, float], list[Cue]]:
    scores = {"fact": 0.0, "inference": 0.0, "opinion": 0.0, "reasoning": 0.0}
    cues: list[Cue] = []

    opinion = _collect(text, OPINION_MARKERS)
    if opinion:
        scores["opinion"] += 2 + len(opinion)
        cues.append(Cue(category=ClaimType.OPINION, kind="marker", value=opinion))

    strong = _collect(text, REASONING_STRONG)
    if strong:
        scores["reasoning"] += 2 + len(strong)
        cues.append(Cue(category=ClaimType.REASONING, kind="connective", value=strong))
    weak = _collect(text, REASONING_WEAK)
    if weak:
        scores["reasoning"] += 1
        cues.append(Cue(category=ClaimType.REASONING, kind="causal-verb", value=weak))

    inference = _collect(text, INFERENCE_MARKERS)
    if inference:
        scores["inference"] += 2 + len(inference)
        cues.append(Cue(category=ClaimType.INFERENCE, kind="hedge", value=inference))

    m = _QUANT.search(text)
    if m:
        scores["fact"] += 2
        cues.append(Cue(category=ClaimType.FACT, kind="quantitative", value=m.group(0)))
    m = _YEAR.search(text)
    if m:
        scores["fact"] += 1.5
        cues.append(Cue(category=ClaimType.FACT, kind="temporal", value=m.group(0)))
    if _ATTR_ZH.search(text) or _ATTR_EN.search(text):
        scores["fact"] += 1.5
        cues.append(Cue(category=ClaimType.FACT, kind="attributed", value="attribution phrase"))

    proper = [w for w in re.split(r"\s+", text)[1:] if _PROPER.match(w)]
    if proper:
        scores["fact"] += min(2, len(proper))
        cues.append(Cue(category=ClaimType.FACT, kind="proper_noun", value=proper))

    return scores, cues


def classify(text: str) -> tuple[ClaimType, float, list[Cue], dict[str, float], str | None]:
    scores, cues = score_claim(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_score = ranked[0]
    second_score = ranked[1][1]

    if top_score == 0:
        return (ClaimType.FACT, 0.4, cues, scores,
                "no explicit markers; treated as bare assertion")

    margin = (top_score - second_score) / (top_score + 1e-9)
    confidence = round(min(0.95, 0.5 + margin * 0.45), 2)
    return ClaimType(top_class), confidence, cues, scores, None


def question_relevance(claim_text: str, question: str | None) -> QuestionAnchor:
    """Lexical overlap between a claim and the question it should answer."""
    if not question:
        return QuestionAnchor(question="", relevance=0.0, reactant=True,
                              note="no question supplied; claim is unanchored")
    q_tokens = tokenize(question)
    c_tokens = tokenize(claim_text)
    rel = round(overlap(q_tokens, c_tokens), 2)
    if not q_tokens:
        return QuestionAnchor(question=question, relevance=0.0, reactant=True,
                              note="question has no discriminative tokens")
    reactant = rel > 0.0
    note = (
        "reacts to the question" if reactant
        else "no token shared with the question; possible off-topic claim"
    )
    return QuestionAnchor(question=question, relevance=rel, reactant=reactant, note=note)


def decompose(answer: str, question: str | None = None) -> tuple[list[Claim], DecompositionStats]:
    claims: list[Claim] = []
    for i, sentence in enumerate(split_claims(answer), start=1):
        ctype, confidence, cues, scores, note = classify(sentence)
        claims.append(
            Claim(
                id=f"C{i}",
                text=sentence,
                type=ctype,
                confidence=confidence,
                cues=cues,
                score_breakdown=scores,
                note=note,
                verifiable=ctype in (ClaimType.FACT, ClaimType.INFERENCE, ClaimType.REASONING),
                question_anchor=question_relevance(sentence, question),
            )
        )

    stats = DecompositionStats(total=len(claims))
    for c in claims:
        setattr(stats, c.type.value, getattr(stats, c.type.value) + 1)
    return claims, stats