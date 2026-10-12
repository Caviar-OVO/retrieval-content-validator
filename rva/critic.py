"""Critical verification (the Critic role), on two tracks.

`critic_backend="deterministic"` (default) is the offline, rule-based critic:
`decompose` -> `verify` -> `evaluate_factuality`, fully auditable and zero-cost.
`critic_backend="llm"` injects the Critic persona (`rva.roles.CRITIC.prompt`,
with a few-shot) and lets the model do the claim extraction / classification and
the five-level verdict in one pass - the Google LLM Auditor critic shape. It is
opt-in (a demo), and falls back to the deterministic critic on any failure or
schema violation.

Both tracks return the **same schema**: a list of `Claim` (with `cues`), the
`Verification` evidence layers, and a `FactualityResult` whose `verdicts` are
`ClaimVerdict` (verdict + `rationale` = justification + `evidence_domains`). The
LLM track therefore keeps `Claim.cues`, `Claim.note` (justification) and
`ClaimVerdict.rationale` populated, so nothing downstream drifts.

The five-level verdict is *not* delegated to an LLM in the default path: the
verdict for the deterministic critic still comes from `evaluate_factuality`, and
`evaluate_factuality` keeps having no judge hook. Only the explicit
`critic_backend="llm"` produces LLM-authored verdicts, reported with
`method="llm_critic"`.

The LLM critic is not re-checked: a critic *audits* rather than *generates new
text*, so the revision->recheck gate does not apply to it (it applies to the
Reviser). Audibility instead rests on the required per-claim `justification` and
`evidence`: a knowledge verdict with no cited source is rejected.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from .evaluate import (
    NOT_APPLICABLE_VERDICT,
    VERIFIABLE_TYPES,
    evaluate_factuality,
    verdict_score,
)
from .models import (
    Claim,
    ClaimType,
    ClaimVerdict,
    Cue,
    DecompositionStats,
    FactualityResult,
    Source,
    Verification,
    Verdict,
)
from .stages.decompose import classify, decompose, question_relevance
from .stages.verify import score_source, verify


#: Method reported on `FactualityResult`, by the engine that produced the verdicts.
DETERMINISTIC_METHOD = "local_source_anchored"
LLM_METHOD = "llm_critic"

#: Fallbacks are logged so a silent downgrade cannot hide behind a green report.
logger = logging.getLogger("rva.critic")

_TYPES = {t.value: t for t in ClaimType}
_VERDICTS = {v.value: v for v in Verdict}

#: Verdicts that assert the claim is substantiated and therefore must cite a source.
_EVIDENCE_REQUIRED = (Verdict.ACCURATE, Verdict.INACCURATE, Verdict.DISPUTED)


@dataclass
class CriticOutput:
    """What a critic returns: claims, evidence layers, verdicts, and stats."""

    claims: list[Claim]
    verifications: list[Verification]
    factuality: FactualityResult
    stats: DecompositionStats
    method: str = DETERMINISTIC_METHOD
    fallback: bool = False


def _clamp(value: Any, default: float = 0.6) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _extract_json(text: str) -> Any:
    """Best-effort JSON recovery from an LLM reply (fences, prose, lists)."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                continue
    return None


def _claim_items(data: Any) -> Optional[list[dict]]:
    """Normalize a critic reply into a list of per-claim dicts, or None."""
    if isinstance(data, dict):
        data = data.get("claims", data.get("items"))
    if not isinstance(data, list):
        return None
    return [item for item in data if isinstance(item, dict)]


def _parse_cues(value: Any) -> list[Cue]:
    cues: list[Cue] = []
    if not isinstance(value, list):
        return cues
    for item in value:
        if not isinstance(item, dict):
            continue
        category = _TYPES.get(str(item.get("category", "")).strip().lower())
        if category is None:
            continue
        cues.append(Cue(category=category, kind=str(item.get("kind", "llm")), value=item.get("value")))
    return cues


class DeterministicCritic:
    """Offline, rule-based critic: decompose -> verify -> evaluate (default)."""

    name = "deterministic"

    def critique(
        self,
        answer: str,
        sources: list[Source],
        *,
        question: str | None = None,
        domain: str = "general",
        registry: dict | None = None,
        retriever=None,
        nli=None,
    ) -> CriticOutput:
        claims, stats = decompose(answer, question=question)
        verifications = verify(
            claims, sources, registry=registry, domain=domain,
            retriever=retriever, nli=nli,
        )
        factuality = evaluate_factuality(claims, verifications)
        return CriticOutput(
            claims=claims, verifications=verifications, factuality=factuality,
            stats=stats, method=DETERMINISTIC_METHOD, fallback=False,
        )


class LLMCritic:
    """Optional LLM critic: claim extraction + five-level verdict in one pass.

    Requires each claim to carry a `justification` and an `evidence` list; a
    knowledge verdict (accurate / inaccurate / disputed) must cite at least one
    source id. Any parse/transport/schema failure falls back to the
    deterministic critic (`CriticOutput.fallback` is then True).
    """

    name = "llm"

    def __init__(self, llm=None, model: str | None = None, temperature: float = 0.0):
        self._client = llm
        self._model = model
        self._temperature = temperature
        self.calls = 0
        self.failures = 0
        self.fallback = False

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

    @staticmethod
    def _system_prompt() -> str:
        from .roles import CRITIC

        few_shot = (
            "\n\n输出格式（严格 JSON，不要多余文字）：\n"
            '{"claims":[{"text":"<原句>","type":"fact|reasoning|opinion|inference",'
            '"confidence":0.0,"verdict":"accurate|inaccurate|disputed|unsupported|not_applicable",'
            '"justification":"<判定理由>","evidence":["S1"],'
            '"cues":[{"category":"fact","kind":"quantitative","value":"420万"}]}]}\n\n'
            "示例：\n"
            "输入：问题「空气污染每年导致多少人过早死亡？」；来源 S1: 「室外空气污染导致全球每年约420万人过早死亡。」\n"
            '输出：{"claims":[{"text":"空气污染每年导致约420万人过早死亡。","type":"fact",'
            '"confidence":0.9,"verdict":"accurate","justification":"S1 明确给出每年约420万人的数字，与陈述一致。",'
            '"evidence":["S1"],"cues":[{"category":"fact","kind":"quantitative","value":"420万"}]}]}\n\n'
            "硬性要求：justification 与 evidence 必填；accurate/inaccurate/disputed 必须至少引用一个来源 id；"
            "主观意见判 not_applicable；只依据所给来源判定，不得引入外部知识。"
        )
        return CRITIC.prompt + few_shot

    @staticmethod
    def _user_prompt(answer: str, question: str | None, sources: list[Source]) -> str:
        lines = [f"问题（claim 锚点）：{question or '（未提供）'}", "", "待审回答：", answer, "", "检索到的来源："]
        for s in sources or []:
            text = (s.text or "").strip().replace("\n", " ")
            lines.append(f"[{s.id}] {text}")
        lines.append("")
        lines.append("请逐条抽取 claim 并给出五级判定，输出上面的严格 JSON。")
        return "\n".join(lines)

    def _call(self, answer: str, question: str | None, sources: list[Source]) -> Optional[list[dict]]:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": self._user_prompt(answer, question, sources)},
            ],
        )
        self.calls += 1
        return _claim_items(_extract_json(self._content(resp.choices[0].message)))

    def _records(self, items: list[dict], question: str | None) -> list[dict]:
        """Validate + normalize replies; raises ValueError on a schema violation."""
        records: list[dict] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if len(text) < 4:
                continue

            justification = str(item.get("justification", "")).strip()
            if not justification:
                raise ValueError("claim is missing the required justification")

            evidence = item.get("evidence")
            if not isinstance(evidence, list):
                raise ValueError("claim is missing the required evidence list")
            evidence_ids = [str(e).strip() for e in evidence if str(e).strip()]

            ctype = _TYPES.get(str(item.get("type", "")).strip().lower())
            if ctype is None:
                ctype, confidence, cues, _, _ = classify(text)
            else:
                confidence = _clamp(item.get("confidence", 0.6))
                cues = _parse_cues(item.get("cues")) or classify(text)[2]

            verdict = _VERDICTS.get(str(item.get("verdict", "")).strip().lower(), Verdict.UNSUPPORTED)
            if ctype is ClaimType.OPINION:
                verdict = Verdict.NOT_APPLICABLE  # opinions stay exempt, as in the rule track
            elif verdict is Verdict.NOT_APPLICABLE:
                verdict = Verdict.UNSUPPORTED  # a knowledge claim cannot be "not applicable"
            if verdict in _EVIDENCE_REQUIRED and not evidence_ids:
                raise ValueError("a substantiated verdict must cite at least one source id")

            records.append({
                "text": text, "type": ctype, "confidence": confidence, "cues": cues,
                "verdict": verdict, "justification": justification, "evidence": evidence_ids,
            })

        if not records:
            raise ValueError("the critic returned no usable claims")
        return records

    def critique(
        self,
        answer: str,
        sources: list[Source],
        *,
        question: str | None = None,
        domain: str = "general",
        registry: dict | None = None,
        retriever=None,
        nli=None,
    ) -> CriticOutput:
        det = DeterministicCritic()
        try:
            items = self._call(answer, question, sources)
            records = self._records(items or [], question)
        except Exception as exc:  # pragma: no cover - depends on live backend
            self.failures += 1
            self.fallback = True
            logger.warning(
                "LLM critic failed (%s: %s); falling back to the deterministic "
                "critic. critic_backend will be reported as 'deterministic'.",
                type(exc).__name__, exc,
            )
            out = det.critique(answer, sources, question=question, domain=domain,
                               registry=registry, retriever=retriever, nli=nli)
            out.fallback = True
            return out

        try:
            claims = self._build_claims(records, question)
            verifications = verify(
                claims, sources, registry=registry, domain=domain,
                retriever=retriever, nli=nli,
            )
            factuality = self._build_factuality(claims, records, sources)
        except Exception as exc:  # pragma: no cover - defensive
            self.failures += 1
            self.fallback = True
            logger.warning(
                "LLM critic output could not be assembled (%s: %s); falling back "
                "to the deterministic critic. critic_backend will be reported as "
                "'deterministic'.",
                type(exc).__name__, exc,
            )
            out = det.critique(answer, sources, question=question, domain=domain,
                               registry=registry, retriever=retriever, nli=nli)
            out.fallback = True
            return out

        stats = DecompositionStats(total=len(claims))
        for c in claims:
            setattr(stats, c.type.value, getattr(stats, c.type.value) + 1)

        self.fallback = False
        return CriticOutput(
            claims=claims, verifications=verifications, factuality=factuality,
            stats=stats, method=LLM_METHOD, fallback=False,
        )

    @staticmethod
    def _build_claims(records: list[dict], question: str | None) -> list[Claim]:
        claims: list[Claim] = []
        for i, rec in enumerate(records, start=1):
            claims.append(
                Claim(
                    id=f"C{i}",
                    text=rec["text"],
                    type=rec["type"],
                    confidence=rec["confidence"],
                    cues=rec["cues"],
                    score_breakdown={"llm": 1.0},
                    note=rec["justification"],
                    verifiable=rec["type"] in VERIFIABLE_TYPES,
                    question_anchor=question_relevance(rec["text"], question),
                )
            )
        return claims

    @staticmethod
    def _build_factuality(
        claims: list[Claim], records: list[dict], sources: list[Source]
    ) -> FactualityResult:
        domains = {}
        for s in sources or []:
            domains[s.id] = score_source(s).domain

        verdicts: list[ClaimVerdict] = []
        judged = 0
        for claim, rec in zip(claims, records):
            if claim.type is ClaimType.OPINION:
                verdict, score, rationale = NOT_APPLICABLE_VERDICT
            else:
                verdict = rec["verdict"]
                score = verdict_score(verdict)
                rationale = rec["justification"]
                judged += 1
            cited = sorted({domains.get(eid, eid) for eid in rec["evidence"] if domains.get(eid, eid)})
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.id, claim_text=claim.text, claim_type=claim.type,
                    verdict=verdict, score=round(score, 2), evidence_domains=cited,
                    rationale=rationale,
                )
            )

        total = len(verdicts)
        accurate = sum(1 for x in verdicts if x.verdict is Verdict.ACCURATE)
        inaccurate = sum(1 for x in verdicts if x.verdict is Verdict.INACCURATE)
        disputed = sum(1 for x in verdicts if x.verdict is Verdict.DISPUTED)
        unsupported = sum(1 for x in verdicts if x.verdict is Verdict.UNSUPPORTED)
        not_applicable = sum(1 for x in verdicts if x.verdict is Verdict.NOT_APPLICABLE)
        score = round((accurate + not_applicable) / total, 2) if total else 0.0

        return FactualityResult(
            score=score, accurate=accurate, inaccurate=inaccurate, disputed=disputed,
            unsupported=unsupported, not_applicable=not_applicable, total_claims=total,
            verifiable_claims=judged, applicable_claims=total - not_applicable,
            verdicts=verdicts, method=LLM_METHOD,
        )


def build_critic(
    name: str = "deterministic",
    *,
    llm=None,
    llm_ok: bool = False,
    has_sources: bool = False,
):
    """Return a critic backend for `name` in {deterministic, llm, auto}.

    * ``deterministic`` (default) -> the offline rule-based critic.
    * ``llm`` -> the LLM critic (needs a key + network); falls back to the
      deterministic critic on any failure.
    * ``auto`` -> LLM only when a key is available **and** there are sources to
      judge against; otherwise deterministic (a downgrade, logged).

    Source-gating is **strict**: ``has_sources`` defaults to ``False``, so ``auto``
    will not select the LLM critic unless the caller confirms there is something to
    judge against.

    An **explicit** ``llm`` / ``deterministic`` always wins over ``auto`` - ``auto``
    never overrides a caller's stated intent. An explicit ``llm`` is honoured even
    without sources (the caller was explicit), but it is logged: with nothing to
    judge the LLM critic can only fall back anyway.
    """
    name = (name or "deterministic").lower()
    if name == "auto":
        if llm_ok and has_sources:
            name = "llm"
        else:
            reason = ("no API key is available" if not llm_ok
                      else "no sources to judge against")
            logger.warning(
                "critic_backend='auto' downgraded to the deterministic critic: %s. "
                "The report will record critic_backend='deterministic'.", reason,
            )
            name = "deterministic"
    if name in ("llm", "openai", "critic"):
        if not has_sources:
            logger.warning(
                "LLM critic requested with no sources to judge against; honouring "
                "the explicit request (an explicit backend beats 'auto') but flagging "
                "it: with no sources the LLM critic can only fall back.",
            )
        return LLMCritic(llm=llm)
    return DeterministicCritic()
