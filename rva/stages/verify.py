"""Stage 2 - evidence & source verification.

Assesses source reliability, checks supporting evidence, detects
popularity-driven selection, and flags contradictions. Offline and auditable.
"""

from __future__ import annotations

import re

from ..models import (
    Claim,
    ClaimType,
    Evidence,
    EvidenceRef,
    PopularityBias,
    ReliabilityProfile,
    Source,
    Verification,
)

#: Built-in source-reliability registry, keyed by registrable domain (a host
#: also matches a key it is a subdomain of, so `news.cctv.com` -> `cctv.com`).
#: Scores are 0..1 and ONLY sources scoring >= RELIABILITY_THRESHOLD (0.7) can
#: make a claim `Accurate` / `Inaccurate`. Anything not listed falls to
#: `unknown` (0.50), i.e. below the bar - which is sharp, not an error: an
#: unclassified source is "not yet rated", so it cannot substantiate a claim.
#: The list is deliberately explicit and replaceable via the `registry`
#: argument (see docs/design.md 3.1); the built-in set covers the domains most
#: commonly seen in practice.
DEFAULT_REGISTRY: dict[str, dict] = {
    # --- primary / official sources ---------------------------------------
    "gov": {"tier": "primary-official", "score": 0.95, "note": "Government / official statistics"},
    "edu": {"tier": "academic", "score": 0.90, "note": "Academic institution"},
    "who.int": {"tier": "primary-official", "score": 0.95, "note": "WHO"},
    "cdc.gov": {"tier": "primary-official", "score": 0.95, "note": "US CDC"},
    "un.org": {"tier": "primary-official", "score": 0.95, "note": "United Nations"},
    "europa.eu": {"tier": "primary-official", "score": 0.94, "note": "EU institutions"},
    "gov.cn": {"tier": "primary-official", "score": 0.94, "note": "China central government"},
    "nhs.uk": {"tier": "primary-official", "score": 0.92, "note": "UK National Health Service"},
    # --- peer-reviewed literature -----------------------------------------
    "nature.com": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "science.org": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "thelancet.com": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "nejm.org": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "bmj.com": {"tier": "peer-reviewed", "score": 0.92, "note": "Peer-reviewed journal"},
    "cell.com": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "pnas.org": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "jamanetwork.com": {"tier": "peer-reviewed", "score": 0.93, "note": "Peer-reviewed journal"},
    "arxiv.org": {"tier": "preprint", "score": 0.70, "note": "Preprint, not yet peer reviewed"},
    # --- wire services (agency reporting) ---------------------------------
    "reuters.com": {"tier": "wire", "score": 0.85, "note": "Wire service"},
    "apnews.com": {"tier": "wire", "score": 0.85, "note": "Wire service"},
    "afp.com": {"tier": "wire", "score": 0.85, "note": "Wire service"},
    "xinhuanet.com": {"tier": "wire", "score": 0.85, "note": "Xinhua state news agency"},
    "news.cn": {"tier": "wire", "score": 0.85, "note": "Xinhua state news agency"},
    # --- established major news organizations -----------------------------
    "bbc.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "bbc.co.uk": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "nytimes.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "washingtonpost.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "theguardian.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "cnn.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "aljazeera.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "bloomberg.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "ft.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "economist.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "npr.org": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "dw.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "people.com.cn": {"tier": "major-news", "score": 0.80, "note": "People's Daily"},
    "cctv.com": {"tier": "major-news", "score": 0.80, "note": "China Central Television"},
    "chinadaily.com.cn": {"tier": "major-news", "score": 0.80, "note": "China Daily"},
    "china.org.cn": {"tier": "major-news", "score": 0.78, "note": "State news portal"},
    "cgtn.com": {"tier": "major-news", "score": 0.78, "note": "State broadcaster"},
    "globaltimes.cn": {"tier": "major-news", "score": 0.75, "note": "State-affiliated newspaper"},
    "scmp.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "thepaper.cn": {"tier": "major-news", "score": 0.80, "note": "Established news outlet"},
    "caixin.com": {"tier": "major-news", "score": 0.82, "note": "Established financial news outlet"},
    "yicai.com": {"tier": "major-news", "score": 0.80, "note": "Established financial news outlet"},
    "jpost.com": {"tier": "major-news", "score": 0.78, "note": "Major news organization"},
    "timesofisrael.com": {"tier": "major-news", "score": 0.78, "note": "Major news organization"},
    "haaretz.com": {"tier": "major-news", "score": 0.80, "note": "Major news organization"},
    "china.com.cn": {"tier": "major-news", "score": 0.75, "note": "State-run news portal"},
    # --- encyclopedic / reference -----------------------------------------
    "wikipedia.org": {"tier": "encyclopedic", "score": 0.65, "note": "Crowd-edited encyclopedia"},
    # --- portals / aggregators (below the bar, not primary reporting) ------
    "sina.com.cn": {"tier": "portal", "score": 0.55, "note": "Aggregator portal"},
    "sohu.com": {"tier": "portal", "score": 0.55, "note": "Aggregator portal"},
    "163.com": {"tier": "portal", "score": 0.55, "note": "Aggregator portal"},
    "qq.com": {"tier": "portal", "score": 0.55, "note": "Aggregator portal"},
    "ifeng.com": {"tier": "portal", "score": 0.55, "note": "Aggregator portal"},
    "baidu.com": {"tier": "portal", "score": 0.50, "note": "Search / aggregator"},
    "toutiao.com": {"tier": "portal", "score": 0.50, "note": "Aggregator feed"},
    # --- self-media / low-credibility -------------------------------------
    "medium.com": {"tier": "blog", "score": 0.40, "note": "Self-published blog"},
    "zhihu.com": {"tier": "ugc", "score": 0.45, "note": "Q&A user-generated content"},
    "weibo.com": {"tier": "social", "score": 0.35, "note": "Social media"},
    "x.com": {"tier": "social", "score": 0.35, "note": "Social media"},
    "blog": {"tier": "blog", "score": 0.35, "note": "Generic blog"},
    "social": {"tier": "ugc", "score": 0.30, "note": "User-generated content"},
    "forum": {"tier": "ugc", "score": 0.30, "note": "Forum / UGC"},
    # --- fallback ---------------------------------------------------------
    "unknown": {"tier": "unknown", "score": 0.50, "note": "Unclassified source"},
}

# Assertion-level denials: these deny the substance of a claim.
NEGATION_STRONG = [
    "not", "no", "never", "false", "denied", "myth", "debunked", "refuted",
    "没有", "并非", "不是", "错误", "辟谣", "否认", "毫无", "并无", "夸大",
    "虚构", "不支持", "并不",
]

# Procedural / meta negations, e.g. "not yet peer reviewed". These concern
# process, not the claim's substance, so they need a very tight window.
NEGATION_WEAK = ["未", "尚未", "not yet", "暂无"]

THRESHOLDS = {
    # `strong_window` / `weak_window` bound the negation-proximity check that the
    # offline NLI heuristic (`rva/nli.py`) uses for refutation detection.
    "strong_window": 14,
    "weak_window": 4,
}

# Retrieval defaults, echoing Google ADK's RAG engine knobs
# (VertexAiRagRetrieval: similarity_top_k / vector_distance_threshold).
DEFAULT_TOP_K = 5
DEFAULT_DISTANCE_THRESHOLD = None  # None -> backend-specific default


def _domain_of(url: str | None) -> str:
    if not url:
        return ""
    m = re.match(r"^https?://([^/]+)", url)
    host = m.group(1) if m else url.split("/")[0]
    return host.replace("www.", "")


def _all_positions(lower: str, markers: list[str], strength: str) -> list[dict]:
    positions = []
    for marker in markers:
        start = lower.find(marker)
        while start != -1:
            positions.append({"at": start, "marker": marker, "strength": strength})
            start = lower.find(marker, start + 1)
    return positions


def relevant_negation(claim_tokens: list[str], source_text: str) -> dict | None:
    """Return a negation that sits near a token shared with the claim.

    Distinguishes assertion-level denials (wide window) from procedural ones
    (tight window) so that e.g. "not yet peer reviewed" does not refute a finding.
    """
    lower = (source_text or "").lower()
    if not lower:
        return None
    negations = (
        _all_positions(lower, NEGATION_STRONG, "strong")
        + _all_positions(lower, NEGATION_WEAK, "weak")
    )
    if not negations:
        return None

    shared = [t for t in dict.fromkeys(claim_tokens) if t in lower]
    for neg in negations:
        window = THRESHOLDS["strong_window"] if neg["strength"] == "strong" else THRESHOLDS["weak_window"]
        for token in shared:
            idx = lower.find(token)
            if idx != -1 and abs(idx - neg["at"]) <= window:
                return {**neg, "token": token}
    return None


def _registry_candidates(domain: str) -> list[str]:
    """Suffixes of `domain` a registry key may match, longest (most specific) first.

    ``news.cctv.com`` -> ``["news.cctv.com", "cctv.com", "com"]`` so an exact
    key beats its registrable parent, and ``www.gov.cn`` still finds ``gov.cn``.
    """
    if not domain:
        return []
    parts = domain.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]


def score_source(source: Source, registry: dict[str, dict] | None = None) -> ReliabilityProfile:
    registry = registry or DEFAULT_REGISTRY
    domain = _domain_of(source.url) or (source.domain or "")

    # Match the URL host (and its suffixes) first, then the caller-supplied
    # `tag`, so an explicit tag acts as an extra signal, not an override.
    signal_groups: list[list[str]] = [_registry_candidates(domain)]
    if source.tag:
        signal_groups.append([source.tag.lower()])

    for signals in signal_groups:
        for sig in signals:
            if sig in registry:
                profile = registry[sig]
                return ReliabilityProfile(
                    tier=profile["tier"], score=profile["score"], note=profile["note"],
                    matched=sig, domain=domain,
                )

    # `gov` / `edu` are registry keys themselves, so the suffix walk above
    # already resolves e.g. `mit.edu` -> `edu` and `x.gov` -> `gov`; a host that
    # merely *contains* those labels (e.g. `gov.cn.evil.com`) matches nothing.

    profile = registry["unknown"]
    return ReliabilityProfile(tier=profile["tier"], score=profile["score"], note=profile["note"], matched="none", domain=domain)


def assess_evidence(claim: Claim, sources: list[Source], registry=None,
                    retriever=None, nli=None) -> Evidence:
    """Retrieve relevant chunks for a claim, then judge each chunk's stance.

    Pipeline: vector retrieval (top-k chunks under a distance threshold) ->
    three-way NLI (support / refute / neutral) -> Evidence buckets. The
    reliability of the *source* is attached to its chunk, so a confidently
    refuting blog does not automatically outweigh a supporting official source
    (that weighing happens downstream in `evaluate.py`).
    """
    from ..nli import Label, build_nli_judge
    from ..retrieval import VectorRetriever

    if nli is None:
        nli = build_nli_judge("heuristic")
    if retriever is None:
        retriever = VectorRetriever(sources or [], similarity_top_k=DEFAULT_TOP_K,
                                    vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD)

    by_id = {s.id: s for s in sources}
    similarities = retriever.similarity_map(claim.text)

    supports: list[EvidenceRef] = []
    partials: list[EvidenceRef] = []
    contradicts: list[EvidenceRef] = []

    for chunk in retriever.retrieve(claim.text):
        source = by_id.get(chunk.source_id)
        if source is None:
            continue
        profile = score_source(source, registry)
        label, reason = nli.judge(claim, chunk.text)

        ref = EvidenceRef(
            source_id=source.id,
            title=source.title,
            url=source.url,
            domain=profile.domain,
            reliability=profile.score,
            tier=profile.tier,
            similarity=round(chunk.similarity, 2),
            coverage=round(chunk.similarity, 2),  # kept for report compatibility
            shared_tokens=0,
            traffic=source.traffic,
            negation=reason if label is Label.REFUTE else None,
            stance=label.value,
        )

        if label is Label.SUPPORT:
            supports.append(ref)
        elif label is Label.REFUTE:
            contradicts.append(ref)
        else:
            partials.append(ref)

    evidence = Evidence(supports=supports, partials=partials, contradicts=contradicts)

    # Candidates the vector search ranked below the distance threshold are still
    # reported so a reader can see what was screened out and why.
    seen = {r.source_id for r in supports + partials + contradicts}
    for source in sources:
        if source.id in seen:
            continue
        profile = score_source(source, registry)
        similarity = round(similarities.get(source.id, 0.0), 2)
        evidence.screened.append(
            EvidenceRef(
                source_id=source.id, title=source.title, url=source.url,
                domain=profile.domain, reliability=profile.score, tier=profile.tier,
                similarity=similarity, coverage=similarity, shared_tokens=0,
                traffic=source.traffic, stance="screened",
            )
        )

    return evidence

def detect_popularity_bias(evidence: Evidence) -> PopularityBias:
    """Flag support that rests on high-traffic but low-reliability sources."""
    refs = evidence.supports + evidence.partials
    if not refs:
        return PopularityBias(biased=False, ratio=0.0, note="no supporting sources")

    with_traffic = [r for r in refs if r.traffic is not None]
    if not with_traffic:
        return PopularityBias(biased=False, ratio=0.0, note="no traffic signal available")

    avg_rel = sum(r.reliability for r in with_traffic) / len(with_traffic)
    avg_traf = sum(r.traffic for r in with_traffic) / len(with_traffic)  # type: ignore[arg-type]
    biased = avg_traf > 0.6 and avg_rel < 0.7
    return PopularityBias(
        biased=biased,
        ratio=round(avg_rel, 2),
        avg_traffic=round(avg_traf, 2),
        note=(
            "support rests on high-traffic but lower-reliability sources"
            if biased else "support not dominated by popularity signal"
        ),
    )


def verify(claims: list[Claim], sources: list[Source], registry=None,
           domain: str = "general", retriever=None, nli=None):
    """Run Stage 2 over every claim.

    `retriever` and `nli` are injected once and shared across claims so the
    corpus is embedded a single time and a single judge (e.g. the LLM one) is
    reused for the whole answer.
    """
    from ..nli import build_nli_judge
    from ..retrieval import VectorRetriever

    if nli is None:
        nli = build_nli_judge("heuristic")
    if retriever is None:
        retriever = VectorRetriever(sources or [], similarity_top_k=DEFAULT_TOP_K,
                                    vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD)

    results: list[Verification] = []
    for claim in claims:
        evidence = assess_evidence(claim, sources, registry, retriever=retriever, nli=nli)
        bias = detect_popularity_bias(evidence)

        independent_domains = len({e.domain for e in evidence.supports})

        anchor = claim.question_anchor
        relevance = anchor.relevance if anchor is not None else 0.0
        # A knowledge claim that shares no vocabulary with the question is
        # off-topic: it reacts to something other than what was asked.
        off_topic = bool(
            anchor is not None
            and anchor.question
            and not anchor.reactant
            and claim.type is not ClaimType.OPINION
        )

        # Per-claim evidence-efficacy checks (authority / freshness / consistency
        # / traffic-vs-validity). Offline and deterministic; shared by both critic
        # tracks since both route through verify().
        from ..efficacy import (
            assess_consistency,
            assess_freshness,
            assess_source_quality,
            score_support,
            suspicious_traffic_ids,
        )

        source_quality = assess_source_quality(evidence, sources)
        freshness = assess_freshness(claim.text, evidence, sources)
        consistency = assess_consistency(claim.text, evidence, sources)
        support_score = score_support(evidence, consistency, bias.biased)
        traffic_suspects = suspicious_traffic_ids(evidence)

        results.append(
            Verification(
                claim_id=claim.id,
                claim_text=claim.text,
                claim_type=claim.type,
                evidence=evidence,
                support_score=round(support_score, 2),
                independent_domains=independent_domains,
                conflict=len(evidence.contradicts) > 0,
                source_quality=source_quality,
                freshness=freshness,
                consistency=consistency,
                popularity_suspect_source_ids=traffic_suspects,
                popularity_bias=bias,
                retrieval_backend=getattr(retriever, "resolved_backend", ""),
                nli_backend=getattr(nli, "name", ""),
                screened_count=len(evidence.screened),
                question_relevance=relevance,
                off_topic=off_topic,
            )
        )
    return results