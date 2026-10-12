"""Compare factuality backends on the bundled example.

Factuality (five-level, external-source-anchored) is always computed locally and
is identical across backends. The `ragas` / `trulens` backends merely add a
*separate* grounding-only report, printed in its own table, so you can see that a
judge score never rewrites the five-level verdict.

Stage 2 (vector retrieval + three-way NLI) is selectable independently, so you
can see how the evidence set changes.

Usage:
    python examples/compare_judges.py                                   # local, offline stage 2
    python examples/compare_judges.py local ragas                       # + real Ragas grounding judge
    python examples/compare_judges.py local ragas --retriever openai --nli llm
    python examples/compare_judges.py local --retriever hash --nli heuristic

Writes examples/comparison_<backend>.json (or
examples/comparison_<backend>__<retriever>-<nli>.json for non-default stage 2)
and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rva import Source, validate  # noqa: E402
from rva.config import get_nli_backend, get_retriever_backend  # noqa: E402

ANSWER_PATH = ROOT / "examples" / "answer.txt"
QUESTION_PATH = ROOT / "examples" / "question.txt"
SOURCES_PATH = ROOT / "examples" / "sources.json"


def _stage2_slug(retriever: str, nli: str) -> str:
    defaults = (get_retriever_backend(), get_nli_backend())
    if (retriever, nli) == defaults:
        return ""  # keep the original filename
    return f"__{retriever}-{nli}"


def run(backend: str, retriever: str, nli: str) -> dict:
    answer = ANSWER_PATH.read_text(encoding="utf-8")
    question = QUESTION_PATH.read_text(encoding="utf-8").strip()
    sources = [Source(**s) for s in json.loads(SOURCES_PATH.read_text(encoding="utf-8"))]
    result = validate(
        answer=answer, question=question, sources=sources, domain="public-health",
        judge=backend, retriever=retriever, nli=nli, revise=True,
    )
    f = result.factuality
    g = result.grounding
    first = result.verifications[0] if result.verifications else None

    payload = {
        "backend": backend,
        "retriever": retriever,
        "nli": nli,
        "stage2": {
            "retrieval_backend": getattr(first, "retrieval_backend", "") if first else "",
            "nli_backend": getattr(first, "nli_backend", "") if first else "",
            "screened_total": sum(v.screened_count for v in result.verifications),
        },
        # factuality is always local + source-anchored, independent of the judge
        "method": f.method,
        "score": f.score,
        "accurate": f.accurate,
        "inaccurate": f.inaccurate,
        "disputed": f.disputed,
        "unsupported": f.unsupported,
        "not_applicable": f.not_applicable,
        "verifiable_claims": f.verifiable_claims,
        "question": result.question,
        "honesty": None if result.honesty is None else {
            "score": result.honesty.score,
            "passed": result.honesty.passed,
            "flags": [{"code": fl.code, "label": fl.label, "claim_ids": fl.claim_ids}
                      for fl in result.honesty.flags],
        },
        "revised_text": None if result.revision is None else result.revision.revised_text,
        "verdicts": [
            {
                "claim_id": v.claim_id,
                "type": v.claim_type.value,
                "verdict": v.verdict.value,
                "score": v.score,
                "rationale": v.rationale,
                "off_topic": next((x.off_topic for x in result.verifications if x.claim_id == v.claim_id), False),
            }
            for v in f.verdicts
        ],
        # grounding-only, separate report (None when backend == local)
        "grounding": None if g is None else {
            "method": g.method,
            "judged_claims": g.judged_claims,
            "grounded_count": g.grounded_count,
            "ungrounded_count": g.ungrounded_count,
            "skipped_claims": g.skipped_claims,
            "mean_score": g.mean_score,
            "items": [
                {
                    "claim_id": it.claim_id,
                    "type": it.claim_type.value,
                    "grounded": it.grounded,
                    "score": it.score,
                    "rationale": it.rationale,
                }
                for it in g.items
            ],
        },
        "evidence": [
            {
                "claim_id": v.claim_id,
                "supports": [(e.source_id, e.stance, e.similarity) for e in v.evidence.supports],
                "neutral": [(e.source_id, e.stance, e.similarity) for e in v.evidence.partials],
                "refutes": [(e.source_id, e.stance, e.similarity) for e in v.evidence.contradicts],
                "screened": [(e.source_id, e.similarity) for e in v.evidence.screened],
            }
            for v in result.verifications
        ],
    }

    out_path = ROOT / "examples" / f"comparison_{backend}{_stage2_slug(retriever, nli)}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _grounding_cell(payload: dict, claim_id: str) -> str:
    g = payload.get("grounding")
    if g is None:
        return "-"
    item = next((x for x in g["items"] if x["claim_id"] == claim_id), None)
    if item is None or item["grounded"] is None:
        return "no verdict"
    label = "grounded" if item["grounded"] else "ungrounded"
    score = item["score"]
    return f"{label} ({score})" if score is not None else label


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RVA factuality backends (and grounding judges)")
    parser.add_argument("backends", nargs="*", default=None,
                        help="backends to compare (default: local); ragas/trulens add a grounding report")
    parser.add_argument("--retriever", default=get_retriever_backend(),
                        choices=["hash", "openai", "auto"],
                        help="Stage 2 embedding backend (default from .env: %(default)s)")
    parser.add_argument("--nli", default=get_nli_backend(),
                        choices=["heuristic", "llm", "auto"],
                        help="Stage 2 entailment judge (default from .env: %(default)s)")
    args = parser.parse_args()

    backends = args.backends or ["local"]
    results = {b: run(b, args.retriever, args.nli) for b in backends}
    first = next(iter(results.values()))

    print("\n=== Factuality: five-level verdicts (identical across backends) ===")
    print(f"Stage 2: retriever={args.retriever}  nli={args.nli}")
    print(f"{'backend':<10}{'method':<26}{'score':>6}")
    for b, r in results.items():
        print(f"{b:<10}{r['method']:<26}{r['score']:>6}")
    print("resolved stage 2: " + str(first["stage2"]))
    print("(the judge never changes these verdicts; see the grounding table below)")

    print("\n=== Per-claim factuality verdicts ===")
    ids = [v["claim_id"] for v in first["verdicts"]]
    header = f"{'claim':<7}{'type':<11}" + "".join(f"{b:<20}" for b in results)
    print(header)
    for cid in ids:
        row = f"{cid:<7}"
        ctype = next(v["type"] for v in first["verdicts"] if v["claim_id"] == cid)
        row += f"{ctype:<11}"
        for b in results:
            v = next(x for x in results[b]["verdicts"] if x["claim_id"] == cid)
            row += f"{v['verdict'] + ' (' + str(v['score']) + ')':<20}"
        print(row)

    if any(r.get("grounding") is not None for r in results.values()):
        print("\n=== Grounding-only judge (separate report; never a factuality verdict) ===")
        header = f"{'claim':<7}{'type':<11}" + "".join(f"{b:<24}" for b in results)
        print(header)
        for cid in ids:
            ctype = next(v["type"] for v in first["verdicts"] if v["claim_id"] == cid)
            row = f"{cid:<7}{ctype:<11}"
            for b in results:
                row += f"{_grounding_cell(results[b], cid):<24}"
            print(row)
        for b, r in results.items():
            g = r.get("grounding")
            if g:
                print(f"  {b}: judged {g['judged_claims']} | grounded {g['grounded_count']} | "
                      f"ungrounded {g['ungrounded_count']} | skipped {g['skipped_claims']} | "
                      f"mean {g['mean_score']}")

    if first.get("honesty"):
        h = first["honesty"]
        print("\n=== Honesty & auditability (separate from factuality) ===")
        print(f"score {h['score']} | {'PASS' if h['passed'] else 'NEEDS WORK'} | "
              + (", ".join(fl["code"] for fl in h["flags"]) or "no flags"))
        for fl in h["flags"]:
            ids = ", ".join(fl["claim_ids"]) or "-"
            print(f"  [{fl['code']}] {fl['label']} ({ids})")
    if first.get("revised_text"):
        print("\n=== Revised answer (Reviser; original never mutated) ===")
        print(first["revised_text"])

    print("\n=== Stage 2 evidence (support / refute / screened) ===")
    for item in first["evidence"]:
        sup = ",".join(f"{sid}@{sim}" for sid, _st, sim in item["supports"]) or "-"
        ref = ",".join(f"{sid}@{sim}" for sid, _st, sim in item["refutes"]) or "-"
        scr = ",".join(f"{sid}@{sim}" for sid, sim in item["screened"]) or "-"
        print(f"  {item['claim_id']}: support[{sup}] refute[{ref}] screened[{scr}]")

    print("\nSaved: " + ", ".join(
        f"examples/comparison_{b}{_stage2_slug(args.retriever, args.nli)}.json" for b in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
