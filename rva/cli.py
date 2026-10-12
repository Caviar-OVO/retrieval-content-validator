"""Command-line entry point.

Usage:
    python -m rva.cli --answer "..." --sources examples/sources.json
    python -m rva.cli --answer-file examples/answer.txt --question "..." --sources examples/sources.json
    python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --revise
    python -m rva.cli --answer-file examples/answer.txt --sources examples/sources.json --judge ragas --grounding

Four independent reports are produced, deliberately kept apart:

  * factuality - five-level verdicts anchored on external reliable sources (always local)
  * grounding  - optional Ragas / TruLens judge (is the claim entailed by the retrieved contexts?)
  * honesty    - does the answer disclose its own limits? (H1-H5)
  * revision   - optional Reviser rewrite (the original answer is never mutated)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (
    format_factuality,
    format_grounding,
    format_honesty,
    format_report,
    validate,
)
from .config import get_default_domain, get_default_judge, get_revise_default
from .roles import format_text as format_roles
from .models import Source


def _force_utf8_output() -> None:
    """Make CJK output readable on consoles whose default code page is not UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _load_sources(path: str | None) -> list[Source]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Source(**item) for item in data]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rva", description="Response Validation Agent")
    parser.add_argument("--answer", help="the LLM answer text")
    parser.add_argument("--answer-file", help="path to a file containing the answer")
    parser.add_argument("--question", help="the user's question (anchors claim relevance)")
    parser.add_argument("--question-file", help="path to a file containing the question")
    parser.add_argument("--sources", help="path to a JSON file with retrieved sources")
    parser.add_argument("--domain", default=get_default_domain(), help="topic domain for credibility weighting")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    parser.add_argument("--summary", action="store_true",
                        help="attach the report-level summary (five-level distribution + "
                             "running rates; NOT a single score)")
    parser.add_argument("--factuality", "--faithfulness", dest="factuality", action="store_true",
                        help="show only the factuality report (--faithfulness is kept as an alias)")
    parser.add_argument("--grounding", action="store_true",
                        help="show only the grounding-only judge report (needs --judge ragas/trulens)")
    parser.add_argument("--honesty", action="store_true", help="show only the honesty / disclosure report")
    parser.add_argument("--revise", action="store_true", default=get_revise_default(),
                        help="run the Reviser stage and emit the revised answer (original is never mutated)")
    parser.add_argument("--revised-text-only", action="store_true",
                        help="print only the revised answer text (implies --revise)")
    parser.add_argument("--critic", choices=["deterministic", "llm", "auto"], default=None,
                        help="critic backend: deterministic=offline rules (default) | llm=Critic persona | auto")
    parser.add_argument("--reviser", choices=["local", "llm", "auto"], default=None,
                        help="revision backend: local=deterministic offline (default) | llm=editor persona | auto")
    parser.add_argument("--no-recheck", action="store_true",
                        help="skip the revision->recheck gate (factuality re-check of the revised text)")
    parser.add_argument("--recheck-threshold", type=float, default=None,
                        help="score below which the recheck gate forces another edit (default 0.7)")
    parser.add_argument("--recheck-rounds", type=int, default=None,
                        help="maximum revision->recheck rounds (default 2)")
    parser.add_argument("--judge", default=get_default_judge(), choices=["local", "ragas", "trulens"],
                        help="grounding-only judge backend (ragas/trulens need network + OPENAI_API_KEY); "
                             "it never changes the factuality verdicts")
    parser.add_argument("--retriever", choices=["hash", "openai", "auto"], default=None,
                        help="Stage 2 embedding backend (hash=offline, openai=real embeddings)")
    parser.add_argument("--nli", choices=["heuristic", "llm", "auto"], default=None,
                        help="Stage 2 entailment judge (heuristic=offline, llm=LLM judge)")
    parser.add_argument("--judge-status", action="store_true", help="show which judge backends are available and exit")
    parser.add_argument("--show-config", action="store_true", help="show the loaded configuration (key redacted) and exit")
    parser.add_argument("--roles", action="store_true", help="print the two-stage role setting (Critic / Reviser) and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)

    if args.roles:
        print(format_roles())
        return 0

    if args.show_config:
        from .config import describe

        print(json.dumps(describe(), indent=2))
        return 0

    if args.judge_status:
        from .integrations import availability

        status = {"config": __import__("rva.config", fromlist=["describe"]).describe(), "backends": availability()}
        print(json.dumps(status, indent=2))
        return 0

    answer = args.answer
    if args.answer_file:
        answer = Path(args.answer_file).read_text(encoding="utf-8")
    if not answer:
        print('Provide a response via --answer "..." or --answer-file <path>', file=sys.stderr)
        return 1

    question = args.question
    if args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8").strip()

    revise = args.revise or args.revised_text_only

    result = validate(
        answer=answer,
        question=question,
        sources=_load_sources(args.sources),
        domain=args.domain,
        judge=args.judge,
        retriever=args.retriever,
        nli=args.nli,
        revise=revise,
        critic_backend=args.critic,
        reviser_backend=args.reviser,
        recheck=(False if args.no_recheck else None),
        recheck_threshold=args.recheck_threshold,
        recheck_max_rounds=args.recheck_rounds,
        summary=args.summary,
    )

    if args.revised_text_only:
        if result.revision is None:
            print("No revision produced.", file=sys.stderr)
            return 1
        print(result.revision.revised_text)
    elif args.json:
        print(result.model_dump_json(indent=2))
    elif args.grounding:
        if result.grounding is None:
            print("No grounding report: pass --judge ragas or --judge trulens.", file=sys.stderr)
            return 1
        print(format_grounding(result.grounding))
    elif args.honesty:
        print(format_honesty(result.honesty))
    elif args.factuality:
        print(format_factuality(result.factuality))
    else:
        if result.summary is not None:
            print(format_summary(result.summary))
            print()
        print(format_report(result.report))
        print()
        print(format_factuality(result.factuality))
        if result.grounding is not None:
            print()
            print(format_grounding(result.grounding))
        if result.honesty is not None:
            print()
            print(format_honesty(result.honesty))
        if result.revision is not None:
            print()
            print(result.revision.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
