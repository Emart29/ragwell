"""Ask a question and see the answer with its citations checked.

The command a person actually runs. Every claim prints with the chunk it came
from, whether the quote was found in that chunk, and where in the source
document to look — so a reader can check a claim without knowing anything about
the system.

Usage::

    python scripts/ask.py "What is the maximum deposit insurance coverage?"
    python scripts/ask.py "..." --strategy vector --top-k 5
    python scripts/ask.py "..." --judge          # run the entailment judge
    python scripts/ask.py "..." --show-chunks    # print the cited text
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer.confidence import score_answer as score_confidence  # noqa: E402
from app.answer.contract import GroundedAnswer  # noqa: E402
from app.answer.generate import AnswerGenerator  # noqa: E402
from app.answer.verify import CitationVerifier  # noqa: E402
from app.config import settings  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIDTH = 88


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(
        text, width=WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--provider", default=None)
    parser.add_argument(
        "--judge", action="store_true",
        help="run the entailment judge as well as the free checks",
    )
    parser.add_argument("--show-chunks", action="store_true",
                        help="print the text of every cited chunk")
    args = parser.parse_args()

    provider = args.provider or settings.DEFAULT_LLM_PROVIDER
    retrieval = Retriever().retrieve(
        args.question, strategy=args.strategy, top_k=args.top_k
    )
    if not retrieval.results:
        print("nothing retrieved. Has anything been ingested?")
        return 1

    result = AnswerGenerator(provider_name=provider).generate(
        args.question, retrieval.results
    )
    if not result.ok:
        # Non-zero, so the command composes in a shell.
        print(f"failed: {result.error}", file=sys.stderr)
        return 1

    verifier = CitationVerifier(judge_provider=provider if args.judge else None)
    verification = None
    if isinstance(result.answer, GroundedAnswer):
        verification = verifier.verify(
            result.answer, result.retrieved, run_judge=args.judge
        )
    confidence = score_confidence(
        args.question, result.answer, result.retrieved, verification
    )

    print()
    print(args.question)
    print("=" * min(len(args.question), WIDTH))
    print()

    if result.declined:
        print("  No answer: the corpus does not support one.")
        print(wrap(result.answer.missing, indent="  "))
        print()
        print(f"  searched {len(result.retrieved)} chunks via {args.strategy}")
        return 0

    by_id = {r.chunk_id: r for r in result.retrieved}
    checks = {c.claim_text: c for c in (verification.checks if verification else [])}

    for i, claim in enumerate(result.answer.claims, 1):
        check = checks.get(claim.text)
        if check is None or check.quote_found is None:
            mark = "?"
        elif check.quote_found:
            mark = "verified"
        else:
            mark = "UNVERIFIED"
        print(f"  {i}. [{mark}] {claim.text}")

        for chunk_id in claim.chunk_ids:
            chunk = by_id.get(chunk_id)
            where = "not retrieved"
            if chunk:
                where = chunk.filename
                if chunk.page_number is not None:
                    where += f", p.{chunk.page_number}"
                if chunk.heading_context:
                    where += f" — {chunk.heading_context}"
            print(f"       source: {where}  [{chunk_id[:12]}]")

        print(wrap(f'quote: "{claim.quote}"', indent="       "))
        if check and check.entailment.value != "unchecked":
            print(f"       judge: {check.entailment.value} — {check.entailment_reason}")
        if args.show_chunks:
            for chunk_id in claim.chunk_ids:
                chunk = by_id.get(chunk_id)
                if chunk:
                    print(wrap(chunk.text[:600], indent="         | "))
        print()

    print(f"  confidence {confidence.score:.2f}")
    for note in confidence.notes:
        print(f"    {note}")
    if verification:
        print(f"  citations: {verification.summary()}")
    print(
        f"  {result.provider}/{result.model}, "
        f"{result.prompt_tokens or 0:,} prompt tokens, "
        f"{result.latency_ms:.0f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
