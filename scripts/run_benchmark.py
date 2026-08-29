"""Runs the benchmark matrix and writes the results the report is built from.

Ingests the corpus at each size, runs both arms, checkpoints every cell.

Usage::

    python scripts/run_benchmark.py                    # the full sweep
    python scripts/run_benchmark.py --sizes 1_report   # one size
    python scripts/run_benchmark.py --arms rag         # one arm
    python scripts/run_benchmark.py --out results.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer.generate import AnswerGenerator  # noqa: E402
from app.answer.verify import CitationVerifier  # noqa: E402
from app.config import settings  # noqa: E402
from app.longcontext.stuff import LongContextAnswerer, load_corpus_chunks  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.storage.vector_store import VectorStore  # noqa: E402
from eval.questions import REPORTS  # noqa: E402
from eval.run import (  # noqa: E402
    CORPUS_SIZES,
    BenchmarkRunner,
    crossover,
    load,
    questions_for,
    save,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_DIR = Path(__file__).resolve().parent.parent / "eval" / "corpus_ndic"

#: Reports newest first. A corpus of n reports is the n most recent, so every
#: size contains the current position and the sweep varies only how much
#: superseded material sits alongside it.
ORDERED = [REPORTS[year] for year in sorted(REPORTS, reverse=True)]


def ingest(reports: int) -> int:
    """Rebuild the index with the n most recent reports.

    The store is reset first. Leaving an earlier size in place would make every
    later cell answer from a corpus larger than its label says.
    """
    VectorStore().reset()
    pipeline = Pipeline()
    chunks = 0
    for stem in ORDERED[:reports]:
        path = CORPUS_DIR / f"{stem}.pdf"
        if not path.exists():
            raise SystemExit(
                f"missing {path.name}. Run: python eval/fetch_corpus.py"
            )
        print(f"    ingesting {path.name} ... ", end="", flush=True)
        result = pipeline.process_document(path)
        if not result.success:
            raise SystemExit(f"ingestion failed: {result.error}")
        chunks += result.chunk_count
        print(f"{result.chunk_count} chunks")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=["rag", "long_context"])
    parser.add_argument(
        "--rag-provider", default="gemini",
        help=(
            "Provider for the retrieval arm. Defaults to gemini so both "
            "arms run on one model and the comparison is retrieval against "
            "stuffing rather than one model against another."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--out", default="benchmark_results.json")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip citation verification")
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "keep cells already in the output file and run only what is "
            "missing. A sweep is half an hour of real requests and a "
            "network blip should not cost the cells that already finished."
        ),
    )
    args = parser.parse_args()

    sizes = [s for s in CORPUS_SIZES if not args.sizes or s[0] in args.sizes]
    if not sizes:
        raise SystemExit(f"no such size. Known: {[s[0] for s in CORPUS_SIZES]}")

    runner = BenchmarkRunner(
        retriever=Retriever(),
        generator=AnswerGenerator(provider_name=args.rag_provider),
        verifier=None if args.no_verify else CitationVerifier(judge_provider=None),
        long_context=LongContextAnswerer(),
        top_k=args.top_k,
        strategy=args.strategy,
    )

    started = time.time()
    cells = []
    done: set[tuple[str, str]] = set()
    if args.resume and Path(args.out).exists():
        previous = load(args.out)
        cells = [c for c in previous['cells'] if c.ran or c.skipped_reason]
        done = {(c.arm, c.corpus_size) for c in cells}
        print(f"resuming: {len(done)} cell(s) already complete")

    for name, reports, label in sizes:
        print(f"\n{name} ({label}, {reports} report(s))")
        chunk_count = ingest(reports)
        chunks = load_corpus_chunks()
        present = set(ORDERED[:reports])
        questions = questions_for(
            runner.long_context.measure_fit(chunks).tokens, present
        )
        dropped = len(questions_for(0)) - len(questions_for(0, present))
        print(
            f"    {chunk_count} chunks indexed, {len(questions)} questions"
            + (f" ({dropped} need reports not in this corpus)" if dropped else "")
        )

        for arm in args.arms:
            if (arm, name) in done:
                print(f"    {arm}/{name}: already complete, skipping")
                continue
            if arm == "rag":
                cell = runner.run_rag(questions, name, reports)
            else:
                cell = runner.run_long_context(questions, name, reports, chunks)
            cells.append(cell)
            print(f"    {cell.summary()}")
            # Checkpointed per cell: a matrix is hours of real requests.
            save(cells, args.out, sampling={
                "top_k": args.top_k,
                "strategy": args.strategy,
                "rag_model": f"{runner.generator.provider.name}/{runner.generator.provider.model}",
                "long_context_model": runner.long_context.provider.model,
                "elapsed_seconds": round(time.time() - started, 1),
            })

    print("\ncrossover")
    print(crossover(cells))
    print(f"\nwrote {args.out} in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
