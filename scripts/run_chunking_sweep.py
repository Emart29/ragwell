"""Measures what chunk size does to citation quality.

Re-ingests the corpus at each chunk size and answers the same questions, so the
only thing varying is how the documents were cut. Retrieval quality is
ragwell's existing comparator and is not re-derived here.

Usage::

    python scripts/run_chunking_sweep.py                  # every size
    python scripts/run_chunking_sweep.py --sizes 256 512
    python scripts/run_chunking_sweep.py --reports 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken  # noqa: E402

from app.answer.contract import GroundedAnswer  # noqa: E402
from app.answer.generate import AnswerGenerator  # noqa: E402
from app.answer.verify import CitationVerifier  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.storage.vector_store import VectorStore  # noqa: E402
from eval.chunking import CHUNK_SIZES, ChunkingCell, citation_precision  # noqa: E402
from eval.questions import REPORTS  # noqa: E402
from eval.run import questions_for, save  # noqa: E402
from eval.score import score_answer  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_DIR = Path(__file__).resolve().parent.parent / "eval" / "corpus_ndic"
ORDERED = [REPORTS[year] for year in sorted(REPORTS, reverse=True)]
ENCODER = tiktoken.get_encoding("cl100k_base")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=list(CHUNK_SIZES))
    parser.add_argument(
        "--reports", type=int, default=2,
        help=(
            "How many reports to index. Two by default: enough superseded "
            "material for the trap to bite, small enough to re-ingest four "
            "times without spending an afternoon."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="chunking_results.json")
    args = parser.parse_args()

    generator = AnswerGenerator()
    verifier = CitationVerifier(judge_provider=None)
    present = set(ORDERED[: args.reports])
    questions = questions_for(0, present)
    cells: list[ChunkingCell] = []
    started = time.time()

    print(f"{len(questions)} questions, {args.reports} report(s), top_k={args.top_k}")

    for chunk_size in args.sizes:
        # Overlap scales with size. A fixed overlap is a tenth of a large chunk
        # and most of a small one, which would confound the comparison with a
        # second variable.
        overlap = max(chunk_size // 10, 16)
        print(f"\nchunk size {chunk_size} (overlap {overlap})")

        VectorStore().reset()
        pipeline = Pipeline()
        cell = ChunkingCell(chunk_size=chunk_size, chunk_overlap=overlap)

        for stem in ORDERED[: args.reports]:
            path = CORPUS_DIR / f"{stem}.pdf"
            if not path.exists():
                raise SystemExit(f"missing {path.name}. Run eval/fetch_corpus.py")
            result = pipeline.process_document(
                path, chunk_size=chunk_size, chunk_overlap=overlap
            )
            if not result.success:
                raise SystemExit(f"ingestion failed: {result.error}")
            cell.chunks_indexed += result.chunk_count
        print(f"    {cell.chunks_indexed} chunks indexed")

        retriever = Retriever()
        cited_tokens: list[int] = []
        claim_spans: list[int] = []

        for question in questions:
            retrieval = retriever.retrieve(
                question.text, strategy="hybrid", top_k=args.top_k
            )
            answer = generator.generate(question.text, retrieval.results)
            if not answer.ok:
                cell.errors += 1
                continue

            verification = None
            if isinstance(answer.answer, GroundedAnswer):
                verification = verifier.verify(
                    answer.answer, answer.retrieved, run_judge=False
                )
                by_id = {r.chunk_id: r for r in answer.retrieved}
                for claim in answer.answer.claims:
                    claim_spans.append(len(claim.chunk_ids))
                    for chunk_id in claim.chunk_ids:
                        if chunk_id in by_id:
                            cited_tokens.append(
                                len(ENCODER.encode(by_id[chunk_id].text))
                            )

            cell.scores.append(asdict(score_answer(
                question, answer.answer, verification,
                retrieved_ids={r.chunk_id for r in answer.retrieved},
            )))

        cell.total_claims = len(claim_spans)
        cell.multi_chunk_claims = sum(1 for n in claim_spans if n > 1)
        cell.mean_cited_chunk_tokens = (
            sum(cited_tokens) / len(cited_tokens) if cited_tokens else 0.0
        )
        cells.append(cell)
        print(f"    {cell.summary()}")

        save(cells, args.out, sampling={
            "reports": args.reports,
            "top_k": args.top_k,
            "model": f"{generator.provider.name}/{generator.provider.model}",
            "elapsed_seconds": round(time.time() - started, 1),
        })

    print()
    print(citation_precision(cells))
    print(f"\nwrote {args.out} in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
