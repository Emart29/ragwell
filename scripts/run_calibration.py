"""Plots confidence against measured accuracy, and reads a threshold off it.

The confidence score built in ``app/answer/confidence.py`` has been recorded
and never acted on, because a threshold guessed before this measurement would
be presented afterwards as though it had been derived from it. This is the
measurement.

It answers two things:

* **Is the score calibrated?** If answers scored at 0.8 are right 40% of the
  time, the score is not a probability and must be presented as a rank.
* **Where should the fallback threshold sit?** Read off the curve, with the
  cost stated: how many answerable questions get declined to avoid how many
  wrong answers.

Usage::

    python scripts/run_calibration.py                 # score every question
    python scripts/run_calibration.py --reports 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer.confidence import score_answer as score_confidence  # noqa: E402
from app.answer.contract import GroundedAnswer  # noqa: E402
from app.answer.generate import AnswerGenerator  # noqa: E402
from app.answer.verify import CitationVerifier  # noqa: E402
from app.calibration_io import save_outcomes  # noqa: E402
from app.answer.calibration import ScoredOutcome, build_curve  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.storage.vector_store import VectorStore  # noqa: E402
from eval.questions import QUESTIONS, REPORTS  # noqa: E402
from eval.run import questions_for  # noqa: E402
from eval.score import score_answer  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS_DIR = Path(__file__).resolve().parent.parent / "eval" / "corpus_ndic"
ORDERED = [REPORTS[year] for year in sorted(REPORTS, reverse=True)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=int, default=2)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--repeats", type=int, default=3,
        help=(
            "Passes over the question set. A calibration curve needs points in "
            "each bin, and sixteen questions once over gives bins of two."
        ),
    )
    parser.add_argument("--target", type=float, default=0.9,
                        help="accuracy the threshold should reach")
    parser.add_argument("--out", default="calibration_results.json")
    args = parser.parse_args()

    VectorStore().reset()
    pipeline = Pipeline()
    for stem in ORDERED[: args.reports]:
        path = CORPUS_DIR / f"{stem}.pdf"
        if not path.exists():
            raise SystemExit(f"missing {path.name}. Run eval/fetch_corpus.py")
        result = pipeline.process_document(path)
        if not result.success:
            raise SystemExit(f"ingestion failed: {result.error}")

    generator = AnswerGenerator(provider_name=args.provider)
    verifier = CitationVerifier(judge_provider=None)
    retriever = Retriever()

    # The full set, not the core subset: a curve wants points, and every
    # question that can be answered from this corpus is one.
    present = set(ORDERED[: args.reports])
    questions = [q for q in QUESTIONS if all(f in present for f in q.source_files)]

    print(
        f"{len(questions)} questions x {args.repeats} passes, "
        f"{args.reports} report(s), {args.provider}"
    )

    outcomes: list[ScoredOutcome] = []
    started = time.time()

    for pass_number in range(args.repeats):
        for question in questions:
            retrieval = retriever.retrieve(
                question.text, strategy="hybrid", top_k=args.top_k
            )
            result = generator.generate(question.text, retrieval.results)
            if not result.ok:
                continue

            verification = None
            if isinstance(result.answer, GroundedAnswer):
                verification = verifier.verify(
                    result.answer, result.retrieved, run_judge=False
                )

            confidence = score_confidence(
                question.text, result.answer, result.retrieved, verification
            )
            scored = score_answer(
                question, result.answer, verification,
                retrieved_ids={r.chunk_id for r in result.retrieved},
            )

            outcomes.append(ScoredOutcome(
                confidence=confidence.score,
                # An ABSENT question has no content to be right about, so a
                # correct decline is not folded in here — build_curve drops
                # declines, and counting them as correct would let a system
                # that declines everything report perfect calibration.
                correct=bool(scored.correct),
                question_id=question.id,
                declined=scored.declined,
            ))
        print(f"  pass {pass_number + 1}: {len(outcomes)} outcomes so far")

    curve = build_curve(outcomes)
    print()
    print(curve.report())

    threshold, cost = curve.threshold_for(args.target)
    print()
    print(f"threshold for {args.target:.0%} accuracy: {threshold:.2f}")
    print(f"  {cost}")

    save_outcomes(outcomes, curve, args.out, {
        "reports": args.reports,
        "repeats": args.repeats,
        "top_k": args.top_k,
        "target_accuracy": args.target,
        "model": f"{generator.provider.name}/{generator.provider.model}",
        "elapsed_seconds": round(time.time() - started, 1),
    })
    print(f"\nwrote {args.out} in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
