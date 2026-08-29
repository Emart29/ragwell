"""The benchmark matrix: both arms, every corpus size.

The headline is the crossover. At what corpus size does retrieval start winning
on accuracy, and does the cost curve cross at the same place? They may not, and
if they do not that is the more useful result — *retrieval is more accurate and
more expensive above X* is a sentence somebody can act on.

Practical constraints, several learned the hard way in the previous project of
this series:

* **Every cell is checkpointed as it finishes.** A matrix is hours of real
  requests and losing the lot to a dropped connection in the last cell means
  paying for it twice.
* **A cell that errored is not a cell that was skipped**, and neither is a cell
  that ran and scored badly. All three are recorded distinctly, because a
  missing row is indistinguishable from one nobody thought to run.
* **The question set shrinks at the largest sizes.** The long-context arm sends
  the whole corpus per question, so its cost is the product of corpus size and
  question count. The core subset keeps every category rather than keeping the
  easy questions, so the top of the sweep measures the same thing as the bottom.
* **Rate limits are per model and per day.** The runner reports what it spent so
  a stopped run can be resumed rather than restarted.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.answer.confidence import score_answer as score_confidence
from app.answer.contract import GroundedAnswer
from app.answer.generate import AnswerGenerator
from app.answer.verify import CitationVerifier
from app.logging_config import get_logger
from app.longcontext.stuff import LongContextAnswerer, load_corpus_chunks
from eval.questions import CORE_QUESTIONS, QUESTIONS, Question
from eval.score import AnswerScore, Scoreboard, score_answer

logger = get_logger(__name__)

#: Corpus sizes, as the number of reports included, newest first. Named so the
#: results table reads without a lookup.
CORPUS_SIZES = (
    ("1_report", 1, "~49k tokens"),
    ("2_reports", 2, "~94k tokens"),
    ("3_reports", 3, "~151k tokens"),
    ("4_reports", 4, "~252k tokens"),
)

#: Above this many tokens the long-context arm runs the core subset only.
#: Chosen so the two cheap sizes carry the full set and the two expensive ones
#: do not, which is where the cost becomes the binding constraint.
FULL_SET_TOKEN_CEILING = 100_000


@dataclass
class CellResult:
    """One arm at one corpus size."""

    arm: str
    corpus_size: str
    reports: int
    question_count: int
    #: Empty unless the cell could not run at all.
    skipped_reason: str = ""
    errors: int = 0
    corpus_tokens: int = 0
    prompt_tokens: int = 0
    latency_ms: float = 0.0
    scores: list[dict] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return not self.skipped_reason and bool(self.scores)

    def board(self) -> Scoreboard:
        return Scoreboard(scores=[AnswerScore(**s) for s in self.scores])

    def summary(self) -> str:
        if self.skipped_reason:
            return f"{self.arm}/{self.corpus_size}: skipped — {self.skipped_reason}"
        board = self.board()
        return (
            f"{self.arm}/{self.corpus_size}: correct {board.correctness:.0%}, "
            f"overreach {board.overreach_rate:.0%}, trap {board.trap_rate:.0%}, "
            f"{self.prompt_tokens:,} prompt tokens, "
            f"{self.latency_ms / max(len(self.scores), 1):.0f}ms/q"
        )


def answerable_at(question: Question, present_reports: set[str]) -> bool:
    """Whether a question can be answered from the reports actually present.

    A question sourced from the 2021 and 2022 reports is unanswerable in a
    corpus holding only 2024, and declining it is then the *correct* response.
    Scoring it as a miss would penalise the small corpus sizes for refusing to
    invent an answer, which is the behaviour the rest of this benchmark is
    trying to reward.

    ABSENT questions carry no sources and are answerable — that is, declinable —
    at every size.
    """
    return all(stem in present_reports for stem in question.source_files)


def questions_for(
    corpus_tokens: int, present_reports: set[str] | None = None
) -> list[Question]:
    """The question set to use at a given corpus size.

    The core subset is used at every size. An earlier version switched sets
    partway up the sweep to save cost, which made the correctness column
    incomparable along its own axis: 100% of ten questions sat directly above
    91% of fifteen, and the difference was the question mix rather than the
    corpus. Holding the set constant costs a little coverage at the cheap end
    and buys a column that can be read.

    Questions whose sources are not in this corpus are still dropped, because
    declining them is correct and scoring that as a miss would penalise the
    smaller sizes for being right.
    """
    pool = CORE_QUESTIONS
    if present_reports is None:
        return pool
    return [q for q in pool if answerable_at(q, present_reports)]


class BenchmarkRunner:
    """Runs both arms across the corpus sizes."""

    def __init__(
        self,
        retriever,
        generator: AnswerGenerator | None = None,
        verifier: CitationVerifier | None = None,
        long_context: LongContextAnswerer | None = None,
        top_k: int = 8,
        strategy: str = "hybrid",
    ) -> None:
        self.retriever = retriever
        self.generator = generator or AnswerGenerator()
        self.verifier = verifier
        self.long_context = long_context or LongContextAnswerer()
        self.top_k = top_k
        self.strategy = strategy

    def run_rag(
        self, questions: list[Question], size_name: str, reports: int
    ) -> CellResult:
        """The retrieval arm: search, then answer from what came back."""
        cell = CellResult(
            arm="rag",
            corpus_size=size_name,
            reports=reports,
            question_count=len(questions),
        )

        for question in questions:
            started = time.perf_counter()
            retrieval = self.retriever.retrieve(
                question.text, strategy=self.strategy, top_k=self.top_k
            )
            result = self.generator.generate(question.text, retrieval.results)
            cell.latency_ms += (time.perf_counter() - started) * 1000

            if not result.ok:
                cell.errors += 1
                logger.warning("%s: %s", question.id, result.error)
                continue

            verification = None
            if self.verifier and isinstance(result.answer, GroundedAnswer):
                verification = self.verifier.verify(
                    result.answer, result.retrieved, run_judge=False
                )

            cell.prompt_tokens += result.prompt_tokens or 0
            cell.scores.append(asdict(score_answer(
                question,
                result.answer,
                verification,
                retrieved_ids={r.chunk_id for r in result.retrieved},
            )))

        return cell

    def run_long_context(
        self, questions: list[Question], size_name: str, reports: int, chunks: list
    ) -> CellResult:
        """The long-context arm: no retrieval, the whole corpus in the prompt."""
        cell = CellResult(
            arm="long_context",
            corpus_size=size_name,
            reports=reports,
            question_count=len(questions),
        )

        fit = self.long_context.measure_fit(chunks)
        cell.corpus_tokens = fit.tokens
        if not fit.fits:
            # Recorded rather than truncated. Truncating would answer a
            # different question from a different corpus and report it as this.
            cell.skipped_reason = f"corpus exceeds the context window: {fit.describe()}"
            return cell

        all_ids = {c.chunk_id for c in chunks}
        for question in questions:
            started = time.perf_counter()
            result, _ = self.long_context.answer(question.text, chunks)
            cell.latency_ms += (time.perf_counter() - started) * 1000

            if not result.ok:
                cell.errors += 1
                logger.warning("%s: %s", question.id, result.error)
                continue

            verification = None
            if self.verifier and isinstance(result.answer, GroundedAnswer):
                verification = self.verifier.verify(
                    result.answer, result.retrieved, run_judge=False
                )

            cell.prompt_tokens += result.prompt_tokens or 0
            cell.scores.append(asdict(score_answer(
                question, result.answer, verification, retrieved_ids=all_ids,
            )))

        return cell


def save(cells: list[CellResult], path: Path | str, sampling: dict | None = None) -> Path:
    """Write the matrix to disk, with the settings that produced it."""
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "sampling": sampling or {},
                "cells": [asdict(c) for c in cells],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def load(path: Path | str) -> dict:
    """Read a saved matrix back."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["cells"] = [CellResult(**c) for c in payload.get("cells", [])]
    return payload


def crossover(cells: list[CellResult]) -> str:
    """Where the two arms cross, on accuracy and on cost.

    Reported as a sentence rather than a number because the two curves may not
    cross at the same place, and "retrieval is more accurate and more expensive
    above X" is the useful form of that result.
    """
    by_size: dict[str, dict[str, CellResult]] = {}
    for cell in cells:
        by_size.setdefault(cell.corpus_size, {})[cell.arm] = cell

    lines = []
    accuracy_crossed = cost_crossed = None
    for name, _, label in CORPUS_SIZES:
        pair = by_size.get(name, {})
        rag, long_context = pair.get("rag"), pair.get("long_context")
        if not (rag and rag.ran):
            continue
        rag_board = rag.board()
        if not (long_context and long_context.ran):
            lines.append(
                f"  {name:<11} rag {rag_board.correctness:>4.0%}  |  long context "
                f"unavailable ({long_context.skipped_reason if long_context else 'not run'})"
            )
            if accuracy_crossed is None:
                accuracy_crossed = name
            continue

        lc_board = long_context.board()
        lines.append(
            f"  {name:<11} rag {rag_board.correctness:>4.0%} "
            f"({rag.prompt_tokens:>7,} tok)  |  long context "
            f"{lc_board.correctness:>4.0%} ({long_context.prompt_tokens:>9,} tok)"
        )
        if accuracy_crossed is None and rag_board.correctness > lc_board.correctness:
            accuracy_crossed = name
        if cost_crossed is None and rag.prompt_tokens < long_context.prompt_tokens:
            cost_crossed = name

    verdict = []
    if accuracy_crossed:
        verdict.append(f"retrieval becomes more accurate at {accuracy_crossed}")
    else:
        verdict.append("long context is at least as accurate at every size measured")
    if cost_crossed:
        verdict.append(f"retrieval becomes cheaper at {cost_crossed}")
    else:
        verdict.append("retrieval is not cheaper at any size measured")

    return "\n".join(lines + ["", "  " + "; ".join(verdict) + "."])
