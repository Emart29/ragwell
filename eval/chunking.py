"""What chunk size does to citation quality, as opposed to retrieval quality.

Ragwell already compares chunking strategies for retrieval — recall, overlap,
size distribution. Nobody has asked what chunking does to a *citation*, and
there is a real tension worth measuring:

* A **large** chunk is more likely to contain the quote a claim needs, and is a
  worse citation. Pointing a reader at two thousand tokens is pointing them at a
  page and telling them to look.
* A **small** chunk is a precise citation and more likely to split the evidence
  for one claim across two chunks, so the claim either cites both or cites the
  half that fits.

The measurement is deliberately narrow. Retrieval quality is ragwell's existing
comparator and is not re-derived here; what this adds is citation precision —
how much text a reader must check to verify one claim — measured beside the
answer accuracy at the same size, so a size that makes citations tighter while
making answers worse shows up as the trade it is.

If chunk size turns out not to matter here, that is the result. A negative
finding on a question people assume the answer to is worth the section.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.answer.verify import VerificationReport
from app.logging_config import get_logger
from eval.score import AnswerScore, Scoreboard

logger = get_logger(__name__)

#: Chunk sizes swept, in tokens. The bottom is small enough to split a
#: paragraph, the top large enough to swallow a whole section — both ends are
#: needed for the trade to be visible.
CHUNK_SIZES = (128, 256, 512, 1024)


@dataclass
class ChunkingCell:
    """One chunk size, measured on both accuracy and citation precision."""

    chunk_size: int
    chunk_overlap: int
    chunks_indexed: int = 0
    #: Mean tokens in the chunks that answers actually cited. The number a
    #: reader pays: how much text they must read to check one claim.
    mean_cited_chunk_tokens: float = 0.0
    #: Claims citing more than one chunk. Rises as chunks shrink and the
    #: evidence for a claim stops fitting in one.
    multi_chunk_claims: int = 0
    total_claims: int = 0
    scores: list[dict] = field(default_factory=list)
    errors: int = 0

    @property
    def split_evidence_rate(self) -> float:
        """Share of claims whose evidence spanned more than one chunk."""
        if not self.total_claims:
            return 0.0
        return self.multi_chunk_claims / self.total_claims

    def board(self) -> Scoreboard:
        return Scoreboard(scores=[AnswerScore(**s) for s in self.scores])

    def summary(self) -> str:
        board = self.board()
        return (
            f"size {self.chunk_size:>4}: {self.chunks_indexed:>4} chunks, "
            f"correct {board.correctness:>4.0%}, trap {board.trap_rate:>4.0%}, "
            f"citation {self.mean_cited_chunk_tokens:>6.0f} tokens, "
            f"split evidence {self.split_evidence_rate:>4.0%}"
        )


def citation_precision(cells: list[ChunkingCell]) -> str:
    """The trade, stated as a table.

    Citation precision and answer accuracy are reported side by side on
    purpose. A chunk size that halves the text a reader must check while
    halving accuracy has not improved anything, and either number alone would
    hide that.
    """
    lines = [
        f"{'chunk size':<12}{'chunks':>8}{'correct':>10}{'trap':>7}"
        f"{'cited tokens':>14}{'split evidence':>16}",
    ]
    for cell in sorted(cells, key=lambda c: c.chunk_size):
        board = cell.board()
        lines.append(
            f"{cell.chunk_size:<12}{cell.chunks_indexed:>8}"
            f"{board.correctness:>10.0%}{board.trap_rate:>7.0%}"
            f"{cell.mean_cited_chunk_tokens:>14.0f}{cell.split_evidence_rate:>16.0%}"
        )

    measured = [c for c in cells if c.total_claims]
    if len(measured) < 2:
        lines.append("")
        lines.append("  too few sizes produced claims to compare.")
        return "\n".join(lines)

    smallest = min(measured, key=lambda c: c.chunk_size)
    largest = max(measured, key=lambda c: c.chunk_size)
    ratio = (
        largest.mean_cited_chunk_tokens / smallest.mean_cited_chunk_tokens
        if smallest.mean_cited_chunk_tokens
        else 0.0
    )
    accuracy_gap = largest.board().correctness - smallest.board().correctness

    lines.append("")
    lines.append(
        f"  a citation at size {largest.chunk_size} asks the reader to check "
        f"{ratio:.1f}x the text of one at size {smallest.chunk_size}"
    )
    if abs(accuracy_gap) < 0.11:
        lines.append(
            "  and answer accuracy is within one question across the sweep, so "
            "the smaller chunk buys precision at no measured cost."
        )
    elif accuracy_gap > 0:
        lines.append(
            f"  but the larger chunk answers {accuracy_gap:.0%} more correctly, "
            "so the precision is bought rather than free."
        )
    else:
        lines.append(
            f"  and the smaller chunk also answers {-accuracy_gap:.0%} more "
            "correctly, so there is no trade to make here."
        )
    return "\n".join(lines)


def measure_cell(
    cell: ChunkingCell,
    verification: VerificationReport | None,
    cited_chunk_tokens: list[int],
    claim_chunk_counts: list[int],
) -> None:
    """Fold one question's citation measurements into a cell."""
    if cited_chunk_tokens:
        seen = cell.mean_cited_chunk_tokens * cell.total_claims
        cell.mean_cited_chunk_tokens = (seen + sum(cited_chunk_tokens)) / (
            cell.total_claims + len(cited_chunk_tokens)
        )
    cell.total_claims += len(claim_chunk_counts)
    cell.multi_chunk_claims += sum(1 for n in claim_chunk_counts if n > 1)
