"""Tests for the chunking-versus-citation measurement.

The property under test: citation precision and answer accuracy are reported
together. A chunk size that halves the text a reader must check while halving
accuracy has improved nothing, and either number quoted alone would hide that.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from eval.chunking import CHUNK_SIZES, ChunkingCell, citation_precision
from eval.questions import Category
from eval.score import AnswerScore


def cell(size: int, correct: int, total: int, cited_tokens: float,
         multi: int = 0, claims: int = 10) -> ChunkingCell:
    scores = [
        asdict(AnswerScore(
            question_id=f"q{i}", category=Category.DIRECT,
            declined=False, correct=i < correct,
        ))
        for i in range(total)
    ]
    return ChunkingCell(
        chunk_size=size, chunk_overlap=size // 10, chunks_indexed=size,
        mean_cited_chunk_tokens=cited_tokens, multi_chunk_claims=multi,
        total_claims=claims, scores=scores,
    )


class TestTheSweepIsUsable:
    def test_it_spans_a_real_range(self):
        """The bottom must be small enough to split a paragraph and the top
        large enough to swallow a section, or the trade is invisible."""
        assert min(CHUNK_SIZES) <= 128
        assert max(CHUNK_SIZES) >= 1024

    def test_overlap_would_scale_with_size(self):
        """A fixed overlap is a tenth of a large chunk and most of a small one,
        which would confound the comparison with a second variable."""
        for size in CHUNK_SIZES:
            assert max(size // 10, 16) < size / 2


class TestSplitEvidence:
    def test_a_claim_citing_two_chunks_counts(self):
        c = cell(128, 8, 10, cited_tokens=140, multi=3, claims=10)
        assert c.split_evidence_rate == 0.3

    def test_no_claims_does_not_divide_by_zero(self):
        c = ChunkingCell(chunk_size=512, chunk_overlap=50)
        assert c.split_evidence_rate == 0.0


class TestTheTradeIsReportedAsATrade:
    def test_precision_and_accuracy_appear_together(self):
        rendered = citation_precision([
            cell(128, 8, 10, cited_tokens=140),
            cell(1024, 8, 10, cited_tokens=1100),
        ])
        assert "correct" in rendered
        assert "cited tokens" in rendered

    def test_free_precision_is_called_free(self):
        """Same accuracy, tighter citation: the smaller chunk costs nothing."""
        rendered = citation_precision([
            cell(128, 8, 10, cited_tokens=140),
            cell(1024, 8, 10, cited_tokens=1120),
        ])
        assert "8.0x" in rendered
        assert "no measured cost" in rendered

    def test_bought_precision_is_called_bought(self):
        """A tighter citation that answers worse is a trade, not a win."""
        rendered = citation_precision([
            cell(128, 5, 10, cited_tokens=140),
            cell(1024, 9, 10, cited_tokens=1120),
        ])
        assert "bought rather than free" in rendered

    def test_a_smaller_chunk_winning_outright_says_so(self):
        rendered = citation_precision([
            cell(128, 9, 10, cited_tokens=140),
            cell(1024, 5, 10, cited_tokens=1120),
        ])
        assert "no trade to make" in rendered

    def test_one_size_cannot_be_compared(self):
        rendered = citation_precision([cell(512, 8, 10, cited_tokens=500)])
        assert "too few sizes" in rendered

    def test_a_size_that_produced_no_claims_is_excluded(self):
        """It has no citation precision to report, and averaging a zero in
        would make the smallest chunk look infinitely precise."""
        empty = ChunkingCell(chunk_size=64, chunk_overlap=16)
        rendered = citation_precision([empty, cell(512, 8, 10, cited_tokens=500)])
        assert "too few sizes" in rendered
