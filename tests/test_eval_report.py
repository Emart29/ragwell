"""Tests for the HTML report.

The report is the artefact a reader judges the project by, so these check the
properties that make it honest rather than the ones that make it pretty: a cell
that did not run says so, every rate carries its n, and nothing depends on a
network.
"""

from __future__ import annotations

import json

import pytest

from eval.report import build, line_chart, load


@pytest.fixture
def measurements(tmp_path):
    """A small benchmark, chunking and calibration set on disk."""
    benchmark = {
        "sampling": {"top_k": 8, "strategy": "hybrid"},
        "cells": [
            {
                "arm": "rag", "corpus_size": "1_report", "reports": 1,
                "question_count": 2, "skipped_reason": "", "errors": 0,
                "corpus_tokens": 0, "prompt_tokens": 5000, "latency_ms": 100.0,
                "scores": [
                    {"question_id": "a", "category": "direct", "declined": False,
                     "correct": True, "missing_facts": [], "trapped_on": [],
                     "citation_accuracy": 1.0, "hallucinated_claims": 0,
                     "total_claims": 1, "judged": False},
                    {"question_id": "b", "category": "misleading", "declined": False,
                     "correct": False, "missing_facts": [], "trapped_on": ["N500,000"],
                     "citation_accuracy": 1.0, "hallucinated_claims": 0,
                     "total_claims": 1, "judged": False},
                ],
            },
            {
                "arm": "long_context", "corpus_size": "1_report", "reports": 1,
                "question_count": 2, "skipped_reason": "corpus exceeds the window",
                "errors": 0, "corpus_tokens": 9_999_999, "prompt_tokens": 0,
                "latency_ms": 0.0, "scores": [],
            },
        ],
    }
    chunking = {
        "sampling": {},
        "cells": [
            {"chunk_size": 128, "chunk_overlap": 16, "chunks_indexed": 900,
             "mean_cited_chunk_tokens": 100.0, "multi_chunk_claims": 0,
             "total_claims": 5, "scores": [], "errors": 0},
            {"chunk_size": 1024, "chunk_overlap": 102, "chunks_indexed": 100,
             "mean_cited_chunk_tokens": 800.0, "multi_chunk_claims": 0,
             "total_claims": 5, "scores": [], "errors": 0},
        ],
    }
    calibration = {
        "sampling": {}, "expected_calibration_error": 0.151,
        "is_calibrated": False,
        "bins": [
            {"low": 0.8, "high": 0.9, "count": 10, "accuracy": 0.8,
             "mean_confidence": 0.85, "gap": 0.05, "is_measurable": True},
            {"low": 0.4, "high": 0.6, "count": 3, "accuracy": 1.0,
             "mean_confidence": 0.52, "gap": -0.48, "is_measurable": False},
        ],
        "outcomes": [],
    }
    for name, payload in (
        ("benchmark_results.json", benchmark),
        ("chunking_results.json", chunking),
        ("calibration_results.json", calibration),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


class TestItIsSelfContained:
    def test_nothing_is_fetched_from_a_network(self, measurements, tmp_path):
        """A report whose graphs need a CDN stops having graphs the day the
        CDN moves."""
        out = build(
            measurements / "benchmark_results.json",
            measurements / "chunking_results.json",
            measurements / "calibration_results.json",
            tmp_path / "report.html",
        )
        html = out.read_text(encoding="utf-8")
        for marker in ("http://", "https://", "<script"):
            assert marker not in html

    def test_the_charts_are_inline_svg(self, measurements, tmp_path):
        out = build(
            measurements / "benchmark_results.json",
            measurements / "chunking_results.json",
            measurements / "calibration_results.json",
            tmp_path / "report.html",
        )
        assert "<svg" in out.read_text(encoding="utf-8")


class TestItIsHonest:
    def _html(self, measurements, tmp_path) -> str:
        return build(
            measurements / "benchmark_results.json",
            measurements / "chunking_results.json",
            measurements / "calibration_results.json",
            tmp_path / "report.html",
        ).read_text(encoding="utf-8")

    def test_a_cell_that_did_not_run_says_why(self, measurements, tmp_path):
        """A missing row is indistinguishable from one nobody thought to run."""
        assert "corpus exceeds the window" in self._html(measurements, tmp_path)

    def test_every_rate_carries_its_n(self, measurements, tmp_path):
        assert "n=2" in self._html(measurements, tmp_path)

    def test_the_accuracy_caveat_is_stated(self, measurements, tmp_path):
        """Ten questions per cell means one question is ten points."""
        assert "noise rather than a finding" in self._html(measurements, tmp_path)

    def test_a_thin_calibration_bin_is_flagged(self, measurements, tmp_path):
        assert "n too small" in self._html(measurements, tmp_path)

    def test_an_uncalibrated_score_is_not_called_a_probability(self, measurements, tmp_path):
        assert "not a probability" in self._html(measurements, tmp_path)

    def test_the_failure_section_names_the_superseded_figure(self, measurements, tmp_path):
        assert "N500,000" in self._html(measurements, tmp_path)

    def test_the_corpus_and_licence_are_credited(self, measurements, tmp_path):
        assert "CC BY-NC" in self._html(measurements, tmp_path)


class TestMissingInputs:
    def test_an_absent_file_omits_its_section_rather_than_failing(self, tmp_path):
        out = build(
            tmp_path / "nope.json", tmp_path / "nope2.json",
            tmp_path / "nope3.json", tmp_path / "report.html",
        )
        assert out.exists()
        assert "Citations you can check" in out.read_text(encoding="utf-8")

    def test_load_returns_none_for_a_missing_file(self, tmp_path):
        assert load(tmp_path / "absent.json") is None


class TestCharts:
    def test_an_empty_series_renders_nothing(self):
        assert line_chart([], "t", "x", "y") == ""
        assert line_chart([("a", [], "#000")], "t", "x", "y") == ""

    def test_a_log_scale_says_so(self):
        chart = line_chart(
            [("a", [(1, 10), (2, 1000)], "#000")], "cost", "x", "y", log_y=True
        )
        assert "log scale" in chart

    def test_the_label_stays_inside_the_viewbox(self):
        """It sits beyond the last point, and was being clipped."""
        chart = line_chart([("retrieval", [(1, 5), (4, 9)], "#000")], "t", "x", "y")
        import re

        width = int(re.search(r'viewBox="0 0 (\d+)', chart).group(1))
        xs = [float(m) for m in re.findall(r'<text x="([\d.]+)"', chart)]
        assert max(xs) < width
