"""Persists a calibration run.

Separate from ``answer/calibration.py`` so the curve itself has no opinion about
files. What is saved is every scored outcome as well as the binned curve: bin
edges are a presentation choice, and a reader who disagrees with mine should be
able to re-bin the same data rather than re-run the measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.answer.calibration import CalibrationCurve, ScoredOutcome, build_curve


def save_outcomes(
    outcomes: list[ScoredOutcome],
    curve: CalibrationCurve,
    path: Path | str,
    sampling: dict[str, Any] | None = None,
) -> Path:
    """Write a calibration run, raw outcomes included."""
    path = Path(path)
    path.write_text(
        json.dumps(
            {
                "sampling": sampling or {},
                "expected_calibration_error": curve.expected_calibration_error,
                "is_calibrated": curve.is_calibrated,
                "bins": [
                    {
                        "low": b.low,
                        "high": b.high,
                        "count": b.count,
                        "accuracy": b.accuracy,
                        "mean_confidence": b.mean_confidence,
                        "gap": b.gap,
                        "is_measurable": b.is_measurable,
                    }
                    for b in curve.bins
                ],
                "outcomes": [asdict(o) for o in outcomes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_outcomes(path: Path | str) -> tuple[list[ScoredOutcome], CalibrationCurve]:
    """Read a calibration run back and rebuild the curve from the raw outcomes.

    Rebuilt rather than read from the stored bins, so a change to the binning
    applies to old runs instead of leaving them frozen in whatever shape they
    were written with.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    outcomes = [ScoredOutcome(**o) for o in payload.get("outcomes", [])]
    return outcomes, build_curve(outcomes)
