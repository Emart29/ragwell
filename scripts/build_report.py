"""Builds the HTML report from whichever measurement files are present.

Usage::

    python scripts/build_report.py
    python scripts/build_report.py --out docs/report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.report import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark_results.json")
    parser.add_argument("--chunking", default="chunking_results.json")
    parser.add_argument("--calibration", default="calibration_results.json")
    parser.add_argument("--out", default="report.html")
    args = parser.parse_args()

    for name in (args.benchmark, args.chunking, args.calibration):
        if not Path(name).exists():
            print(f"  {name} is absent; its section will be omitted")

    path = build(args.benchmark, args.chunking, args.calibration, args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f}KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
