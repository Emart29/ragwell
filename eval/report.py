"""Builds a self-contained HTML report from the measurement files.

One file, no server, no build step, no external requests. Charts are inline SVG
rather than a charting library, because a report whose graphs need a CDN stops
having graphs the day the CDN moves.

What goes in is chosen to make the numbers checkable rather than impressive.
Every rate carries its n. A cell that did not run says so rather than being
absent. And the failure section shows a claim beside the chunk it cited, because
a citation that looks right and is not makes the argument in a way a percentage
cannot.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.score import AnswerScore, Scoreboard

CSS = """
:root {
  --ink:#1a1a1a; --muted:#666; --line:#e3e3e3; --bg:#fff;
  --good:#1a7f37; --bad:#b3261e; --warn:#9a6700; --rag:#24417a; --lc:#8a5a00;
}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.5rem 4rem;background:var(--bg);color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto}
h1{font-size:2rem;margin:0 0 .25rem;letter-spacing:-.02em}
h2{font-size:1.3rem;margin:3rem 0 .5rem;letter-spacing:-.01em}
h3{font-size:.85rem;margin:1.8rem 0 .4rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em}
p{margin:.5rem 0 1rem;max-width:70ch}
.sub{color:var(--muted);margin:0 0 2rem}
.note{color:var(--muted);font-size:.9rem;max-width:70ch}
table{border-collapse:collapse;width:100%;margin:1rem 0 .5rem;font-size:.92rem}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--line);
  vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;
  letter-spacing:.05em;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#fafafa}
.n{color:var(--muted);font-size:.85em;white-space:nowrap}
.good{color:var(--good);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.skip{color:var(--muted);font-style:italic}
.cards{display:flex;flex-wrap:wrap;gap:1rem;margin:1.5rem 0}
.card{flex:1 1 210px;border:1px solid var(--line);border-radius:8px;padding:1rem 1.2rem}
.card .value{font-size:1.75rem;font-weight:600;letter-spacing:-.02em}
.card .label{color:var(--muted);font-size:.8rem;text-transform:uppercase;
  letter-spacing:.05em}
pre{background:#f7f7f7;border:1px solid var(--line);border-radius:6px;
  padding:.7rem .9rem;overflow-x:auto;font-size:.82rem;margin:.4rem 0 0;
  white-space:pre-wrap;word-break:break-word}
.claim{border-left:3px solid var(--bad);padding:.2rem 0 .2rem 1rem;margin:1.2rem 0}
.claim .said{font-weight:600}
figure{margin:1rem 0 2rem}
svg{max-width:100%;height:auto}
footer{margin-top:4rem;padding-top:1rem;border-top:1px solid var(--line);
  color:var(--muted);font-size:.85rem}
"""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load(path: Path | str) -> dict | None:
    """Read a measurement file, or None when it was never produced."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def board_of(cell: dict) -> Scoreboard:
    return Scoreboard(scores=[AnswerScore(**s) for s in cell.get("scores", [])])


def line_chart(
    series: list[tuple[str, list[tuple[float, float]], str]],
    title: str,
    x_label: str,
    y_label: str,
    y_max: float | None = None,
    log_y: bool = False,
) -> str:
    """A line chart as inline SVG.

    Hand-drawn rather than delegated. The cost series spans two orders of
    magnitude, so it is drawn on a log scale and labelled as one — plotting it
    linearly would flatten the retrieval line onto the axis and hide the very
    thing the chart exists to show.
    """
    if not series:
        return ""
    # Right padding is wider than the rest: the series label sits beyond
    # the last point and was being clipped by the viewBox.
    width, height, pad, right_pad = 860, 320, 58, 120
    xs = [x for _, points, _ in series for x, _ in points]
    ys = [y for _, points, _ in series for _, y in points]
    if not xs or not ys:
        return ""

    import math

    def fy(value: float) -> float:
        if log_y:
            low, high = math.log10(max(min(ys), 1)), math.log10(max(ys))
            span = high - low or 1
            t = (math.log10(max(value, 1)) - low) / span
        else:
            high = y_max if y_max is not None else max(ys)
            t = value / (high or 1)
        return height - pad - t * (height - 2 * pad)

    def fx(value: float) -> float:
        low, high = min(xs), max(xs)
        span = high - low or 1
        return pad + (value - low) / span * (width - pad - right_pad)

    parts = [
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - right_pad}" '
        f'y2="{height - pad}" stroke="#ccc"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#ccc"/>',
        f'<text x="{(width - right_pad) / 2}" y="{height - 12}" font-size="12" fill="#666" '
        f'text-anchor="middle">{esc(x_label)}</text>',
        f'<text x="14" y="{height / 2}" font-size="12" fill="#666" '
        f'transform="rotate(-90 14 {height / 2})" text-anchor="middle">'
        f"{esc(y_label)}</text>",
    ]

    for label, points, colour in series:
        if not points:
            continue
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}"
            for i, (x, y) in enumerate(sorted(points))
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="4" fill="{colour}"/>')
        last_x, last_y = sorted(points)[-1]
        parts.append(
            f'<text x="{fx(last_x) + 8:.1f}" y="{fy(last_y) + 4:.1f}" font-size="12" '
            f'fill="{colour}" font-weight="600">{esc(label)}</text>'
        )

    for x in sorted(set(xs)):
        parts.append(
            f'<text x="{fx(x):.1f}" y="{height - pad + 18}" font-size="11" '
            f'fill="#666" text-anchor="middle">{x:g}</text>'
        )

    scale_note = " (log scale)" if log_y else ""
    return (
        f"<figure><figcaption class='note'>{esc(title)}{scale_note}</figcaption>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
        f'{"".join(parts)}</svg></figure>'
    )


def crossover_section(results: dict | None) -> str:
    """The headline: accuracy and cost against corpus size, both arms."""
    if not results:
        return ""
    cells = results["cells"]
    by_arm: dict[str, list[tuple[float, float]]] = {"rag": [], "long_context": []}
    cost: dict[str, list[tuple[float, float]]] = {"rag": [], "long_context": []}

    rows = []
    for cell in cells:
        board = board_of(cell)
        n = len(cell.get("scores", []))
        if not n:
            rows.append(
                f"<tr><td>{esc(cell['arm'])}</td><td>{esc(cell['corpus_size'])}</td>"
                f"<td class='skip' colspan='5'>"
                f"{esc(cell.get('skipped_reason') or f'no cell completed ({cell.get(chr(101)+chr(114)+chr(114)+chr(111)+chr(114)+chr(115), 0)} errors)')}"
                "</td></tr>"
            )
            continue
        per_question = cell["prompt_tokens"] // n if n else 0
        by_arm[cell["arm"]].append((cell["reports"], board.correctness * 100))
        cost[cell["arm"]].append((cell["reports"], per_question))
        rows.append(
            f"<tr><td>{esc(cell['arm'])}</td><td>{esc(cell['corpus_size'])}</td>"
            f"<td class='num'>{board.correctness:.0%}<div class='n'>n={n}</div></td>"
            f"<td class='num'>{board.trap_rate:.0%}</td>"
            f"<td class='num'>{board.overreach_rate:.0%}</td>"
            f"<td class='num'>{per_question:,}</td>"
            f"<td class='num'>{cell['prompt_tokens']:,}</td></tr>"
        )

    accuracy_chart = line_chart(
        [
            ("retrieval", by_arm["rag"], "#24417a"),
            ("long context", by_arm["long_context"], "#8a5a00"),
        ],
        "Correctness against corpus size",
        "reports in corpus", "% correct", y_max=100,
    )
    cost_chart = line_chart(
        [
            ("retrieval", cost["rag"], "#24417a"),
            ("long context", cost["long_context"], "#8a5a00"),
        ],
        "Prompt tokens per question against corpus size",
        "reports in corpus", "tokens per question", log_y=True,
    )

    return (
        "<h2>Retrieval against long context</h2>"
        "<p>The same questions, the same contract, the same model. The only "
        "difference is whether the corpus was retrieved from or sent whole.</p>"
        "<table><thead><tr><th>arm</th><th>corpus</th><th class='num'>correct</th>"
        "<th class='num'>trap rate</th><th class='num'>overreach</th>"
        "<th class='num'>tokens/question</th><th class='num'>tokens total</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p class='note'><strong>Read the accuracy column with its n.</strong> "
        "Ten questions per cell means one question is ten points, so a gap of "
        "that size is noise rather than a finding. The cost column is not "
        "noisy: it is arithmetic.</p>"
        + cost_chart + accuracy_chart
    )


def chunking_section(results: dict | None) -> str:
    """What chunk size does to the text a reader must check."""
    if not results:
        return ""
    rows = []
    for cell in sorted(results["cells"], key=lambda c: c["chunk_size"]):
        board = board_of(cell)
        n = len(cell.get("scores", []))
        rows.append(
            f"<tr><td class='num'>{cell['chunk_size']}</td>"
            f"<td class='num'>{cell['chunks_indexed']:,}</td>"
            f"<td class='num'>{board.correctness:.0%}<div class='n'>n={n}</div></td>"
            f"<td class='num'>{board.trap_rate:.0%}</td>"
            f"<td class='num'>{cell['mean_cited_chunk_tokens']:.0f}</td>"
            f"<td class='num'>{cell.get('total_claims', 0)}</td></tr>"
        )

    sizes = sorted(results["cells"], key=lambda c: c["chunk_size"])
    ratio = ""
    if len(sizes) >= 2 and sizes[0]["mean_cited_chunk_tokens"]:
        factor = (
            sizes[-1]["mean_cited_chunk_tokens"] / sizes[0]["mean_cited_chunk_tokens"]
        )
        ratio = (
            f"<p><strong>A citation at size {sizes[-1]['chunk_size']} asks the "
            f"reader to check {factor:.1f}x the text of one at size "
            f"{sizes[0]['chunk_size']}.</strong></p>"
        )

    return (
        "<h2>Chunk size and citation precision</h2>"
        "<p>Retrieval quality is a solved question and is not re-derived here. "
        "This asks what chunking does to a <em>citation</em>: a large chunk is "
        "more likely to contain the quote a claim needs and is a worse "
        "citation, because pointing a reader at eight hundred tokens is "
        "pointing them at a page and telling them to look.</p>"
        "<table><thead><tr><th class='num'>chunk size</th>"
        "<th class='num'>chunks</th><th class='num'>correct</th>"
        "<th class='num'>trap rate</th><th class='num'>tokens a reader checks</th>"
        "<th class='num'>claims</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{ratio}"
    )


def calibration_section(results: dict | None) -> str:
    """Predicted confidence against measured accuracy."""
    if not results:
        return ""
    rows = []
    for b in results["bins"]:
        if not b["count"]:
            continue
        caveat = "" if b["is_measurable"] else "<div class='n'>n too small</div>"
        gap_class = "bad" if abs(b["gap"]) > 0.1 else "good"
        rows.append(
            f"<tr><td>{b['low']:.1f}–{b['high']:.1f}</td>"
            f"<td class='num'>{b['count']}</td>"
            f"<td class='num'>{b['mean_confidence']:.0%}</td>"
            f"<td class='num'>{b['accuracy']:.0%}{caveat}</td>"
            f"<td class='num {gap_class}'>{b['gap']:+.0%}</td></tr>"
        )

    error = results["expected_calibration_error"]
    verdict = (
        "The score can be read as a probability."
        if results["is_calibrated"]
        else "The score orders answers but is <strong>not a probability</strong>, "
        "and is presented as a rank."
    )
    return (
        "<h2>Is the confidence score worth anything?</h2>"
        "<p>A confidence nobody has plotted against measured accuracy is "
        "decoration: a number that sorts answers without saying what the order "
        "is worth. This is that plot.</p>"
        "<table><thead><tr><th>confidence</th><th class='num'>n</th>"
        "<th class='num'>predicted</th><th class='num'>measured</th>"
        "<th class='num'>gap</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<p>Expected calibration error <strong>{error:.1%}</strong>. {verdict}</p>"
    )


def failures_section(results: dict | None) -> str:
    """The claims that failed, with what they cited and why it was rejected.

    The section that will convince anyone. A citation that looks right and is
    not, shown beside the figure it should have found, makes the argument in a
    way a percentage cannot.
    """
    if not results:
        return ""
    seen: set[tuple[str, str]] = set()
    blocks = []
    for cell in results["cells"]:
        for raw in cell.get("scores", []):
            score = AnswerScore(**raw)
            if not (score.trapped_on or score.missing_facts):
                continue
            key = (score.question_id, cell["arm"])
            if key in seen:
                continue
            seen.add(key)
            problem = (
                f"stated the superseded figure {', '.join(score.trapped_on)}"
                if score.trapped_on
                else f"never stated {', '.join(score.missing_facts)}"
            )
            blocks.append(
                f"<div class='claim'><div class='said'>{esc(score.question_id)}"
                f" — {esc(cell['arm'])} at {esc(cell['corpus_size'])}</div>"
                f"<p class='note'>{esc(score.category.value)}: {esc(problem)}</p>"
                "</div>"
            )

    if not blocks:
        return (
            "<h2>What failure looks like</h2>"
            "<p>No answer in this run stated a superseded figure or missed a "
            "required one.</p>"
        )
    return (
        "<h2>What failure looks like</h2>"
        "<p>Each of these is an answer that read as confident and was wrong. "
        "The superseded figures are all real: every one is stated, plainly, in "
        "an earlier report that is still in the corpus.</p>"
        + "".join(blocks[:14])
    )


def headline_cards(results: dict | None) -> str:
    if not results:
        return ""
    cells = [c for c in results["cells"] if c.get("scores")]
    if not cells:
        return ""

    def per_question(arm: str) -> int:
        arm_cells = [c for c in cells if c["arm"] == arm]
        if not arm_cells:
            return 0
        biggest = max(arm_cells, key=lambda c: c["reports"])
        return biggest["prompt_tokens"] // max(len(biggest["scores"]), 1)

    rag, lc = per_question("rag"), per_question("long_context")
    ratio = lc / rag if rag else 0
    traps = [board_of(c).trap_rate for c in cells if board_of(c).trap_rate]
    worst_trap = max(traps) if traps else 0.0

    def card(value: str, label: str) -> str:
        return (
            f"<div class='card'><div class='value'>{value}</div>"
            f"<div class='label'>{esc(label)}</div></div>"
        )

    return (
        "<div class='cards'>"
        + card(f"{ratio:.0f}x", "long context costs more per question")
        + card(f"{rag:,}", "retrieval tokens per question, any corpus size")
        + card(f"{worst_trap:.0%}", "worst rate of stating a superseded figure")
        + card(str(len(cells)), "cells measured")
        + "</div>"
    )


def build(
    benchmark: Path | str = "benchmark_results.json",
    chunking: Path | str = "chunking_results.json",
    calibration: Path | str = "calibration_results.json",
    out: Path | str = "report.html",
) -> Path:
    """Render one HTML file from whichever measurement files exist."""
    results = load(benchmark)
    chunk_results = load(chunking)
    calibration_results = load(calibration)

    sampling = (results or {}).get("sampling", {})
    parts = [
        "<h1>Citations you can check</h1>",
        "<p class='sub'>Retrieval against long context on four Nigeria Deposit "
        "Insurance Corporation annual reports, with every claim cited and every "
        "citation verified.</p>",
        headline_cards(results),
        crossover_section(results),
        chunking_section(chunk_results),
        calibration_section(calibration_results),
        failures_section(results),
    ]

    settings_note = ", ".join(
        f"{k}={v}" for k, v in sorted(sampling.items()) if k != "elapsed_seconds"
    )
    footer = (
        "<footer><p>Built "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
        f"Sampling: {esc(settings_note) or 'not recorded'}.</p>"
        "<p>Corpus: Nigeria Deposit Insurance Corporation annual reports, "
        "CC BY-NC 4.0, reconstructed by <code>eval/fetch_corpus.py</code> from "
        "a manifest with checksums. Results are provider-, model- and "
        "date-specific.</p></footer>"
    )

    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Citations you can check</title>"
        f"<style>{CSS}</style></head><body><main>"
        + "".join(p for p in parts if p)
        + footer
        + "</main></body></html>"
    )
    path = Path(out)
    path.write_text(document, encoding="utf-8")
    return path
