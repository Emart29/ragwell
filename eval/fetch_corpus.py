"""Downloads the NDIC corpus the benchmark is labelled against.

The PDFs are roughly 450MB and are not committed. What is committed is this
manifest, so anyone can reconstruct the exact corpus the numbers were measured
on. Each entry carries a SHA-256 of the file as downloaded on the date in
``FETCHED``: a regulator that quietly replaces a PDF would otherwise change the
corpus under a benchmark that claims to be reproducible, and the checksum turns
that from an invisible problem into a loud one.

Source: Nigeria Deposit Insurance Corporation, https://ndic.gov.ng/
Licence: Creative Commons Attribution-NonCommercial 4.0 International.
Free to read, download, copy, distribute and print with attribution, for
non-commercial use. This benchmark is non-commercial and attributes the source.

Usage::

    python eval/fetch_corpus.py            # download anything missing
    python eval/fetch_corpus.py --verify   # check what is already there
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_ndic"

#: When these checksums were taken. Quoted in the results, because the corpus is
#: only the corpus as of a date.
FETCHED = "2026-08-28"


@dataclass(frozen=True)
class Document:
    """One report, and how to get it."""

    year: int
    filename: str
    url: str
    sha256: str = ""
    note: str = ""


DOCUMENTS = (
    Document(
        year=2024,
        filename="2024-Annual-Report.pdf",
        sha256="da068fb16775a7b2603a720c008295fa9765c65dc6c5f4e92c83cbe32334508e",
        url="https://ndic.gov.ng/wp-content/uploads/2025/10/2024-Annual-Report.pdf",
        note=(
            "The current position. States the coverage limits after the 2024 "
            "review that raised DMB cover from N500,000 to N5,000,000."
        ),
    ),
    Document(
        year=2022,
        filename="2022-Annual-Report.pdf",
        sha256="1ba108b2e6073d6ad798e4b8f45bbf38147b94e0ad8fb9c2fd94356e3b21f14f",
        url="https://ndic.gov.ng/wp-content/uploads/2024/08/2022-Annual-Report.pdf",
        note="States the pre-review limits, and 35 DMBs in operation.",
    ),
    Document(
        year=2021,
        filename="2021-Annual-Report.pdf",
        sha256="5a1f19a151655be48e532de39744a1f60497fc0974c94d6c0033d858e7c83b92",
        url="https://ndic.gov.ng/wp-content/uploads/2023/01/2021-Annual-Report.pdf",
        note="States 32 DMBs, which contradicts the later reports.",
    ),
    Document(
        year=2020,
        filename="NDIC-2020-Annual-Report.pdf",
        sha256="1643c423d8026f1fa9c2c72388ff39f1169e15b6e2a7976c60d6c2a25baccf97",
        url="https://ndic.gov.ng/wp-content/uploads/2021/11/NDIC-2020-Annual-Report.pdf",
        note="The oldest in the set; furthest from the current position.",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(document: Document, destination: Path) -> Path:
    """Fetch one report, streaming it rather than holding it in memory."""
    target = destination / document.filename
    print(f"  {document.filename} ... ", end="", flush=True)
    response = requests.get(document.url, stream=True, timeout=300)
    response.raise_for_status()
    with target.open("wb") as handle:
        for block in response.iter_content(chunk_size=1 << 20):
            handle.write(block)
    print(f"{target.stat().st_size / 1e6:.0f}MB")
    return target


def verify(destination: Path) -> list[str]:
    """Check what is present against the recorded checksums.

    Returns:
        Problems found. A checksum mismatch means the published document has
        changed since these labels were written, and the labels may no longer
        describe it.
    """
    problems = []
    for document in DOCUMENTS:
        path = destination / document.filename
        if not path.exists():
            problems.append(f"{document.filename}: missing")
            continue
        if not document.sha256:
            continue
        actual = sha256_of(path)
        if actual != document.sha256:
            problems.append(
                f"{document.filename}: checksum differs from {FETCHED}. The "
                "published document has changed; the labels may not match it."
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check, do not download")
    parser.add_argument("--dir", default=str(CORPUS_DIR))
    parser.add_argument(
        "--print-checksums", action="store_true",
        help="print checksums of what is present, for updating this manifest",
    )
    args = parser.parse_args()

    destination = Path(args.dir)
    destination.mkdir(parents=True, exist_ok=True)

    if args.print_checksums:
        for document in DOCUMENTS:
            path = destination / document.filename
            if path.exists():
                print(f'    sha256="{sha256_of(path)}",  # {document.filename}')
        return 0

    if args.verify:
        problems = verify(destination)
        for problem in problems:
            print(f"  {problem}")
        print("corpus verified" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0

    print(f"NDIC corpus -> {destination}")
    for document in DOCUMENTS:
        if (destination / document.filename).exists():
            print(f"  {document.filename} (already present)")
            continue
        try:
            download(document, destination)
        except Exception as exc:  # noqa: BLE001
            print(f"failed: {exc}")
            return 1

    print(
        "\nSource: Nigeria Deposit Insurance Corporation (ndic.gov.ng), "
        "CC BY-NC 4.0."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
