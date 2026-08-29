import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import fitz
import requests
from docx import Document

API_URL = "http://localhost:8000"
DATA_DIR = Path("data/demo_samples")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_json(response: requests.Response):
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _request(method: str, url: str, quiet: bool = False, **kwargs):
    """Make a request, reporting transport failures rather than raising.

    ``quiet`` suppresses the failure line for callers where a refused
    connection is the expected state -- polling a server that is still
    starting up, where the message would otherwise print once a second and
    make a healthy first run look broken.
    """
    try:
        return requests.request(method, url, timeout=kwargs.pop("timeout", 20), **kwargs)
    except requests.exceptions.ConnectionError:
        if not quiet:
            print(f"[FAIL] Connection error contacting {url}. Is API running?")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"[FAIL] Request failed for {url}: {exc}")
        return None


def clean_previous_data():
    print("[>>] Cleaning previous data ...")
    db_path = Path("data/ragwell.db")
    chroma_path = Path("data/chroma")

    if db_path.exists():
        try:
            db_path.unlink()
            print(f"  Removed {db_path}")
        except PermissionError:
            print(f"  [WARN] Could not remove {db_path} (file in use)")

    for suffix in ["-journal", "-shm", "-wal"]:
        p = Path(f"{db_path}{suffix}")
        if p.exists():
            try:
                p.unlink()
            except PermissionError:
                pass

    if chroma_path.exists():
        shutil.rmtree(chroma_path, ignore_errors=True)
        print(f"  Removed {chroma_path}/")


def start_api():
    health = _request("GET", f"{API_URL}/api/health", timeout=2, quiet=True)
    if health is not None and health.status_code == 200:
        print("[OK] API already running")
        return None

    print(">> Starting API server ...")
    # Pass environment variables to subprocess
    env = os.environ.copy()
    
    # Start API with visible output for debugging
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000", "--log-level", "info"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        env=env,
    )

    for i in range(30):
        time.sleep(1)
        health = _request("GET", f"{API_URL}/api/health", timeout=2, quiet=True)
        if health is not None and health.status_code == 200:
            print("[OK] API started")
            return process
        print(f"  waiting for API ({i + 1}/30)")

    print("[FAIL] API did not become healthy in time")
    return process


def generate_samples():
    print(">> Generating sample documents ...")

    # PDF with varied content
    pdf_path = DATA_DIR / "company_report.pdf"
    doc = fitz.open()
    
    paragraphs = [
        "The architecture of modern retrieval systems combines multiple search strategies to deliver optimal results.",
        "Vector embeddings capture semantic meaning and enable finding conceptually similar content.",
        "Keyword search using inverted indices provides fast exact matching for specific terms.",
        "Hybrid approaches merge vector and keyword results to leverage the strengths of both methods.",
        "Re-ranking models refine initial search results by scoring relevance with more sophisticated algorithms.",
        "Production deployments require careful tuning of chunk sizes to balance context and granularity.",
        "Evaluation metrics like MRR and recall at different cutoffs help measure search quality objectively.",
        "Latency considerations often involve trading off between search depth and response time.",
        "Data preprocessing steps include parsing multiple formats and extracting meaningful metadata.",
        "Storage solutions must handle both structured metadata and high-dimensional vector representations.",
        "Query expansion techniques can improve recall by generating variations of the original search terms.",
        "Cross-encoder models provide high-quality relevance scores but require more computational resources.",
    ]
    
    for p in range(4):
        page = doc.new_page()
        page.insert_text((50, 40), f"Company Report Section {p + 1}", fontsize=16)
        y = 70
        for i in range(12):
            para_idx = (p * 12 + i) % len(paragraphs)
            line = f"{i + 1}. {paragraphs[para_idx]} Additional details include implementation specifics and performance benchmarks for enterprise use cases."
            page.insert_text((50, y), line, fontsize=10)
            y += 14
            if y > 760:
                page = doc.new_page()
                y = 40
    doc.save(str(pdf_path))
    doc.close()

    # DOCX with varied meeting content
    docx_path = DATA_DIR / "meeting_notes.docx"
    docx = Document()
    docx.add_heading("Engineering Sync Minutes", 0)
    
    action_items = [
        "Optimize vector index parameters for faster search performance",
        "Implement fallback mechanisms when external APIs are unavailable",
        "Add monitoring dashboards for retrieval latency metrics",
        "Review and update chunking strategies based on user feedback",
        "Conduct load testing with concurrent document processing",
        "Document API endpoints and response schemas for frontend team",
        "Set up alerting for failed document processing jobs",
        "Evaluate alternative embedding models for comparison",
        "Implement caching layer for frequently accessed embeddings",
        "Create runbook for common operational issues and resolutions",
    ]
    
    for section in ["Action Items", "Retrieval Improvements", "Evaluation Plan"]:
        docx.add_heading(section, level=1)
        for i, item in enumerate(action_items[:6]):
            docx.add_paragraph(f"{i + 1}. {item}")
    docx.save(str(docx_path))

    csv_path = DATA_DIR / "sales_data.csv"
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    notes_templates = [
        "Q{i} performance review shows strong momentum in {region} market",
        "Customer acquisition costs decreased significantly this period in {region}",
        "New product launches drove growth across multiple {region} territories",
        "Strategic partnerships expanded market presence throughout {region}",
        "Operational efficiency improvements benefited {region} division results",
        "Market conditions in {region} remained favorable for sustained expansion",
        "Competitive analysis revealed opportunities in {region} segment",
        "Revenue diversification strategy successful in {region} operations",
        "Customer retention rates improved substantially across {region}",
        "Investment in {region} infrastructure supported scalability goals",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "region", "sales", "target", "notes"])
        for i in range(50):  # Reduced from 300 to avoid repetition issues
            region = regions[i % len(regions)]
            sales = 800 + i * 11
            note_template = notes_templates[i % len(notes_templates)]
            note = note_template.format(i=i+1, region=region)
            writer.writerow(
                [
                    f"TXN-{1000 + i}",
                    region,
                    sales,
                    1500,
                    note,
                ]
            )

    print("[OK] Generated: company_report.pdf, meeting_notes.docx, sales_data.csv")
    return [pdf_path, docx_path, csv_path]


def upload_and_wait(file_path: Path):
    with file_path.open("rb") as fh:
        form = {
            "chunk_strategy": (None, "recursive"),
            "chunk_size": (None, "500"),
            "chunk_overlap": (None, "50"),
        }
        files = {"file": (file_path.name, fh)}
        resp = _request("POST", f"{API_URL}/api/ingest", files=files, data={k: v[1] for k, v in form.items()})

    if resp is None:
        return None
    if resp.status_code != 200:
        body = resp.text[:200]
        print(f"[FAIL] Ingest failed for {file_path.name}: HTTP {resp.status_code} {body}")
        return None

    data = _safe_json(resp)
    if not data:
        print(f"[FAIL] Ingest returned non-JSON for {file_path.name}: {resp.text[:200]}")
        return None

    doc_id = data.get("document_id")
    if not doc_id:
        print(f"[FAIL] Missing document_id for {file_path.name}")
        return None

    print(f"  Uploaded {file_path.name} -> {doc_id}")

    for _ in range(120):
        time.sleep(1)
        status_resp = _request("GET", f"{API_URL}/api/ingest/{doc_id}/status", timeout=10)
        if status_resp is None:
            continue

        if status_resp.status_code != 200:
            continue

        status_data = _safe_json(status_resp)
        if not status_data:
            continue

        status = status_data.get("status")
        if status == "completed":
            return doc_id
        if status == "failed":
            print(f"[FAIL] Processing failed for {file_path.name}: {status_data.get('error')}")
            return doc_id

    print(f"[FAIL] Processing timeout for {file_path.name}")
    return doc_id


def print_chunk_counts(document_ids: list[str]):
    resp = _request("GET", f"{API_URL}/api/documents", timeout=10)
    if resp is None or resp.status_code != 200:
        print("[FAIL] Could not fetch /api/documents for chunk counts")
        return

    data = _safe_json(resp)
    if not data:
        print("[FAIL] /api/documents returned non-JSON")
        return

    docs = data.get("documents", [])
    by_id = {d.get("id"): d for d in docs}

    print("\nChunk counts from /api/documents:")
    for doc_id in document_ids:
        doc = by_id.get(doc_id)
        if not doc:
            print(f"  {doc_id}: not found in current page")
            continue

        details = _request("GET", f"{API_URL}/api/documents/{doc_id}/chunks?per_page=1", timeout=10)
        chunk_total = None
        if details is not None and details.status_code == 200:
            details_json = _safe_json(details)
            if details_json:
                chunk_total = details_json.get("total")

        print(f"  {doc.get('filename')}: {chunk_total if chunk_total is not None else 'unknown'} chunks")


def run_queries():
    print("\n" + "=" * 72)
    print("QUERY CHECK")
    print("=" * 72)

    queries = [
        ("What is the benefit of hybrid search?", "hybrid"),
        ("cross encoder reranking", "vector"),
        ("Action items for engineering sync", "keyword"),
        ("Improving recall in RAG", "hyde"),
    ]

    for query, strategy in queries:
        resp = _request(
            "POST",
            f"{API_URL}/api/query",
            json={"query": query, "strategy": strategy, "top_k": 5, "rerank": True},
            timeout=30,
        )
        if resp is None:
            continue
        if resp.status_code != 200:
            print(f"[FAIL] {strategy} query HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        data = _safe_json(resp)
        if not data:
            print(f"[FAIL] {strategy} query returned non-JSON: {resp.text[:200]}")
            continue

        results = data.get("results", [])
        total_ms = (data.get("search_time_ms", 0) + data.get("rerank_time_ms", 0))
        print(f"{strategy:8} | {len(results):2d} results | {total_ms:.1f}ms | {query}")


def run_evaluation():
    print("\n" + "=" * 72)
    print("EVALUATION")
    print("=" * 72)

    try:
        subprocess.run([sys.executable, "-m", "evaluation.generate_test_set"], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] test set generation failed: {exc}")
        return

    resp = _request("POST", f"{API_URL}/api/evaluate", timeout=180)
    if resp is None:
        return
    if resp.status_code != 200:
        print(f"[FAIL] /api/evaluate HTTP {resp.status_code}: {resp.text[:200]}")
        return

    data = _safe_json(resp)
    if not data:
        print(f"[FAIL] /api/evaluate returned non-JSON: {resp.text[:200]}")
        return

    results = data.get("results", {})
    if not results:
        print("[FAIL] No evaluation results returned")
        return

    print(f"{'Strategy':<18} {'MRR':>8} {'Recall@1':>10} {'Recall@10':>10} {'Latency':>10}")
    print("-" * 62)
    for name, item in results.items():
        m = item.get("metrics", {})
        print(
            f"{name:<18} {m.get('mrr', 0):>8.4f} {m.get('recall_1', 0):>10.4f} "
            f"{m.get('recall_10', 0):>10.4f} {m.get('latency_ms', 0):>8.1f}ms"
        )


def main():
    clean_previous_data()
    files = generate_samples()

    process = start_api()
    try:
        print("\n[>>] Uploading and processing documents sequentially ...")
        doc_ids = []
        for file_path in files:
            doc_id = upload_and_wait(file_path)
            if doc_id:
                doc_ids.append(doc_id)

        print_chunk_counts(doc_ids)
        run_queries()
        run_evaluation()

        print("\n" + "=" * 72)
        print("DEMO COMPLETE")
        print("Frontend: http://localhost:8000")
        print("API docs:  http://localhost:8000/docs")
        print("=" * 72)
    finally:
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    main()
