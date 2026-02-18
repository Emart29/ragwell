"""Quick test: upload 3 docs with wait=true and verify all complete."""
import requests, time, subprocess, sys, shutil
from pathlib import Path

# Clean
for p in [Path("data/ragwell.db")]:
    if p.exists():
        p.unlink()
for suffix in ["-journal", "-shm", "-wal"]:
    p = Path(f"data/ragwell.db{suffix}")
    if p.exists():
        p.unlink()
for d in [Path("data/chroma"), Path("data/embedding_cache"), Path("data/uploads")]:
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

print("Starting API...", flush=True)
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

for i in range(20):
    time.sleep(1)
    try:
        r = requests.get("http://localhost:8000/api/health", timeout=1)
        if r.status_code == 200:
            print(f"API ready after {i+1}s", flush=True)
            break
    except Exception:
        pass
else:
    print("FAIL: API never started", flush=True)
    proc.terminate()
    sys.exit(1)

files = [
    Path("data/demo_samples/company_report.pdf"),
    Path("data/demo_samples/meeting_notes.docx"),
    Path("data/demo_samples/sales_data.csv"),
]

doc_ids = []
for f in files:
    if not f.exists():
        print(f"SKIP: {f}", flush=True)
        continue
    print(f"  Uploading {f.name}...", end=" ", flush=True)
    t0 = time.time()
    with open(f, "rb") as fh:
        r = requests.post(
            "http://localhost:8000/api/ingest",
            files={"file": (f.name, fh)},
            data={"chunk_strategy": "recursive", "chunk_size": "500", "wait": "true"},
            timeout=300,
        )
    data = r.json()
    doc_ids.append(data["document_id"])
    elapsed = time.time() - t0
    print(f"{data['status']} ({data.get('chunk_count','?')} chunks, {elapsed:.1f}s)", flush=True)

print(f"\nAll {len(doc_ids)} documents processed!", flush=True)

# Verify via status endpoint
for doc_id in doc_ids:
    r = requests.get(f"http://localhost:8000/api/ingest/{doc_id}/status", timeout=10)
    s = r.json()
    print(f"  {s['filename']}: {s['status']} ({s.get('chunk_count', 0)} chunks)", flush=True)

proc.terminate()
print("Done", flush=True)
