"""Test multi-document upload processing."""
import requests, time, subprocess, sys, os, shutil
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

API_URL = "http://localhost:8000"

# Clean data
for p in [Path("data/ragwell.db")]:
    if p.exists():
        try: p.unlink()
        except: pass
for d in [Path("data/chroma"), Path("data/embedding_cache")]:
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
print("Cleaned data", flush=True)

# Start API
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)

# Wait for API
for i in range(20):
    try:
        time.sleep(1)
        r = requests.get(f"{API_URL}/api/health", timeout=1)
        if r.status_code == 200:
            print("API ready", flush=True)
            break
    except:
        print(f"  Waiting... ({i+1}/20)", flush=True)

# Upload 3 files
files = [
    Path("data/demo_samples/company_report.pdf"),
    Path("data/demo_samples/meeting_notes.docx"),
    Path("data/demo_samples/sales_data.csv"),
]

doc_ids = []
for f in files:
    if not f.exists():
        print(f"File not found: {f}", flush=True)
        continue
    with open(f, "rb") as fh:
        r = requests.post(
            f"{API_URL}/api/ingest",
            files={"file": (f.name, fh)},
            data={"chunk_strategy": "recursive", "chunk_size": "500"}
        )
        data = r.json()
        doc_ids.append(data["document_id"])
        print(f"Uploaded {f.name}: {data['document_id'][:8]}...", flush=True)

# Poll all docs
print("\nPolling for status...", flush=True)
for i in range(60):
    time.sleep(5)
    statuses = []
    for doc_id in doc_ids:
        r = requests.get(f"{API_URL}/api/ingest/{doc_id}/status", timeout=10)
        d = r.json()
        statuses.append(d)

    line = f"[{i*5:3d}s]"
    all_done = True
    for d in statuses:
        s = d.get("status", "?")
        c = d.get("chunk_count") or 0
        fn = d.get("filename", "?")[:15]
        line += f"  {fn}={s}({c})"
        if s not in ("completed", "failed"):
            all_done = False
    print(line, flush=True)

    if all_done:
        print("\nAll done!", flush=True)
        for d in statuses:
            print(f"  {d['filename']}: {d['chunk_count']} chunks", flush=True)
        break

proc.terminate()
print("Done", flush=True)
