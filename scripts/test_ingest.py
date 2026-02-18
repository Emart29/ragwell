"""Quick test to check if document processing works through the API."""
import requests, time, subprocess, sys, os
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

API_URL = "http://localhost:8000"

# Clean data
import shutil
for p in [Path("data/ragwell.db"), Path("data/ragwell.db-journal"), Path("data/ragwell.db-shm"), Path("data/ragwell.db-wal")]:
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

# Upload a small test file
test_file = Path("data/demo_samples/meeting_notes.docx")
if not test_file.exists():
    print(f"Test file not found: {test_file}", flush=True)
    proc.terminate()
    sys.exit(1)

print(f"Uploading {test_file.name}...", flush=True)
with open(test_file, "rb") as f:
    r = requests.post(
        f"{API_URL}/api/ingest",
        files={"file": (test_file.name, f)},
        data={"chunk_strategy": "recursive", "chunk_size": "500"}
    )
    data = r.json()
    doc_id = data["document_id"]
    print(f"Uploaded: {doc_id}", flush=True)

# Poll for status
for i in range(60):
    time.sleep(5)
    try:
        r = requests.get(f"{API_URL}/api/ingest/{doc_id}/status", timeout=10)
        data = r.json()
        status = data.get("status")
        chunks = data.get("chunk_count", 0)
        print(f"  [{i*5}s] status={status}, chunks={chunks}", flush=True)
        if status in ("completed", "failed"):
            print(f"Final: {data}", flush=True)
            break
    except Exception as e:
        print(f"  [{i*5}s] Error: {e}", flush=True)

proc.terminate()
print("Done", flush=True)
