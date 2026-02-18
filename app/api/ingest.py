"""Ingestion API for document upload and processing."""
import logging
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text

from app.pipeline import Pipeline
from app.storage.database import DatabaseManager
from app.storage.models import Document


logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
VALID_FILE_TYPES = {".pdf", ".docx", ".doc", ".txt", ".csv", ".html", ".htm"}


# ---------------------------------------------------------------------------
# Background worker (daemon thread + queue)
# ---------------------------------------------------------------------------

_task_queue: queue.Queue = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()
_processing_lock = threading.Lock()


def _run_pipeline(
    file_path: Path,
    document_id: str,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Run the processing pipeline for a single document."""
    # Global lock ensures we never process two documents concurrently,
    # even if sync wait=true and queue worker overlap.
    with _processing_lock:
        logger.info("[WORKER] Starting processing for document %s", document_id)
        try:
            logger.info("[WORKER] Setting status to 'processing' for %s", document_id)
            with DatabaseManager() as db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.processing_status = "processing"
                    db.commit()

            logger.info("[WORKER] Initializing pipeline for %s", document_id)
            pipeline = Pipeline()

            with DatabaseManager() as db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                original_filename = doc.filename if doc else None

            logger.info("[WORKER] Starting document processing for %s", document_id)
            result = pipeline.process_document(
                filepath=file_path,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                original_filename=original_filename,
                document_id=document_id,
            )
            logger.info("[WORKER] Document processing completed for %s with success=%s", document_id, result.success)

            with DatabaseManager() as db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    if result.success:
                        doc.processing_status = "completed"
                        logger.info("Document %s done - %s chunks", document_id, result.chunk_count)
                    else:
                        doc.processing_status = "failed"
                        doc.processing_error = result.error
                        logger.error("Document %s failed - %s", document_id, result.error)
                    db.commit()

            if file_path.exists():
                file_path.unlink()

            return result

        except Exception as exc:
            import traceback

            logger.error("Document %s exception - %s", document_id, exc)
            logger.error(traceback.format_exc())

            try:
                with DatabaseManager() as db:
                    doc = db.query(Document).filter(Document.id == document_id).first()
                    if doc:
                        doc.processing_status = "failed"
                        doc.processing_error = str(exc)
                        db.commit()
            except Exception:
                pass

            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass

            return None


def _worker_loop():
    """Daemon thread: drains the task queue sequentially."""
    while True:
        item = _task_queue.get()
        if item is None:
            break
        _run_pipeline(*item)
        _task_queue.task_done()


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        worker = threading.Thread(target=_worker_loop, daemon=True, name="ingest-worker")
        worker.start()
        _worker_started = True
        logger.info("Ingestion worker thread started")


def enqueue_document(file_path, document_id, chunk_strategy, chunk_size, chunk_overlap):
    _ensure_worker()
    _task_queue.put((file_path, document_id, chunk_strategy, chunk_size, chunk_overlap))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    document_id: str
    status: str
    filename: str
    chunk_count: Optional[int] = None


class BatchIngestResponse(BaseModel):
    batch_id: str
    document_ids: list[str]
    count: int


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: Optional[int] = None
    chunks_rejected: Optional[int] = None
    cache_hit_rate: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    wait: bool = Form(False),
):
    """Upload and process a document.

    Set wait=true to process synchronously (blocks until done).
    Default is async processing via background worker.
    
    Limits:
        - Maximum file size: 10MB
        - Supported formats: PDF, DOCX, DOC, TXT, CSV, HTML
    """
    # Validate file type
    file_ext = Path(file.filename or "unknown").suffix.lower()
    if file_ext not in VALID_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(VALID_FILE_TYPES)}"
        )
    
    if chunk_strategy not in ["recursive", "semantic"]:
        raise HTTPException(status_code=400, detail="chunk_strategy must be 'recursive' or 'semantic'")

    document_id = str(uuid.uuid4())

    upload_path = UPLOAD_DIR / f"{document_id}{file_ext}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Validate file size after upload
    file_size = upload_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        upload_path.unlink()  # Delete the file
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB"
        )

    file_type = file_ext.lstrip(".")

    with DatabaseManager() as db:
        doc = Document(
            id=document_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            processing_status="pending",
        )
        db.add(doc)
        db.commit()

    if wait:
        result = _run_pipeline(upload_path, document_id, chunk_strategy, chunk_size, chunk_overlap)
        status = "completed" if (result and result.success) else "failed"
        chunk_count = result.chunk_count if (result and result.success) else None
        return IngestResponse(
            document_id=document_id,
            status=status,
            filename=file.filename,
            chunk_count=chunk_count,
        )

    enqueue_document(upload_path, document_id, chunk_strategy, chunk_size, chunk_overlap)
    return IngestResponse(document_id=document_id, status="processing", filename=file.filename)


@router.post("/ingest/batch", response_model=BatchIngestResponse)
def ingest_batch(
    files: list[UploadFile] = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
):
    """Upload and process multiple documents (async)."""
    batch_id = str(uuid.uuid4())
    document_ids = []

    for file in files:
        document_id = str(uuid.uuid4())
        document_ids.append(document_id)

        file_ext = Path(file.filename).suffix
        upload_path = UPLOAD_DIR / f"{document_id}{file_ext}"
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = upload_path.stat().st_size
        file_type = file_ext.lower().lstrip(".")

        with DatabaseManager() as db:
            doc = Document(
                id=document_id,
                filename=file.filename,
                file_type=file_type,
                file_size=file_size,
                processing_status="pending",
            )
            db.add(doc)
            db.commit()

        enqueue_document(upload_path, document_id, chunk_strategy, chunk_size, chunk_overlap)

    return BatchIngestResponse(batch_id=batch_id, document_ids=document_ids, count=len(document_ids))


@router.get("/ingest/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str):
    """Get processing status for a document."""
    with DatabaseManager() as db:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        chunk_count = db.execute(
            text("SELECT COUNT(*) FROM chunks WHERE document_id = :doc_id"),
            {"doc_id": document_id},
        ).scalar()

        return DocumentStatusResponse(
            document_id=doc.id,
            filename=doc.filename,
            status=doc.processing_status,
            chunk_count=chunk_count if doc.processing_status == "completed" else None,
            error=doc.processing_error,
        )
