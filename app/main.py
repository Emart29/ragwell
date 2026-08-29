from app.config import settings
"""FastAPI main application for Ragwell."""
import os
# Load environment variables from .env file first
from dotenv import load_dotenv
load_dotenv()

# Set offline mode for HuggingFace models before any imports
os.environ["HF_HUB_OFFLINE"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api import answer,  ingest, query, documents, health, compare, evaluate
from app.storage.database import create_tables
from app.storage.vector_store import VectorStore
from app.logging_config import setup_logging, request_id_ctx, generate_request_id


# Configure logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("=" * 60)
    logger.info("RAGWELL API STARTING UP")
    logger.info("=" * 60)
    
    # Check environment configuration
    jina_key = settings.JINA_API_KEY or ""
    groq_key = settings.GROQ_API_KEY or ""
    logger.info(f"Environment: JINA_API_KEY={'SET' if jina_key else 'NOT SET'}, GROQ_API_KEY={'SET' if groq_key else 'NOT SET'}")
    
    if not jina_key:
        logger.warning("WARNING: JINA_API_KEY not set! Document processing will fail.")
    
    # Initialize SQLite tables
    logger.info("Initializing SQLite database...")
    create_tables()
    logger.info("SQLite database ready")
    
    # Initialize ChromaDB
    logger.info("Initializing ChromaDB...")
    vector_store = VectorStore()
    stats = vector_store.get_stats()
    logger.info(f"ChromaDB ready: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
    
    logger.info("RAGWELL API READY")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("RAGWELL API SHUTTING DOWN")


app = FastAPI(
    title="Ragwell API",
    description="RAG Document Ingestion & Retrieval Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_request_id(request, call_next):
    """Middleware to inject request ID into context."""
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    token = request_id_ctx.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_ctx.reset(token)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(evaluate.router, prefix="/api")
app.include_router(answer.router, prefix="/api")
# Mounted only when the directory is present. StaticFiles raises at import
# time if it is missing, so an unconditional mount makes the whole application
# unimportable on a fresh clone — the frontend is built separately and is not
# required to serve the API.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend"
    )
else:
    logger.info("No frontend directory; serving the API only")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """No icon, but a 204 rather than a 404.

        A console error on a demo page reads as something being wrong,
        and a missing favicon is not.
        """
        return Response(status_code=204)

    @app.get("/", include_in_schema=False)
    async def root():
        """Point a browser at the docs when no frontend is built."""
        return {
            "service": "ragwell",
            "docs": "/docs",
            "api": "/api",
        }
