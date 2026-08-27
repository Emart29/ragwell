"""SQLAlchemy database models for Ragwell."""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, 
    ForeignKey, JSON, Float, Text, event
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from app.config import settings, ensure_data_directories


# Ensure data directories exist
ensure_data_directories()

# SQLAlchemy base
Base = declarative_base()


class Document(Base):
    """Represents an uploaded document."""
    __tablename__ = 'documents'
    
    id = Column(String(36), primary_key=True)  # UUID string
    filename = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=True)
    title = Column(String(512), nullable=True)
    author = Column(String(255), nullable=True)
    language = Column(String(10), nullable=True)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    processing_status = Column(
        String(20), 
        default='pending', 
        nullable=False
    )  # pending/processing/completed/failed
    processing_error = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, status={self.processing_status})>"


class Chunk(Base):
    """Represents a text chunk extracted from a document."""
    __tablename__ = 'chunks'
    
    id = Column(String(36), primary_key=True)  # UUID string
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    chunk_strategy = Column(String(50), nullable=True)  # recursive/semantic
    page_number = Column(Integer, nullable=True)
    heading_context = Column(String(512), nullable=True)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


# FTS5 table creation SQL
CREATE_FTS5_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    id UNINDEXED,
    document_id UNINDEXED
);
"""

# Triggers keeping the FTS5 index in step with the chunks table.
#
# Deletes use a plain DELETE rather than FTS5's "delete" command. That command —
# INSERT INTO chunks_fts(chunks_fts, rowid, ...) VALUES('delete', ...) — is only
# valid for external-content tables, ones declared with content=. This table
# stores its own content, so the command raises "SQL logic error" and takes the
# whole transaction with it. The symptom is that deleting any chunk fails, which
# also means a document cannot be deleted and a corpus cannot be re-ingested.
CREATE_FTS5_TRIGGERS = """
-- Insert trigger
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(text, id, document_id, rowid)
    VALUES (NEW.text, NEW.id, NEW.document_id, NEW.rowid);
END;

-- Delete trigger
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = OLD.rowid;
END;

-- Update trigger
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = OLD.rowid;
    INSERT INTO chunks_fts(text, id, document_id, rowid)
    VALUES (NEW.text, NEW.id, NEW.document_id, NEW.rowid);
END;
"""


def create_fts5_tables(engine):
    """Create FTS5 virtual table and triggers."""
    with engine.connect() as conn:
        conn.execute(CREATE_FTS5_TABLE)
        conn.execute(CREATE_FTS5_TRIGGERS)
        conn.commit()
