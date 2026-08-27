"""Database connection and session management."""
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.config import settings
from app.storage.models import Base, CREATE_FTS5_TABLE, CREATE_FTS5_TRIGGERS


from app.logging_config import get_logger

logger = get_logger(__name__)

# Create database engine with connection pooling.
# WAL mode + busy_timeout allow the worker thread and the API event-loop
# to read/write concurrently without blocking each other.
engine = create_engine(
    f"sqlite:///{settings.DATABASE_PATH}",
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,  # seconds to wait on a locked database
    },
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)


def _enable_wal(dbapi_conn, connection_record):
    """Enable WAL journal mode on every new raw connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


from sqlalchemy import event as _sa_event
_sa_event.listen(engine, "connect", _enable_wal)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables."""
    # Ensure directory exists
    db_path = Path(settings.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create FTS5 virtual table and triggers
    create_fts5_tables()


def create_fts5_tables():
    """Create FTS5 virtual table and triggers for full-text search."""
    try:
        with engine.connect() as conn:
            # We use a script to ensure the table and triggers exist without wiping data if already there.
            # However, to change schema safely we might need to drop, but here we'll just ensure it's synced.
            # The triggers are dropped before being recreated. CREATE TRIGGER
            # IF NOT EXISTS leaves an existing one alone, so a database made by
            # an earlier version would keep the delete trigger that raises
            # "SQL logic error" and stays unable to delete a chunk. Triggers
            # hold no data, so recreating them costs nothing.
            setup_script = f"""
            {CREATE_FTS5_TABLE};

            DROP TRIGGER IF EXISTS chunks_fts_insert;
            DROP TRIGGER IF EXISTS chunks_fts_delete;
            DROP TRIGGER IF EXISTS chunks_fts_update;

            {CREATE_FTS5_TRIGGERS};

            -- Sync existing data if any was missed
            INSERT INTO chunks_fts(text, id, document_id, rowid)
            SELECT text, id, document_id, rowid FROM chunks
            WHERE rowid NOT IN (SELECT rowid FROM chunks_fts);
            """
            
            raw_conn = conn.connection.dbapi_connection
            cursor = raw_conn.cursor()
            cursor.executescript(setup_script)
            cursor.close()
            conn.commit()
            logger.info("FTS5 tables and triggers verified/synced.")
    except Exception as e:
        logger.error(f"Failed to setup FTS5: {e}")


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


class DatabaseManager:
    """Context manager for database sessions."""
    
    def __init__(self):
        self.db: Session | None = None
    
    def __enter__(self) -> Session:
        self.db = SessionLocal()
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            if exc_type:
                self.db.rollback()
            else:
                self.db.commit()
            self.db.close()


# Initialize tables on module load
create_tables()
