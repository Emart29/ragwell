from app.storage.database import DatabaseManager
from sqlalchemy import text

with DatabaseManager() as db:
    chunks_count = db.execute(text("SELECT COUNT(*) FROM chunks")).fetchone()[0]
    fts_count = db.execute(text("SELECT COUNT(*) FROM chunks_fts")).fetchone()[0]
    print(f"Chunks count: {chunks_count}")
    print(f"FTS count: {fts_count}")
    
    if fts_count > 0:
        sample = db.execute(text("SELECT text FROM chunks_fts LIMIT 1")).fetchone()[0]
        print(f"Sample FTS text: {sample[:100]}...")
        
        # Try a test match
        words = sample.split()[:2]
        query = " ".join(words)
        print(f"Testing MATCH with: {query}")
        match_count = db.execute(text("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH :q"), {"q": query}).fetchone()[0]
        print(f"Match count: {match_count}")
