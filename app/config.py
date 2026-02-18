"""Ragwell configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    # API Keys
    JINA_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    # Database Configuration
    DATABASE_PATH: str = './data/ragwell.db'
    CHROMA_PATH: str = './data/chroma'
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = 'jina-embeddings-v3'
    EMBEDDING_DIMENSIONS: int = 384


# Global settings instance
settings = Settings()


def ensure_data_directories():
    """Create data directories if they don't exist."""
    data_dir = Path('./data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_dir = Path(settings.CHROMA_PATH)
    chroma_dir.mkdir(parents=True, exist_ok=True)
