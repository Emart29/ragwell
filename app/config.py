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
    
    # API Keys. Every one the application uses is declared here, and every
    # consumer reads them from this object rather than from os.environ.
    # pydantic-settings loads .env into these fields and exports nothing to the
    # process environment, so a module calling os.getenv sees a key only when
    # something else happened to set it first — which made key availability
    # depend on import order, and tests pass or fail depending on what ran
    # before them.
    JINA_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    
    # Database Configuration
    DATABASE_PATH: str = './data/ragwell.db'
    CHROMA_PATH: str = './data/chroma'
    
    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = 'jina-embeddings-v3'
    EMBEDDING_DIMENSIONS: int = 384

    # Generation. Both model ids this project previously pinned have since been
    # withdrawn by their providers — llama-3.3-70b-versatile and
    # gemini-1.5-flash — so they are configuration rather than constants, and
    # the Gemini default is a rolling alias for the same reason.
    DEFAULT_LLM_PROVIDER: str = 'groq'
    GROQ_MODEL: str = 'openai/gpt-oss-20b'
    GEMINI_MODEL: str = 'gemini-flash-latest'

    #: Model for the long-context arm. Named separately from GEMINI_MODEL so
    #: the comparison cannot be changed by accident when the default moves.
    LONG_CONTEXT_MODEL: str = 'gemini-flash-latest'

    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_TIMEOUT: float = 30.0


# Global settings instance
settings = Settings()


def ensure_data_directories():
    """Create data directories if they don't exist."""
    data_dir = Path('./data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_dir = Path(settings.CHROMA_PATH)
    chroma_dir.mkdir(parents=True, exist_ok=True)
