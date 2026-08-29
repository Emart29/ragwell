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
    
    #: Extract tables from PDFs with pdfplumber during parsing.
    #: Off by default because it is the single most expensive step in the
    #: pipeline - roughly six minutes on a 262-page report against 29
    #: seconds for the text itself - and nothing downstream reads the
    #: result. The tables are carried into document metadata and never
    #: consulted by chunking, retrieval, or the API. Turn it on when
    #: something needs them.
    EXTRACT_PDF_TABLES: bool = False

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
    # Pinned rather than a rolling alias. `gemini-flash-latest` was
    # persistently 503 while pinned models of the same generation served
    # normally, so the alias buys currency at the cost of availability -
    # a poor trade for a benchmark that has to complete.
    GEMINI_MODEL: str = 'gemini-3.5-flash-lite'

    #: Model for the long-context arm. Named separately from GEMINI_MODEL so
    #: the comparison cannot be changed by accident when the default moves.
    # The free tier caps gemini-3.6-flash at 20 requests a day, which a
    # four-size sweep exhausts before it reaches the third size. The lite
    # model has the headroom to finish a run.
    LONG_CONTEXT_MODEL: str = 'gemini-3.5-flash-lite'

    GENERATION_TEMPERATURE: float = 0.7
    GENERATION_TIMEOUT: float = 30.0

    #: Attempts for a failure that retrying can fix. Capacity errors are
    #: common enough on a free tier that one attempt loses whole cells of a
    #: benchmark to a problem that clears in seconds.
    GENERATION_RETRIES: int = 4


# Global settings instance
settings = Settings()


def ensure_data_directories():
    """Create data directories if they don't exist."""
    data_dir = Path('./data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_dir = Path(settings.CHROMA_PATH)
    chroma_dir.mkdir(parents=True, exist_ok=True)
