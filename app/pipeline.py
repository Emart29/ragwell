"""Full document processing pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
from app.parsers import parse_file, extract_all_metadata
from app.parsers.base import ParsedDocument
from app.chunking.recursive import RecursiveChunker
from app.chunking.semantic import SemanticChunker
from app.validation.chunk_validator import ChunkValidator, QualityResult
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing."""
    document_id: str
    filename: str
    chunk_count: int
    chunks_rejected: int
    cache_hit_rate: float
    processing_time_seconds: float
    success: bool
    error: Optional[str] = None


class Pipeline:
    """End-to-end document processing pipeline.
    
    Steps: parse -> extract metadata -> chunk -> validate -> embed -> store
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.vector_store = VectorStore()
        self.validator = ChunkValidator()
        logger.info("Pipeline initialized")
    
    def process_document(
        self,
        filepath: Path,
        chunk_strategy: str = 'recursive',
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        original_filename: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> ProcessingResult:
        """Process a document end-to-end.

        Args:
            filepath: Path to document file
            chunk_strategy: 'recursive' or 'semantic'
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            document_id: Existing document ID (skip creating a new doc record)

        Returns:
            ProcessingResult with details
        """
        start_time = time.time()
        filepath = Path(filepath)
        
        try:
            logger.info(f"Processing document: {filepath}")
            
            # Step 1: Parse document
            logger.info("Step 1: Parsing document...")
            parsed_doc = parse_file(filepath)
            
            # Step 2: Extract metadata
            logger.info("Step 2: Extracting metadata...")
            metadata = extract_all_metadata(filepath, {
                'text': parsed_doc.text,
                'metadata': parsed_doc.metadata,
                'tables': parsed_doc.tables,
                'pages': parsed_doc.pages
            })
            
            # Step 3: Chunk document
            logger.info(f"Step 3: Chunking with '{chunk_strategy}' strategy...")
            chunks = self._chunk_document(
                parsed_doc,
                chunk_strategy,
                chunk_size,
                chunk_overlap
            )
            
            if not chunks:
                raise ValueError("No chunks generated from document")
            
            logger.info(f"Generated {len(chunks)} chunks")
            
            # Step 4: Validate quality
            logger.info("Step 4: Validating chunk quality...")
            validation_results = self.validator.validate_batch(chunks)
            
            # Filter out bad chunks
            good_chunks = []
            quality_scores = []
            
            for chunk, result in zip(chunks, validation_results):
                if result.passed:
                    good_chunks.append(chunk)
                    quality_scores.append(result.score)
                else:
                    logger.warning(
                        f"Rejected chunk {chunk.chunk_index}: {result.issues}"
                    )
            
            chunks_rejected = len(chunks) - len(good_chunks)
            logger.info(f"Accepted {len(good_chunks)} chunks, rejected {chunks_rejected}")
            
            if not good_chunks:
                raise ValueError("All chunks failed quality validation")
            
            # Step 5: Embed chunks
            logger.info("Step 5: Embedding chunks...")
            try:
                embedder = Embedder.get_instance()
                embedder.reset_cache_stats()
                
                chunk_texts = [chunk.text for chunk in good_chunks]
                embeddings = embedder.embed_batch(chunk_texts)
                
                cache_stats = embedder.get_cache_stats()
                logger.info(f"Embedding cache hit rate: {cache_stats['hit_rate']:.1%}")
            except Exception as e:
                logger.error(f"Embedding step failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise ValueError(f"Failed to embed chunks: {str(e)}")
            
            # Step 6: Store in database
            logger.info("Step 6: Storing in database...")
            final_filename = original_filename or filepath.name

            if document_id is None:
                # No pre-existing document record — create one
                document_id = self.vector_store.store_document(
                    parsed_doc={'text': parsed_doc.text, 'metadata': parsed_doc.metadata},
                    metadata=metadata,
                    filename=final_filename,
                    file_type=filepath.suffix.lower().lstrip('.'),
                    file_size=filepath.stat().st_size
                )
            
            chunk_ids = self.vector_store.store_chunks(
                document_id=document_id,
                chunks=good_chunks,
                embeddings=embeddings,
                quality_scores=quality_scores
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document processing complete: {document_id}")
            
            return ProcessingResult(
                document_id=document_id,
                filename=filepath.name,
                chunk_count=len(good_chunks),
                chunks_rejected=chunks_rejected,
                cache_hit_rate=cache_stats['hit_rate'],
                processing_time_seconds=round(processing_time, 2),
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Document processing failed: {e}")
            
            return ProcessingResult(
                document_id="",
                filename=filepath.name,
                chunk_count=0,
                chunks_rejected=0,
                cache_hit_rate=0.0,
                processing_time_seconds=round(processing_time, 2),
                success=False,
                error=str(e)
            )
    
    def _chunk_document(
        self,
        parsed_doc: ParsedDocument,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> list:
        """Chunk document using specified strategy.
        
        Args:
            parsed_doc: Parsed document
            strategy: 'recursive' or 'semantic'
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        if strategy == 'semantic':
            chunker = SemanticChunker(max_tokens=chunk_size)
        else:
            chunker = RecursiveChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        return chunker.chunk_document(parsed_doc)


def process_document(
    filepath: Path,
    chunk_strategy: str = 'recursive',
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    original_filename: Optional[str] = None,
    document_id: Optional[str] = None
) -> ProcessingResult:
    """Convenience function to process a document."""
    pipeline = Pipeline()
    return pipeline.process_document(
        filepath=filepath,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        original_filename=original_filename,
        document_id=document_id
    )
