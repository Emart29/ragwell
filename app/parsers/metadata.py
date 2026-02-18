"""Metadata extraction utilities for documents."""
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def extract_file_metadata(file_path: Path) -> dict:
    """Extract file system metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    file_path = Path(file_path)
    stat = file_path.stat()
    
    return {
        'filename': file_path.name,
        'extension': file_path.suffix.lower(),
        'size_bytes': stat.st_size,
        'size_human': _format_file_size(stat.st_size),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
    }


def extract_content_metadata(text: str, language: Optional[str] = None) -> dict:
    """Extract content-based metadata.
    
    Args:
        text: Document text content
        language: Optional detected language code
        
    Returns:
        Dictionary with content metadata
    """
    if not text:
        return {
            'word_count': 0,
            'char_count': 0,
            'line_count': 0,
            'estimated_reading_time_minutes': 0,
            'language': language or 'unknown'
        }
    
    word_count = len(text.split())
    char_count = len(text)
    line_count = text.count('\n') + 1
    
    # Estimate reading time (average 200 words per minute)
    reading_time = round(word_count / 200, 1)
    
    metadata = {
        'word_count': word_count,
        'char_count': char_count,
        'line_count': line_count,
        'estimated_reading_time_minutes': reading_time,
        'language': language or detect_language(text)
    }
    
    return metadata


def extract_structural_metadata(parsed_doc: dict) -> dict:
    """Extract structural metadata from parsed document.
    
    Args:
        parsed_doc: Parsed document data with metadata
        
    Returns:
        Dictionary with structural metadata
    """
    metadata = {}
    
    # Get heading outline if available
    doc_metadata = parsed_doc.get('metadata', {})
    heading_outline = doc_metadata.get('heading_outline', [])
    
    if heading_outline:
        metadata['heading_outline'] = heading_outline
        metadata['heading_count'] = len(heading_outline)
        metadata['section_count'] = len(
            [h for h in heading_outline if h.get('level', 99) <= 2]
        )
    
    # Table count
    tables = parsed_doc.get('tables', [])
    metadata['table_count'] = len(tables)
    
    # Page count
    pages = parsed_doc.get('pages', [])
    if pages:
        metadata['page_count'] = len(pages)
    elif 'page_count' in doc_metadata:
        metadata['page_count'] = doc_metadata['page_count']
    
    return metadata


def detect_language(text: str) -> str:
    """Detect document language.
    
    Uses simple heuristic first, falls back to langdetect if available.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code (e.g., 'en', 'es', 'fr')
    """
    if not text:
        return 'unknown'
    
    # Simple heuristic: check for common words
    text_lower = text.lower()
    
    # English indicators
    english_words = ['the', 'and', 'is', 'of', 'to', 'in', 'that', 'have']
    english_score = sum(1 for word in english_words if word in text_lower.split())
    
    # If strong English indicators, return early
    if english_score >= 3:
        return 'en'
    
    # Try langdetect if available
    try:
        from langdetect import detect
        sample = text[:10000]  # Use first 10K chars for speed
        return detect(sample)
    except ImportError:
        pass
    except Exception:
        pass
    
    return 'unknown'


def merge_metadata(
    file_metadata: dict,
    content_metadata: dict,
    structural_metadata: dict,
    parser_metadata: dict
) -> dict:
    """Merge all metadata sources into flat dictionary.
    
    Args:
        file_metadata: File system metadata
        content_metadata: Content-based metadata
        structural_metadata: Structural metadata
        parser_metadata: Metadata from parser
        
    Returns:
        Flattened metadata dictionary for Document.metadata_json
    """
    merged = {}
    
    # File metadata (prefixed to avoid conflicts)
    for key, value in file_metadata.items():
        merged[f'file_{key}'] = value
    
    # Content metadata
    merged.update(content_metadata)
    
    # Structural metadata
    merged.update(structural_metadata)
    
    # Parser metadata (overwrite if conflicts)
    for key, value in parser_metadata.items():
        if key not in ['heading_outline', 'tables']:  # Skip large objects
            merged[key] = value
    
    return merged


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def extract_all_metadata(file_path: Path, parsed_doc: dict) -> dict:
    """Extract comprehensive metadata from file and parsed content.
    
    This is the main entry point for metadata extraction.
    
    Args:
        file_path: Path to the document
        parsed_doc: Parsed document from parser
        
    Returns:
        Flattened metadata dictionary
    """
    file_meta = extract_file_metadata(file_path)
    content_meta = extract_content_metadata(
        parsed_doc.get('text', ''),
        parsed_doc.get('metadata', {}).get('language')
    )
    structural_meta = extract_structural_metadata(parsed_doc)
    parser_meta = parsed_doc.get('metadata', {})
    
    return merge_metadata(file_meta, content_meta, structural_meta, parser_meta)
