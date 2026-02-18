"""Plain text document parser with encoding detection."""
import chardet
from pathlib import Path
from app.parsers.base import BaseParser, ParsedDocument
from app.logging_config import get_logger

logger = get_logger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files."""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.rst', '.text'}
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse text file with automatic encoding detection.
        
        Handles BOM markers and mixed line endings.
        """
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Read file
            text = self._read_text_file(file_path, encoding)
            
            # Normalize line endings
            text = self._normalize_line_endings(text)
            
            # Calculate basic metadata
            line_count = text.count('\n') + 1 if text else 0
            word_count = len(text.split()) if text else 0
            char_count = len(text)
            
            metadata = {
                'encoding': encoding,
                'char_count': char_count,
                'line_count': line_count,
                'word_count': word_count,
                'file_type': file_path.suffix.lower().lstrip('.')
            }
            
            # Split into pages (use double newlines as section breaks)
            pages = self._split_into_pages(text)
            
            if not text.strip():
                logger.warning(f"Extracted text from {file_path} is empty.")
            
            return ParsedDocument(
                text=text,
                tables=[],  # Text files don't have structured tables
                metadata=metadata,
                pages=pages if pages else [text]
            )
        except Exception as e:
            logger.error(f"Failed to parse TXT {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not parse TXT: {str(e)}")
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
            # Check for BOM
            if raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'  # UTF-8 with BOM
            elif raw_data.startswith(b'\xff\xfe'):
                return 'utf-16-le'
            elif raw_data.startswith(b'\xfe\xff'):
                return 'utf-16-be'
            
            # Use chardet
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # Fallback strategies
            if confidence < 0.5 or not encoding:
                # Try utf-8 first
                try:
                    raw_data.decode('utf-8')
                    return 'utf-8'
                except UnicodeDecodeError:
                    pass
                
                # Try latin-1 as last resort
                return 'latin-1'
            
            return encoding
    
    def _read_text_file(self, file_path: Path, encoding: str) -> str:
        """Read text file with specified encoding."""
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return f.read()
    
    def _normalize_line_endings(self, text: str) -> str:
        """Normalize line endings to Unix style."""
        # Convert Windows (CRLF) and old Mac (CR) to Unix (LF)
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        return text
    
    def _split_into_pages(self, text: str) -> list[str]:
        """Split text into pages/sections."""
        if not text:
            return []
        
        # Split by multiple consecutive newlines
        sections = text.split('\n\n\n')
        
        # Clean up and filter empty sections
        sections = [s.strip() for s in sections if s.strip()]
        
        # If text is very long, split by paragraphs
        if len(text) > 10000 and len(sections) < 2:
            paragraphs = text.split('\n\n')
            # Group paragraphs into sections of ~5000 chars
            sections = []
            current_section = []
            current_length = 0
            
            for para in paragraphs:
                if para.strip():
                    current_section.append(para.strip())
                    current_length += len(para)
                    
                    if current_length > 5000:
                        sections.append('\n\n'.join(current_section))
                        current_section = []
                        current_length = 0
            
            if current_section:
                sections.append('\n\n'.join(current_section))
        
        return sections if sections else [text]
