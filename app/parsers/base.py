"""Base class for document parsers."""
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod


@dataclass
class ParsedDocument:
    """Result of parsing a document."""
    text: str
    tables: list[dict]  # {headers: [], rows: [[]]}
    metadata: dict      # title, author, page_count, etc.
    pages: list[str]    # text split by page where applicable
    
    def __post_init__(self):
        """Ensure defaults for optional fields."""
        if self.tables is None:
            self.tables = []
        if self.metadata is None:
            self.metadata = {}
        if self.pages is None:
            self.pages = []


class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    SUPPORTED_EXTENSIONS: set[str] = set()
    
    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document file and return structured content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParsedDocument with text, tables, metadata, and pages
        """
        pass
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file extension is supported
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
