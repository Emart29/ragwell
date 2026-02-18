"""Parser registry for routing files to appropriate parsers."""
from pathlib import Path
from typing import Type
from app.parsers.base import BaseParser, ParsedDocument
from app.parsers.pdf_parser import PDFParser
from app.parsers.docx_parser import DOCXParser
from app.parsers.html_parser import HTMLParser
from app.parsers.csv_parser import CSVParser
from app.parsers.txt_parser import TXTParser


class ParserRegistry:
    """Registry for document parsers."""
    
    def __init__(self):
        self._parsers: dict[str, Type[BaseParser]] = {}
        self._instances: dict[str, BaseParser] = {}
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """Register all built-in parsers."""
        self.register(PDFParser)
        self.register(DOCXParser)
        self.register(HTMLParser)
        self.register(CSVParser)
        self.register(TXTParser)
    
    def register(self, parser_class: Type[BaseParser]):
        """Register a parser class.
        
        Args:
            parser_class: Parser class to register
        """
        for ext in parser_class.SUPPORTED_EXTENSIONS:
            self._parsers[ext.lower()] = parser_class
    
    def get_parser(self, file_path: Path) -> BaseParser:
        """Get appropriate parser for file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser instance for the file type
            
        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        parser_class = self._parsers.get(ext)
        if not parser_class:
            supported = ', '.join(sorted(self._parsers.keys()))
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types: {supported}"
            )
        
        # Use cached instance if available
        if ext not in self._instances:
            self._instances[ext] = parser_class()
        
        return self._instances[ext]
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file can be parsed.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if a parser is available for this file type
        """
        return file_path.suffix.lower() in self._parsers
    
    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return sorted(self._parsers.keys())


# Global registry instance
_registry = ParserRegistry()


def parse_file(file_path: Path) -> ParsedDocument:
    """Parse a file using the appropriate parser.
    
    This is the main entry point for document parsing.
    
    Args:
        file_path: Path to the file to parse
        
    Returns:
        ParsedDocument with extracted content
        
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    parser = _registry.get_parser(file_path)
    return parser.parse(file_path)


def get_supported_types() -> list[str]:
    """Get list of supported file types."""
    return _registry.get_supported_extensions()
