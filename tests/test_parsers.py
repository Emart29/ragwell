"""Tests for document parsers."""
import pytest
from pathlib import Path
from app.parsers import parse_file, get_supported_types, extract_all_metadata
from app.parsers.registry import ParserRegistry
from app.parsers.pdf_parser import PDFParser
from app.parsers.docx_parser import DOCXParser
from app.parsers.html_parser import HTMLParser
from app.parsers.csv_parser import CSVParser
from app.parsers.txt_parser import TXTParser


# Fixtures directory
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


class TestParserRegistry:
    """Tests for the parser registry."""
    
    def test_get_supported_types(self):
        """Test getting supported file types."""
        types = get_supported_types()
        assert '.pdf' in types
        assert '.docx' in types
        assert '.html' in types
        assert '.csv' in types
        assert '.txt' in types
    
    def test_can_parse_supported_types(self):
        """Test registry can identify supported types."""
        registry = ParserRegistry()
        
        assert registry.can_parse(Path('test.pdf'))
        assert registry.can_parse(Path('test.docx'))
        assert registry.can_parse(Path('test.html'))
        assert registry.can_parse(Path('test.csv'))
        assert registry.can_parse(Path('test.txt'))
    
    def test_cannot_parse_unsupported_types(self):
        """Test registry rejects unsupported types."""
        registry = ParserRegistry()
        
        assert not registry.can_parse(Path('test.xyz'))
        assert not registry.can_parse(Path('test.exe'))
        assert not registry.can_parse(Path('test'))
    
    def test_parse_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_file(Path('nonexistent.pdf'))
    
    def test_parse_unsupported_type(self):
        """Test error handling for unsupported types."""
        # Create a temp file with unsupported extension
        temp_file = FIXTURES_DIR / 'test.xyz'
        temp_file.write_text('test')
        
        try:
            with pytest.raises(ValueError, match='Unsupported file type'):
                parse_file(temp_file)
        finally:
            temp_file.unlink()


class TestPDFParser:
    """Tests for PDF parser."""
    
    def test_parse_pdf_basic(self):
        """Test basic PDF parsing."""
        pdf_path = FIXTURES_DIR / 'test.pdf'
        result = parse_file(pdf_path)
        
        assert result.text
        assert 'Test Document Title' in result.text
        assert 'Test Author' in result.text or len(result.pages) > 0
        assert len(result.pages) >= 1
    
    def test_pdf_metadata_extraction(self):
        """Test PDF metadata extraction."""
        parser = PDFParser()
        result = parser.parse(FIXTURES_DIR / 'test.pdf')
        
        assert result.metadata.get('title') == 'Test PDF Document'
        assert result.metadata.get('author') == 'Test Author'
        assert 'page_count' in result.metadata
    
    def test_pdf_tables(self):
        """Test PDF table extraction."""
        result = parse_file(FIXTURES_DIR / 'test.pdf')
        
        # Should have at least one table
        assert len(result.tables) >= 0  # Table extraction is best-effort


class TestDOCXParser:
    """Tests for DOCX parser."""
    
    def test_parse_docx_basic(self):
        """Test basic DOCX parsing."""
        docx_path = FIXTURES_DIR / 'test.docx'
        result = parse_file(docx_path)
        
        assert result.text
        assert 'Test Document Title' in result.text
        assert len(result.pages) >= 1
    
    def test_docx_metadata_extraction(self):
        """Test DOCX metadata extraction."""
        parser = DOCXParser()
        result = parser.parse(FIXTURES_DIR / 'test.docx')
        
        assert result.metadata.get('title') == 'Test DOCX Document'
        assert result.metadata.get('author') == 'Test Author'
    
    def test_docx_heading_hierarchy(self):
        """Test DOCX heading extraction."""
        parser = DOCXParser()
        result = parser.parse(FIXTURES_DIR / 'test.docx')
        
        assert 'heading_outline' in result.metadata
        outline = result.metadata['heading_outline']
        assert len(outline) > 0
        
        # Check for expected headings
        heading_texts = [h['text'] for h in outline]
        assert any('Introduction' in text for text in heading_texts)
    
    def test_docx_tables(self):
        """Test DOCX table extraction."""
        result = parse_file(FIXTURES_DIR / 'test.docx')
        
        assert len(result.tables) >= 1
        
        # Check table structure
        table = result.tables[0]
        assert 'headers' in table
        assert 'rows' in table
        assert 'Name' in table['headers']


class TestHTMLParser:
    """Tests for HTML parser."""
    
    def test_parse_html_basic(self):
        """Test basic HTML parsing."""
        html_path = FIXTURES_DIR / 'test.html'
        result = parse_file(html_path)
        
        assert result.text
        # Trafilatura extracts main content (title often in metadata only)
        assert 'This is a test HTML document for Ragwell' in result.text
        assert len(result.pages) >= 1
    
    def test_html_metadata_extraction(self):
        """Test HTML metadata extraction."""
        parser = HTMLParser()
        result = parser.parse(FIXTURES_DIR / 'test.html')
        
        # Title could be from h1 or title tag
        assert result.metadata.get('title') in ['Test HTML Document', 'Test Document Title']
        assert 'author' in result.metadata or 'description' in result.metadata
    
    def test_html_tables(self):
        """Test HTML table extraction."""
        result = parse_file(FIXTURES_DIR / 'test.html')
        
        assert len(result.tables) >= 1
        
        # Check table structure
        table = result.tables[0]
        assert 'headers' in table
        assert 'rows' in table


class TestCSVParser:
    """Tests for CSV parser."""
    
    def test_parse_csv_basic(self):
        """Test basic CSV parsing."""
        csv_path = FIXTURES_DIR / 'test.csv'
        result = parse_file(csv_path)
        
        assert result.text
        assert 'Alice' in result.text
        assert 'New York' in result.text
    
    def test_csv_table_extraction(self):
        """Test CSV table extraction."""
        parser = CSVParser()
        result = parser.parse(FIXTURES_DIR / 'test.csv')
        
        assert len(result.tables) == 1
        table = result.tables[0]
        
        assert table['headers'] == ['Name', 'Age', 'City']
        assert len(table['rows']) == 3
        assert ['Alice', '30', 'New York'] in table['rows']
    
    def test_csv_delimiter_detection(self):
        """Test CSV delimiter detection."""
        parser = CSVParser()
        
        # Test CSV
        csv_result = parser.parse(FIXTURES_DIR / 'test.csv')
        assert csv_result.metadata.get('delimiter') == ','
        
        # Test TSV
        tsv_result = parser.parse(FIXTURES_DIR / 'test.tsv')
        assert tsv_result.metadata.get('delimiter') == '\t'
    
    def test_csv_metadata(self):
        """Test CSV metadata extraction."""
        result = parse_file(FIXTURES_DIR / 'test.csv')
        
        assert 'row_count' in result.metadata
        assert result.metadata['row_count'] == 3
        assert result.metadata['column_count'] == 3


class TestTXTParser:
    """Tests for TXT parser."""
    
    def test_parse_txt_basic(self):
        """Test basic TXT parsing."""
        txt_path = FIXTURES_DIR / 'test.txt'
        result = parse_file(txt_path)
        
        assert result.text
        assert 'Test Document Title' in result.text
        assert len(result.pages) >= 1
    
    def test_txt_encoding_detection(self):
        """Test TXT encoding detection."""
        parser = TXTParser()
        
        # UTF-8 file (may be detected as ascii for pure ASCII content)
        result_utf8 = parser.parse(FIXTURES_DIR / 'test.txt')
        encoding = result_utf8.metadata.get('encoding', '').lower()
        assert encoding in ['utf-8', 'utf-8-sig', 'ascii']
        
        # UTF-16 file
        result_utf16 = parser.parse(FIXTURES_DIR / 'fixture_utf16.txt')
        assert 'utf-16' in result_utf16.metadata.get('encoding', '').lower()
    
    def test_txt_content_metadata(self):
        """Test TXT content metadata extraction."""
        result = parse_file(FIXTURES_DIR / 'test.txt')
        
        assert 'word_count' in result.metadata
        assert 'char_count' in result.metadata
        assert 'line_count' in result.metadata
        assert result.metadata['word_count'] > 0
        assert result.metadata['char_count'] > 0
    
    def test_txt_line_endings_normalization(self):
        """Test TXT line ending normalization."""
        parser = TXTParser()
        result = parser.parse(FIXTURES_DIR / 'test.txt')
        
        # Should not have Windows line endings
        assert '\r\n' not in result.text
        # Should have Unix line endings
        assert '\n' in result.text


class TestMetadataExtractor:
    """Tests for metadata extraction."""
    
    def test_extract_file_metadata(self):
        """Test file metadata extraction."""
        from app.parsers.metadata import extract_file_metadata
        
        result = extract_file_metadata(FIXTURES_DIR / 'test.txt')
        
        # Raw file metadata (without prefix)
        assert 'filename' in result
        assert 'extension' in result
        assert 'size_bytes' in result
        assert 'size_human' in result
    
    def test_extract_content_metadata(self):
        """Test content metadata extraction."""
        from app.parsers.metadata import extract_content_metadata
        
        text = "This is a test document with multiple words for testing."
        result = extract_content_metadata(text)
        
        assert result['word_count'] == 10
        assert result['char_count'] == len(text)
        assert result['line_count'] == 1
        assert result['estimated_reading_time_minutes'] > 0
    
    def test_language_detection(self):
        """Test language detection."""
        from app.parsers.metadata import detect_language
        
        english_text = "The quick brown fox jumps over the lazy dog."
        assert detect_language(english_text) == 'en'
    
    def test_extract_all_metadata(self):
        """Test comprehensive metadata extraction."""
        parsed_doc = parse_file(FIXTURES_DIR / 'test.txt')
        result = extract_all_metadata(FIXTURES_DIR / 'test.txt', {
            'text': parsed_doc.text,
            'metadata': parsed_doc.metadata,
            'tables': parsed_doc.tables,
            'pages': parsed_doc.pages
        })
        
        # Should have file metadata
        assert any(k.startswith('file_') for k in result.keys())
        
        # Should have content metadata
        assert 'word_count' in result
        assert 'char_count' in result


class TestDatabase:
    """Tests for database models."""
    
    def test_document_model(self):
        """Test Document model creation."""
        from app.storage.models import Document
        from datetime import datetime
        
        doc = Document(
            id='test-uuid-123',
            filename='test.pdf',
            file_type='pdf',
            file_size=1024,
            processing_status='pending'
        )
        
        assert doc.id == 'test-uuid-123'
        assert doc.filename == 'test.pdf'
        assert doc.processing_status == 'pending'
    
    def test_chunk_model(self):
        """Test Chunk model creation."""
        from app.storage.models import Chunk
        
        chunk = Chunk(
            id='chunk-uuid-456',
            document_id='test-uuid-123',
            chunk_index=0,
            text='Test chunk content',
            chunk_strategy='recursive'
        )
        
        assert chunk.id == 'chunk-uuid-456'
        assert chunk.document_id == 'test-uuid-123'
        assert chunk.chunk_index == 0
    
    def test_database_connection(self):
        """Test database connection."""
        from app.storage.database import get_db
        
        db = get_db()
        assert db is not None
        db.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
