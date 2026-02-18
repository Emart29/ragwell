"""PDF document parser using PyMuPDF and pdfplumber."""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from datetime import datetime
from app.parsers.base import BaseParser, ParsedDocument
from app.logging_config import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """Parser for PDF documents."""
    
    SUPPORTED_EXTENSIONS = {'.pdf'}
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse PDF file and extract text, tables, and metadata.
        
        Uses PyMuPDF for text extraction with layout preservation
        and pdfplumber for table detection.
        """
        file_path = Path(file_path)
        
        try:
            # Extract text and metadata with PyMuPDF
            doc = fitz.open(file_path)
            
            pages_text = []
            full_text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with position info for layout preservation
                blocks = page.get_text("blocks")
                
                # Sort blocks by vertical position (y0), then horizontal (x0)
                # This handles multi-column layouts
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                page_text_parts = []
                for block in blocks:
                    # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) >= 5 and block[6] == 0:  # type 0 = text
                        text = block[4].strip()
                        if text:
                            page_text_parts.append(text)
                
                page_text = '\n'.join(page_text_parts)
                pages_text.append(page_text)
                full_text_parts.append(page_text)
            
            full_text = '\n\n'.join(full_text_parts)
            
            # Extract metadata from PDF
            metadata = self._extract_metadata(doc)
            metadata['page_count'] = len(doc)
            
            doc.close()
            
            # Extract tables with pdfplumber
            tables = self._extract_tables(file_path)
            
            if not full_text.strip():
                logger.warning(f"Extracted text from {file_path} is empty. Scanned PDF?")
            
            return ParsedDocument(
                text=full_text,
                tables=tables,
                metadata=metadata,
                pages=pages_text
            )
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not parse PDF: {str(e)}")
    
    def _extract_metadata(self, doc: fitz.Document) -> dict:
        """Extract metadata from PDF document."""
        pdf_metadata = doc.metadata
        
        metadata = {}
        
        if pdf_metadata.get('title'):
            metadata['title'] = pdf_metadata['title']
        if pdf_metadata.get('author'):
            metadata['author'] = pdf_metadata['author']
        if pdf_metadata.get('subject'):
            metadata['subject'] = pdf_metadata['subject']
        if pdf_metadata.get('creator'):
            metadata['creator'] = pdf_metadata['creator']
        if pdf_metadata.get('producer'):
            metadata['producer'] = pdf_metadata['producer']
        if pdf_metadata.get('creationDate'):
            try:
                # PDF dates are typically in format D:YYYYMMDDHHmmSS
                date_str = pdf_metadata['creationDate']
                if date_str.startswith('D:'):
                    date_str = date_str[2:14]  # Get YYYYMMDDHHmmSS
                    metadata['creation_date'] = datetime.strptime(
                        date_str[:14], '%Y%m%d%H%M%S'
                    ).isoformat()
            except (ValueError, IndexError):
                pass
        
        return metadata
    
    def _extract_tables(self, file_path: Path) -> list[dict]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table in page_tables:
                        if not table or len(table) < 2:
                            continue
                        
                        # First row as headers
                        headers = table[0] if table else []
                        rows = table[1:] if len(table) > 1 else []
                        
                        # Clean up headers and rows
                        headers = [str(h).strip() if h else '' for h in headers]
                        cleaned_rows = []
                        
                        for row in rows:
                            cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                            if any(cleaned_row):  # Skip empty rows
                                cleaned_rows.append(cleaned_row)
                        
                        if headers or cleaned_rows:
                            tables.append({
                                'headers': headers,
                                'rows': cleaned_rows,
                                'page_number': page_num + 1
                            })
        except Exception as e:
            # Table extraction is best-effort
            pass
        
        return tables
