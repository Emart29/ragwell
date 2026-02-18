"""DOCX document parser using python-docx."""
from pathlib import Path
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from app.parsers.base import BaseParser, ParsedDocument
from app.logging_config import get_logger

logger = get_logger(__name__)


class DOCXParser(BaseParser):
    """Parser for DOCX documents."""
    
    SUPPORTED_EXTENSIONS = {'.docx'}
    
    # Heading style names in Word
    HEADING_STYLES = {'Heading 1', 'Heading 2', 'Heading 3', 'Heading 4', 
                      'Heading 5', 'Heading 6', 'Title'}
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse DOCX file and extract text, tables, and metadata."""
        file_path = Path(file_path)
        try:
            doc = Document(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(doc)
            
            # Extract text with heading hierarchy
            text_parts = []
            heading_hierarchy = []
            pages = []  # DOCX doesn't have explicit pages, we'll use sections
            current_section = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                # Track headings
                style_name = para.style.name if para.style else ''
                if style_name in self.HEADING_STYLES:
                    level = self._get_heading_level(style_name)
                    heading_hierarchy.append({
                        'text': text,
                        'level': level,
                        'style': style_name
                    })
                    
                    # Start a new "page" for major sections
                    if level <= 2 and current_section:
                        pages.append('\n'.join(current_section))
                        current_section = []
                
                text_parts.append(text)
                current_section.append(text)
            
            # Add last section
            if current_section:
                pages.append('\n'.join(current_section))
            
            # If no sections created, use all text as one page
            if not pages and text_parts:
                pages = ['\n'.join(text_parts)]
            
            full_text = '\n\n'.join(text_parts)
            
            # Extract tables
            tables = self._extract_tables(doc)
            
            # Add structural metadata
            metadata['heading_outline'] = heading_hierarchy
            metadata['heading_count'] = len(heading_hierarchy)
            
            if not full_text.strip():
                logger.warning(f"Extracted text from {file_path} is empty.")
            
            return ParsedDocument(
                text=full_text,
                tables=tables,
                metadata=metadata,
                pages=pages if pages else [full_text]
            )
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not parse DOCX: {str(e)}")
    
    def _extract_metadata(self, doc: Document) -> dict:
        """Extract metadata from DOCX document."""
        metadata = {}
        
        # Core properties
        if doc.core_properties:
            props = doc.core_properties
            if props.title:
                metadata['title'] = props.title
            if props.author:
                metadata['author'] = props.author
            if props.subject:
                metadata['subject'] = props.subject
            if props.created:
                metadata['created'] = props.created.isoformat()
            if props.modified:
                metadata['modified'] = props.modified.isoformat()
            if props.last_modified_by:
                metadata['last_modified_by'] = props.last_modified_by
            if props.revision:
                metadata['revision'] = props.revision
        
        return metadata
    
    def _get_heading_level(self, style_name: str) -> int:
        """Get heading level from style name."""
        if style_name == 'Title':
            return 0
        if style_name.startswith('Heading '):
            try:
                return int(style_name.split(' ')[1])
            except (IndexError, ValueError):
                pass
        return 99  # Unknown level
    
    def _extract_tables(self, doc: Document) -> list[dict]:
        """Extract tables from DOCX document."""
        tables = []
        
        for table in doc.tables:
            if not table.rows:
                continue
            
            # Try to detect header row
            rows_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                rows_data.append(row_data)
            
            if not rows_data:
                continue
            
            # Use first row as headers if it looks like headers
            # (shorter text, different formatting, etc.)
            headers = rows_data[0] if rows_data else []
            rows = rows_data[1:] if len(rows_data) > 1 else []
            
            tables.append({
                'headers': headers,
                'rows': rows
            })
        
        return tables
