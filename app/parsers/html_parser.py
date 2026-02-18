"""HTML document parser using trafilatura and BeautifulSoup."""
import trafilatura
from bs4 import BeautifulSoup
from pathlib import Path
from app.parsers.base import BaseParser, ParsedDocument
from app.logging_config import get_logger

logger = get_logger(__name__)


class HTMLParser(BaseParser):
    """Parser for HTML documents."""
    
    SUPPORTED_EXTENSIONS = {'.html', '.htm'}
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse HTML file and extract text, tables, and metadata.
        
        Uses trafilatura as primary extractor (strips boilerplate)
        and BeautifulSoup as fallback.
        """
        try:
            html_content = file_path.read_text(encoding='utf-8', errors='replace')
            
            # Try trafilatura first for main content extraction
            trafilatura_result = self._extract_with_trafilatura(html_content)
            
            if trafilatura_result['text']:
                # Use trafilatura results
                text = trafilatura_result['text']
                metadata = trafilatura_result['metadata']
            else:
                # Fallback to BeautifulSoup
                text, metadata = self._extract_with_bs4(html_content)
            
            # Extract tables using BeautifulSoup
            tables = self._extract_tables(html_content)
            
            # Split into pages/sections (use major headings as dividers)
            pages = self._split_into_pages(text)
            
            if not text.strip():
                logger.warning(f"Extracted text from {file_path} is empty.")
            
            return ParsedDocument(
                text=text,
                tables=tables,
                metadata=metadata,
                pages=pages
            )
        except Exception as e:
            logger.error(f"Failed to parse HTML {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not parse HTML: {str(e)}")
    
    def _extract_with_trafilatura(self, html_content: str) -> dict:
        """Extract content using trafilatura."""
        result = {
            'text': '',
            'metadata': {}
        }
        
        try:
            # Extract main text content
            text = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False,  # We'll handle tables separately
                no_fallback=False,
                output_format='txt'
            )
            
            if text:
                result['text'] = text.strip()
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html_content)
            if metadata:
                if metadata.title:
                    result['metadata']['title'] = metadata.title
                if metadata.author:
                    result['metadata']['author'] = metadata.author
                if metadata.description:
                    result['metadata']['description'] = metadata.description
                if metadata.sitename:
                    result['metadata']['sitename'] = metadata.sitename
                if metadata.url:
                    result['metadata']['source_url'] = metadata.url
                if metadata.date:
                    result['metadata']['date'] = metadata.date
        except Exception:
            pass
        
        return result
    
    def _extract_with_bs4(self, html_content: str) -> tuple[str, dict]:
        """Extract content using BeautifulSoup as fallback."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Extract metadata
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Meta tags
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and meta_description.get('content'):
            metadata['description'] = meta_description['content']
        
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            metadata['author'] = meta_author['content']
        
        # Get text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text, metadata
    
    def _extract_tables(self, html_content: str) -> list[dict]:
        """Extract tables from HTML."""
        tables = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for table in soup.find_all('table'):
            headers = []
            rows = []
            
            # Try to find headers
            header_row = table.find('thead')
            if header_row:
                ths = header_row.find_all(['th', 'td'])
                headers = [th.get_text(strip=True) for th in ths]
            else:
                # First row might be headers
                first_row = table.find('tr')
                if first_row:
                    ths = first_row.find_all('th')
                    if ths:
                        headers = [th.get_text(strip=True) for th in ths]
            
            # Get all rows
            for tr in table.find_all('tr'):
                # Skip header row if we already extracted headers
                if tr.find_parent('thead'):
                    continue
                
                cells = tr.find_all(['td', 'th'])
                if not cells:
                    continue
                
                row_data = [cell.get_text(strip=True) for cell in cells]
                
                # If this is the first row and we don't have headers yet
                if not headers and tr == table.find('tr'):
                    # Check if it looks like a header row
                    all_th = all(cell.name == 'th' for cell in cells)
                    if all_th:
                        headers = row_data
                        continue
                
                if any(row_data):  # Skip empty rows
                    rows.append(row_data)
            
            if headers or rows:
                tables.append({
                    'headers': headers,
                    'rows': rows
                })
        
        return tables
    
    def _split_into_pages(self, text: str) -> list[str]:
        """Split text into pages/sections based on major headings."""
        if not text:
            return []
        
        # Simple splitting by double newlines for now
        # Could be enhanced to split by H1/H2 headings
        sections = text.split('\n\n\n')
        
        # Filter out empty sections
        sections = [s.strip() for s in sections if s.strip()]
        
        return sections if sections else [text]
