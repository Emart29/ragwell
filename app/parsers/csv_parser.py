"""CSV/TSV document parser with auto-detection."""
import csv
import chardet
from pathlib import Path
from io import StringIO
from app.parsers.base import BaseParser, ParsedDocument
from app.logging_config import get_logger

logger = get_logger(__name__)


class CSVParser(BaseParser):
    """Parser for CSV and TSV files."""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.tsv'}
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse CSV/TSV file and extract structured data.
        
        Auto-detects delimiter and encoding.
        Converts table to readable text with column context.
        """
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            
            # Detect delimiter
            delimiter = self._detect_delimiter(file_path, encoding)
            
            # Read and parse the file
            rows, headers = self._parse_csv(file_path, encoding, delimiter)
            
            if not rows and not headers:
                logger.warning(f"Extracted CSV data from {file_path} is empty.")
            
            # Create structured table
            tables = [{
                'headers': headers,
                'rows': rows,
                'delimiter': delimiter,
                'row_count': len(rows),
                'column_count': len(headers) if headers else 0
            }] if headers or rows else []
            
            # Convert to readable text
            text_parts = []
            text_parts.append(f"Table with {len(rows)} rows and {len(headers)} columns:\n")
            
            for i, row in enumerate(rows, 1):
                row_text_parts = []
                # Use headers if available, otherwise use indexed columns
                for j, value in enumerate(row):
                    header = headers[j] if j < len(headers) else f"Column {j+1}"
                    row_text_parts.append(f"{header}: {value}")
                
                text_parts.append(f"Row {i}: {'; '.join(row_text_parts)}")
            
            full_text = '\n'.join(text_parts)
            
            # Metadata
            metadata = {
                'delimiter': delimiter,
                'encoding': encoding,
                'row_count': len(rows),
                'column_count': len(headers) if headers else 0,
                'file_type': 'tsv' if delimiter == '\t' else 'csv'
            }
            
            return ParsedDocument(
                text=full_text,
                tables=tables,
                metadata=metadata,
                pages=[full_text]
            )
        except Exception as e:
            logger.error(f"Failed to parse CSV {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not parse CSV: {str(e)}")
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # Fallback to utf-8 if confidence is low
            if confidence < 0.5 or not encoding:
                encoding = 'utf-8'
            
            return encoding
    
    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Detect CSV delimiter using csv.Sniffer."""
        # Check if it's a TSV file by extension
        if file_path.suffix.lower() == '.tsv':
            return '\t'
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(8192)  # Read first 8KB
            
            if not sample:
                return ','
            
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                return dialect.delimiter
            except csv.Error:
                # Fallback to comma
                return ','
    
    def _parse_csv(self, file_path: Path, encoding: str, delimiter: str) -> tuple[list[list[str]], list[str]]:
        """Parse CSV file and return rows and headers."""
        rows = []
        headers = []
        
        with open(file_path, 'r', encoding=encoding, errors='replace', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            for i, row in enumerate(reader):
                # Clean up cells
                row = [cell.strip() for cell in row]
                
                if i == 0:
                    # First row as headers
                    headers = row
                else:
                    # Pad row with empty strings if needed
                    while len(row) < len(headers):
                        row.append('')
                    rows.append(row)
        
        return rows, headers
