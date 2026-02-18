import pytest
from pathlib import Path
import csv
import os
import fitz  # PyMuPDF
from docx import Document
from app.pipeline import Pipeline
from app.retrieval.retriever import Retriever
from app.logging_config import setup_logging

# Ensure logging is setup for tests
setup_logging()

class TestRealWorldMessiness:
    """Test the pipeline with realistic, messy documents."""
    
    @pytest.fixture(scope="class")
    def pipeline(self):
        return Pipeline()
    
    @pytest.fixture(scope="class")
    def retriever(self):
        return Retriever()
    
    @pytest.fixture(scope="class")
    def messy_data_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("messy_data")

    def test_complex_pdf(self, pipeline, retriever, messy_data_dir):
        """1. Complex PDF: multi-column, headers, footers."""
        pdf_path = messy_data_dir / "complex_report.pdf"
        
        # Create a complex PDF using fitz
        doc = fitz.open()
        
        for p in range(3):
            page = doc.new_page()
            # Header
            page.insert_text((50, 30), f"Ragwell Confidential Report - Page {p+1}", fontsize=8)
            
            # Left Column
            rect1 = fitz.Rect(50, 60, 280, 700)
            page.insert_textbox(rect1, "The quick brown fox jumps over the lazy dog. " * 50)
            
            # Right Column
            rect2 = fitz.Rect(310, 60, 540, 700)
            page.insert_textbox(rect2, "RAG (Retrieval-Augmented Generation) is a technique for enhancing LLM outputs. " * 40)
            
            # Footer
            page.insert_text((50, 750), "Copyright (c) 2025 Ragwell Systems", fontsize=8)
        
        doc.save(str(pdf_path))
        doc.close()
        
        self._run_pipeline_check(pipeline, retriever, pdf_path, "Complex PDF")

    def test_long_docx(self, pipeline, retriever, messy_data_dir):
        """2. Long DOCX: nested headings, lists, tables."""
        docx_path = messy_data_dir / "long_manual.docx"
        doc = Document()
        doc.add_heading('Technical Specification v1.0', 0)
        
        for i in range(1, 4):
            doc.add_heading(f'Section {i}: Introduction', level=1)
            doc.add_paragraph(f'This is the main introduction for section {i}. ' * 10)
            
            doc.add_heading(f'Subsection {i}.1: Details', level=2)
            doc.add_paragraph('Supporting details here with a list:')
            doc.add_paragraph('Primary feature alpha', style='List Bullet')
            doc.add_paragraph('Secondary feature beta', style='List Bullet')
            
            doc.add_heading(f'Deep dive {i}.1.1', level=3)
            doc.add_paragraph('More technical depth here. ' * 100) # Ensure it's long
            
        table = doc.add_table(rows=3, cols=3)
        for r in range(3):
            for c in range(3):
                table.rows[r].cells[c].text = f"Data {r},{c}"
                
        doc.save(str(docx_path))
        
        self._run_pipeline_check(pipeline, retriever, docx_path, "Long DOCX")

    def test_messy_csv(self, pipeline, retriever, messy_data_dir):
        """3. Messy CSV: empty rows, special chars, quoted fields."""
        csv_path = messy_data_dir / "messy_sales.csv"
        
        content = [
            ["ID", "Product", "Price", "Description"],
            ["1", "Widget A", "10.00", "Standard widget"],
            ["", "", "", ""], # Empty row
            ["2", "Widget B", "$15.99", "Special \"Quoted\" Name"],
            ["3", "Widget C", "20.50", "Multi-line\nDescription"],
            ["4", "Gäme Station", "299,00", "Unicode characters"],
            [",,,,", "", "", ""] # Junk row
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(content)
            
        self._run_pipeline_check(pipeline, retriever, csv_path, "Messy CSV")

    def test_boilerplate_html(self, pipeline, retriever, messy_data_dir):
        """4. Boilerplate HTML: nav, sidebar, footer wrapping article."""
        html_path = messy_data_dir / "article.html"
        
        html_content = """
        <html>
            <head><title>The Future of AI</title></head>
            <body>
                <nav><ul><li>Home</li><li>About</li></ul></nav>
                <aside><div class="ad">Buy our product!</div></aside>
                <main>
                    <h1>The Future of AI</h1>
                    <p>Artificial Intelligence is evolving rapidly. Large Language Models are changing how we interact with technology.</p>
                    <p>RAG pipelines allow models to access private data without retraining, reducing hallucinations effectively.</p>
                    <p>Vector databases are the storage backbone of modern AI applications.</p>
                </main>
                <footer>&copy; 2025 News Corp. All rights reserved.</footer>
            </body>
        </html>
        """
        html_path.write_text(html_content)
        
        self._run_pipeline_check(pipeline, retriever, html_path, "Boilerplate HTML")

    def test_weird_txt(self, pipeline, retriever, messy_data_dir):
        """5. Weird TXT: mixed line endings, BOM, unicode."""
        txt_path = messy_data_dir / "weird.txt"
        
        # UTF-8 with BOM
        bom = b'\xef\xbb\xbf'
        content = "Line 1: Unix Ending\nLine 2: Windows Ending\r\nLine 3: Legacy Mac\rLine 4: Unicode \u2728 Stars"
        
        with open(txt_path, 'wb') as f:
            f.write(bom)
            f.write(content.encode('utf-8'))
            
        self._run_pipeline_check(pipeline, retriever, txt_path, "Weird TXT")

    def _run_pipeline_check(self, pipeline, retriever, filepath, label):
        """Helper to run and assert pipeline stats."""
        print(f"\n--- Testing {label} ---")
        
        # 1. Process document
        result = pipeline.process_document(filepath)
        assert result.success, f"{label} failed: {result.error}"
        assert result.chunk_count >= 3, f"{label} produced only {result.chunk_count} chunks"
        
        # 2. Verify searchability
        # Use a term likely found in the generated content
        search_query = "RAG" if label != "Messy CSV" else "Widget"
        retrieval = retriever.retrieve(search_query, strategy='hybrid', top_k=5)
        
        assert len(retrieval.results) > 0, f"{label}: search for '{search_query}' returned no results"
        
        print(f"STATUS: PASS")
        print(f"Chunks: {result.chunk_count}")
        print(f"Time: {result.processing_time_seconds}s")
        print(f"Best Search Score: {retrieval.results[0].score:.4f}")
        print(f"Top result: {retrieval.results[0].text[:100]}...")
