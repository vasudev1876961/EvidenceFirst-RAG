"""PDF Parser for scientific literature extraction using PyMuPDF or pdfplumber."""

from __future__ import annotations
import os
import re
import logging
from typing import Dict, Any, List, Optional
from src.ingestion.section_parser import SectionParser

logger = logging.getLogger(__name__)


class PDFParser:
    """Extracts text, metadata, and sections from PDF files."""

    def __init__(self, section_parser: Optional[SectionParser] = None):
        self.section_parser = section_parser or SectionParser()

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a PDF file into metadata and sections."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        extracted_text = ""
        metadata: Dict[str, Any] = {}

        # 1. Try PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(pdf_path)
            meta = doc.metadata or {}
            metadata["title"] = meta.get("title", "")
            metadata["author"] = meta.get("author", "")
            metadata["creation_date"] = meta.get("creationDate", "")

            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            extracted_text = "\n".join(pages_text)
            doc.close()
        except ImportError:
            logger.info("fitz (PyMuPDF) not found. Trying pdfplumber...")
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    pages_text = [p.extract_text() or "" for p in pdf.pages]
                    extracted_text = "\n".join(pages_text)
            except ImportError:
                raise ImportError("Neither pymupdf nor pdfplumber is installed. Please install one: pip install pymupdf")

        # Fallback title if not in PDF metadata
        filename = os.path.basename(pdf_path).replace(".pdf", "")
        if not metadata.get("title"):
            first_lines = [l.strip() for l in extracted_text.split("\n") if l.strip()][:3]
            metadata["title"] = first_lines[0] if first_lines else filename

        # Estimate year from text or filename
        year_match = re.search(r"(?:19|20)\d{2}", extracted_text[:1000] + " " + filename)
        year = int(year_match.group(0)) if year_match else 2024
        metadata["year"] = year

        # Section parsing
        sections = self.section_parser.parse(extracted_text)

        return {
            "title": metadata["title"],
            "year": year,
            "raw_text": extracted_text,
            "sections": sections,
            "metadata": metadata,
        }
