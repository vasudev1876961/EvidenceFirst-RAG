"""Paper loader: loads, converts, and standardizes scientific papers into Paper models."""

from __future__ import annotations
import os
import json
import logging
from typing import List, Optional
from src.models import Paper
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.section_parser import SectionParser

logger = logging.getLogger(__name__)


class PaperLoader:
    """Loads papers from local directories or structured JSON/PDF files."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.pdf_parser = PDFParser(SectionParser())

    def load_paper_file(self, filepath: str, paper_id: Optional[str] = None) -> Paper:
        """Load a single file (.pdf, .json, .txt) and return a Paper object."""
        ext = os.path.splitext(filepath)[1].lower()
        base_id = paper_id or os.path.splitext(os.path.basename(filepath))[0]

        if ext == ".pdf":
            parsed = self.pdf_parser.parse_pdf(filepath)
            abstract = parsed["sections"].get("Abstract", parsed["raw_text"][:400])
            return Paper(
                paper_id=base_id,
                title=parsed.get("title") or base_id,
                authors=[],
                abstract=abstract,
                year=parsed.get("year", 2024),
                sections=parsed.get("sections", {}),
                metadata=parsed.get("metadata", {}),
            )

        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "paper_id" not in data:
                data["paper_id"] = base_id
            return Paper(**data)

        elif ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            parser = SectionParser()
            sections = parser.parse(content)
            title_line = content.strip().split("\n")[0] if content.strip() else base_id
            abstract = sections.get("Abstract", content[:300])
            return Paper(
                paper_id=base_id,
                title=title_line,
                abstract=abstract,
                year=2024,
                sections=sections,
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_all_from_directory(self, directory: Optional[str] = None) -> List[Paper]:
        """Load all papers from directory."""
        target_dir = directory or self.raw_dir
        if not os.path.exists(target_dir):
            return []

        papers: List[Paper] = []
        for filename in sorted(os.listdir(target_dir)):
            filepath = os.path.join(target_dir, filename)
            if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in [".pdf", ".json", ".txt", ".md"]:
                try:
                    paper = self.load_paper_file(filepath)
                    papers.append(paper)
                    logger.info(f"Loaded paper: {paper.paper_id} - '{paper.title}'")
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        return papers

    def save_processed_papers(self, papers: List[Paper], output_dir: Optional[str] = None) -> str:
        """Save structured papers to processed JSON directory."""
        target = output_dir or self.processed_dir
        os.makedirs(target, exist_ok=True)
        out_file = os.path.join(target, "papers_catalog.json")

        serializable = [p.model_dump() for p in papers]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        return out_file

    def load_processed_papers(self, catalog_path: Optional[str] = None) -> List[Paper]:
        """Load previously processed papers catalog."""
        path = catalog_path or os.path.join(self.processed_dir, "papers_catalog.json")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Paper(**item) for item in data]
