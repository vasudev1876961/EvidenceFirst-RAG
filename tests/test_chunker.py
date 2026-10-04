"""Tests for Section-aware Semantic Chunker."""

import pytest
from src.models import Paper
from src.preprocessing.chunker import SemanticChunker


def test_chunker_section_preservation():
    paper = Paper(
        paper_id="test_001",
        title="Test Vision Transformer in Low-Light",
        abstract="Test abstract for low light vision.",
        year=2025,
        sections={
            "Introduction": "Object detection in low-light environments is challenging for computer vision models. " * 15,
            "Experiments": "We benchmarked on DarkFace and ExDark datasets using mAP@0.5 metric. " * 15,
        }
    )

    chunker = SemanticChunker(target_chunk_size=100, chunk_overlap=15)
    chunks = chunker.chunk_paper(paper)

    assert len(chunks) >= 2
    assert any(c.section == "Introduction" for c in chunks)
    assert any(c.section == "Experiments" for c in chunks)

    # Check entity extraction
    exp_chunks = [c for c in chunks if c.section == "Experiments"]
    assert exp_chunks[0].dataset in ["DarkFace", "ExDark"]
    assert exp_chunks[0].evaluation_metric == "mAP@0.5"
