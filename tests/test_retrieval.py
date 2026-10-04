"""Tests for BM25, Dense, and Hybrid Retrieval."""

import pytest
from src.models import SemanticChunk
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


@pytest.fixture
def sample_chunks():
    return [
        SemanticChunk(
            chunk_id="chunk_01",
            paper_id="paper_01",
            paper_title="ViT for Low Light",
            section="Experiments",
            year=2024,
            text="Vision Transformers achieve 44.2% mAP on the DarkFace benchmark under extreme low illumination.",
            dataset="DarkFace",
            model_architecture="ViT",
            evaluation_metric="mAP@0.5"
        ),
        SemanticChunk(
            chunk_id="chunk_02",
            paper_id="paper_02",
            paper_title="YOLO Real Time",
            section="Results",
            year=2026,
            text="YOLOv8 runs at 68.4 FPS on edge Jetson hardware with low power consumption.",
            dataset="NightOwls",
            model_architecture="YOLOv8",
            evaluation_metric="FPS"
        ),
        SemanticChunk(
            chunk_id="chunk_03",
            paper_id="paper_03",
            paper_title="Illumination Enhancement",
            section="Abstract",
            year=2025,
            text="Illumination preprocessing provides no significant improvement on ExDark dataset.",
            dataset="ExDark",
            model_architecture="Faster R-CNN",
            evaluation_metric="mAP@0.5:0.95"
        ),
    ]


def test_bm25_retrieval(sample_chunks):
    bm25 = BM25Retriever(sample_chunks)
    results = bm25.search("DarkFace low illumination", top_k=2)
    assert len(results) > 0
    assert results[0][0].chunk_id == "chunk_01"


def test_hybrid_retrieval(sample_chunks):
    bm25 = BM25Retriever(sample_chunks)
    dense = DenseRetriever()
    dense.index_chunks(sample_chunks)

    hybrid = HybridRetriever(bm25, dense, alpha=0.5)
    candidates = hybrid.retrieve("real-time edge FPS", top_candidates=3)
    assert len(candidates) > 0
    assert any(c.chunk.chunk_id == "chunk_02" for c in candidates)
