"""End-to-end integration test for EvidenceFirst-RAG pipeline."""

import pytest
from src.pipeline import EvidenceFirstPipeline
from src.ingestion.paper_loader import PaperLoader
from src.evaluation.benchmark import BenchmarkRunner


def test_end_to_end_pipeline():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    assert len(papers) >= 3

    pipeline.index_papers(papers)

    query = "What are the limitations of Vision Transformers for low-light object detection?"
    report = pipeline.run(query)

    assert report is not None
    assert len(report.claims) > 0
    assert report.citation_audit.total_claims > 0
    assert report.citation_audit.evidence_coverage_score >= 0.0
    assert len(report.references) > 0


def test_ablation_benchmarking():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    benchmarker = BenchmarkRunner()
    results = benchmarker.run_ablation_study(pipeline, questions=["Test query on vision transformers."])

    assert len(results) == 5
    assert results[0]["system"] == "Basic Dense RAG"
    assert results[4]["system"] == "EvidenceFirst-RAG (Full Framework)"
    # EvidenceFirst should achieve highest citation accuracy and conflict detection rate
    assert results[4]["conflict_detection_rate"] > results[0]["conflict_detection_rate"]
    assert results[4]["citation_accuracy"] > results[0]["citation_accuracy"]
