"""Unit and Integration Tests for Multi-Domain Corpora and IR Benchmarking."""

import pytest
from src.pipeline import EvidenceFirstPipeline
from src.ingestion.paper_loader import PaperLoader
from src.evaluation.benchmark import BenchmarkRunner


def test_multi_domain_paper_loading():
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    assert len(papers) >= 10

    domains = {p.metadata.get("domain") for p in papers if "domain" in p.metadata}
    assert "Computer Vision / Deep Learning" in domains
    assert "NLP / LLMs (ACL)" in domains
    assert "Biomedical Literature (PubMed)" in domains


def test_domain_filtering_pipeline():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    
    # 1. Index all papers
    pipeline.index_papers(papers)
    assert len(pipeline.indexed_papers) >= 10
    total_chunks = len(pipeline.indexed_chunks)

    # 2. Filter to Computer Vision
    pipeline.filter_by_domain("Computer Vision / Deep Learning")
    assert len(pipeline.indexed_papers) == 5
    assert len(pipeline.indexed_chunks) < total_chunks

    # 3. Filter to NLP
    pipeline.filter_by_domain("NLP / LLMs (ACL)")
    assert len(pipeline.indexed_papers) == 3

    # 4. Filter to Biomedical
    pipeline.filter_by_domain("Biomedical Literature (PubMed)")
    assert len(pipeline.indexed_papers) == 2


def test_benchmark_queries_loading_and_ir_eval():
    benchmarker = BenchmarkRunner()
    queries = benchmarker.load_benchmark_queries()
    assert len(queries) >= 6

    cv_queries = benchmarker.load_benchmark_queries(domain="Computer Vision / Deep Learning")
    assert len(cv_queries) >= 3

    nlp_queries = benchmarker.load_benchmark_queries(domain="NLP / LLMs (ACL)")
    assert len(nlp_queries) >= 2

    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    # Run IR benchmark on CV domain
    ir_results = benchmarker.run_ir_benchmark(pipeline, domain="Computer Vision / Deep Learning", top_k=3)
    assert "macro_mrr" in ir_results
    assert "macro_ndcg_at_k" in ir_results
    assert "macro_precision_at_k" in ir_results
    assert "macro_recall_at_k" in ir_results
    assert ir_results["macro_mrr"] > 0.0
    assert ir_results["evaluated_queries_count"] >= 3
