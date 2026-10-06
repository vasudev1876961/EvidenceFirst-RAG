"""Unit and Integration Tests for Advanced EvidenceFirst-RAG Capabilities."""

import pytest
from fastapi.testclient import TestClient

from src.models import Paper, FactualClaim, EvidenceItem, SemanticChunk
from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.synthesis.meta_analysis import EmpiricalMetaAnalysisEngine
from src.verification.robustness import RobustnessEngine
from src.synthesis.latex_exporter import LatexExporter
from src.pipeline import EvidenceFirstPipeline
from app.api import app, lifespan


def test_arxiv_fetcher_offline():
    fetcher = ArxivFetcher(offline_mode=True)
    papers = fetcher.search("vision transformer low light", max_results=2)
    assert len(papers) == 2
    assert "arxiv_" in papers[0].paper_id
    assert "Abstract" in papers[0].sections
    assert "Results" in papers[0].sections
    assert len(papers[0].authors) > 0

    paper_by_id = fetcher.fetch_by_id("2405.08912")
    assert paper_by_id is not None
    assert "arxiv_" in paper_by_id.paper_id


def test_empirical_meta_analysis():
    engine = EmpiricalMetaAnalysisEngine()
    dummy_claim = FactualClaim(
        claim_id="C001",
        statement="Vision Transformers achieve 44.2% mAP on DarkFace.",
    )
    meta = engine.extract_from_papers_and_claims(papers=[], claims=[dummy_claim])

    assert len(meta.records) > 0
    assert len(meta.dataset_comparison_table) > 0
    assert len(meta.model_leaderboard) > 0
    assert len(meta.key_empirical_insights) > 0
    assert "DarkFace" in meta.cross_dataset_variance


def test_robustness_leave_one_out():
    engine = RobustnessEngine()

    claim = FactualClaim(
        claim_id="C1",
        statement="Illumination enhancement improves detection.",
        evidence_status="Conflicting",
        confidence_score=0.75,
    )
    claim.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_1", chunk_id="chk_1", paper_id="paper_002",
            paper_title="Paper 002", section="Results", year=2025,
            text_snippet="Enhancement improves mAP by 8.2%",
            composite_score=0.85, relationship="SUPPORTS",
        )
    )
    claim.contradicting_evidence.append(
        EvidenceItem(
            evidence_id="ev_2", chunk_id="chk_2", paper_id="paper_003",
            paper_title="Paper 003", section="Results", year=2025,
            text_snippet="Enhancement yields -0.4% regression on ExDark",
            composite_score=0.82, relationship="CONTRADICTS",
        )
    )

    report = engine.evaluate_robustness([claim])
    assert 0.0 <= report.evidence_robustness_index <= 1.0
    assert len(report.critical_papers) >= 1
    assert "paper_002" in [p["paper_id"] for p in report.critical_papers]
    assert len(report.counterfactual_scenarios) > 0


def test_latex_exporter():
    from src.ingestion.paper_loader import PaperLoader
    exporter = LatexExporter()
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)
    report = pipeline.run("What are the limitations of Vision Transformers for low-light object detection?")

    latex = exporter.generate_latex(report)
    assert "\\documentclass" in latex
    assert "\\begin{document}" in latex
    assert "\\begin{table}" in latex
    assert "\\end{document}" in latex

    bibtex = exporter.generate_bibtex(report)
    assert "@article{" in bibtex


def test_package_exports():
    from src.ingestion import ArxivFetcher as IngestionArxiv
    from src.synthesis import EmpiricalMetaAnalysisEngine as SynthesisMeta, LatexExporter as SynthesisLatex
    from src.verification import RobustnessEngine as VerificationRobustness

    assert IngestionArxiv is not None
    assert SynthesisMeta is not None
    assert SynthesisLatex is not None
    assert VerificationRobustness is not None


def test_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. Health
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["indexed_papers_count"] >= 3

        # 2. Corpus
        res_corpus = client.get("/api/v1/corpus")
        assert res_corpus.status_code == 200
        corpus_data = res_corpus.json()
        assert corpus_data["total_papers"] >= 3

        # 3. Domains
        res_domains = client.get("/api/v1/domains")
        assert res_domains.status_code == 200
        domain_data = res_domains.json()
        assert len(domain_data["domains"]) >= 1

        # 4. Arxiv Search
        res_arxiv = client.post("/api/v1/arxiv/search", json={"query": "vision transformers", "max_results": 2, "auto_index": False})
        assert res_arxiv.status_code == 200
        arxiv_data = res_arxiv.json()
        assert len(arxiv_data["papers"]) > 0

        # 5. Query Analysis with Domain Filter
        res_query = client.post("/api/v1/query", json={"query": "Vision Transformers in extreme low-light", "alpha": 0.5, "domain": "Computer Vision / Deep Learning"})
        assert res_query.status_code == 200
        q_data = res_query.json()
        assert "executive_summary" in q_data
        assert "claims" in q_data
        assert "evidence_robustness_index" in q_data

        # 6. HTML Export
        res_html = client.post("/api/v1/export/html", json={"query": "Vision Transformers in low-light", "alpha": 0.5})
        assert res_html.status_code == 200
        html_data = res_html.json()
        assert "<!DOCTYPE html>" in html_data["html_code"]

        # 7. IR Benchmark
        res_ir = client.post("/api/v1/benchmark/ir?domain=Computer%20Vision%20%2F%20Deep%20Learning")
        assert res_ir.status_code == 200
        ir_data = res_ir.json()
        assert "macro_mrr" in ir_data
        assert ir_data["macro_mrr"] > 0.0

