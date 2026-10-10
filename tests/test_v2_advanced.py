"""Unit and Integration Tests for EvidenceFirst-RAG v2.0 Advanced Features:

1. Agentic Iterative Verification Trajectory
2. Automated Scientific Hypothesis Generation
3. Methodological Rigor & Evidence Hierarchy Analyzer
4. Multi-Stage Sankey & Physics Network Visualizers
5. New REST API Endpoints
"""

import pytest
from fastapi.testclient import TestClient

from src.models import (
    Paper,
    SemanticChunk,
    FactualClaim,
    EvidenceItem,
    ConflictAttribution,
)
from src.verification.agentic_verifier import AgenticVerifier
from src.synthesis.hypothesis_generator import ScientificHypothesisGenerator
from src.verification.rigor_analyzer import MethodologyRigorAnalyzer
from src.claims.graph_visualizer import EvidenceGraphVisualizer
from src.ingestion.paper_loader import PaperLoader
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_agentic_verifier_loop():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    verifier = pipeline.agentic_verifier

    # Claim with conflicting status
    claim = FactualClaim(
        claim_id="C_TEST",
        statement="Vision Transformers suffer performance degradation under extreme low-light on DarkFace.",
        evidence_status="Conflicting",
        confidence_score=0.55,
    )
    claim.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_test_1", chunk_id="chk_test_1", paper_id="paper_002",
            paper_title="Paper 002", section="Results", year=2025,
            text_snippet="Vision transformers face quadratic degradation in dark noise.",
            composite_score=0.60, relationship="SUPPORTS",
        )
    )

    claims, trajectory = verifier.run_verification_loop([claim], max_iterations=1)

    assert len(claims) == 1
    assert trajectory.iterations_performed == 1
    assert len(trajectory.steps) >= 1
    step = trajectory.steps[0]
    assert step.claim_id == "C_TEST"
    assert len(step.query_generated) > 0
    assert step.retrieved_chunks_count >= 0
    assert len(step.reasoning) > 0


def test_hypothesis_generator():
    generator = ScientificHypothesisGenerator()
    conflict = ConflictAttribution(
        claim_id="C01",
        claim_statement="Illumination preprocessing improves object detection in dark images.",
        likely_reason="DarkFace extreme darkness versus ExDark urban night natural lighting.",
        divergent_factors=[{"factor": "Dataset Lux", "supporting_side": "DarkFace (<0.1 lux)", "contradicting_side": "ExDark (>1.0 lux)"}],
    )

    hypotheses = generator.generate_hypotheses(
        query="What are the limitations of Vision Transformers for low-light object detection?",
        conflicts=[conflict],
        claims=[],
    )

    assert len(hypotheses) >= 1
    h = hypotheses[0]
    assert "HYP-" in h.hypothesis_id
    assert len(h.title) > 0
    assert len(h.statement) > 0
    assert len(h.rationale) > 0
    assert len(h.proposed_methodology) > 0
    assert len(h.testable_predictions) > 0
    assert len(h.falsification_criteria) > 0
    assert h.feasibility_tier in ["High", "Medium", "Exploratory"]
    assert h.novelty_impact_tier in ["Breakthrough", "High", "Incremental"]
    assert 0.0 <= h.estimated_impact_score <= 1.0


def test_methodology_rigor_analyzer():
    analyzer = MethodologyRigorAnalyzer()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    assert len(papers) >= 3

    # Assess corpus
    report = analyzer.analyze_corpus(papers)
    assert 0.0 <= report.overall_corpus_rigor <= 1.0
    assert 0.0 <= report.high_rigor_paper_ratio <= 1.0
    assert len(report.assessments) == len(papers)
    assert len(report.methodology_distribution) > 0

    first = report.assessments[0]
    assert first.evidence_weight_multiplier >= 0.8
    assert len(first.strengths) > 0

    # Test evidence weighting adjustment
    dummy_ev = EvidenceItem(
        evidence_id="ev_rigor_1", chunk_id="chk_1", paper_id=first.paper_id,
        paper_title=first.paper_title, section="Results", year=first.year,
        text_snippet="Test snippet", composite_score=0.70,
    )
    assessments_map = {a.paper_id: a for a in report.assessments}
    adjusted = analyzer.apply_rigor_weighting([dummy_ev], assessments_map)
    assert adjusted[0].composite_score > 0.0


def test_graph_visualizer():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)
    report = pipeline.run("What are the limitations of Vision Transformers for low-light object detection?")

    visualizer = pipeline.graph_visualizer

    # 1. Sankey data and HTML
    sankey_data = visualizer.build_sankey_data(report)
    assert len(sankey_data["nodes"]) > 0
    assert len(sankey_data["sources"]) == len(sankey_data["targets"])
    assert len(sankey_data["values"]) == len(sankey_data["sources"])

    sankey_html = visualizer.generate_sankey_html(report)
    assert "<!DOCTYPE html>" in sankey_html
    assert "plotly" in sankey_html
    assert "sankey" in sankey_html

    # 2. Physics Network HTML
    net_html = visualizer.generate_network_html(report)
    assert "<!DOCTYPE html>" in net_html
    assert "vis-network" in net_html
    assert "query_root" in net_html


def test_v2_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. Hypotheses endpoint
        res_hypo = client.post("/api/v1/hypotheses", json={"query": "Vision Transformers in extreme low-light"})
        assert res_hypo.status_code == 200
        data_hypo = res_hypo.json()
        assert data_hypo["hypotheses_count"] >= 1
        assert "HYP-" in data_hypo["hypotheses"][0]["hypothesis_id"]

        # 2. Rigor endpoint
        res_rigor = client.get("/api/v1/rigor/assess")
        assert res_rigor.status_code == 200
        data_rigor = res_rigor.json()
        assert "overall_corpus_rigor" in data_rigor
        assert len(data_rigor["assessments"]) >= 3

        # 3. Iterative verification endpoint
        res_iter = client.post("/api/v1/verify/iterative", json={"query": "Vision Transformers in extreme low-light"})
        assert res_iter.status_code == 200
        data_iter = res_iter.json()
        assert "claims" in data_iter

        # 4. Sankey visualization endpoint
        res_sankey = client.post("/api/v1/visualize/sankey", json={"query": "Vision Transformers in low-light"})
        assert res_sankey.status_code == 200
        data_sankey = res_sankey.json()
        assert "sankey_data" in data_sankey
        assert "<!DOCTYPE html>" in data_sankey["html_code"]

        # 5. Network visualization endpoint
        res_net = client.post("/api/v1/visualize/network", json={"query": "Vision Transformers in low-light"})
        assert res_net.status_code == 200
        data_net = res_net.json()
        assert data_net["nodes_count"] > 0
        assert "<!DOCTYPE html>" in data_net["html_code"]
