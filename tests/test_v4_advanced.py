"""Unit and Integration Tests for EvidenceFirst-RAG v4.0 Advanced Autonomous Meta-Scientist Capabilities:

1. Causal Directed Acyclic Graph (DAG) & Do-Calculus Counterfactual Reasoning
2. Autonomous Open Science & Reproducibility Protocol Verifier
3. Temporal Knowledge Horizon & Living Research Roadmap
4. Autonomous Research Dossier & Public Press Briefing Generator
5. v4.0 REST API Endpoints
6. Module and Package Export Integrity
"""

import pytest
from fastapi.testclient import TestClient

from src.models import (
    Paper,
    FactualClaim,
    EvidenceItem,
    ConflictAttribution,
    CitationAudit,
    EvidenceGraphData,
    SynthesisReport,
    CausalDAGReport,
    CorpusReproducibilityReport,
    TemporalRoadmapReport,
    AutonomousResearchDossier,
)
from src.claims.causal_graph import CausalGraphEngine
from src.verification.reproducibility import ReproducibilityVerifier
from src.synthesis.temporal_roadmap import TemporalRoadmapEngine
from src.synthesis.dossier_generator import DossierGenerator
from src.ingestion.paper_loader import PaperLoader
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_causal_graph_engine():
    engine = CausalGraphEngine()
    
    claims = [
        FactualClaim(
            claim_id="C1",
            statement="Quadratic self-attention in Vision Transformers degrades under severe shot noise.",
            evidence_status="Strong support",
            confidence_score=0.91,
        ),
        FactualClaim(
            claim_id="C2",
            statement="Zero-DCE illumination enhancement improves detection by +8.2% on DarkFace.",
            evidence_status="Conflicting",
            confidence_score=0.75,
        ),
    ]
    conflicts = [
        ConflictAttribution(
            claim_id="C2",
            claim_statement="Zero-DCE illumination enhancement improves detection.",
            likely_reason="Benchmark lux disparity (<0.1 lux on DarkFace vs >1.0 lux on ExDark).",
        )
    ]

    causal_report = engine.build_causal_dag(
        query="What are the limitations of Vision Transformers for low-light object detection?",
        claims=claims,
        conflicts=conflicts,
    )

    assert isinstance(causal_report, CausalDAGReport)
    assert len(causal_report.nodes) >= 4
    assert len(causal_report.edges) >= 3
    assert 0.0 <= causal_report.overall_causal_soundness_score <= 1.0
    assert 0.0 <= causal_report.confounding_vulnerability_index <= 1.0
    assert len(causal_report.spurious_correlations_detected) >= 1
    assert len(causal_report.interventional_insights) >= 2
    assert "graph LR" in causal_report.mermaid_diagram

    # Check that variables have distinct roles (treatment, outcome, confounder, mediator)
    types_found = {n.variable_type for n in causal_report.nodes}
    assert "treatment" in types_found
    assert "outcome" in types_found
    assert "confounder" in types_found


def test_reproducibility_verifier():
    verifier = ReproducibilityVerifier()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    assert len(papers) >= 3

    # Audit corpus
    corpus_report = verifier.audit_corpus(papers)

    assert isinstance(corpus_report, CorpusReproducibilityReport)
    assert corpus_report.corpus_size == len(papers)
    assert 0.0 <= corpus_report.mean_reproducibility_risk_index <= 1.0
    assert 0.0 <= corpus_report.open_code_ratio <= 1.0
    assert 0.0 <= corpus_report.open_data_ratio <= 1.0
    assert corpus_report.overall_open_science_grade in ["A+", "A", "B", "C", "D", "F"]
    assert len(corpus_report.paper_audits) == len(papers)
    assert len(corpus_report.critical_reproducibility_bottlenecks) >= 1
    assert len(corpus_report.remediation_patch_protocol) >= 1

    # Audit single paper
    p_audit = corpus_report.paper_audits[0]
    assert len(p_audit.paper_id) > 0
    assert p_audit.open_science_grade in ["A", "B", "C", "D", "F"]
    assert p_audit.gpu_tpu_footprint_hours is not None
    assert len(p_audit.audit_notes) >= 1


def test_temporal_roadmap_engine():
    engine = TemporalRoadmapEngine()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")

    roadmap = engine.build_roadmap(
        query="Vision Transformers in low-light object detection",
        papers=papers,
    )

    assert isinstance(roadmap, TemporalRoadmapReport)
    assert len(roadmap.paradigm_trends) >= 3
    assert len(roadmap.decaying_approaches) >= 1
    assert len(roadmap.emerging_frontiers) >= 1
    assert len(roadmap.projected_milestones) >= 2
    assert len(roadmap.strategic_takeaways) >= 1

    # Check milestone attributes
    m1 = roadmap.projected_milestones[0]
    assert m1.milestone_id.startswith("MS-")
    assert len(m1.technical_objective) > 0
    assert len(m1.success_metric) > 0
    assert len(m1.strategic_grant_or_patent_relevance) > 0


def test_dossier_generator():
    generator = DossierGenerator()

    dummy_audit = CitationAudit(
        total_claims=3,
        supported_claims=3,
        conflicting_claims=1,
        unsupported_claims=0,
        evidence_coverage_score=1.0,
        citation_integrity_score=1.0,
        citation_faithfulness=1.0,
    )
    dummy_report = SynthesisReport(
        query="What are the limitations of Vision Transformers for low-light object detection?",
        decomposed_subqueries=["SubQ 1", "SubQ 2"],
        overall_confidence=0.90,
        executive_summary="Vision Transformers suffer under severe photon starvation.",
        citation_audit=dummy_audit,
        evidence_graph=EvidenceGraphData(),
        claims=[
            FactualClaim(claim_id="C1", statement="ViTs exhibit degradation under noise.", evidence_status="Strong support", confidence_score=0.92),
            FactualClaim(claim_id="C2", statement="Enhancement amplifies Poisson noise.", evidence_status="Conflicting", confidence_score=0.85),
        ],
    )

    dossier = generator.build_dossier(dummy_report)

    assert isinstance(dossier, AutonomousResearchDossier)
    assert dossier.dossier_id.startswith("DOSSIER-")
    assert len(dossier.executive_lead_summary) > 0
    assert len(dossier.layman_public_briefing) > 0
    assert len(dossier.technical_breakthrough_table) >= 2
    assert len(dossier.scientific_myths_debunked) >= 2
    assert len(dossier.two_year_actionable_mandate) >= 3
    assert "<!DOCTYPE html>" in dossier.html_dossier
    assert "Autonomous Research Dossier" in dossier.html_dossier
    assert "# Autonomous Research Dossier" in dossier.markdown_dossier


def test_v4_pipeline_end_to_end():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    report = pipeline.run("What are the limitations of Vision Transformers for low-light object detection?")

    # Verify v4.0 components are populated in SynthesisReport
    assert report.causal_dag is not None
    assert len(report.causal_dag.nodes) >= 3
    assert len(report.causal_dag.edges) >= 2

    assert report.reproducibility is not None
    assert report.reproducibility.corpus_size >= 3
    assert report.reproducibility.mean_reproducibility_risk_index >= 0.0

    assert report.temporal_roadmap is not None
    assert len(report.temporal_roadmap.paradigm_trends) >= 3
    assert len(report.temporal_roadmap.projected_milestones) >= 2

    assert report.dossier is not None
    assert "<!DOCTYPE html>" in report.dossier.html_dossier
    assert len(report.dossier.markdown_dossier) > 0


def test_v4_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. Causal DAG endpoint
        res_causal = client.post("/api/v1/causal/dag", json={"query": "Vision Transformers in low-light"})
        assert res_causal.status_code == 200
        data_causal = res_causal.json()
        assert "nodes" in data_causal
        assert "edges" in data_causal
        assert len(data_causal["nodes"]) >= 3
        assert data_causal["overall_causal_soundness_score"] >= 0.50

        # 2. Reproducibility audit endpoint
        res_repro = client.get("/api/v1/audit/reproducibility")
        assert res_repro.status_code == 200
        data_repro = res_repro.json()
        assert data_repro["corpus_size"] >= 3
        assert "mean_reproducibility_risk_index" in data_repro
        assert len(data_repro["paper_audits"]) >= 3

        # 3. Temporal roadmap endpoint
        res_roadmap = client.get("/api/v1/roadmap/temporal")
        assert res_roadmap.status_code == 200
        data_roadmap = res_roadmap.json()
        assert "paradigm_trends" in data_roadmap
        assert len(data_roadmap["paradigm_trends"]) >= 3
        assert len(data_roadmap["projected_milestones"]) >= 2

        # 4. Dossier export endpoint
        res_dossier = client.post("/api/v1/export/dossier", json={"query": "Vision Transformers in low-light"})
        assert res_dossier.status_code == 200
        data_dossier = res_dossier.json()
        assert "dossier_id" in data_dossier
        assert "<!DOCTYPE html>" in data_dossier["html_dossier"]
        assert len(data_dossier["markdown_dossier"]) > 0


def test_package_exports_v4():
    from src.claims import CausalGraphEngine as ExportCausal
    from src.verification import ReproducibilityVerifier as ExportRepro
    from src.synthesis import TemporalRoadmapEngine as ExportRoadmap, DossierGenerator as ExportDossier
    import src

    assert ExportCausal is not None
    assert ExportRepro is not None
    assert ExportRoadmap is not None
    assert ExportDossier is not None
    assert src.__version__ == "0.4.0"
