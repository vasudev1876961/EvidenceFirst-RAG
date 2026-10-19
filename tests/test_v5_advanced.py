"""Unit and Integration Tests for EvidenceFirst-RAG v5.0 Frontier Scientific Intelligence Capabilities:

1. In Silico Laboratory Experiment Simulator & Monte Carlo Benchmarking
2. Cross-Domain Conceptual Transfer & Interdisciplinary Analogy Engine
3. Academic Novelty, Prior Art Collision & Semantic Drift Sentinel
4. End-to-End Pipeline Integration for v5.0 Schemas
5. v5.0 REST API Endpoints
6. Module and Package Export Integrity
"""

import pytest
from fastapi.testclient import TestClient

from src.models import (
    Paper,
    FactualClaim,
    ScientificHypothesis,
    EmpiricalMetaAnalysis,
    SynthesisReport,
    InSilicoExperimentReport,
    InterdisciplinaryTransferReport,
    NoveltySentinelReport,
)
from src.synthesis.in_silico_simulator import InSilicoSimulator
from src.claims.analogy_engine import CrossDomainAnalogyEngine
from src.verification.novelty_sentinel import NoveltySentinel
from src.ingestion.paper_loader import PaperLoader
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_in_silico_simulator():
    simulator = InSilicoSimulator(random_seed=42)

    dummy_hyp = ScientificHypothesis(
        hypothesis_id="HYP-001",
        title="Wavelet-Fourier Attention Gating under Poisson Shot Noise",
        statement="Filtering high-frequency shot noise prior to query-key dot products eliminates quadratic degradation.",
        tension_addressed="DarkFace vs ExDark discrepancy",
        rationale="Decoupling spatial-frequency subbands preserves structural contours.",
        proposed_methodology="Dual-domain wavelet shrinkage attention layer.",
        falsification_criteria="Negative mAP delta on DarkFace benchmark.",
    )

    report = simulator.simulate(
        query="What are the limitations of Vision Transformers for low-light object detection?",
        hypothesis=dummy_hyp,
        trials_count=1000,
    )

    assert isinstance(report, InSilicoExperimentReport)
    assert report.simulation_id.startswith("SIM-")
    assert report.hypothesis_id == "HYP-001"
    assert report.trials_count == 1000
    assert report.mean_baseline_performance > 0.0
    assert report.mean_simulated_performance > report.mean_baseline_performance
    assert report.projected_net_gain_pct > 0.0
    assert len(report.confidence_interval_95) == 2
    assert report.confidence_interval_95[0] < report.confidence_interval_95[1]
    assert report.std_dev > 0.0

    # Power analysis assertions
    pa = report.power_analysis
    assert 0.80 <= pa.statistical_power_achieved <= 1.0
    assert pa.cohens_d_effect_size >= 0.50
    assert pa.hedges_g_effect_size >= 0.50
    assert pa.critical_sample_size_required > 10
    assert 0.0 <= pa.hypothesis_falsification_risk <= 0.40

    # Parameter sweeps and hardware projections
    assert len(report.parameter_sweeps) >= 4
    assert len(report.hardware_projections) >= 4
    assert len(report.sample_trials) >= 5

    # Artifact generators
    assert "torch" in report.executable_python_benchmark
    assert "FrequencyGuidedSparseAttention" in report.executable_python_benchmark
    assert "<!DOCTYPE html>" in report.interactive_lab_html
    assert "In Silico Laboratory Simulator" in report.interactive_lab_html


def test_cross_domain_analogy_engine():
    engine = CrossDomainAnalogyEngine()

    report = engine.discover_analogies(
        query="What are the limitations of Vision Transformers for low-light object detection?",
    )

    assert isinstance(report, InterdisciplinaryTransferReport)
    assert len(report.neighboring_domains_scanned) >= 4
    assert len(report.analogies) >= 4
    assert report.isomorphism_density_score >= 0.70
    assert report.top_recommended_transfer is not None
    assert len(report.interdisciplinary_takeaways) >= 2

    # Check that Biomedical Low-Dose CT analogy is captured
    ct_analogy = next((a for a in report.analogies if "Computed Tomography" in a.source_domain), None)
    assert ct_analogy is not None
    assert "Poisson" in ct_analogy.shared_mathematical_form
    assert "Wavelet" in ct_analogy.proposed_hybrid_architecture
    assert ct_analogy.novelty_score >= 0.85
    assert ct_analogy.feasibility_score >= 0.80
    assert len(ct_analogy.validation_protocol) > 0


def test_novelty_sentinel():
    sentinel = NoveltySentinel()

    report = sentinel.audit_novelty(
        query="What are the limitations of Vision Transformers for low-light object detection?",
    )

    assert isinstance(report, NoveltySentinelReport)
    assert 0.70 <= report.mean_novelty_index <= 1.0
    assert len(report.novelty_quadrant_distribution) == 4
    assert len(report.frontier_breakthroughs) >= 1
    assert len(report.speculative_frontiers) >= 1
    assert len(report.established_consensuses) >= 1

    # Prior Art Collisions
    assert len(report.prior_art_collisions) >= 3
    col1 = report.prior_art_collisions[0]
    assert len(col1.differentiation_argument) > 0
    assert col1.infringement_or_redundancy_risk in ["Low", "Moderate", "High"]

    # Semantic Drift Records
    assert len(report.semantic_drift_records) >= 4
    terms_audited = [d.term for d in report.semantic_drift_records]
    assert any("Zero-Shot" in t for t in terms_audited)
    assert any("Hallucination" in t for t in terms_audited)
    assert any("Self-Supervised" in t for t in terms_audited)

    # Check dilution properties
    for dr in report.semantic_drift_records:
        assert 0.0 <= dr.dilution_score <= 1.0
        assert dr.verdict in ["Mathematically Rigorous", "Moderate Drift", "Severe Dilution / Buzzword"]
        assert len(dr.audit_recommendation) > 0


def test_v5_pipeline_end_to_end():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    report = pipeline.run("What are the limitations of Vision Transformers for low-light object detection?")

    # Verify v5.0 components are populated in SynthesisReport
    assert report.in_silico_simulation is not None
    assert report.in_silico_simulation.trials_count == 1000
    assert report.in_silico_simulation.power_analysis.statistical_power_achieved >= 0.80

    assert report.cross_domain_transfer is not None
    assert len(report.cross_domain_transfer.analogies) >= 3

    assert report.novelty_sentinel is not None
    assert report.novelty_sentinel.mean_novelty_index >= 0.70
    assert len(report.novelty_sentinel.prior_art_collisions) >= 2


def test_v5_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. In Silico Simulation endpoint
        res_sim = client.post("/api/v1/simulation/in-silico", json={"query": "Vision Transformers in low-light", "trials_count": 500})
        assert res_sim.status_code == 200
        data_sim = res_sim.json()
        assert "simulation_id" in data_sim
        assert data_sim["trials_count"] == 500
        assert data_sim["projected_net_gain_pct"] > 0
        assert "power_analysis" in data_sim

        # 2. Simulation Harness code endpoint
        res_harness = client.get("/api/v1/simulation/harness")
        assert res_harness.status_code == 200
        data_harness = res_harness.json()
        assert data_harness["filename"] == "in_silico_benchmark.py"
        assert "torch" in data_harness["code"]

        # 3. Simulation Lab HTML endpoint
        res_lab = client.get("/api/v1/export/simulation-lab")
        assert res_lab.status_code == 200
        data_lab = res_lab.json()
        assert "<!DOCTYPE html>" in data_lab["interactive_lab_html"]

        # 4. Cross-Domain Transfer endpoint
        res_transfer = client.post("/api/v1/transfer/cross-domain", json={"query": "Vision Transformers in low-light"})
        assert res_transfer.status_code == 200
        data_transfer = res_transfer.json()
        assert "analogies" in data_transfer
        assert len(data_transfer["analogies"]) >= 3

        # 5. Novelty Sentinel endpoint
        res_novelty = client.post("/api/v1/audit/novelty", json={"query": "Vision Transformers in low-light"})
        assert res_novelty.status_code == 200
        data_novelty = res_novelty.json()
        assert data_novelty["mean_novelty_index"] >= 0.70
        assert "prior_art_collisions" in data_novelty
        assert "semantic_drift_records" in data_novelty


def test_package_exports_v5():
    from src.synthesis import InSilicoSimulator as ExportSim
    from src.claims import CrossDomainAnalogyEngine as ExportAnalogy
    from src.verification import NoveltySentinel as ExportSentinel
    import src

    assert ExportSim is not None
    assert ExportAnalogy is not None
    assert ExportSentinel is not None
    assert src.__version__ in ["0.5.0", "0.6.0", "0.7.0"]
