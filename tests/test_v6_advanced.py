"""Unit and Integration Tests for EvidenceFirst-RAG v6.0 Frontier Meta-Science Capabilities:

1. Automated Theorem & Symbolic Formal Proof Verifier (SymPy)
2. Epistemic Red-Teaming & Adversarial Hallucination Stress-Tester
3. Autonomous Systematic Review & PRISMA 2020 Meta-Analysis Synthesizer
4. End-to-End Pipeline Integration for v6.0 Schemas
5. v6.0 REST API Endpoints
6. Module and Package Export Integrity
"""

import pytest
from fastapi.testclient import TestClient

from src.models import (
    Paper,
    FactualClaim,
    SynthesisReport,
    SymbolicProofReport,
    SymbolicProofCertificate,
    AdversarialStressReport,
    AdversarialAttackVector,
    PrismaMetaAnalysisReport,
    PrismaFlowRecord,
    CochraneRiskOfBiasItem,
    ForestPlotStudyRecord,
)
from src.verification.symbolic_verifier import SymbolicProofEngine
from src.verification.adversarial_tester import AdversarialStressTester
from src.synthesis.prisma_synthesizer import PrismaMetaAnalysisEngine
from src.ingestion.paper_loader import PaperLoader
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_symbolic_proof_engine():
    engine = SymbolicProofEngine()

    report = engine.verify_query_theorems(
        query="What are the limitations of Vision Transformers for low-light object detection?",
    )

    assert isinstance(report, SymbolicProofReport)
    assert report.proof_system_id.startswith("SYM-PROOF-")
    assert report.theorems_evaluated >= 4
    assert len(report.proof_certificates) >= 4
    assert report.formal_consistency_score >= 0.75
    assert len(report.critical_theoretical_breakdowns) >= 2

    # Test Theorem 1: Attention logit variance
    cert_attn = next((c for c in report.proof_certificates if c.proof_id == "PRF-001"), None)
    assert cert_attn is not None
    assert "lux" in cert_attn.sympy_formulation
    assert "O(1 / lux^2)" in cert_attn.asymptotic_bound
    assert cert_attn.verdict == "VERIFIED_THEOREM"
    assert len(cert_attn.formal_derivation_steps) >= 4
    assert "\\begin{theorem}" in cert_attn.latex_proof_render

    # Test Theorem 2: Lipschitz continuity
    cert_lip = next((c for c in report.proof_certificates if c.proof_id == "PRF-002"), None)
    assert cert_lip is not None
    assert "L = 1" in cert_lip.sympy_formulation or "L = sup" in cert_lip.sympy_formulation
    assert cert_lip.verdict == "VERIFIED_THEOREM"

    # Test Theorem 3: Gamma pedestal singularity
    cert_gamma = next((c for c in report.proof_certificates if c.proof_id == "PRF-003"), None)
    assert cert_gamma is not None
    assert cert_gamma.verdict in ["VERIFIED_THEOREM", "CONDITIONALLY_BOUNDED"]
    assert len(cert_gamma.unstated_assumptions) > 0

    # Test Theorem 4: Sparse attention asymptotics
    cert_asymp = next((c for c in report.proof_certificates if c.proof_id == "PRF-004"), None)
    assert cert_asymp is not None
    assert cert_asymp.verdict == "VERIFIED_THEOREM"

    # Test script generator
    assert "import sympy as sp" in report.executable_sympy_script
    assert "run_symbolic_proof_audit" in report.executable_sympy_script


def test_adversarial_stress_tester():
    tester = AdversarialStressTester()

    report = tester.run_stress_test(
        query="What are the limitations of Vision Transformers for low-light object detection?",
    )

    assert isinstance(report, AdversarialStressReport)
    assert report.stress_test_id.startswith("ADV-STRESS-")
    assert report.total_attacks_simulated == 5
    assert report.attacks_successfully_defended >= 4
    assert report.defense_rate_pct >= 80.0
    assert 0.0 <= report.epistemic_vulnerability_index <= 0.30
    assert report.adversarial_resilience_verdict in [
        "Hardened Research Defense",
        "Moderate Epistemic Armor",
    ]

    # Check Simpson's Paradox Audit
    simp = report.simpsons_paradox_audit
    assert "paradox_type" in simp
    assert simp["aggregate_apparent_gain_pct"] > 0
    assert simp["subgroup_near_lux_zero"]["isolated_delta_map_pct"] < 0

    # Check Attack Vectors
    categories = [a.attack_category for a in report.attack_vectors]
    assert "Simpsons_Paradox_Reversal" in categories
    assert "Metric_Cherry_Picking_Inversion" in categories
    assert "P_Hacking_Subgroup_Dredging" in categories
    assert "Adversarial_Citation_Poisoning" in categories
    assert "Semantic_Hallucination_Trap" in categories

    for atk in report.attack_vectors:
        assert atk.pipeline_defended is True
        assert len(atk.observed_failure_or_defense) > 0
        assert len(atk.remediation_patch) > 0

    # Automated rules
    assert len(report.automated_defense_rules) >= 4
    assert any("Benjamini-Hochberg" in r for r in report.automated_defense_rules)


def test_prisma_meta_analysis_engine():
    engine = PrismaMetaAnalysisEngine()

    report = engine.synthesize_systematic_review(
        query="What are the limitations of Vision Transformers for low-light object detection?",
    )

    assert isinstance(report, PrismaMetaAnalysisReport)
    assert report.prisma_id.startswith("PRISMA-")

    # PRISMA flow checks
    flow = report.prisma_flow
    assert flow.identification_count >= 10
    assert flow.screening_count >= flow.included_synthesis_count
    assert flow.included_synthesis_count >= 5
    assert len(flow.flow_exclusion_reasons) > 0

    # Cochrane Risk of Bias checks
    assert len(report.cochrane_risk_of_bias) >= 4
    for rob in report.cochrane_risk_of_bias:
        assert rob.domain_1_randomization_selection in ["Low", "Some Concerns", "High"]
        assert rob.domain_4_measurement_outcome in ["Low", "Some Concerns", "High"]
        assert rob.overall_risk_of_bias in ["Low Risk", "Some Concerns", "High Risk"]
        assert len(rob.rationale) > 0

    # Forest plot studies
    assert len(report.forest_plot_studies) >= 4
    for study in report.forest_plot_studies:
        assert study.sample_size > 0
        assert len(study.confidence_interval_95) == 2
        assert study.confidence_interval_95[0] < study.confidence_interval_95[1]
        assert study.weight_pct > 0

    # Random-effects calculations
    assert report.pooled_random_effects_estimate > 0.0
    assert len(report.pooled_confidence_interval_95) == 2
    assert report.pooled_confidence_interval_95[0] < report.pooled_confidence_interval_95[1]
    assert 0.0 <= report.heterogeneity_i2_pct <= 100.0
    assert report.cochran_q_statistic > 0.0
    assert report.cochran_q_p_value >= 0.0
    assert report.tau_squared_variance >= 0.0

    # Egger's test
    assert report.eggers_p_value > 0.0
    assert report.publication_bias_detected in [True, False]

    # HTML visualization
    assert "<!DOCTYPE html>" in report.interactive_forest_html
    assert "PRISMA 2020 Systematic Review" in report.interactive_forest_html
    assert "Forest Plot" in report.interactive_forest_html


def test_v6_pipeline_end_to_end():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    report = pipeline.run("What are the limitations of Vision Transformers for low-light object detection?")

    # Verify v6.0 components are populated in SynthesisReport
    assert report.symbolic_proof_report is not None
    assert report.symbolic_proof_report.formal_consistency_score >= 0.70
    assert len(report.symbolic_proof_report.proof_certificates) >= 4

    assert report.adversarial_stress_report is not None
    assert report.adversarial_stress_report.defense_rate_pct >= 80.0
    assert len(report.adversarial_stress_report.attack_vectors) == 5

    assert report.prisma_meta_analysis is not None
    assert report.prisma_meta_analysis.prisma_flow.included_synthesis_count >= 5
    assert len(report.prisma_meta_analysis.forest_plot_studies) >= 4


def test_v6_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. Symbolic Proof endpoint
        res_sym = client.post("/api/v1/verification/symbolic", json={"query": "Vision Transformers in low-light"})
        assert res_sym.status_code == 200
        data_sym = res_sym.json()
        assert "proof_certificates" in data_sym
        assert len(data_sym["proof_certificates"]) >= 4
        assert data_sym["formal_consistency_score"] >= 0.70

        # 2. Symbolic Verification Script endpoint
        res_script = client.get("/api/v1/verification/symbolic/script")
        assert res_script.status_code == 200
        data_script = res_script.json()
        assert data_script["filename"] == "symbolic_proof_verification.py"
        assert "sympy" in data_script["script"]

        # 3. Adversarial Stress Test endpoint
        res_adv = client.post("/api/v1/verification/adversarial-stress-test", json={"query": "Vision Transformers in low-light"})
        assert res_adv.status_code == 200
        data_adv = res_adv.json()
        assert "defense_rate_pct" in data_adv
        assert data_adv["defense_rate_pct"] >= 80.0
        assert "simpsons_paradox_audit" in data_adv
        assert len(data_adv["attack_vectors"]) == 5

        # 4. PRISMA Systematic Review endpoint
        res_prisma = client.post("/api/v1/synthesis/prisma-review", json={"query": "Vision Transformers in low-light"})
        assert res_prisma.status_code == 200
        data_prisma = res_prisma.json()
        assert "prisma_flow" in data_prisma
        assert "cochrane_risk_of_bias" in data_prisma
        assert "forest_plot_studies" in data_prisma
        assert data_prisma["pooled_random_effects_estimate"] > 0

        # 5. PRISMA HTML Export endpoint
        res_html = client.get("/api/v1/export/prisma-report")
        assert res_html.status_code == 200
        data_html = res_html.json()
        assert "<!DOCTYPE html>" in data_html["interactive_forest_html"]


def test_package_exports_v6():
    from src.verification import SymbolicProofEngine as ExportSym, AdversarialStressTester as ExportAdv
    from src.synthesis import PrismaMetaAnalysisEngine as ExportPrisma
    import src

    assert ExportSym is not None
    assert ExportAdv is not None
    assert ExportPrisma is not None
    assert src.__version__ in ["0.6.0", "0.7.0"]
