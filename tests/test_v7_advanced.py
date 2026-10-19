"""Unit and Integration Tests for v7.0 Advanced Capabilities.

Validates:
1. Sequential Bayesian Epistemic Belief Updating & Bayes Factor Calibration.
2. Automated OSF Preregistration Protocol Synthesizer & Cryptographic SHA-256 Hash.
3. Quantum-Inspired Hilbert Space Epistemic State & Bloch Sphere Superposition.
4. End-to-End Pipeline Integration & FastAPI REST Endpoints.
"""

import math
import pytest
from fastapi.testclient import TestClient

from src.models import (
    FactualClaim,
    EvidenceItem,
    ConflictAttribution,
    ScientificHypothesis,
    Paper,
    MethodologyRigorReport,
    MethodologyAssessment,
)
from src.verification.bayesian_updater import BayesianBeliefUpdater
from src.synthesis.preregistration_synthesizer import PreregistrationSynthesizer
from src.claims.quantum_epistemic import QuantumEpistemicEngine
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_bayesian_belief_updater_supporting_and_refuting():
    updater = BayesianBeliefUpdater(default_prior=0.50)

    # Claim 1: Strong Support
    c1 = FactualClaim(
        claim_id="C001",
        statement="Frequency gating prevents attention logit dispersion in low-light conditions.",
        confidence_score=0.88,
    )
    c1.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_01",
            chunk_id="chk_01",
            paper_id="paper_001",
            paper_title="ViT Low Light",
            section="Results",
            year=2024,
            text_snippet="mAP improves by 8.4% on DarkFace.",
            composite_score=0.85,
            entailment_prob=0.89,
            contradiction_prob=0.04,
            relationship="SUPPORTS",
        )
    )

    # Claim 2: Contradicting / Refuting
    c2 = FactualClaim(
        claim_id="C002",
        statement="Gamma-curve amplification reliably recovers fine semantic textures.",
        confidence_score=0.25,
    )
    c2.contradicting_evidence.append(
        EvidenceItem(
            evidence_id="ev_02",
            chunk_id="chk_02",
            paper_id="paper_002",
            paper_title="Noise Explosion in Dark RAW",
            section="Analysis",
            year=2025,
            text_snippet="Gamma curve induces 4x noise amplification singularity.",
            composite_score=0.90,
            entailment_prob=0.05,
            contradiction_prob=0.92,
            relationship="CONTRADICTS",
        )
    )

    report = updater.evaluate_beliefs(claims=[c1, c2])

    assert report.claims_evaluated == 2
    assert 0.0 <= report.consensus_convergence_rate <= 1.0
    assert report.mean_information_gain_nats >= 0.0

    b1 = report.claim_beliefs[0]
    assert b1.posterior_credence > 0.50
    assert b1.bayes_factor_aggregate_10 > 1.0
    assert "Settled" in b1.epistemic_state_label or "Consensus" in b1.epistemic_state_label
    assert b1.credible_interval_95_hdi[0] <= b1.credible_interval_95_hdi[1]

    b2 = report.claim_beliefs[1]
    assert b2.posterior_credence < 0.50
    assert b2.bayes_factor_aggregate_10 < 1.0
    assert "Refuted" in b2.epistemic_state_label or "Debunked" in b2.epistemic_state_label


def test_preregistration_synthesizer_cryptographic_hash():
    synthesizer = PreregistrationSynthesizer(default_alpha=0.01, target_power=0.90)

    conflict = ConflictAttribution(
        claim_id="C_CONF_01",
        claim_statement="Illumination pre-enhancement improves object detection in extreme night.",
        supporting_papers=[{"paper_id": "paper_A", "paper_title": "Paper A"}],
        contradicting_papers=[{"paper_id": "paper_B", "paper_title": "Paper B"}],
        divergent_factors=[{"factor": "Lux disparity (0.05 lux vs 5.0 lux)"}],
        likely_reason="Benchmark illuminance disparity between DarkFace and ExDark.",
    )

    hyp = ScientificHypothesis(
        hypothesis_id="HYP-001",
        title="Wavelet Frequency Isolation in Vision Transformers",
        statement="Pre-attention wavelet frequency gating yields non-divergent Lipschitz bounds.",
        tension_addressed="Conflict between spatial enhancement and feature collapse.",
        rationale="Wavelet shrinkage isolates high-frequency noise from structural tokens.",
        proposed_methodology="Multi-scale 2D Haar wavelet transformation prior to token projection.",
        testable_predictions=["mAP gain >= 3.0%", "False positive drop >= 20%"],
        falsification_criteria="mAP gain <= 0.5% under extreme darkness < 0.1 lux",
    )

    dossier = synthesizer.synthesize_dossier(
        claims=[],
        conflicts=[conflict],
        hypotheses=[hyp],
    )

    assert dossier.total_protocols_synthesized == 2
    assert dossier.open_science_compliance_index >= 0.85
    assert len(dossier.anti_phacking_safeguards) >= 3

    p1 = dossier.protocols[0]
    assert p1.protocol_id.startswith("PRG-2026-CONF")
    assert len(p1.cryptographic_sha256_hash) == 64
    assert p1.osf_badge_audit.preregistered_badge is True
    assert p1.sample_size_determination["alpha_level"] == 0.01
    assert p1.sample_size_determination["power_target"] == 0.90

    # Ensure markdown dossier contains protocol details
    assert "# Formal Open Science Preregistration Dossier" in dossier.preregistration_markdown_document
    assert p1.cryptographic_sha256_hash in dossier.preregistration_markdown_document


def test_quantum_epistemic_bloch_coordinates_and_entropy():
    engine = QuantumEpistemicEngine(decoherence_factor=0.10)

    # Pure Support Claim
    c_sup = FactualClaim(
        claim_id="C_SUP",
        statement="Vision Transformers achieve state-of-the-art mAP on daytime benchmarks.",
        confidence_score=0.95,
    )
    c_sup.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_s",
            chunk_id="chk_s",
            paper_id="p1",
            paper_title="Daytime ViT",
            section="Results",
            year=2023,
            text_snippet="52.4% mAP achieved on COCO val.",
            composite_score=0.92,
            relationship="SUPPORTS",
        )
    )

    # Superposed Controversial Claim (equal support and contradiction)
    c_super = FactualClaim(
        claim_id="C_SUPER",
        statement="Pre-enhancement models are strictly required before ViT backbone feature extraction.",
        confidence_score=0.52,
    )
    c_super.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_s2",
            chunk_id="chk_s2",
            paper_id="p2",
            paper_title="Pre-Enhancement Win",
            section="Results",
            year=2024,
            text_snippet="Pre-enhancement yields +8.2% mAP on DarkFace.",
            composite_score=0.80,
            relationship="SUPPORTS",
        )
    )
    c_super.contradicting_evidence.append(
        EvidenceItem(
            evidence_id="ev_c2",
            chunk_id="chk_c2",
            paper_id="p3",
            paper_title="Pre-Enhancement Failure",
            section="Results",
            year=2025,
            text_snippet="Pre-enhancement yields -0.4% mAP regression on ExDark.",
            composite_score=0.80,
            relationship="CONTRADICTS",
        )
    )

    conflict = ConflictAttribution(
        claim_id="C_SUPER",
        claim_statement=c_super.statement,
        divergent_factors=[{"factor": "Dataset luxury and lux levels"}],
        likely_reason="Sensor noise amplification",
    )

    q_report = engine.evaluate_epistemic_states(
        claims=[c_sup, c_super],
        conflicts=[conflict],
    )

    assert q_report.total_claims_in_superposition >= 1
    assert 0.0 <= q_report.mean_von_neumann_entropy <= math.log(2.0)
    assert 0.0 <= q_report.systemic_coherence_index <= 1.0

    # Test coordinate geometry on unit sphere (x^2 + y^2 + z^2 == 1.0)
    for qs in q_report.quantum_states:
        coord = qs.bloch_coordinates
        norm_sq = coord.x ** 2 + coord.y ** 2 + coord.z ** 2
        assert math.isclose(norm_sq, 1.0, abs_tol=0.05)
        assert 0.50 <= qs.superposition_purity <= 1.0
        assert 0.0 <= qs.quantum_coherence <= 1.0

    # Ensure HTML dashboard was generated with SVG
    assert "<!DOCTYPE html>" in q_report.bloch_sphere_html
    assert "Bloch Sphere Radar" in q_report.bloch_sphere_html
    assert "<svg" in q_report.bloch_sphere_html


def test_pipeline_v7_integration():
    from src.ingestion.paper_loader import PaperLoader

    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    query = "What are the limitations of Vision Transformers for low-light object detection?"
    report = pipeline.run(query)

    # Validate v7.0 reports exist and are populated
    assert report.bayesian_belief_report is not None
    assert report.bayesian_belief_report.claims_evaluated > 0
    assert len(report.bayesian_belief_report.claim_beliefs) > 0

    assert report.preregistration_dossier is not None
    assert report.preregistration_dossier.total_protocols_synthesized > 0
    assert len(report.preregistration_dossier.protocols) > 0

    assert report.quantum_epistemic_report is not None
    assert len(report.quantum_epistemic_report.quantum_states) > 0
    assert len(report.quantum_epistemic_report.bloch_sphere_html) > 0


def test_fastapi_v7_endpoints():
    with TestClient(app) as client:
        # 1. Bayesian beliefs endpoint
        res_bayes = client.post(
            "/api/v1/verification/bayesian-beliefs",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
        )
        assert res_bayes.status_code == 200
        data_bayes = res_bayes.json()
        assert "report_id" in data_bayes
        assert "claim_beliefs" in data_bayes
        assert "mean_information_gain_nats" in data_bayes

        # 2. OSF Preregistration endpoint
        res_prereg = client.post(
            "/api/v1/synthesis/osf-preregistration",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
        )
        assert res_prereg.status_code == 200
        data_prereg = res_prereg.json()
        assert "dossier_id" in data_prereg
        assert "protocols" in data_prereg
        assert len(data_prereg["protocols"]) > 0

        # 3. Quantum Epistemic endpoint
        res_q = client.post(
            "/api/v1/claims/quantum-epistemic",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
        )
        assert res_q.status_code == 200
        data_q = res_q.json()
        assert "quantum_states" in data_q
        assert "mean_von_neumann_entropy" in data_q

        # 4. Export Quantum Bloch Radar HTML
        res_html = client.get("/api/v1/export/quantum-radar?query=Vision%20Transformers")
        assert res_html.status_code == 200
        data_html = res_html.json()
        assert "<!DOCTYPE html>" in data_html["bloch_sphere_html"]


def test_package_exports_v7():
    import src
    from src.verification import BayesianBeliefUpdater as ExpBayes
    from src.synthesis import PreregistrationSynthesizer as ExpPrereg
    from src.claims import QuantumEpistemicEngine as ExpQuantum

    assert ExpBayes is not None
    assert ExpPrereg is not None
    assert ExpQuantum is not None
    assert src.__version__ == "0.7.0"

