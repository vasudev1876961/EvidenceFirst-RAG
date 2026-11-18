"""Unit and Integration Tests for v8.0 Advanced Capabilities.

Validates:
1. Conformal Risk Control (CRC) & Finite-Sample Hallucination Error Bounds (Angelopoulos et al. 2024; Kumar et al. 2024).
2. Semantic Entropy & Meaning-Space Epistemic Uncertainty (Farquhar et al., Nature 2024).
3. Astute Parametric-Evidence Conflict Arbiter & Misinformation Resistance (Wang et al. 2024).
4. RAGChecker Tripartite Diagnostic Benchmark Evaluation Suite (Hu et al. 2024 - AWS AI / SJTU / Westlake).
5. End-to-End Master Pipeline Integration & FastAPI v8.0 REST Endpoints.
"""

import math
import pytest
from fastapi.testclient import TestClient

from src.models import (
    FactualClaim,
    EvidenceItem,
    ConflictAttribution,
    Paper,
    SynthesisReport,
)
from src.verification.conformal_risk import ConformalRiskEngine
from src.claims.semantic_entropy import SemanticEntropyEngine
from src.synthesis.astute_arbiter import AstuteConflictArbiter
from src.evaluation.ragchecker import RAGCheckerEvaluator
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_conformal_risk_engine():
    engine = ConformalRiskEngine(target_alpha=0.05, confidence_level=0.95)

    c1 = FactualClaim(
        claim_id="C001",
        statement="Vision Transformers improve detection in low light.",
        confidence_score=0.90,
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
            composite_score=0.88,
            entailment_prob=0.92,
            contradiction_prob=0.02,
            relationship="SUPPORTS",
        )
    )

    c2 = FactualClaim(
        claim_id="C002",
        statement="Gamma-curve amplification introduces noise artifacts.",
        confidence_score=0.30,
    )
    c2.contradicting_evidence.append(
        EvidenceItem(
            evidence_id="ev_02",
            chunk_id="chk_02",
            paper_id="paper_002",
            paper_title="Enhance Fail",
            section="Results",
            year=2025,
            text_snippet="Gamma curves fail in extreme dark.",
            composite_score=0.85,
            entailment_prob=0.05,
            contradiction_prob=0.89,
            relationship="CONTRADICTS",
        )
    )

    report = engine.evaluate_claims([c1, c2], alpha=0.05)

    assert report.report_id.startswith("CRC-")
    assert report.target_error_budget_alpha == 0.05
    assert report.statistical_guarantee_confidence == 0.95
    assert report.conformal_threshold_lambda > 0.0
    assert report.total_claims_evaluated == 2
    assert len(report.claims_breakdown) == 2

    # Claim 1 with high support should have lower nonconformity score than claim 2
    rec1 = report.claims_breakdown[0]
    rec2 = report.claims_breakdown[1]
    assert rec1.nonconformity_score < rec2.nonconformity_score
    assert rec1.is_certified_safe is True
    assert "Under exchangeability" in report.formal_theorem_guarantee


def test_semantic_entropy_engine():
    engine = SemanticEntropyEngine(entropy_threshold=0.45)

    c_consensus = FactualClaim(
        claim_id="C101",
        statement="Wavelet frequency gating stabilizes vision transformer attention in low lux.",
        confidence_score=0.85,
    )

    c_ambiguous = FactualClaim(
        claim_id="C102",
        statement="Illumination enhancement improves detection.",
        confidence_score=0.60,
    )
    c_ambiguous.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_10",
            chunk_id="chk_10",
            paper_id="p1",
            paper_title="P1",
            section="Results",
            year=2024,
            text_snippet="Enhancement improves mAP by 8.2%",
            composite_score=0.80,
            relationship="SUPPORTS",
        )
    )
    c_ambiguous.contradicting_evidence.append(
        EvidenceItem(
            evidence_id="ev_11",
            chunk_id="chk_11",
            paper_id="p2",
            paper_title="P2",
            section="Results",
            year=2025,
            text_snippet="Enhancement yields -0.4% regression on ExDark",
            composite_score=0.85,
            relationship="CONTRADICTS",
        )
    )

    report = engine.evaluate_entropy_report([c_consensus, c_ambiguous])

    assert report.report_id.startswith("SEM-ENT-")
    assert report.total_claims_evaluated == 2
    assert report.mean_semantic_entropy_nats >= 0.0
    assert len(report.claim_evaluations) == 2

    # Ambiguous claim with contradictions should exhibit multiple clusters & higher entropy
    amb_eval = report.claim_evaluations[1]
    assert amb_eval.cluster_count >= 2
    assert amb_eval.semantic_entropy > report.claim_evaluations[0].semantic_entropy


def test_astute_conflict_arbiter():
    arbiter = AstuteConflictArbiter()

    # Claim with overwhelming literature evidence
    c_evid = FactualClaim(
        claim_id="C201",
        statement="Attention logits explode quadratically under photon starvation.",
        confidence_score=0.88,
    )
    c_evid.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_20",
            chunk_id="chk_20",
            paper_id="p_vit",
            paper_title="ViT Starvation",
            section="Method",
            year=2024,
            text_snippet="Logit variance exhibits O(1/lux^2) explosion.",
            composite_score=0.91,
            relationship="SUPPORTS",
        )
    )

    # Claim with strong contradiction
    c_contra = FactualClaim(
        claim_id="C202",
        statement="Direct RAW image feeding outperforms pre-enhanced RGB images.",
        confidence_score=0.65,
    )
    c_contra.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_21",
            chunk_id="chk_21",
            paper_id="p_raw",
            paper_title="Raw ViT",
            section="Results",
            year=2024,
            text_snippet="RAW feeding yields +4.2% mAP.",
            composite_score=0.82,
            relationship="SUPPORTS",
        )
    )
    c_contra.contradicting_evidence.append(
        EvidenceItem(
            evidence_id="ev_22",
            chunk_id="chk_22",
            paper_id="p_rgb",
            paper_title="RGB ViT",
            section="Results",
            year=2025,
            text_snippet="Pre-enhanced RGB yields higher mAP when trained with dark priors.",
            composite_score=0.79,
            relationship="CONTRADICTS",
        )
    )

    report = arbiter.arbitrate_corpus([c_evid, c_contra])

    assert report.report_id.startswith("ASTUTE-")
    assert report.total_claims_arbitrated == 2
    assert report.misinformation_immunity_index >= 0.80

    rec_evid = report.claim_arbitrations[0]
    rec_contra = report.claim_arbitrations[1]

    assert rec_evid.arbitration_verdict == "RELY_ON_EXTERNAL_EVIDENCE"
    assert rec_contra.arbitration_verdict == "DIALECTICAL_SYNTHESIS"


def test_ragchecker_evaluator():
    evaluator = RAGCheckerEvaluator()

    from src.models import CitationAudit, EvidenceGraphData

    dummy_report = SynthesisReport(
        query="What are the limitations of Vision Transformers in low light?",
        overall_confidence=0.85,
        executive_summary="Vision Transformers suffer from attention logit dispersion.",
        citation_audit=CitationAudit(
            total_claims=5,
            supported_claims=5,
            conflicting_claims=0,
            unsupported_claims=0,
            evidence_coverage_score=0.85,
            citation_integrity_score=0.94,
            citation_faithfulness=0.95,
            hallucination_rate=0.03,
            audit_details=[],
        ),
        evidence_graph=EvidenceGraphData(nodes=[], edges=[]),
        claims=[
            FactualClaim(
                claim_id="C1",
                statement="Vision Transformers suffer in extreme darkness.",
                confidence_score=0.88,
                supporting_evidence=[
                    EvidenceItem(
                        evidence_id="e1", chunk_id="chk1", paper_id="p1",
                        paper_title="ViT", section="Results", year=2024,
                        text_snippet="ViT fails at <0.05 lux.", composite_score=0.88,
                    )
                ]
            )
        ],
    )

    ragck = evaluator.evaluate_pipeline_output(dummy_report)

    assert ragck.report_id.startswith("RAGCK-")
    assert ragck.proposed_system_metrics.overall_f1 > ragck.baseline_standard_dense_metrics.overall_f1
    assert ragck.proposed_system_metrics.hallucination_rate < ragck.baseline_standard_dense_metrics.hallucination_rate
    assert ragck.proposed_system_metrics.faithfulness > ragck.baseline_standard_dense_metrics.faithfulness
    assert ragck.relative_f1_gain_percent > 0.0
    assert ragck.relative_hallucination_reduction_percent > 50.0


def test_pipeline_v8_integration():
    from src.ingestion.paper_loader import PaperLoader

    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    query = "What are the limitations of Vision Transformers for low-light object detection?"
    report = pipeline.run(query)

    # Validate all v8.0 reports are populated
    assert report.conformal_risk_report is not None
    assert report.conformal_risk_report.total_claims_evaluated > 0
    assert report.conformal_risk_report.certified_safe_claims_count > 0

    assert report.semantic_entropy_report is not None
    assert report.semantic_entropy_report.total_claims_evaluated > 0
    assert len(report.semantic_entropy_report.claim_evaluations) > 0

    assert report.astute_arbitration_report is not None
    assert report.astute_arbitration_report.total_claims_arbitrated > 0
    assert len(report.astute_arbitration_report.claim_arbitrations) > 0

    assert report.ragchecker_report is not None
    assert report.ragchecker_report.proposed_system_metrics.overall_f1 > 0.50
    assert report.ragchecker_report.relative_f1_gain_percent > 0.0


def test_fastapi_v8_endpoints():
    with TestClient(app) as client:
        # 1. Conformal Risk endpoint
        res_crc = client.post(
            "/api/v1/verification/conformal-risk",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
            params={"alpha": 0.05},
        )
        assert res_crc.status_code == 200
        data_crc = res_crc.json()
        assert "report_id" in data_crc
        assert "conformal_threshold_lambda" in data_crc
        assert "claims_breakdown" in data_crc

        # 2. Semantic Entropy endpoint
        res_se = client.post(
            "/api/v1/claims/semantic-entropy",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
        )
        assert res_se.status_code == 200
        data_se = res_se.json()
        assert "report_id" in data_se
        assert "mean_semantic_entropy_nats" in data_se
        assert "claim_evaluations" in data_se

        # 3. Astute Arbitration endpoint
        res_astute = client.post(
            "/api/v1/synthesis/astute-arbitration",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
        )
        assert res_astute.status_code == 200
        data_astute = res_astute.json()
        assert "report_id" in data_astute
        assert "external_evidence_reliance_rate" in data_astute
        assert "claim_arbitrations" in data_astute

        # 4. RAGChecker Diagnostics endpoint
        res_rc = client.post(
            "/api/v1/evaluation/ragchecker",
            json={"query": "Vision Transformers in low-light", "alpha": 0.5},
        )
        assert res_rc.status_code == 200
        data_rc = res_rc.json()
        assert "report_id" in data_rc
        assert "proposed_system_metrics" in data_rc
        assert "relative_f1_gain_percent" in data_rc


def test_package_exports_v8():
    import src
    from src.verification import ConformalRiskEngine as ExpCRC
    from src.claims import SemanticEntropyEngine as ExpSE
    from src.synthesis import AstuteConflictArbiter as ExpAstute
    from src.evaluation import RAGCheckerEvaluator as ExpRC

    assert ExpCRC is not None
    assert ExpSE is not None
    assert ExpAstute is not None
    assert ExpRC is not None
    assert src.__version__ == "0.8.0"
