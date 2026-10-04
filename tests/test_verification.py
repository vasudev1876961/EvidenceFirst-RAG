"""Tests for Evidence Verification, Contradiction Detection, and Citation Auditing."""

import pytest
from src.models import FactualClaim, RetrievalCandidate, SemanticChunk
from src.verification.entailment import NLIEntailmentChecker
from src.verification.evidence_score import EvidenceScorer
from src.verification.contradiction import ContradictionEngine
from src.verification.citation_checker import CitationAuditor


def test_evidence_scoring():
    scorer = EvidenceScorer()
    score = scorer.compute_score(
        semantic_sim=0.85,
        entailment_prob=0.90,
        source_quality=1.0,
        year=2025,
    )
    assert score > 0.80
    assert scorer.classify_status(score) == "Strong support"


def test_contradiction_detection():
    nli = NLIEntailmentChecker(use_fallback=True)
    engine = ContradictionEngine(nli_checker=nli, contradiction_threshold=0.60)

    claim = FactualClaim(
        claim_id="C001",
        statement="Vision Transformers outperform CNNs in low-light object detection."
    )

    supporting_chunk = SemanticChunk(
        chunk_id="c1", paper_id="pA", paper_title="Paper A", section="Results", year=2025,
        text="Vision Transformers consistently outperform CNNs in low-light object detection across tests.",
        dataset="DarkFace", evaluation_metric="mAP"
    )
    contradicting_chunk = SemanticChunk(
        chunk_id="c2", paper_id="pB", paper_title="Paper B", section="Results", year=2025,
        text="Vision Transformers do not outperform CNNs and fail to improve detection accuracy.",
        dataset="ExDark", evaluation_metric="mAP"
    )

    candidates = [
        RetrievalCandidate(chunk=supporting_chunk, hybrid_score=0.9),
        RetrievalCandidate(chunk=contradicting_chunk, hybrid_score=0.88),
    ]

    sup, contra, neut = engine.evaluate_evidence_relationships(claim, candidates)
    assert len(sup) >= 1
    assert len(contra) >= 1

    attribution = engine.attribute_conflict(claim, sup, contra)
    assert attribution is not None
    assert "DarkFace" in str(attribution.divergent_factors) or "ExDark" in str(attribution.divergent_factors) or len(attribution.likely_reason) > 0


def test_citation_auditor():
    nli = NLIEntailmentChecker(use_fallback=True)
    auditor = CitationAuditor(nli)

    claim = FactualClaim(
        claim_id="C001",
        statement="Vision Transformers achieve high accuracy.",
    )
    # Give claim 1 supporting item
    from src.models import EvidenceItem
    claim.supporting_evidence.append(
        EvidenceItem(
            evidence_id="ev_1", chunk_id="c1", paper_id="p1", paper_title="P1",
            section="Results", year=2025, text_snippet="High accuracy achieved.",
            entailment_prob=0.88, relationship="SUPPORTS"
        )
    )

    audit = auditor.audit([claim])
    assert audit.total_claims == 1
    assert audit.supported_claims == 1
    assert audit.evidence_coverage_score == 1.0
    assert audit.citation_integrity_score == 1.0
