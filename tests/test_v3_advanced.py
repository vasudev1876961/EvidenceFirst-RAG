"""Unit and Integration Tests for EvidenceFirst-RAG v3.0 Advanced Capabilities:

1. Conference Peer Review & Soundness Audit (NeurIPS / ICLR Style)
2. Multi-Agent Scientific Consensus Council Deliberation
3. Bibliometric Co-Citation Network & PageRank Influence
4. Academic Presentation Slide Deck Generator (Interactive HTML & Marp)
5. v3.0 REST API Endpoints
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
)
from src.synthesis.peer_review import ScientificPeerReviewer
from src.verification.consensus_council import ScientificConsensusCouncil
from src.claims.citation_network import BibliometricNetworkAnalyzer
from src.synthesis.slide_generator import SlideDeckGenerator
from src.ingestion.paper_loader import PaperLoader
from src.pipeline import EvidenceFirstPipeline
from app.api import app


def test_peer_review_engine():
    reviewer = ScientificPeerReviewer()
    
    # Create mock synthesis report
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
        decomposed_subqueries=["SubQ 1", "SubQ 2", "SubQ 3"],
        overall_confidence=0.88,
        executive_summary="Vision Transformers struggle with quadratic noise attention in sub-lux regimes.",
        citation_audit=dummy_audit,
        evidence_graph=EvidenceGraphData(),
        claims=[
            FactualClaim(claim_id="C1", statement="ViTs exhibit degradation under shot noise.", evidence_status="Strong support", confidence_score=0.9),
            FactualClaim(claim_id="C2", statement="Enhancement improves detection by 8.2% on DarkFace.", evidence_status="Conflicting", confidence_score=0.8),
        ],
        conflicting_findings=[
            ConflictAttribution(
                claim_id="C2",
                claim_statement="Enhancement improves detection.",
                likely_reason="Lighting lux disparity across DarkFace and ExDark.",
            )
        ],
    )

    review = reviewer.review_synthesis(dummy_report)

    assert review.conference_venue == "NeurIPS / ICLR Academic Review Track"
    assert 1 <= review.score_card.soundness_score <= 5
    assert 1 <= review.score_card.presentation_score <= 5
    assert 1 <= review.score_card.contribution_score <= 5
    assert 1 <= review.score_card.overall_recommendation <= 10
    assert 1 <= review.score_card.confidence_score <= 5
    assert len(review.strengths) >= 2
    assert len(review.weaknesses) >= 2
    assert len(review.questions_for_authors) >= 2
    assert len(review.reproducibility_audit) > 0
    assert len(review.ethical_and_societal_implications) > 0


def test_consensus_council_deliberation():
    council = ScientificConsensusCouncil()
    
    conflict = ConflictAttribution(
        claim_id="C_CONF",
        claim_statement="Illumination preprocessing improves object detection in darkness.",
        likely_reason="DarkFace photon starvation (<0.1 lux) vs ExDark natural ambient light (>1.0 lux).",
    )

    council_report = council.deliberate(
        query="Does illumination enhancement consistently improve low-light detection?",
        conflicts=[conflict],
    )

    assert council_report.active_claim_id == "C_CONF"
    assert len(council_report.participating_agents) == 3
    assert len(council_report.debate_rounds) == 3
    
    # Check turns
    for r in council_report.debate_rounds:
        assert len(r.turns) == 3
        for t in r.turns:
            assert len(t.content) > 0
            assert len(t.speaker_name) > 0

    # Check verdict
    v = council_report.verdict
    assert v.verdict_type in ["Conditional Consensus", "Distribution Shift Confirmed"]
    assert v.confidence >= 0.85
    assert len(v.golden_synthesis_principle) > 0
    assert len(v.concrete_boundary_rule) > 0
    assert v.unanimous_vote is True


def test_bibliometric_network_analyzer():
    analyzer = BibliometricNetworkAnalyzer()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    assert len(papers) >= 3

    claim = FactualClaim(
        claim_id="C1",
        statement="Vision transformers need inductive priors.",
        supporting_evidence=[
            EvidenceItem(
                evidence_id="ev_1", chunk_id="chk_1", paper_id=papers[0].paper_id,
                paper_title=papers[0].title, section="Results", year=papers[0].year,
                text_snippet="Snippet 1", composite_score=0.9,
            ),
            EvidenceItem(
                evidence_id="ev_2", chunk_id="chk_2", paper_id=papers[1].paper_id,
                paper_title=papers[1].title, section="Results", year=papers[1].year,
                text_snippet="Snippet 2", composite_score=0.85,
            )
        ]
    )

    biblio = analyzer.analyze_network(papers=papers, claims=[claim])

    assert biblio.total_papers_analyzed == len(papers)
    assert biblio.graph_density >= 0.0
    assert len(biblio.paper_metrics) == len(papers)
    assert len(biblio.foundational_anchors) >= 1
    
    first_metric = biblio.paper_metrics[0]
    assert first_metric.pagerank_centrality > 0.0
    assert first_metric.impact_tier in ["Foundational Anchor", "Methodological Bridge", "Specialized Frontier"]


def test_slide_deck_generator():
    pipeline = EvidenceFirstPipeline()
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline.index_papers(papers)

    report = pipeline.run("What are the limitations of Vision Transformers for low-light object detection?")
    generator = SlideDeckGenerator()

    deck = generator.build_slide_deck(report)

    assert deck.total_slides == 10
    assert len(deck.slides) == 10
    assert "<!DOCTYPE html>" in deck.html_deck
    assert "progress-bar" in deck.html_deck
    assert "controls-btn" in deck.html_deck
    assert "marp: true" in deck.marp_markdown
    assert "EvidenceFirst-RAG" in deck.marp_markdown


def test_v3_fastapi_endpoints():
    with TestClient(app) as client:
        # 1. Peer review endpoint
        res_review = client.post("/api/v1/review/peer-review", json={"query": "Vision Transformers in low-light"})
        assert res_review.status_code == 200
        data_review = res_review.json()
        assert "score_card" in data_review
        assert data_review["score_card"]["overall_recommendation"] >= 6
        assert len(data_review["strengths"]) >= 1

        # 2. Consensus Council deliberation endpoint
        res_council = client.post("/api/v1/council/deliberate", json={"query": "Vision Transformers in extreme low-light"})
        assert res_council.status_code == 200
        data_council = res_council.json()
        assert len(data_council["debate_rounds"]) == 3
        assert "verdict" in data_council
        assert data_council["verdict"]["confidence"] >= 0.85

        # 3. Bibliometrics endpoint
        res_bib = client.get("/api/v1/network/bibliometrics")
        assert res_bib.status_code == 200
        data_bib = res_bib.json()
        assert data_bib["total_papers_analyzed"] >= 3
        assert len(data_bib["paper_metrics"]) >= 3

        # 4. Slide deck export endpoint
        res_slides = client.post("/api/v1/export/slides", json={"query": "Vision Transformers in low-light"})
        assert res_slides.status_code == 200
        data_slides = res_slides.json()
        assert data_slides["total_slides"] == 10
        assert "<!DOCTYPE html>" in data_slides["html_deck"]
        assert "marp: true" in data_slides["marp_markdown"]


def test_package_exports_v3():
    from src.synthesis import ScientificPeerReviewer as ExportPeerReview, SlideDeckGenerator as ExportSlide
    from src.verification import ScientificConsensusCouncil as ExportCouncil
    from src.claims import BibliometricNetworkAnalyzer as ExportBiblio

    assert ExportPeerReview is not None
    assert ExportSlide is not None
    assert ExportCouncil is not None
    assert ExportBiblio is not None
