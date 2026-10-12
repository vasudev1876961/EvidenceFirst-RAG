"""Core data models and schemas for EvidenceFirst-RAG."""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field


class PaperSection(BaseModel):
    title: str
    content: str
    subsections: Optional[List[PaperSection]] = None


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str
    year: int
    venue: Optional[str] = None
    url: Optional[str] = None
    sections: Dict[str, str] = Field(default_factory=dict)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    page: Optional[int] = None
    year: int
    text: str
    dataset: Optional[str] = None
    model_architecture: Optional[str] = None
    evaluation_metric: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    chunk: SemanticChunk
    bm25_score: float = 0.0
    dense_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


class EvidenceItem(BaseModel):
    evidence_id: str
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    year: int
    text_snippet: str
    semantic_similarity: float = 0.0
    entailment_prob: float = 0.0
    contradiction_prob: float = 0.0
    neutral_prob: float = 0.0
    source_score: float = 1.0
    recency_score: float = 1.0
    composite_score: float = 0.0
    relationship: Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"] = "NEUTRAL"
    extracted_attributes: Dict[str, Any] = Field(default_factory=dict)


class ConflictAttribution(BaseModel):
    claim_id: str
    claim_statement: str
    supporting_papers: List[Dict[str, Any]] = Field(default_factory=list)
    contradicting_papers: List[Dict[str, Any]] = Field(default_factory=list)
    divergent_factors: List[Dict[str, str]] = Field(default_factory=list)
    likely_reason: str


class FactualClaim(BaseModel):
    claim_id: str
    statement: str
    subquery_source: Optional[str] = None
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)
    neutral_evidence: List[EvidenceItem] = Field(default_factory=list)
    confidence_score: float = 0.0
    evidence_status: Literal["Strong support", "Moderate support", "Conflicting", "Weak support", "Unsupported"] = "Unsupported"
    conflict_attribution: Optional[ConflictAttribution] = None


class CitationAudit(BaseModel):
    total_claims: int
    supported_claims: int
    conflicting_claims: int
    unsupported_claims: int
    evidence_coverage_score: float  # ECS = supported / total
    citation_integrity_score: float  # CIS = verified_citations / total_citations
    citation_faithfulness: float
    audit_details: List[Dict[str, Any]] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    label: str
    node_type: Literal["query", "claim", "paper", "dataset", "method", "evidence"]
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: Literal["SUBQUERY", "EXTRACTED_FROM", "SUPPORTS", "CONTRADICTS", "USES_DATASET", "USES_METHOD"]
    weight: float = 1.0


class EvidenceGraphData(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class EmpiricalBenchmarkRecord(BaseModel):
    model_name: str
    paper_id: str
    dataset: str
    metric_name: str
    metric_value: float
    unit: str = "%"
    fps: Optional[float] = None
    params_m: Optional[float] = None
    notes: Optional[str] = None


class EmpiricalMetaAnalysis(BaseModel):
    records: List[EmpiricalBenchmarkRecord] = Field(default_factory=list)
    dataset_comparison_table: List[Dict[str, Any]] = Field(default_factory=list)
    model_leaderboard: List[Dict[str, Any]] = Field(default_factory=list)
    cross_dataset_variance: Dict[str, float] = Field(default_factory=dict)
    key_empirical_insights: List[str] = Field(default_factory=list)


class RobustnessReport(BaseModel):
    evidence_robustness_index: float  # ERI score between 0.0 and 1.0
    consensus_divergence_score: float  # measure of variance in evidence directions
    critical_papers: List[Dict[str, Any]] = Field(default_factory=list)
    claim_stability_scores: Dict[str, float] = Field(default_factory=dict)
    temporal_drift_findings: List[str] = Field(default_factory=list)
    counterfactual_scenarios: List[Dict[str, Any]] = Field(default_factory=list)


class ArxivPaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str
    published_year: int
    pdf_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    extracted_datasets: List[str] = Field(default_factory=list)
    extracted_methods: List[str] = Field(default_factory=list)


class VerificationStep(BaseModel):
    step_number: int
    claim_id: str
    target_statement: str
    query_generated: str
    retrieved_chunks_count: int
    initial_status: str
    updated_status: str
    confidence_delta: float
    reasoning: str


class VerificationTrajectory(BaseModel):
    iterations_performed: int
    steps: List[VerificationStep] = Field(default_factory=list)
    claims_resolved: int = 0
    trajectory_summary: str = ""


class ScientificHypothesis(BaseModel):
    hypothesis_id: str
    title: str
    statement: str
    tension_addressed: str
    rationale: str
    proposed_methodology: str
    testable_predictions: List[str] = Field(default_factory=list)
    falsification_criteria: str
    feasibility_tier: Literal["High", "Medium", "Exploratory"] = "High"
    novelty_impact_tier: Literal["Breakthrough", "High", "Incremental"] = "High"
    estimated_impact_score: float = 0.85


class MethodologyAssessment(BaseModel):
    paper_id: str
    paper_title: str
    year: int
    methodology_type: Literal[
        "Large-Scale Empirical Benchmark",
        "Controlled Ablation Study",
        "Theoretical / Algorithmic Proof",
        "Observational / Case Study",
        "Qualitative / Conceptual Review"
    ]
    rigor_score: float  # 0.0 to 1.0
    evidence_weight_multiplier: float  # 0.80 to 1.25
    sample_size_or_datasets_count: int
    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class MethodologyRigorReport(BaseModel):
    overall_corpus_rigor: float
    high_rigor_paper_ratio: float
    methodology_distribution: Dict[str, int] = Field(default_factory=dict)
    assessments: List[MethodologyAssessment] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SynthesisReport(BaseModel):
    query: str
    decomposed_subqueries: List[str] = Field(default_factory=list)
    overall_confidence: float
    executive_summary: str
    key_findings: List[Dict[str, Any]] = Field(default_factory=list)
    conflicting_findings: List[ConflictAttribution] = Field(default_factory=list)
    claims: List[FactualClaim] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    citation_audit: CitationAudit
    evidence_graph: EvidenceGraphData
    references: List[Dict[str, Any]] = Field(default_factory=list)
    meta_analysis: Optional[EmpiricalMetaAnalysis] = None
    robustness: Optional[RobustnessReport] = None
    latex_code: Optional[str] = None
    bibtex_code: Optional[str] = None
    verification_trajectory: Optional[VerificationTrajectory] = None
    hypotheses: List[ScientificHypothesis] = Field(default_factory=list)
    rigor_report: Optional[MethodologyRigorReport] = None
    sankey_data: Optional[Dict[str, Any]] = None
    peer_review: Optional[PeerReviewReport] = None
    consensus_council: Optional[ConsensusCouncilReport] = None
    bibliometrics: Optional[BibliometricNetworkReport] = None
    slide_deck: Optional[SlideDeckReport] = None
    causal_dag: Optional[CausalDAGReport] = None
    reproducibility: Optional[CorpusReproducibilityReport] = None
    temporal_roadmap: Optional[TemporalRoadmapReport] = None
    dossier: Optional[AutonomousResearchDossier] = None


# --- v3.0 Advanced Research Intelligence Schemas ---

class PeerReviewScoreCard(BaseModel):
    soundness_score: int = Field(default=4, ge=1, le=5, description="Technical soundness and empirical validity (1-5)")
    presentation_score: int = Field(default=5, ge=1, le=5, description="Clarity, structure and rigor of exposition (1-5)")
    contribution_score: int = Field(default=4, ge=1, le=5, description="Novelty and significance of scientific contributions (1-5)")
    overall_recommendation: int = Field(default=8, ge=1, le=10, description="Overall conference recommendation (1-10: Strong Reject to Strong Accept)")
    confidence_score: int = Field(default=5, ge=1, le=5, description="Reviewer confidence (1-5)")
    recommendation_label: str = "Accept (Poster / Spotlight)"


class PeerReviewReport(BaseModel):
    conference_venue: str = "NeurIPS / ICLR Academic Review Track"
    score_card: PeerReviewScoreCard = Field(default_factory=PeerReviewScoreCard)
    literature_summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    questions_for_authors: List[str] = Field(default_factory=list)
    missing_baselines: List[str] = Field(default_factory=list)
    reproducibility_audit: Dict[str, Any] = Field(default_factory=dict)
    ethical_and_societal_implications: str = ""


class DebateTurn(BaseModel):
    speaker_id: str
    speaker_name: str
    persona_title: str
    turn_type: Literal["opening", "critique", "counter_evidence", "synthesis"] = "opening"
    content: str
    cited_papers: List[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    round_number: int
    round_theme: str
    turns: List[DebateTurn] = Field(default_factory=list)


class ConsensusResolutionVerdict(BaseModel):
    claim_id: str
    target_statement: str
    verdict_type: Literal[
        "Conditional Consensus",
        "Irreconcilable Empirical Discrepancy",
        "Methodological Artifact",
        "Distribution Shift Confirmed",
        "Unsubstantiated Premise"
    ] = "Conditional Consensus"
    confidence: float = 0.90
    golden_synthesis_principle: str = ""
    concrete_boundary_rule: str = ""
    unanimous_vote: bool = True
    dissenting_opinions: List[str] = Field(default_factory=list)


class ConsensusCouncilReport(BaseModel):
    council_topic: str
    active_claim_id: Optional[str] = None
    participating_agents: List[Dict[str, str]] = Field(default_factory=list)
    debate_rounds: List[DebateRound] = Field(default_factory=list)
    verdict: ConsensusResolutionVerdict
    council_summary: str = ""


class BibliometricPaperMetric(BaseModel):
    paper_id: str
    paper_title: str
    year: int
    pagerank_centrality: float = 0.0
    citation_degree: int = 0
    co_citation_strength: float = 0.0
    impact_tier: Literal["Foundational Anchor", "Methodological Bridge", "Specialized Frontier"] = "Foundational Anchor"
    inbound_claim_links: int = 0
    outbound_citation_links: int = 0


class BibliometricNetworkReport(BaseModel):
    total_papers_analyzed: int = 0
    graph_density: float = 0.0
    foundational_anchors: List[str] = Field(default_factory=list)
    methodological_bridges: List[str] = Field(default_factory=list)
    paper_metrics: List[BibliometricPaperMetric] = Field(default_factory=list)
    co_citation_clusters: Dict[str, List[str]] = Field(default_factory=dict)


class SlideItem(BaseModel):
    slide_id: int
    title: str
    subtitle: Optional[str] = None
    layout_type: Literal["title", "split", "metrics", "verdict", "table", "conclusion"] = "split"
    bullets: List[str] = Field(default_factory=list)
    highlight_metric: Optional[Dict[str, str]] = None
    accent_badge: Optional[str] = None


class SlideDeckReport(BaseModel):
    deck_title: str
    total_slides: int = 0
    slides: List[SlideItem] = Field(default_factory=list)
    html_deck: str = ""
    marp_markdown: str = ""


# --- v4.0 Autonomous Meta-Scientist & Research Frontier Schemas ---

class CausalNode(BaseModel):
    id: str
    name: str
    variable_type: Literal["treatment", "outcome", "mediator", "confounder", "moderator"] = "treatment"
    description: str = ""
    domain_entity: Optional[str] = None
    observed_mean_impact: float = 0.0


class CausalEdge(BaseModel):
    source: str
    target: str
    relation_type: Literal["CAUSES", "MEDIATES", "CONFOUNDS", "MODERATES", "INHIBITS"] = "CAUSES"
    effect_direction: Literal["POSITIVE", "NEGATIVE", "NON_LINEAR", "AMBIGUOUS"] = "POSITIVE"
    causal_confidence: float = 0.85
    do_intervention_effect: float = 0.75  # P(Y | do(X)) intervention effect
    confounding_variables: List[str] = Field(default_factory=list)
    supporting_papers: List[str] = Field(default_factory=list)
    empirical_evidence_snippet: str = ""


class CausalDAGReport(BaseModel):
    target_query: str
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)
    overall_causal_soundness_score: float = 0.85
    confounding_vulnerability_index: float = 0.25
    spurious_correlations_detected: List[Dict[str, str]] = Field(default_factory=list)
    interventional_insights: List[str] = Field(default_factory=list)
    mermaid_diagram: str = ""


class PaperReproducibilityAudit(BaseModel):
    paper_id: str
    paper_title: str
    year: int
    open_code_available: bool = True
    open_code_url: Optional[str] = None
    open_weights_available: bool = True
    open_data_available: bool = True
    hardware_documented: bool = True
    gpu_tpu_footprint_hours: Optional[float] = None
    random_seeds_reported: bool = True
    error_bars_present: bool = True
    statistical_significance_tested: bool = True
    reproducibility_risk_score: float = 0.15  # 0.0 (perfect) to 1.0 (unreplicable)
    open_science_grade: Literal["A", "B", "C", "D", "F"] = "A"
    audit_notes: List[str] = Field(default_factory=list)


class CorpusReproducibilityReport(BaseModel):
    corpus_size: int = 0
    mean_reproducibility_risk_index: float = 0.25  # 0.0 to 1.0
    open_code_ratio: float = 0.80
    open_data_ratio: float = 0.75
    statistical_rigor_ratio: float = 0.70
    overall_open_science_grade: Literal["A+", "A", "B", "C", "D", "F"] = "A"
    paper_audits: List[PaperReproducibilityAudit] = Field(default_factory=list)
    critical_reproducibility_bottlenecks: List[str] = Field(default_factory=list)
    remediation_patch_protocol: List[str] = Field(default_factory=list)


class TemporalTrendItem(BaseModel):
    technology_or_paradigm: str
    temporal_status: Literal["ACCELERATING_FRONTIER", "MATURE_CONSOLIDATING", "DECAYING_PARADIGM", "EMERGING_CHALLENGER"]
    start_year: int
    peak_or_current_year: int
    velocity_score: float = 0.85  # rate of publication/claim acceleration
    empirical_support_trend: str
    key_contributing_papers: List[str] = Field(default_factory=list)


class ResearchHorizonMilestone(BaseModel):
    milestone_id: str
    target_timeframe: Literal["0-6 Months (Near-Term)", "6-12 Months (Mid-Term)", "12-24 Months (Long-Term Frontier)"]
    milestone_title: str
    technical_objective: str
    success_metric: str
    strategic_grant_or_patent_relevance: str
    recommended_methods: List[str] = Field(default_factory=list)


class TemporalRoadmapReport(BaseModel):
    analysis_domain: str
    timespan_analyzed: str
    paradigm_trends: List[TemporalTrendItem] = Field(default_factory=list)
    decaying_approaches: List[str] = Field(default_factory=list)
    emerging_frontiers: List[str] = Field(default_factory=list)
    projected_milestones: List[ResearchHorizonMilestone] = Field(default_factory=list)
    strategic_takeaways: List[str] = Field(default_factory=list)


class AutonomousResearchDossier(BaseModel):
    dossier_title: str
    dossier_id: str
    generated_timestamp: str
    target_inquiry: str
    executive_lead_summary: str
    layman_public_briefing: str
    technical_breakthrough_table: List[Dict[str, Any]] = Field(default_factory=list)
    scientific_myths_debunked: List[Dict[str, str]] = Field(default_factory=list)
    causal_and_reproducibility_verdict: str
    two_year_actionable_mandate: List[str] = Field(default_factory=list)
    html_dossier: str = ""
    markdown_dossier: str = ""




