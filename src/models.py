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
    in_silico_simulation: Optional[InSilicoExperimentReport] = None
    cross_domain_transfer: Optional[InterdisciplinaryTransferReport] = None
    novelty_sentinel: Optional[NoveltySentinelReport] = None
    symbolic_proof_report: Optional[SymbolicProofReport] = None
    adversarial_stress_report: Optional[AdversarialStressReport] = None
    prisma_meta_analysis: Optional[PrismaMetaAnalysisReport] = None
    bayesian_belief_report: Optional[BayesianBeliefReport] = None
    preregistration_dossier: Optional[PreregistrationDossier] = None
    quantum_epistemic_report: Optional[QuantumEpistemicReport] = None



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


# --- v5.0 Frontier Scientific Intelligence & In Silico Assistant Schemas ---

class SimulationParamSweep(BaseModel):
    parameter_name: str
    unit: str
    min_value: float
    max_value: float
    simulated_steps: int
    optimal_operating_point: float
    degradation_threshold: float


class MonteCarloTrialResult(BaseModel):
    trial_id: int
    noise_sigma: float
    lux_level: float
    baseline_metric: float  # e.g., mAP baseline
    proposed_metric: float  # e.g., mAP with proposed method
    relative_gain_pct: float
    latency_ms: float
    converged: bool = True


class HardwareScalingEstimate(BaseModel):
    target_hardware: str  # e.g. "NVIDIA RTX 4090", "NVIDIA A100 (80GB)", "NVIDIA H100 SXM5", "Google TPU v5e"
    vram_required_gb: float
    throughput_samples_sec: float
    training_time_hours: float
    estimated_cost_usd: float
    co2_kg_footprint: float


class PowerAnalysisReport(BaseModel):
    alpha: float = 0.05
    statistical_power_achieved: float = 0.88  # 1 - beta
    cohens_d_effect_size: float = 1.15
    hedges_g_effect_size: float = 1.12
    critical_sample_size_required: int = 42
    hypothesis_falsification_risk: float = 0.12  # probability of failing to reject null


class InSilicoExperimentReport(BaseModel):
    simulation_id: str
    hypothesis_id: str
    target_hypothesis_title: str
    simulated_domain: str
    trials_count: int = 1000
    mean_baseline_performance: float
    mean_simulated_performance: float
    projected_net_gain_pct: float
    confidence_interval_95: List[float] = Field(default_factory=list)  # [lower, upper]
    std_dev: float
    power_analysis: PowerAnalysisReport
    parameter_sweeps: List[SimulationParamSweep] = Field(default_factory=list)
    sample_trials: List[MonteCarloTrialResult] = Field(default_factory=list)
    hardware_projections: List[HardwareScalingEstimate] = Field(default_factory=list)
    executable_python_benchmark: str = ""
    interactive_lab_html: str = ""


class CrossDomainAnalogy(BaseModel):
    analogy_id: str
    source_domain: str
    target_domain: str
    isomorphic_challenge: str
    shared_mathematical_form: str
    source_solution_mechanism: str
    proposed_hybrid_architecture: str
    feasibility_score: float = 0.85
    novelty_score: float = 0.90
    expected_empirical_gain: str
    validation_protocol: str


class InterdisciplinaryTransferReport(BaseModel):
    active_inquiry: str
    primary_domain: str
    neighboring_domains_scanned: List[str] = Field(default_factory=list)
    analogies: List[CrossDomainAnalogy] = Field(default_factory=list)
    isomorphism_density_score: float = 0.78
    top_recommended_transfer: Optional[CrossDomainAnalogy] = None
    interdisciplinary_takeaways: List[str] = Field(default_factory=list)


class SemanticDriftRecord(BaseModel):
    term: str
    original_formal_definition: str
    observed_literature_usage: str
    dilution_score: float  # 0.0 (rigorous) to 1.0 (buzzword inflation)
    verdict: Literal["Mathematically Rigorous", "Moderate Drift", "Severe Dilution / Buzzword"]
    audit_recommendation: str


class PriorArtCollision(BaseModel):
    collision_id: str
    overlapping_paper_id: str
    overlapping_paper_title: str
    similarity_score: float
    shared_mechanisms: List[str] = Field(default_factory=list)
    differentiation_argument: str
    infringement_or_redundancy_risk: Literal["Low", "Moderate", "High"] = "Low"


class NoveltySentinelReport(BaseModel):
    mean_novelty_index: float  # 0.0 to 1.0
    novelty_quadrant_distribution: Dict[str, int] = Field(default_factory=dict)
    frontier_breakthroughs: List[str] = Field(default_factory=list)
    speculative_frontiers: List[str] = Field(default_factory=list)
    established_consensuses: List[str] = Field(default_factory=list)
    redundant_clutters: List[str] = Field(default_factory=list)
    prior_art_collisions: List[PriorArtCollision] = Field(default_factory=list)
    semantic_drift_records: List[SemanticDriftRecord] = Field(default_factory=list)
    overall_sentinel_verdict: str


# --- v6.0 Autonomous Meta-Science & Epistemic Synthesis Schemas ---

class SymbolicProofCertificate(BaseModel):
    proof_id: str
    target_claim_or_theorem: str
    domain_formalism: str  # e.g., "Poisson Shot Noise & Attention Degradation Bounds"
    sympy_formulation: str
    formal_derivation_steps: List[str] = Field(default_factory=list)
    asymptotic_bound: str
    lipschitz_constant_or_gradient_bound: Optional[str] = None
    verdict: Literal["VERIFIED_THEOREM", "CONDITIONALLY_BOUNDED", "FALSIFIED_COUNTEREXAMPLE"] = "VERIFIED_THEOREM"
    unstated_assumptions: List[str] = Field(default_factory=list)
    counterexample_conditions: Optional[str] = None
    latex_proof_render: str = ""


class SymbolicProofReport(BaseModel):
    proof_system_id: str
    analyzed_query: str
    theorems_evaluated: int
    proof_certificates: List[SymbolicProofCertificate] = Field(default_factory=list)
    formal_consistency_score: float  # 0.0 to 1.0
    critical_theoretical_breakdowns: List[str] = Field(default_factory=list)
    executable_sympy_script: str = ""


class AdversarialAttackVector(BaseModel):
    attack_id: str
    attack_category: Literal[
        "Simpsons_Paradox_Reversal",
        "Metric_Cherry_Picking_Inversion",
        "P_Hacking_Subgroup_Dredging",
        "Adversarial_Citation_Poisoning",
        "Semantic_Hallucination_Trap"
    ]
    attack_prompt_or_perturbation: str
    intended_deception_target: str
    pipeline_defended: bool = True
    vulnerability_severity: Literal["Low", "Medium", "High", "Critical"] = "Low"
    observed_failure_or_defense: str
    remediation_patch: str


class AdversarialStressReport(BaseModel):
    stress_test_id: str
    epistemic_vulnerability_index: float  # 0.0 (fully defended) to 1.0 (vulnerable)
    total_attacks_simulated: int
    attacks_successfully_defended: int
    defense_rate_pct: float
    attack_vectors: List[AdversarialAttackVector] = Field(default_factory=list)
    simpsons_paradox_audit: Dict[str, Any] = Field(default_factory=dict)
    metric_robustness_gap: float = 0.12
    adversarial_resilience_verdict: Literal[
        "Hardened Research Defense",
        "Moderate Epistemic Armor",
        "Vulnerable to Subtle Fallacies"
    ] = "Hardened Research Defense"
    automated_defense_rules: List[str] = Field(default_factory=list)


class PrismaFlowRecord(BaseModel):
    identification_count: int = 14
    screening_count: int = 14
    excluded_screening_count: int = 3
    eligibility_assessed_count: int = 11
    excluded_eligibility_count: int = 2
    included_synthesis_count: int = 9
    flow_exclusion_reasons: Dict[str, int] = Field(default_factory=dict)


class CochraneRiskOfBiasItem(BaseModel):
    study_id: str
    study_title: str
    domain_1_randomization_selection: Literal["Low", "Some Concerns", "High"] = "Low"
    domain_2_deviations_interventions: Literal["Low", "Some Concerns", "High"] = "Low"
    domain_3_missing_outcome_data: Literal["Low", "Some Concerns", "High"] = "Low"
    domain_4_measurement_outcome: Literal["Low", "Some Concerns", "High"] = "Low"
    domain_5_selective_reporting: Literal["Low", "Some Concerns", "High"] = "Low"
    overall_risk_of_bias: Literal["Low Risk", "Some Concerns", "High Risk"] = "Low Risk"
    rationale: str


class ForestPlotStudyRecord(BaseModel):
    study_id: str
    study_label: str
    year: int
    sample_size: int
    effect_size: float  # Standardized Mean Difference (Hedges' g)
    confidence_interval_95: List[float] = Field(default_factory=list)  # [ci_lower, ci_upper]
    weight_pct: float
    favors_direction: Literal["Favors Proposed", "Favors Baseline", "Neutral"] = "Favors Proposed"


class PrismaMetaAnalysisReport(BaseModel):
    prisma_id: str
    inquiry_domain: str
    prisma_flow: PrismaFlowRecord
    cochrane_risk_of_bias: List[CochraneRiskOfBiasItem] = Field(default_factory=list)
    forest_plot_studies: List[ForestPlotStudyRecord] = Field(default_factory=list)
    pooled_random_effects_estimate: float
    pooled_confidence_interval_95: List[float] = Field(default_factory=list)
    heterogeneity_i2_pct: float
    cochran_q_statistic: float
    cochran_q_p_value: float
    tau_squared_variance: float
    eggers_intercept_beta0: float
    eggers_p_value: float
    publication_bias_detected: bool
    overall_synthesis_strength: Literal[
        "High Certainty Evidence (Grade A)",
        "Moderate Certainty Evidence (Grade B)",
        "Low Certainty Evidence (Grade C)"
    ] = "High Certainty Evidence (Grade A)"
    interactive_forest_html: str = ""



# --- v7.0 Bayesian Epistemics, OSF Preregistration & Quantum Epistemic Schemas ---

class BayesFactorRecord(BaseModel):
    evidence_id: str
    paper_id: str
    paper_title: str
    direction: Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"]
    raw_bayes_factor_10: float  # BF10 = P(E | H1) / P(E | H0)
    log10_bayes_factor: float
    evidence_weight: float
    sample_size: int
    jeffreys_interpretation: Literal[
        "Decisive Evidence for H1",
        "Very Strong Evidence for H1",
        "Strong Evidence for H1",
        "Moderate Evidence for H1",
        "Anecdotal Evidence for H1",
        "Inconclusive / Neutral",
        "Evidence Favors Null H0 (Contradictory)"
    ]


class BayesianTrajectoryPoint(BaseModel):
    step_index: int
    paper_id: str
    paper_title: str
    prior_prob: float
    likelihood_ratio: float
    posterior_prob: float
    log_odds: float
    information_gain_kl: float  # D_KL in nats


class BayesianClaimBelief(BaseModel):
    claim_id: str
    statement: str
    prior_credence: float = 0.50
    posterior_credence: float = 0.50
    bayes_factor_aggregate_10: float = 1.0
    epistemic_information_gain_nats: float = 0.0
    credible_interval_95_hdi: List[float] = Field(default_factory=lambda: [0.0, 1.0])
    belief_shift_magnitude: float = 0.0
    epistemic_state_label: Literal[
        "Empirically Settled (High Credence)",
        "Probable Consensus",
        "High Epistemic Uncertainty",
        "Substantially Debunked",
        "Strongly Refuted (Low Credence)"
    ] = "High Epistemic Uncertainty"
    evidence_updates: List[BayesFactorRecord] = Field(default_factory=list)
    trajectory: List[BayesianTrajectoryPoint] = Field(default_factory=list)


class BayesianBeliefReport(BaseModel):
    report_id: str
    overall_epistemic_entropy_prior: float
    overall_epistemic_entropy_posterior: float
    mean_information_gain_nats: float
    claims_evaluated: int
    claim_beliefs: List[BayesianClaimBelief] = Field(default_factory=list)
    consensus_convergence_rate: float
    calibration_quality_score: float  # Brier-calibrated score 0.0 to 1.0
    epistemic_insights: List[str] = Field(default_factory=list)


class OSFBadgeAudit(BaseModel):
    open_data_badge: bool = True
    open_materials_badge: bool = True
    preregistered_badge: bool = True
    preregistration_status: Literal[
        "Preregistered Confirmatory",
        "Preregistered with Analysis Plan",
        "Exploratory Unregistered"
    ] = "Preregistered Confirmatory"
    badge_criteria_met: List[str] = Field(default_factory=list)


class PreregistrationProtocol(BaseModel):
    protocol_id: str  # e.g. PRG-2026-VIT-001
    title: str
    target_tension_or_hypothesis: str
    study_type: Literal[
        "Confirmatory Benchmark Evaluation",
        "Double-Blind Replication Study",
        "Multi-Center Cross-Dataset Trial",
        "Factorial Ablation Experiment"
    ] = "Confirmatory Benchmark Evaluation"
    primary_hypothesis_h1: str
    secondary_hypotheses: List[str] = Field(default_factory=list)
    null_hypothesis_h0: str
    independent_variables: List[Dict[str, str]] = Field(default_factory=list)
    dependent_variables: List[Dict[str, str]] = Field(default_factory=list)
    sample_size_determination: Dict[str, Any] = Field(default_factory=dict)
    data_exclusion_rules: List[str] = Field(default_factory=list)
    prespecified_statistical_pipeline: List[str] = Field(default_factory=list)
    fdr_correction_method: str = "Benjamini-Hochberg (q=0.05)"
    cryptographic_sha256_hash: str
    osf_badge_audit: OSFBadgeAudit = Field(default_factory=OSFBadgeAudit)
    registration_timestamp: str = "2026-10-19T00:00:00Z"


class PreregistrationDossier(BaseModel):
    dossier_id: str
    protocols: List[PreregistrationProtocol] = Field(default_factory=list)
    total_protocols_synthesized: int
    open_science_compliance_index: float  # 0.0 to 1.0
    preregistration_markdown_document: str = ""
    anti_phacking_safeguards: List[str] = Field(default_factory=list)


class BlochSphereCoordinate(BaseModel):
    theta_rad: float  # polar angle [0, pi]
    phi_rad: float    # azimuthal angle [0, 2*pi]
    x: float          # x = sin(theta)*cos(phi)
    y: float          # y = sin(theta)*sin(phi)
    z: float          # z = cos(theta)
    latitude_deg: float
    longitude_deg: float


class QuantumInterferenceRecord(BaseModel):
    pair_id: str
    study_a_id: str
    study_b_id: str
    interference_type: Literal["Constructive Consensus", "Destructive Contradiction", "Orthogonal Decoupled"]
    interference_magnitude: float  # 2 * Re( <Support|rho|Contradict> )
    phase_shift_deg: float
    qualitative_explanation: str


class QuantumEpistemicState(BaseModel):
    claim_id: str
    statement: str
    support_amplitude_alpha: float  # |alpha|^2 + |beta|^2 = 1
    contradiction_amplitude_beta: float
    superposition_purity: float     # Tr(rho^2), 0.5 to 1.0
    von_neumann_entropy: float      # -Tr(rho ln rho), 0.0 to ln(2)=0.693
    quantum_coherence: float        # 2 * |rho_01|, 0.0 to 1.0
    bloch_coordinates: BlochSphereCoordinate
    epistemic_regime: Literal[
        "Pure Support Eigenstate",
        "Pure Contradiction Eigenstate",
        "Superposed Disagreement (Equatorial)",
        "Decoherent Mixed Evidence"
    ]
    interfering_pairs: List[QuantumInterferenceRecord] = Field(default_factory=list)


class QuantumEpistemicReport(BaseModel):
    report_id: str
    mean_von_neumann_entropy: float
    total_claims_in_superposition: int
    systemic_coherence_index: float
    quantum_states: List[QuantumEpistemicState] = Field(default_factory=list)
    bloch_sphere_html: str = ""
    epistemic_phase_insights: List[str] = Field(default_factory=list)


# Rebuild model for forward references
SynthesisReport.model_rebuild()





