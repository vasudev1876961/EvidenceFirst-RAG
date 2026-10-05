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

