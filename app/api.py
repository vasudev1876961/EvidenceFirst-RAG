"""FastAPI REST API for EvidenceFirst-RAG."""

from __future__ import annotations
import os
import sys
import yaml
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    Paper,
    SynthesisReport,
    RobustnessReport,
    EmpiricalMetaAnalysis,
    CitationAudit,
)
from src.pipeline import EvidenceFirstPipeline
from src.ingestion.paper_loader import PaperLoader
from src.evaluation.benchmark import BenchmarkRunner

# Global Pipeline Instance
pipeline_instance: Optional[EvidenceFirstPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline and index seed corpus on application startup."""
    global pipeline_instance
    config_path = "configs/config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    pipeline_instance = EvidenceFirstPipeline(config=config)
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipeline_instance.index_papers(papers)
    yield


app = FastAPI(
    title="EvidenceFirst-RAG API",
    description="RESTful API for Claim-Level Evidence Verification, In Silico Laboratory Experiment Simulation, Cross-Domain Analogy Transfer, Novelty & Semantic Drift Sentinel, Causal DAG Reasoning, Reproducibility Auditing, and Research Dossiers",
    version="0.5.0",
    lifespan=lifespan,
)


# --- Request & Response Schemas ---
class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What are the limitations of Vision Transformers for low-light object detection?"})
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Balance between BM25 (1.0) and Dense (0.0)")
    domain: Optional[str] = Field(None, description="Optional scientific domain filter")



class ArxivSearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "low-light object detection vision transformer"})
    max_results: int = Field(3, ge=1, le=10)
    auto_index: bool = Field(True, description="Whether to immediately index retrieved papers into active corpus")


class PaperIngestRequest(BaseModel):
    paper: Paper


@app.get("/health", tags=["System"])
def health_check():
    """Health status and corpus indexing metrics."""
    if not pipeline_instance:
        return {"status": "initializing"}
    return {
        "status": "healthy",
        "service": "EvidenceFirst-RAG",
        "version": "0.3.0",
        "indexed_papers_count": len(pipeline_instance.indexed_papers),
        "indexed_chunks_count": len(pipeline_instance.indexed_chunks),
    }


@app.post("/api/v1/query", response_model=Dict[str, Any], tags=["Research Synthesis"])
def analyze_query(req: QueryRequest):
    """Execute end-to-end EvidenceFirst-RAG pipeline on a scientific query."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    pipeline_instance.hybrid.alpha = req.alpha
    report = pipeline_instance.run(req.query)

    return {
        "query": report.query,
        "executive_summary": report.executive_summary,
        "overall_confidence": report.overall_confidence,
        "decomposed_subqueries": report.decomposed_subqueries,
        "citation_audit": {
            "evidence_coverage_score": report.citation_audit.evidence_coverage_score,
            "citation_integrity_score": report.citation_audit.citation_integrity_score,
            "supported_claims": report.citation_audit.supported_claims,
            "total_claims": report.citation_audit.total_claims,
            "conflicting_claims": report.citation_audit.conflicting_claims,
        },
        "claims": [
            {
                "claim_id": c.claim_id,
                "statement": c.statement,
                "status": c.evidence_status,
                "confidence": c.confidence_score,
                "supporting_count": len(c.supporting_evidence),
                "contradicting_count": len(c.contradicting_evidence),
            }
            for c in report.claims
        ],
        "conflicting_findings": [
            {
                "claim_id": cf.claim_id,
                "statement": cf.claim_statement,
                "likely_reason": cf.likely_reason,
                "supporting_papers": [p.get("paper_id") for p in cf.supporting_papers],
                "contradicting_papers": [p.get("paper_id") for p in cf.contradicting_papers],
                "divergent_factors": cf.divergent_factors,
            }
            for cf in report.conflicting_findings
        ],
        "research_gaps": report.research_gaps,
        "empirical_insights": report.meta_analysis.key_empirical_insights if report.meta_analysis else [],
        "evidence_robustness_index": report.robustness.evidence_robustness_index if report.robustness else 1.0,
        "hypotheses": [h.model_dump() for h in report.hypotheses],
        "verification_trajectory": report.verification_trajectory.model_dump() if report.verification_trajectory else None,
        "rigor_report": report.rigor_report.model_dump() if report.rigor_report else None,
        "peer_review": report.peer_review.model_dump() if report.peer_review else None,
        "consensus_council": report.consensus_council.model_dump() if report.consensus_council else None,
        "bibliometrics": report.bibliometrics.model_dump() if report.bibliometrics else None,
        "slide_deck": report.slide_deck.model_dump() if report.slide_deck else None,
    }




@app.post("/api/v1/arxiv/search", tags=["Live Ingestion"])
def search_arxiv(req: ArxivSearchRequest):
    """Query live ArXiv API and optionally index found papers immediately."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.auto_index:
        papers = pipeline_instance.fetch_and_index_arxiv(req.query, max_results=req.max_results)
    else:
        papers = pipeline_instance.arxiv_fetcher.search(req.query, max_results=req.max_results)

    return {
        "query": req.query,
        "retrieved_count": len(papers),
        "auto_indexed": req.auto_index,
        "total_corpus_size": len(pipeline_instance.indexed_papers),
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "venue": p.venue,
                "url": p.url,
                "abstract": p.abstract[:200] + "...",
                "metadata": p.metadata,
            }
            for p in papers
        ]
    }


@app.post("/api/v1/ingest/paper", tags=["Live Ingestion"])
def ingest_paper(req: PaperIngestRequest):
    """Ingest a single structured Paper into active retrieval index."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    existing = [p for p in pipeline_instance.indexed_papers if p.paper_id != req.paper.paper_id]
    existing.append(req.paper)
    pipeline_instance.index_papers(existing)

    return {
        "status": "success",
        "ingested_paper_id": req.paper.paper_id,
        "total_papers": len(pipeline_instance.indexed_papers),
        "total_chunks": len(pipeline_instance.indexed_chunks),
    }


@app.get("/api/v1/corpus", tags=["Corpus"])
def get_corpus():
    """List all papers currently indexed in the active pipeline."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return {
        "total_papers": len(pipeline_instance.indexed_papers),
        "papers": [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "venue": p.venue,
                "dataset": p.metadata.get("dataset"),
                "model": p.metadata.get("model_architecture"),
                "metric": p.metadata.get("evaluation_metric"),
            }
            for p in pipeline_instance.indexed_papers
        ]
    }


@app.get("/api/v1/benchmark", tags=["Evaluation"])
def run_benchmark():
    """Run systematic ablation study across 5 RAG architectures."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    benchmarker = BenchmarkRunner()
    results = benchmarker.run_ablation_study(pipeline_instance)
    return {"benchmark_results": results}


@app.post("/api/v1/export/latex", tags=["Export"])
def export_latex(req: QueryRequest):
    """Execute query and generate publication-ready LaTeX & BibTeX artifacts."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    report = pipeline_instance.run(req.query)
    return {
        "query": report.query,
        "latex_code": report.latex_code,
        "bibtex_code": report.bibtex_code,
    }


@app.post("/api/v1/robustness", tags=["Robustness"])
def evaluate_robustness(req: QueryRequest):
    """Execute query and calculate counterfactual leave-one-out claim stability & critical papers."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    report = pipeline_instance.run(req.query)
    return {
        "query": report.query,
        "robustness": report.robustness.model_dump() if report.robustness else None,
    }


@app.get("/api/v1/domains", tags=["Corpus"])
def get_domains():
    """List all supported scientific domains and paper counts."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    all_papers = getattr(pipeline_instance, "all_papers", pipeline_instance.indexed_papers)
    domain_counts: Dict[str, int] = {}
    for p in all_papers:
        d = p.metadata.get("domain", "General")
        domain_counts[d] = domain_counts.get(d, 0) + 1

    return {
        "active_domain_filter": getattr(pipeline_instance, "current_domain", "All Domains"),
        "domains": [
            {"domain": dom, "paper_count": count}
            for dom, count in domain_counts.items()
        ]
    }


@app.post("/api/v1/benchmark/ir", tags=["Evaluation"])
def run_ir_benchmark(domain: Optional[str] = Query(None, description="Optional domain filter")):
    """Run formal Information Retrieval benchmark (MRR, nDCG, Precision, Recall)."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    benchmarker = BenchmarkRunner()
    results = benchmarker.run_ir_benchmark(pipeline_instance, domain=domain)
    return results


@app.post("/api/v1/export/html", tags=["Export"])
def export_html(req: QueryRequest):
    """Execute query and generate publication-grade standalone HTML report."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    html_code = pipeline_instance.report_generator.format_html(report)
    return {
        "query": report.query,
        "html_code": html_code,
    }


@app.post("/api/v1/hypotheses", tags=["Scientific Intelligence"])
def generate_hypotheses(req: QueryRequest):
    """Synthesize formal testable scientific hypotheses resolving observed literature contradictions."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    return {
        "query": report.query,
        "hypotheses_count": len(report.hypotheses),
        "hypotheses": [h.model_dump() for h in report.hypotheses],
    }


@app.get("/api/v1/rigor/assess", tags=["Methodology"])
def assess_methodological_rigor():
    """Assess methodological tiers and evidence hierarchy weights across active corpus."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    report = pipeline_instance.rigor_analyzer.analyze_corpus(pipeline_instance.indexed_papers)
    return report.model_dump()


@app.post("/api/v1/verify/iterative", tags=["Agentic Verification"])
def run_iterative_verification(req: QueryRequest):
    """Execute multi-hop agentic verification loop and return auditable trajectory."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    return {
        "query": report.query,
        "trajectory": report.verification_trajectory.model_dump() if report.verification_trajectory else None,
        "claims": [
            {
                "claim_id": c.claim_id,
                "statement": c.statement,
                "status": c.evidence_status,
                "confidence": c.confidence_score,
            }
            for c in report.claims
        ],
    }


@app.post("/api/v1/visualize/sankey", tags=["Visualization"])
def export_sankey_visualization(req: QueryRequest):
    """Generate standalone interactive Sankey diagram HTML visualizing literature verification flow."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    html_code = pipeline_instance.graph_visualizer.generate_sankey_html(report)
    return {
        "query": report.query,
        "sankey_data": report.sankey_data,
        "html_code": html_code,
    }


@app.post("/api/v1/visualize/network", tags=["Visualization"])
def export_network_visualization(req: QueryRequest):
    """Generate standalone interactive Physics Network Graph HTML visualizing evidence entailment & contradictions."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    html_code = pipeline_instance.graph_visualizer.generate_network_html(report)
    return {
        "query": report.query,
        "nodes_count": len(report.evidence_graph.nodes),
        "edges_count": len(report.evidence_graph.edges),
        "html_code": html_code,
    }


@app.post("/api/v1/review/peer-review", tags=["Peer Review"])
def run_peer_review(req: QueryRequest):
    """Conduct formal NeurIPS / ICLR conference-grade peer review and soundness audit."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    review = pipeline_instance.peer_reviewer.review_synthesis(report, pipeline_instance.indexed_papers)
    return {
        "query": req.query,
        "conference_venue": review.conference_venue,
        "score_card": review.score_card.model_dump(),
        "literature_summary": review.literature_summary,
        "strengths": review.strengths,
        "weaknesses": review.weaknesses,
        "questions_for_authors": review.questions_for_authors,
        "missing_baselines": review.missing_baselines,
        "reproducibility_audit": review.reproducibility_audit,
        "ethical_and_societal_implications": review.ethical_and_societal_implications,
    }


@app.post("/api/v1/council/deliberate", tags=["Consensus Council"])
def deliberate_consensus_council(req: QueryRequest):
    """Convene 3-agent scientific debate council (Empirical Skeptic, Methodological Purist, Theoretical Synthesizer) to resolve contradictions."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    council = pipeline_instance.consensus_council.deliberate(
        query=req.query,
        conflicts=report.conflicting_findings,
        claims=report.claims,
    )
    return {
        "query": req.query,
        "council_topic": council.council_topic,
        "participating_agents": council.participating_agents,
        "debate_rounds": [r.model_dump() for r in council.debate_rounds],
        "verdict": council.verdict.model_dump(),
        "council_summary": council.council_summary,
    }


@app.get("/api/v1/network/bibliometrics", tags=["Bibliometrics"])
@app.post("/api/v1/network/bibliometrics", tags=["Bibliometrics"])
def get_bibliometrics_network(req: Optional[QueryRequest] = None):
    """Compute PageRank centrality, impact tiers, and co-citation clusters across the indexed corpus."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    claims = []
    if req and req.query:
        if req.domain:
            pipeline_instance.filter_by_domain(req.domain)
        report = pipeline_instance.run(req.query)
        claims = report.claims

    biblio = pipeline_instance.bibliometric_analyzer.analyze_network(
        papers=pipeline_instance.indexed_papers,
        claims=claims,
    )
    return biblio.model_dump()


@app.post("/api/v1/export/slides", tags=["Export"])
def export_slide_deck(req: QueryRequest):
    """Generate academic slide deck in interactive HTML and Marp Markdown formats."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    deck = pipeline_instance.slide_generator.build_slide_deck(report)
    return {
        "query": req.query,
        "deck_title": deck.deck_title,
        "total_slides": deck.total_slides,
        "slides": [s.model_dump() for s in deck.slides],
        "html_deck": deck.html_deck,
        "marp_markdown": deck.marp_markdown,
    }


# --- v4.0 Autonomous Meta-Scientist Endpoints ---

@app.post("/api/v1/causal/dag", tags=["Causal Reasoning"])
def get_causal_dag(req: QueryRequest):
    """Construct Causal Directed Acyclic Graph (DAG) with Do-Calculus interventional scoring."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    return report.causal_dag.model_dump() if report.causal_dag else {}


@app.get("/api/v1/audit/reproducibility", tags=["Reproducibility"])
@app.post("/api/v1/audit/reproducibility", tags=["Reproducibility"])
def audit_reproducibility(req: Optional[QueryRequest] = None):
    """Conduct comprehensive Open Science and Statistical Reproducibility Audit across active corpus."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req and req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    repro_report = pipeline_instance.reproducibility_verifier.audit_corpus(
        papers=pipeline_instance.indexed_papers
    )
    return repro_report.model_dump()


@app.get("/api/v1/roadmap/temporal", tags=["Research Roadmap"])
@app.post("/api/v1/roadmap/temporal", tags=["Research Roadmap"])
def get_temporal_roadmap(req: Optional[QueryRequest] = None):
    """Forecast living research milestones, paradigm decay, and frontier velocity."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    query = req.query if req else "Vision Transformers for low-light object detection"
    if req and req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(query)
    return report.temporal_roadmap.model_dump() if report.temporal_roadmap else {}


@app.post("/api/v1/export/dossier", tags=["Export"])
def export_research_dossier(req: QueryRequest):
    """Generate official Autonomous Research Dossier & Press Briefing in HTML and Markdown formats."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    dossier = pipeline_instance.dossier_generator.build_dossier(report)
    return dossier.model_dump()


# --- v5.0 Frontier Scientific Intelligence Endpoints ---

class InSilicoSimulationRequest(BaseModel):
    query: str = Field("Vision Transformers in low-light object detection", description="Target research query")
    trials_count: int = Field(1000, ge=10, le=5000, description="Monte Carlo trials count")
    domain: Optional[str] = Field(None, description="Domain filter")


@app.post("/api/v1/simulation/in-silico", tags=["In Silico Laboratory Simulator"])
def run_in_silico_simulation(req: InSilicoSimulationRequest):
    """Execute prospective in silico empirical experiment simulation, Monte Carlo trials, and statistical power analysis."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    top_hyp = report.hypotheses[0] if report.hypotheses else None

    sim_report = pipeline_instance.in_silico_simulator.simulate(
        query=req.query,
        hypothesis=top_hyp,
        claims=report.claims,
        meta_analysis=report.meta_analysis,
        trials_count=req.trials_count,
    )
    return sim_report.model_dump()


@app.get("/api/v1/simulation/harness", tags=["In Silico Laboratory Simulator"])
def get_simulation_harness():
    """Download executable PyTorch benchmark simulation reproduction script."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    sim_report = pipeline_instance.in_silico_simulator.simulate(
        query="Vision Transformers in low-light object detection",
        trials_count=100,
    )
    return {
        "filename": "in_silico_benchmark.py",
        "code": sim_report.executable_python_benchmark,
    }


@app.get("/api/v1/export/simulation-lab", tags=["Export"])
def export_simulation_lab(query: str = "Vision Transformers in low-light object detection"):
    """Export standalone interactive HTML In Silico Laboratory Simulator."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    sim_report = pipeline_instance.in_silico_simulator.simulate(
        query=query,
        trials_count=1000,
    )
    return {
        "query": query,
        "interactive_lab_html": sim_report.interactive_lab_html,
    }


@app.post("/api/v1/transfer/cross-domain", tags=["Cross-Domain Analogies"])
def discover_cross_domain_transfer(req: QueryRequest):
    """Discover isomorphic structural patterns and interdisciplinary architectural transfer blueprints."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    transfer_report = pipeline_instance.analogy_engine.discover_analogies(
        query=req.query,
        claims=report.claims,
        hypotheses=report.hypotheses,
    )
    return transfer_report.model_dump()


@app.post("/api/v1/audit/novelty", tags=["Novelty & Semantic Drift Sentinel"])
def audit_novelty_and_drift(req: QueryRequest):
    """Run academic novelty index, prior art collision detection, and terminology semantic drift audit."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if req.domain:
        pipeline_instance.filter_by_domain(req.domain)

    report = pipeline_instance.run(req.query)
    sentinel_report = pipeline_instance.novelty_sentinel.audit_novelty(
        query=req.query,
        claims=report.claims,
        hypotheses=report.hypotheses,
        papers=pipeline_instance.indexed_papers,
    )
    return sentinel_report.model_dump()




