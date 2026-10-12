"""Master Orchestration Pipeline for EvidenceFirst-RAG."""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from src.models import (
    Paper,
    SemanticChunk,
    RetrievalCandidate,
    FactualClaim,
    SynthesisReport,
)
from src.llm.llm_client import BaseLLMClient, get_llm_client
from src.ingestion.paper_loader import PaperLoader
from src.preprocessing.chunker import SemanticChunker
from src.query.query_decomposer import QueryDecomposer
from src.query.query_expander import QueryExpander
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.claims.extractor import ClaimExtractor
from src.claims.normalizer import ClaimNormalizer
from src.claims.claim_graph import EvidenceGraphBuilder
from src.verification.entailment import NLIEntailmentChecker
from src.verification.evidence_score import EvidenceScorer
from src.verification.contradiction import ContradictionEngine
from src.verification.citation_checker import CitationAuditor
from src.synthesis.evidence_synthesizer import EvidenceSynthesizer
from src.synthesis.conflict_analyzer import ConflictAnalyzer
from src.synthesis.report_generator import ReportGenerator
from src.synthesis.meta_analysis import EmpiricalMetaAnalysisEngine
from src.verification.robustness import RobustnessEngine
from src.verification.agentic_verifier import AgenticVerifier
from src.verification.rigor_analyzer import MethodologyRigorAnalyzer
from src.synthesis.latex_exporter import LatexExporter
from src.synthesis.hypothesis_generator import ScientificHypothesisGenerator
from src.claims.graph_visualizer import EvidenceGraphVisualizer
from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.synthesis.peer_review import ScientificPeerReviewer
from src.verification.consensus_council import ScientificConsensusCouncil
from src.claims.citation_network import BibliometricNetworkAnalyzer
from src.synthesis.slide_generator import SlideDeckGenerator
from src.claims.causal_graph import CausalGraphEngine
from src.verification.reproducibility import ReproducibilityVerifier
from src.synthesis.temporal_roadmap import TemporalRoadmapEngine
from src.synthesis.dossier_generator import DossierGenerator

logger = logging.getLogger(__name__)



class EvidenceFirstPipeline:
    """End-to-end Claim-Level Evidence Verification and Contradiction-Aware RAG Pipeline."""


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = self._load_default_config()
        self.config = config or {}

        # 1. LLM Client
        self.llm = get_llm_client(self.config.get("llm", {}))

        # 2. Ingestion & Preprocessing
        self.loader = PaperLoader()
        chunk_cfg = self.config.get("chunking", {})
        self.chunker = SemanticChunker(
            target_chunk_size=chunk_cfg.get("target_chunk_size", 350),
            chunk_overlap=chunk_cfg.get("chunk_overlap", 50),
        )

        # 3. Query Components
        self.decomposer = QueryDecomposer(llm_client=self.llm)
        self.expander = QueryExpander()

        # 4. Retrieval Components
        ret_cfg = self.config.get("retrieval", {})
        bm25_cfg = ret_cfg.get("bm25", {})
        self.bm25 = BM25Retriever(k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75))

        dense_cfg = ret_cfg.get("dense", {})
        self.dense = DenseRetriever(
            model_name=dense_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        )

        hybrid_cfg = ret_cfg.get("hybrid", {})
        self.hybrid = HybridRetriever(
            bm25_retriever=self.bm25,
            dense_retriever=self.dense,
            alpha=hybrid_cfg.get("alpha", 0.5),
        )

        rerank_cfg = ret_cfg.get("reranker", {})
        self.reranker = CrossEncoderReranker(
            model_name=rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

        # 5. Claims & Graph
        self.claim_extractor = ClaimExtractor(llm_client=self.llm)
        self.claim_normalizer = ClaimNormalizer()
        self.graph_builder = EvidenceGraphBuilder()

        # 6. Verification
        ver_cfg = self.config.get("verification", {})
        weights = ver_cfg.get("weights", {})
        self.nli = NLIEntailmentChecker(
            model_name=ver_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-small"),
            llm_client=self.llm,
            use_fallback=ver_cfg.get("use_fallback", False),
        )
        self.scorer = EvidenceScorer(
            w_semantic=weights.get("w_semantic", 0.35),
            w_entailment=weights.get("w_entailment", 0.45),
            w_source=weights.get("w_source", 0.10),
            w_recency=weights.get("w_recency", 0.10),
        )
        self.contradiction_engine = ContradictionEngine(
            nli_checker=self.nli,
            llm_client=self.llm,
        )
        self.auditor = CitationAuditor(nli_checker=self.nli)

        # 7. Synthesis & Reporting
        self.synthesizer = EvidenceSynthesizer(llm_client=self.llm)
        self.conflict_analyzer = ConflictAnalyzer()
        self.report_generator = ReportGenerator()

        # 8. Advanced Scientific Modules
        self.meta_analysis_engine = EmpiricalMetaAnalysisEngine()
        self.robustness_engine = RobustnessEngine(scorer=self.scorer)
        self.latex_exporter = LatexExporter()
        self.arxiv_fetcher = ArxivFetcher()
        self.agentic_verifier = AgenticVerifier(
            hybrid_retriever=self.hybrid,
            reranker=self.reranker,
            contradiction_engine=self.contradiction_engine,
            scorer=self.scorer,
            llm_client=self.llm,
        )
        self.rigor_analyzer = MethodologyRigorAnalyzer()
        self.hypothesis_generator = ScientificHypothesisGenerator(llm_client=self.llm)
        self.graph_visualizer = EvidenceGraphVisualizer()
        self.peer_reviewer = ScientificPeerReviewer(llm_client=self.llm)
        self.consensus_council = ScientificConsensusCouncil(llm_client=self.llm)
        self.bibliometric_analyzer = BibliometricNetworkAnalyzer()
        self.slide_generator = SlideDeckGenerator()
        self.causal_engine = CausalGraphEngine(llm_client=self.llm)
        self.reproducibility_verifier = ReproducibilityVerifier()
        self.temporal_roadmap_engine = TemporalRoadmapEngine()
        self.dossier_generator = DossierGenerator()

        # State

        self.indexed_papers: List[Paper] = []
        self.indexed_chunks: List[SemanticChunk] = []


    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from configs/config.yaml."""
        import os
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    def index_papers(self, papers: List[Paper], domain: Optional[str] = None):
        """Ingest, clean, chunk, and index a collection of papers, optionally filtered by domain."""
        self.all_papers = papers
        if domain and domain != "All Domains":
            domain_lower = domain.lower()
            filtered = [
                p for p in papers
                if domain_lower in p.metadata.get("domain", "").lower()
                or (("vision" in domain_lower or "cv" in domain_lower) and "vision" in p.metadata.get("domain", "").lower())
                or (("nlp" in domain_lower or "language" in domain_lower) and "nlp" in p.metadata.get("domain", "").lower())
                or (("biomedical" in domain_lower or "med" in domain_lower) and "biomedical" in p.metadata.get("domain", "").lower())
            ]
            self.indexed_papers = filtered if filtered else papers
        else:
            self.indexed_papers = papers

        all_chunks: List[SemanticChunk] = []
        for paper in self.indexed_papers:
            chunks = self.chunker.chunk_paper(paper)
            all_chunks.extend(chunks)

        self.indexed_chunks = all_chunks
        logger.info(f"Indexed {len(self.indexed_papers)} papers into {len(all_chunks)} semantic chunks (Domain: {domain or 'All'}).")

        # Update retrieval indices
        self.bm25.index_chunks(all_chunks)
        self.dense.index_chunks(all_chunks)

    def filter_by_domain(self, domain: Optional[str] = None):
        """Filter and re-index active corpus by scientific domain."""
        source_papers = getattr(self, "all_papers", None) or self.indexed_papers
        self.index_papers(source_papers, domain=domain)


    def fetch_and_index_arxiv(self, query: str, max_results: int = 3) -> List[Paper]:
        """Dynamically retrieve recent research from ArXiv and index into the active corpus."""
        papers = self.arxiv_fetcher.search(query=query, max_results=max_results)
        existing_ids = {p.paper_id for p in self.indexed_papers}
        new_papers = [p for p in papers if p.paper_id not in existing_ids]
        if new_papers:
            self.index_papers(self.indexed_papers + new_papers)
            logger.info(f"Ingested and indexed {len(new_papers)} new ArXiv papers into active pipeline.")
        return papers

    def run(self, query: str) -> SynthesisReport:
        """Execute full EvidenceFirst-RAG pipeline on a user query."""
        logger.info(f"Initiating EvidenceFirst-RAG for query: '{query}'")

        # Step 1: Query Decomposition & Expansion
        subqueries = self.decomposer.decompose(query)
        expanded_subqueries = [self.expander.get_expanded_query_string(sq) for sq in subqueries]
        all_search_queries = list(dict.fromkeys(subqueries + expanded_subqueries))

        # Step 2: Hybrid Retrieval across decomposed queries
        hybrid_candidates = self.hybrid.retrieve_multi(all_search_queries, top_candidates_per_query=20)

        # Step 3: Neural Cross-Encoder Reranking
        reranked_candidates = self.reranker.rerank(query, hybrid_candidates, top_n=14)

        # Step 4: Claim Extraction
        raw_claims = self.claim_extractor.extract_claims(query, reranked_candidates)
        claims: List[FactualClaim] = [self.claim_normalizer.normalize(c) for c in raw_claims]

        # Step 5: Claim -> Evidence Verification & Contradiction Detection
        for claim in claims:
            sup, contra, neut = self.contradiction_engine.evaluate_evidence_relationships(
                claim=claim, candidates=reranked_candidates
            )

            # Score each evidence item
            for ev in sup:
                ev.composite_score = self.scorer.compute_score(
                    semantic_sim=ev.semantic_similarity,
                    entailment_prob=ev.entailment_prob,
                    year=ev.year,
                )
            for ev in contra:
                ev.composite_score = self.scorer.compute_score(
                    semantic_sim=ev.semantic_similarity,
                    entailment_prob=ev.contradiction_prob,
                    year=ev.year,
                )

            claim.supporting_evidence = sorted(sup, key=lambda x: x.composite_score, reverse=True)
            claim.contradicting_evidence = sorted(contra, key=lambda x: x.composite_score, reverse=True)
            claim.neutral_evidence = neut

            # Compute claim confidence
            if sup:
                claim.confidence_score = max(e.composite_score for e in sup)
            elif contra:
                claim.confidence_score = max(e.composite_score for e in contra)
            else:
                claim.confidence_score = 0.25

            # Set status
            has_contradictions = len(contra) > 0 and len(sup) > 0
            claim.evidence_status = self.scorer.classify_status(
                confidence=claim.confidence_score,
                has_contradictions=has_contradictions,
            )

            # Conflict Attribution
            if has_contradictions:
                claim.conflict_attribution = self.contradiction_engine.attribute_conflict(
                    claim=claim, supporting=sup, contradicting=contra
                )

        # Step 5b: Agentic Iterative Verification Loop (resolves uncertain & conflicting claims)
        claims, trajectory = self.agentic_verifier.run_verification_loop(claims, max_iterations=1)

        # Step 5c: Methodological Rigor Calibration (weights evidence based on empirical rigor)
        corpus_rigor_report = self.rigor_analyzer.analyze_corpus(self.indexed_papers)
        corpus_assessments = {a.paper_id: a for a in corpus_rigor_report.assessments}
        for claim in claims:
            claim.supporting_evidence = self.rigor_analyzer.apply_rigor_weighting(claim.supporting_evidence, corpus_assessments)
            claim.contradicting_evidence = self.rigor_analyzer.apply_rigor_weighting(claim.contradicting_evidence, corpus_assessments)

        # Step 6: Evidence Knowledge Graph
        graph_data = self.graph_builder.build_graph(query=query, claims=claims)

        # Step 7: Multi-Hop Evidence Synthesis
        synthesis_result = self.synthesizer.synthesize(query=query, claims=claims)
        executive_summary = synthesis_result.get("executive_summary", "")
        research_gaps = synthesis_result.get("research_gaps", [])

        # Step 8: Conflict Analysis
        conflicts = self.conflict_analyzer.analyze_conflicts(claims)

        # Step 9: Citation & Reliability Audit (ECS, CIS, Faithfulness)
        citation_audit = self.auditor.audit(claims)

        # Step 10: Compile Final Synthesis Report
        report = self.report_generator.generate_report(
            query=query,
            decomposed_subqueries=subqueries,
            claims=claims,
            executive_summary=executive_summary,
            conflicts=conflicts,
            research_gaps=research_gaps,
            citation_audit=citation_audit,
            evidence_graph=graph_data,
            papers=self.indexed_papers,
        )

        # Step 11: Cross-Paper Empirical Meta-Analysis Matrix
        report.meta_analysis = self.meta_analysis_engine.extract_from_papers_and_claims(
            papers=self.indexed_papers, claims=claims
        )

        # Step 12: Robustness & Counterfactual Leave-One-Out Evaluation
        report.robustness = self.robustness_engine.evaluate_robustness(
            claims=claims, papers_pool=self.indexed_papers
        )

        # Step 13: Export LaTeX & BibTeX Publication Artifacts
        report.latex_code = self.latex_exporter.generate_latex(report)
        report.bibtex_code = self.latex_exporter.generate_bibtex(report)

        # Step 14: Agentic Verification Trajectory
        report.verification_trajectory = trajectory

        # Step 15: Corpus Methodological Rigor Assessment
        report.rigor_report = corpus_rigor_report

        # Step 16: Automated Scientific Hypothesis Generation
        report.hypotheses = self.hypothesis_generator.generate_hypotheses(
            query=query,
            conflicts=conflicts,
            claims=claims,
            meta_analysis=report.meta_analysis,
            research_gaps=research_gaps,
        )

        # Step 17: Interactive Sankey Diagram Data
        report.sankey_data = self.graph_visualizer.build_sankey_data(report)

        # Step 18: Bibliometric Co-Citation & PageRank Network
        report.bibliometrics = self.bibliometric_analyzer.analyze_network(
            papers=self.indexed_papers, claims=claims
        )

        # Step 19: Multi-Agent Scientific Consensus Council Deliberation
        report.consensus_council = self.consensus_council.deliberate(
            query=query, conflicts=conflicts, claims=claims
        )

        # Step 20: Conference Peer Review & Soundness Audit (NeurIPS / ICLR Style)
        report.peer_review = self.peer_reviewer.review_synthesis(
            report=report, papers=self.indexed_papers
        )

        # Step 21: Academic Presentation Slide Deck (Interactive HTML & Marp)
        report.slide_deck = self.slide_generator.build_slide_deck(report)

        # Step 22: Causal Directed Acyclic Graph (DAG) & Do-Calculus Counterfactual Reasoning
        report.causal_dag = self.causal_engine.build_causal_dag(
            query=query, claims=claims, conflicts=conflicts, papers=self.indexed_papers
        )

        # Step 23: Autonomous Reproducibility & Open Science Protocol Audit
        report.reproducibility = self.reproducibility_verifier.audit_corpus(
            papers=self.indexed_papers
        )

        # Step 24: Temporal Knowledge Horizon & Living Research Roadmap
        report.temporal_roadmap = self.temporal_roadmap_engine.build_roadmap(
            query=query, papers=self.indexed_papers, claims=claims
        )

        # Step 25: Autonomous Research Dossier & Executive Press Briefing
        report.dossier = self.dossier_generator.build_dossier(report)

        return report



