"""Agentic iterative claim verification engine with multi-hop retrieval trajectory."""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional, Set
from src.models import (
    FactualClaim,
    SemanticChunk,
    RetrievalCandidate,
    EvidenceItem,
    VerificationStep,
    VerificationTrajectory,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.verification.contradiction import ContradictionEngine
from src.verification.evidence_score import EvidenceScorer
from src.llm.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class AgenticVerifier:
    """Iterative multi-hop claim verification engine that generates targeted

    counter-queries to resolve scientific ambiguities and under-supported claims.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        contradiction_engine: ContradictionEngine,
        scorer: EvidenceScorer,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.contradiction_engine = contradiction_engine
        self.scorer = scorer
        self.llm_client = llm_client

    def generate_targeted_query(self, claim: FactualClaim, focus: str = "precision") -> str:
        """Formulate a targeted query focused on confirming, refuting, or clarifying a claim."""
        statement = claim.statement
        if focus == "refutation":
            return f"failure case limitation discrepancy benchmark counter-evidence {statement}"
        elif focus == "dataset_ablation":
            return f"ablation experiment dataset comparison results {statement}"
        else:
            # precision confirmation
            words = [w for w in statement.split() if len(w) > 3 and w.lower() not in {"this", "that", "with", "from", "have"}]
            keyword_subset = " ".join(words[:6])
            return f"empirical verification experimental finding {keyword_subset}"

    def run_verification_loop(
        self,
        claims: List[FactualClaim],
        max_iterations: int = 2,
    ) -> Tuple[List[FactualClaim], VerificationTrajectory]:
        """Execute iterative verification loop across uncertain or conflicting claims."""
        steps: List[VerificationStep] = []
        resolved_count = 0

        # Prioritize claims needing deeper investigation: Conflicting, Weak support, or Unsupported
        target_claims = [
            c for c in claims
            if c.evidence_status in ["Conflicting", "Weak support", "Unsupported"] or c.confidence_score < 0.65
        ]

        if not target_claims:
            return claims, VerificationTrajectory(
                iterations_performed=0,
                steps=[],
                claims_resolved=0,
                trajectory_summary="All claims satisfied strong evidence thresholds on initial pass.",
            )

        step_idx = 1
        for iteration in range(1, max_iterations + 1):
            for claim in target_claims:
                initial_status = claim.evidence_status
                initial_conf = claim.confidence_score

                # Determine targeted focus
                if claim.evidence_status == "Conflicting":
                    focus = "refutation" if (iteration % 2 == 1) else "dataset_ablation"
                else:
                    focus = "precision"

                targeted_q = self.generate_targeted_query(claim, focus=focus)

                # Execute targeted retrieval
                retrieved_candidates = self.hybrid_retriever.retrieve(targeted_q, top_candidates=15)
                reranked = self.reranker.rerank(targeted_q, retrieved_candidates, top_n=6)


                # Filter out chunks already present in existing evidence for this claim
                existing_chunk_ids: Set[str] = {
                    e.chunk_id for e in (claim.supporting_evidence + claim.contradicting_evidence + claim.neutral_evidence)
                }
                new_candidates = [c for c in reranked if c.chunk.chunk_id not in existing_chunk_ids]

                if not new_candidates:
                    continue

                # Run contradiction/entailment evaluation on new candidates
                new_sup, new_contra, new_neut = self.contradiction_engine.evaluate_evidence_relationships(
                    claim=claim, candidates=new_candidates
                )

                # Score and append
                for ev in new_sup:
                    ev.composite_score = self.scorer.compute_score(
                        semantic_sim=ev.semantic_similarity,
                        entailment_prob=ev.entailment_prob,
                        year=ev.year,
                    )
                for ev in new_contra:
                    ev.composite_score = self.scorer.compute_score(
                        semantic_sim=ev.semantic_similarity,
                        entailment_prob=ev.contradiction_prob,
                        year=ev.year,
                    )

                claim.supporting_evidence.extend(new_sup)
                claim.contradicting_evidence.extend(new_contra)
                claim.neutral_evidence.extend(new_neut)

                # Re-sort evidence lists
                claim.supporting_evidence.sort(key=lambda x: x.composite_score, reverse=True)
                claim.contradicting_evidence.sort(key=lambda x: x.composite_score, reverse=True)

                # Update confidence and status
                if claim.supporting_evidence:
                    top_sup = claim.supporting_evidence[0].composite_score
                    claim.confidence_score = max(claim.confidence_score, top_sup)
                if claim.contradicting_evidence and not claim.supporting_evidence:
                    claim.confidence_score = max(claim.confidence_score, claim.contradicting_evidence[0].composite_score)

                has_contradictions = len(claim.contradicting_evidence) > 0 and len(claim.supporting_evidence) > 0
                new_status = self.scorer.classify_status(
                    confidence=claim.confidence_score,
                    has_contradictions=has_contradictions,
                )
                claim.evidence_status = new_status

                conf_delta = round(claim.confidence_score - initial_conf, 3)
                if new_status != initial_status or conf_delta > 0.05:
                    resolved_count += 1
                    reasoning = f"Retrieved {len(new_candidates)} targeted chunks; status changed from '{initial_status}' to '{new_status}' (ΔConf: {conf_delta:+.2f})."
                else:
                    reasoning = f"Targeted retrieval reinforced status as '{new_status}' with {len(new_candidates)} additional corroborating evidence chunks."

                steps.append(
                    VerificationStep(
                        step_number=step_idx,
                        claim_id=claim.claim_id,
                        target_statement=claim.statement,
                        query_generated=targeted_q,
                        retrieved_chunks_count=len(new_candidates),
                        initial_status=initial_status,
                        updated_status=new_status,
                        confidence_delta=conf_delta,
                        reasoning=reasoning,
                    )
                )
                step_idx += 1

        summary = (
            f"Executed {len(steps)} agentic multi-hop verification steps across {len(target_claims)} targeted claims. "
            f"Successfully updated/corroborated {resolved_count} claims with high-precision evidence."
        )

        trajectory = VerificationTrajectory(
            iterations_performed=max_iterations,
            steps=steps,
            claims_resolved=resolved_count,
            trajectory_summary=summary,
        )

        return claims, trajectory
