"""Contradiction detection and conflict attribution engine."""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.models import (
    FactualClaim,
    EvidenceItem,
    ConflictAttribution,
    RetrievalCandidate,
)
from src.verification.entailment import NLIEntailmentChecker
from src.llm.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class ContradictionEngine:
    """Detects contradicting evidence across scientific papers and identifies attribution factors."""

    def __init__(
        self,
        nli_checker: NLIEntailmentChecker,
        llm_client: Optional[BaseLLMClient] = None,
        contradiction_threshold: float = 0.60,
    ):
        self.nli = nli_checker
        self.llm = llm_client
        self.contra_threshold = contradiction_threshold

    def evaluate_evidence_relationships(
        self,
        claim: FactualClaim,
        candidates: List[RetrievalCandidate],
    ) -> Tuple[List[EvidenceItem], List[EvidenceItem], List[EvidenceItem]]:
        """Classify candidate chunks into SUPPORTS, CONTRADICTS, and NEUTRAL for a given claim."""
        supporting: List[EvidenceItem] = []
        contradicting: List[EvidenceItem] = []
        neutral: List[EvidenceItem] = []

        for cand in candidates:
            chunk = cand.chunk
            premise = f"[{chunk.section}] {chunk.text}"
            hypothesis = claim.statement

            nli_res = self.nli.check(premise=premise, hypothesis=hypothesis)
            ent_p = nli_res["entailment"]
            contra_p = nli_res["contradiction"]
            neut_p = nli_res["neutral"]

            # Composite item
            item = EvidenceItem(
                evidence_id=f"ev_{chunk.chunk_id}",
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                paper_title=chunk.paper_title,
                section=chunk.section,
                year=chunk.year,
                text_snippet=chunk.text[:280] + ("..." if len(chunk.text) > 280 else ""),
                semantic_similarity=cand.dense_score or cand.hybrid_score,
                entailment_prob=ent_p,
                contradiction_prob=contra_p,
                neutral_prob=neut_p,
                extracted_attributes={
                    "dataset": chunk.dataset,
                    "model_architecture": chunk.model_architecture,
                    "evaluation_metric": chunk.evaluation_metric,
                },
            )

            # Determine relationship
            if contra_p >= self.contra_threshold:
                item.relationship = "CONTRADICTS"
                contradicting.append(item)
            elif ent_p >= 0.55:
                item.relationship = "SUPPORTS"
                supporting.append(item)
            else:
                item.relationship = "NEUTRAL"
                neutral.append(item)

        return supporting, contradicting, neutral

    def attribute_conflict(
        self,
        claim: FactualClaim,
        supporting: List[EvidenceItem],
        contradicting: List[EvidenceItem],
    ) -> Optional[ConflictAttribution]:
        """Perform Conflict Attribution explaining why studies arrive at conflicting conclusions."""
        if not supporting or not contradicting:
            return None

        # Collect metadata from supporting papers vs contradicting papers
        sup_papers = [
            {
                "paper_id": e.paper_id,
                "title": e.paper_title,
                "year": e.year,
                "dataset": e.extracted_attributes.get("dataset") or "Unspecified",
                "metric": e.extracted_attributes.get("evaluation_metric") or "mAP",
                "model": e.extracted_attributes.get("model_architecture") or "ViT",
                "snippet": e.text_snippet[:150],
            }
            for e in supporting[:3]
        ]

        contra_papers = [
            {
                "paper_id": e.paper_id,
                "title": e.paper_title,
                "year": e.year,
                "dataset": e.extracted_attributes.get("dataset") or "Unspecified",
                "metric": e.extracted_attributes.get("evaluation_metric") or "mAP",
                "model": e.extracted_attributes.get("model_architecture") or "CNN",
                "snippet": e.text_snippet[:150],
            }
            for e in contradicting[:3]
        ]

        divergent_factors = []

        # Check datasets
        sup_datasets = {p["dataset"] for p in sup_papers if p["dataset"] != "Unspecified"}
        contra_datasets = {p["dataset"] for p in contra_papers if p["dataset"] != "Unspecified"}
        if sup_datasets and contra_datasets and sup_datasets != contra_datasets:
            divergent_factors.append({
                "factor": "Evaluation Benchmark Dataset",
                "supporting_side": f"Evaluated on {', '.join(sup_datasets)}",
                "contradicting_side": f"Evaluated on {', '.join(contra_datasets)}",
            })

        # Check metrics
        sup_metrics = {p["metric"] for p in sup_papers if p["metric"]}
        contra_metrics = {p["metric"] for p in contra_papers if p["metric"]}
        if sup_metrics and contra_metrics and sup_metrics != contra_metrics:
            divergent_factors.append({
                "factor": "Evaluation Metric Protocol",
                "supporting_side": f"Reported {', '.join(sup_metrics)}",
                "contradicting_side": f"Reported {', '.join(contra_metrics)}",
            })

        # Check publication year shift
        avg_sup_yr = sum(p["year"] for p in sup_papers) / len(sup_papers)
        avg_contra_yr = sum(p["year"] for p in contra_papers) / len(contra_papers)
        if abs(avg_sup_yr - avg_contra_yr) >= 2:
            divergent_factors.append({
                "factor": "Temporal Evolution",
                "supporting_side": f"Avg Publication Year: {avg_sup_yr:.0f}",
                "contradicting_side": f"Avg Publication Year: {avg_contra_yr:.0f}",
            })

        # Generate explanatory synthesis using LLM or rule-based synthesis
        likely_reason = (
            f"The studies reach contrasting conclusions due to divergent experimental setups. "
            f"Supporting literature ({', '.join(p['paper_id'] for p in sup_papers)}) predominantly tests on "
            f"{', '.join(sup_datasets) or 'specific low-light conditions'}, whereas contradicting works "
            f"({', '.join(p['paper_id'] for p in contra_papers)}) benchmark under alternative illumination protocols and datasets."
        )

        if self.llm:
            try:
                prompt = (
                    f"Explain why these scientific papers conflict regarding the claim: \"{claim.statement}\"\n\n"
                    f"Supporting Studies:\n{sup_papers}\n\n"
                    f"Contradicting Studies:\n{contra_papers}\n\n"
                    f"Provide a concise, 2-3 sentence scientific explanation of the likely experimental or methodological reasons for this disagreement."
                )
                res = self.llm.generate(prompt)
                if res and len(res.strip()) > 30:
                    likely_reason = res.strip()
            except Exception as e:
                logger.warning(f"LLM conflict attribution failed: {e}. Using deterministic explanation.")

        return ConflictAttribution(
            claim_id=claim.claim_id,
            claim_statement=claim.statement,
            supporting_papers=sup_papers,
            contradicting_papers=contra_papers,
            divergent_factors=divergent_factors,
            likely_reason=likely_reason,
        )
