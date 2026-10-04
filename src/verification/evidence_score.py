"""Evidence scoring engine combining semantic similarity, NLI entailment, source quality, and recency."""

from __future__ import annotations
from typing import Dict, Any, Literal
from src.models import EvidenceItem


class EvidenceScorer:
    """Computes composite evidence strength score using the weighted formulation:
    EvidenceScore = w1 * S_semantic + w2 * S_entailment + w3 * S_source + w4 * S_recency
    """

    def __init__(
        self,
        w_semantic: float = 0.35,
        w_entailment: float = 0.45,
        w_source: float = 0.10,
        w_recency: float = 0.10,
        threshold_strong: float = 0.80,
        threshold_moderate: float = 0.60,
        threshold_weak: float = 0.40,
    ):
        self.w_semantic = w_semantic
        self.w_entailment = w_entailment
        self.w_source = w_source
        self.w_recency = w_recency

        self.threshold_strong = threshold_strong
        self.threshold_moderate = threshold_moderate
        self.threshold_weak = threshold_weak

    def calculate_recency_score(self, year: int, current_year: int = 2026) -> float:
        """Normalized temporal score with exponential decay for older literature."""
        diff = max(0, current_year - year)
        return max(0.2, 1.0 - (0.07 * diff))

    def compute_score(
        self,
        semantic_sim: float,
        entailment_prob: float,
        source_quality: float = 1.0,
        year: int = 2024,
    ) -> float:
        """Calculate composite evidence score."""
        recency = self.calculate_recency_score(year)
        score = (
            (self.w_semantic * semantic_sim)
            + (self.w_entailment * entailment_prob)
            + (self.w_source * source_quality)
            + (self.w_recency * recency)
        )
        return round(float(score), 4)

    def classify_status(
        self,
        confidence: float,
        has_contradictions: bool = False
    ) -> Literal["Strong support", "Moderate support", "Conflicting", "Weak support", "Unsupported"]:
        """Classify overall claim status."""
        if has_contradictions:
            return "Conflicting"
        if confidence >= self.threshold_strong:
            return "Strong support"
        elif confidence >= self.threshold_moderate:
            return "Moderate support"
        elif confidence >= self.threshold_weak:
            return "Weak support"
        else:
            return "Unsupported"
