"""Citation and Faithfulness evaluation metrics."""

from __future__ import annotations
from typing import List, Dict, Any


class CitationFaithfulnessEvaluator:
    """Computes Citation Faithfulness, Evidence Coverage Score (ECS), and Citation Integrity Score (CIS)."""

    @staticmethod
    def compute_ecs(supported_claims: int, total_claims: int) -> float:
        """Evidence Coverage Score: ECS = supported_claims / total_claims."""
        if total_claims == 0:
            return 0.0
        return round(supported_claims / total_claims, 4)

    @staticmethod
    def compute_cis(verified_citations: int, total_citations: int) -> float:
        """Citation Integrity Score: CIS = correctly supported cited claims / all cited factual claims."""
        if total_citations == 0:
            return 0.0
        return round(verified_citations / total_citations, 4)
