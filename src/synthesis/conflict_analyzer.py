"""Conflict analyzer: analyzes patterns across contradictory findings."""

from __future__ import annotations
from typing import List
from src.models import FactualClaim, ConflictAttribution


class ConflictAnalyzer:
    """Consolidates and summarizes conflicting findings across all evaluated claims."""

    def analyze_conflicts(self, claims: List[FactualClaim]) -> List[ConflictAttribution]:
        """Extract and format conflict attributions for all claims exhibiting contradictions."""
        conflicts = []
        for c in claims:
            if c.conflict_attribution:
                conflicts.append(c.conflict_attribution)
            elif len(c.contradicting_evidence) > 0:
                # If conflict was flagged without full attribution, build basic one
                sup = [{"paper_id": e.paper_id, "title": e.paper_title, "snippet": e.text_snippet[:120]} for e in c.supporting_evidence]
                contra = [{"paper_id": e.paper_id, "title": e.paper_title, "snippet": e.text_snippet[:120]} for e in c.contradicting_evidence]
                conflicts.append(
                    ConflictAttribution(
                        claim_id=c.claim_id,
                        claim_statement=c.statement,
                        supporting_papers=sup,
                        contradicting_papers=contra,
                        divergent_factors=[{"factor": "Benchmark Evaluation", "details": "Opposing experimental outcomes reported across papers"}],
                        likely_reason="The studies test under distinct illumination protocols, camera noise levels, or detector backbones."
                    )
                )
        return conflicts
