"""Citation audit and faithfulness verification."""

from __future__ import annotations
from typing import List, Dict, Any
from src.models import FactualClaim, CitationAudit
from src.verification.entailment import NLIEntailmentChecker


class CitationAuditor:
    """Audits generated synthesis and claims to measure factual evidence coverage and citation faithfulness."""

    def __init__(self, nli_checker: NLIEntailmentChecker):
        self.nli = nli_checker

    def audit(self, claims: List[FactualClaim]) -> CitationAudit:
        """Calculate Evidence Coverage Score (ECS) and Citation Integrity Score (CIS)."""
        if not claims:
            return CitationAudit(
                total_claims=0,
                supported_claims=0,
                conflicting_claims=0,
                unsupported_claims=0,
                evidence_coverage_score=0.0,
                citation_integrity_score=0.0,
                citation_faithfulness=0.0,
                audit_details=[],
            )

        total_claims = len(claims)
        supported_count = 0
        conflicting_count = 0
        unsupported_count = 0
        details = []

        total_citations = 0
        verified_citations = 0

        for c in claims:
            num_sup = len(c.supporting_evidence)
            num_contra = len(c.contradicting_evidence)

            if num_contra > 0:
                status = "CONFLICTING"
                conflicting_count += 1
            elif num_sup > 0:
                status = "SUPPORTED"
                supported_count += 1
            else:
                status = "UNSUPPORTED"
                unsupported_count += 1

            # Check individual citations
            claim_citations = []
            for ev in c.supporting_evidence:
                total_citations += 1
                if ev.entailment_prob >= 0.50:
                    verified_citations += 1
                    cited_valid = True
                else:
                    cited_valid = False
                claim_citations.append({
                    "paper_id": ev.paper_id,
                    "section": ev.section,
                    "entailment_score": round(ev.entailment_prob, 3),
                    "valid": cited_valid
                })

            details.append({
                "claim_id": c.claim_id,
                "statement": c.statement,
                "status": status,
                "supporting_count": num_sup,
                "contradicting_count": num_contra,
                "citations": claim_citations,
            })

        # Calculate metrics as defined in proposal
        # ECS = number of claims supported by valid evidence / total number of factual claims
        ecs = round(supported_count / total_claims, 4) if total_claims > 0 else 0.0

        # CIS = correctly supported cited claims / all cited factual claims
        cis = round(verified_citations / total_citations, 4) if total_citations > 0 else (1.0 if supported_count > 0 else 0.0)

        faithfulness = round((supported_count + 0.5 * conflicting_count) / total_claims, 4) if total_claims > 0 else 0.0

        return CitationAudit(
            total_claims=total_claims,
            supported_claims=supported_count,
            conflicting_claims=conflicting_count,
            unsupported_claims=unsupported_count,
            evidence_coverage_score=ecs,
            citation_integrity_score=cis,
            citation_faithfulness=faithfulness,
            audit_details=details,
        )
