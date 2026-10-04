"""Final report generator: formats full EvidenceFirst scientific synthesis report."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from src.models import (
    FactualClaim,
    SynthesisReport,
    CitationAudit,
    ConflictAttribution,
    EvidenceGraphData,
    Paper,
)


class ReportGenerator:
    """Compiles findings, contradictions, citation audits, and graphs into a comprehensive scientific report."""

    def generate_report(
        self,
        query: str,
        decomposed_subqueries: List[str],
        claims: List[FactualClaim],
        executive_summary: str,
        conflicts: List[ConflictAttribution],
        research_gaps: List[str],
        citation_audit: CitationAudit,
        evidence_graph: EvidenceGraphData,
        papers: List[Paper],
    ) -> SynthesisReport:
        """Construct full structured SynthesisReport."""
        # Calculate overall confidence
        if claims:
            avg_conf = sum(c.confidence_score for c in claims) / len(claims)
        else:
            avg_conf = 0.0

        key_findings = []
        for i, c in enumerate(claims, 1):
            key_findings.append({
                "index": i,
                "claim_id": c.claim_id,
                "statement": c.statement,
                "aspect": c.subquery_source or f"Finding {i}",
                "confidence_score": round(c.confidence_score, 2),
                "confidence_label": "High" if c.confidence_score >= 0.80 else ("Moderate" if c.confidence_score >= 0.60 else "Weak"),
                "status": c.evidence_status,
                "supporting_count": len(c.supporting_evidence),
                "contradicting_count": len(c.contradicting_evidence),
                "evidence_snippets": [
                    {
                        "paper_id": ev.paper_id,
                        "paper_title": ev.paper_title,
                        "section": ev.section,
                        "year": ev.year,
                        "score": round(ev.composite_score, 2),
                        "snippet": ev.text_snippet,
                    }
                    for ev in c.supporting_evidence[:2]
                ]
            })

        # Build reference list
        seen_refs = set()
        references = []
        for p in papers:
            if p.paper_id not in seen_refs:
                seen_refs.add(p.paper_id)
                references.append({
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "year": p.year,
                    "venue": p.venue or "arXiv / CVPR / ICCV",
                    "url": p.url or f"https://arxiv.org/abs/{p.paper_id}",
                })

        return SynthesisReport(
            query=query,
            decomposed_subqueries=decomposed_subqueries,
            overall_confidence=round(avg_conf, 2),
            executive_summary=executive_summary,
            key_findings=key_findings,
            conflicting_findings=conflicts,
            claims=claims,
            research_gaps=research_gaps,
            citation_audit=citation_audit,
            evidence_graph=evidence_graph,
            references=references,
        )

    def format_markdown(self, report: SynthesisReport) -> str:
        """Format SynthesisReport as clean academic GitHub markdown."""
        lines = []
        lines.append(f"# EvidenceFirst-RAG: Scientific Literature Synthesis")
        lines.append(f"**Research Query**: *\"{report.query}\"*\n")
        lines.append(f"**Overall Confidence Score**: `{report.overall_confidence:.2f} / 1.00`\n")

        lines.append("## 1. Executive Summary")
        lines.append(f"{report.executive_summary}\n")

        if report.decomposed_subqueries:
            lines.append("### Query Decomposition Subqueries")
            for sq in report.decomposed_subqueries:
                lines.append(f"- 🔍 {sq}")
            lines.append("")

        lines.append("## 2. Key Findings & Claim-Level Evidence")
        for f in report.key_findings:
            status_emoji = "✅" if "Strong" in f["status"] else ("⚠️" if "Conflicting" in f["status"] else "ℹ️")
            lines.append(f"### Finding {f['index']}: {f['aspect'].title()} ({f['claim_id']})")
            lines.append(f"> **Claim**: {f['statement']}")
            lines.append(f">")
            lines.append(f"> **Evidence Strength**: `{f['confidence_label']}` (Confidence: {f['confidence_score']:.2f}) {status_emoji}")
            lines.append(f"> **Supporting Studies**: {f['supporting_count']} | **Contradicting Studies**: {f['contradicting_count']}")
            if f.get("evidence_snippets"):
                lines.append(">\n> **Evidence Citations**:")
                for snip in f["evidence_snippets"]:
                    lines.append(f"> - **[{snip['paper_id']}]** *{snip['paper_title']}* ({snip['section']}, {snip['year']}): \"{snip['snippet']}\"")
            lines.append("")

        if report.conflicting_findings:
            lines.append("## 3. Conflicting Evidence & Conflict Attribution")
            for c in report.conflicting_findings:
                lines.append(f"### Conflict in Claim {c.claim_id}")
                lines.append(f"**Claim**: *\"{c.claim_statement}\"*")
                lines.append("")
                lines.append(f"**Supporting Studies**: {', '.join([p['paper_id'] for p in c.supporting_papers]) if c.supporting_papers else 'None'}")
                lines.append(f"**Contradicting Studies**: {', '.join([p['paper_id'] for p in c.contradicting_papers]) if c.contradicting_papers else 'None'}")
                lines.append("")
                lines.append(f"**Likely Reason for Disagreement**:")
                lines.append(f"{c.likely_reason}\n")
                if c.divergent_factors:
                    lines.append("| Divergent Factor | Supporting Evidence Setup | Contradicting Evidence Setup |")
                    lines.append("| :--- | :--- | :--- |")
                    for factor in c.divergent_factors:
                        lines.append(f"| {factor.get('factor', 'Experimental')} | {factor.get('supporting_side', 'N/A')} | {factor.get('contradicting_side', 'N/A')} |")
                    lines.append("")

        if report.research_gaps:
            lines.append("## 4. Identified Research Gaps")
            for gap in report.research_gaps:
                lines.append(f"- 🔬 {gap}")
            lines.append("")

        lines.append("## 5. Citation Faithfulness & Reliability Audit")
        audit = report.citation_audit
        lines.append(f"- **Total Evaluated Claims**: `{audit.total_claims}`")
        lines.append(f"- **Fully Supported Claims**: `{audit.supported_claims}`")
        lines.append(f"- **Conflicting Claims**: `{audit.conflicting_claims}`")
        lines.append(f"- **Unsupported Claims**: `{audit.unsupported_claims}`")
        lines.append(f"- **Evidence Coverage Score (ECS)**: `{audit.evidence_coverage_score * 100:.1f}%`")
        lines.append(f"- **Citation Integrity Score (CIS)**: `{audit.citation_integrity_score * 100:.1f}%`")
        lines.append(f"- **Citation Faithfulness Score**: `{audit.citation_faithfulness * 100:.1f}%`")
        lines.append("")

        if report.references:
            lines.append("## 6. References")
            for ref in report.references:
                lines.append(f"- **[{ref['paper_id']}]** {ref['title']}. *{ref['venue']}*, {ref['year']}.")

        return "\n".join(lines)
