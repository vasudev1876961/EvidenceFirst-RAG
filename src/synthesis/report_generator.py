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

    def format_html(self, report: SynthesisReport) -> str:
        """Format SynthesisReport as publication-grade standalone HTML document."""
        audit = report.citation_audit
        conflicts_html = ""
        if report.conflicting_findings:
            conflicts_cards = []
            for c in report.conflicting_findings:
                sup_p = ", ".join([p.get("paper_id", "") for p in c.supporting_papers]) or "None"
                contra_p = ", ".join([p.get("paper_id", "") for p in c.contradicting_papers]) or "None"
                factors_rows = ""
                for f in c.divergent_factors:
                    factors_rows += f"<tr><td><b>{f.get('factor', 'Aspect')}</b></td><td>{f.get('support', f.get('supporting_side', 'N/A'))}</td><td>{f.get('contradict', f.get('contradicting_side', 'N/A'))}</td></tr>"

                factors_table = f"""
                <table class="data-table">
                    <thead><tr><th>Divergent Factor</th><th>Supporting Setup</th><th>Contradicting Setup</th></tr></thead>
                    <tbody>{factors_rows}</tbody>
                </table>
                """ if factors_rows else ""

                conflicts_cards.append(f"""
                <div class="conflict-box">
                    <h3>⚡ Contradiction in Claim {c.claim_id}</h3>
                    <p class="claim-stmt">"{c.claim_statement}"</p>
                    <p><b>Supporting Papers:</b> <span class="badge-sup">{sup_p}</span> &nbsp;|&nbsp; <b>Contradicting Papers:</b> <span class="badge-contra">{contra_p}</span></p>
                    <div class="attribution-reason"><b>Likely Disagreement Cause:</b> {c.likely_reason}</div>
                    {factors_table}
                </div>
                """)
            conflicts_html = f"<h2>3. Contradiction Detection & Conflict Attribution</h2>{''.join(conflicts_cards)}"

        findings_html = ""
        for f in report.key_findings:
            status_class = "badge-sup" if "Strong" in f["status"] else ("badge-contra" if "Conflicting" in f["status"] else "badge-mod")
            snippets_html = ""
            for s in f.get("evidence_snippets", []):
                snippets_html += f"<blockquote><b>[{s.get('paper_id')}]</b> {s.get('snippet')} <small>({s.get('section')}, {s.get('year')})</small></blockquote>"

            findings_html += f"""
            <div class="finding-card">
                <span class="badge {status_class}">{f['status'].upper()}</span>
                <h4>{f['aspect'].title()} (Claim {f['claim_id']})</h4>
                <p class="statement">{f['statement']}</p>
                <div class="meta">Confidence: <b>{f['confidence_score']:.2f}</b> | Supporting: <b>{f['supporting_count']}</b> | Contradicting: <b>{f['contradicting_count']}</b></div>
                {snippets_html}
            </div>
            """

        refs_html = "".join([f"<li><b>[{r['paper_id']}]</b> {r['title']}. <i>{r.get('venue', '')}</i> ({r.get('year', '')})</li>" for r in report.references])

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EvidenceFirst-RAG: Literature Synthesis</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1E293B; background: #F8FAFC; margin: 0; padding: 40px 20px; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #FFFFFF; border-radius: 12px; padding: 40px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }}
    h1 {{ font-size: 2rem; color: #0F172A; border-bottom: 2px solid #3B82F6; padding-bottom: 12px; }}
    h2 {{ font-size: 1.4rem; color: #1E293B; margin-top: 32px; border-bottom: 1px solid #E2E8F0; padding-bottom: 8px; }}
    .metrics-bar {{ display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }}
    .metric-item {{ flex: 1; min-width: 140px; background: #F1F5F9; border-radius: 8px; padding: 12px 16px; text-align: center; }}
    .metric-num {{ font-size: 1.6rem; font-weight: 700; color: #2563EB; }}
    .metric-label {{ font-size: 0.8rem; color: #64748B; font-weight: 600; text-transform: uppercase; }}
    .finding-card {{ background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    .conflict-box {{ background: #FFF1F2; border-left: 4px solid #E11D48; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }}
    .badge-sup {{ background: #DCFCE7; color: #15803D; }}
    .badge-contra {{ background: #FFE4E6; color: #BE123C; }}
    .badge-mod {{ background: #DBEAFE; color: #1D4ED8; }}
    .data-table {{ width: 100%; border-collapse: collapse; margin-top: 14px; background: white; }}
    .data-table th, .data-table td {{ border: 1px solid #CBD5E1; padding: 10px 14px; text-align: left; font-size: 0.9rem; }}
    .data-table th {{ background: #F1F5F9; font-weight: 600; }}
    blockquote {{ border-left: 3px solid #94A3B8; margin: 10px 0; padding-left: 12px; font-style: italic; color: #475569; }}
</style>
</head>
<body>
<div class="container">
    <h1>🔬 EvidenceFirst-RAG: Scientific Literature Synthesis</h1>
    <p><b>Research Query:</b> <i>"{report.query}"</i></p>

    <div class="metrics-bar">
        <div class="metric-item"><div class="metric-num">{report.overall_confidence * 100:.0f}%</div><div class="metric-label">Confidence</div></div>
        <div class="metric-item"><div class="metric-num">{audit.evidence_coverage_score * 100:.0f}%</div><div class="metric-label">Evidence Coverage (ECS)</div></div>
        <div class="metric-item"><div class="metric-num">{audit.citation_integrity_score * 100:.0f}%</div><div class="metric-label">Citation Integrity (CIS)</div></div>
        <div class="metric-item"><div class="metric-num">{len(report.conflicting_findings)}</div><div class="metric-label">Conflicts Detected</div></div>
    </div>

    <h2>1. Executive Summary</h2>
    <p>{report.executive_summary}</p>

    <h2>2. Verified Factual Claims & Evidence</h2>
    {findings_html}

    {conflicts_html}

    <h2>4. Academic References</h2>
    <ul>{refs_html}</ul>
</div>
</body>
</html>
"""
        return html_template

