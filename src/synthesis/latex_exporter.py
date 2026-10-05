"""Publication-Grade Academic LaTeX & BibTeX Exporter for EvidenceFirst-RAG."""

from __future__ import annotations
import re
from typing import Dict, Any, List
from src.models import SynthesisReport


class LatexExporter:
    """Exports literature syntheses into standard IEEE/ACM/NeurIPS LaTeX and BibTeX documents."""

    def generate_latex(self, report: SynthesisReport) -> str:
        """Generate a complete, compilable LaTeX article from a SynthesisReport."""
        safe_query = self._escape_latex(report.query)
        safe_summary = self._escape_latex(report.executive_summary)

        # Build Claims Subsections
        claims_latex = []
        for claim in report.claims:
            safe_stmt = self._escape_latex(claim.statement)
            citations = [f"\\cite{{{ev.paper_id}}}" for ev in claim.supporting_evidence]
            cite_str = " " + "".join(citations) if citations else ""
            status_color = "teal" if "Strong" in claim.evidence_status else ("red" if "Conflicting" in claim.evidence_status else "blue")

            claim_block = f"""
\\paragraph{{{self._escape_latex(claim.claim_id)}: {safe_stmt}}}
\\textbf{{Status:}} \\textcolor{{{status_color}}}{{{claim.evidence_status}}} (Confidence: {claim.confidence_score:.2f}){cite_str}.
"""
            if claim.supporting_evidence:
                claim_block += "\\begin{itemize}\n"
                for ev in claim.supporting_evidence[:2]:
                    claim_block += f"    \\item \\textit{{{self._escape_latex(ev.paper_title)}}} ({ev.year}): ``{self._escape_latex(ev.text_snippet[:150])}...'' \\cite{{{ev.paper_id}}}\n"
                claim_block += "\\end{itemize}\n"

            claims_latex.append(claim_block)

        claims_str = "\n".join(claims_latex)

        # Build Contradictions Table
        conflict_rows = []
        for c in report.conflicting_findings:
            sup_ids = ", ".join([p.get("paper_id", "") for p in c.supporting_papers])
            contra_ids = ", ".join([p.get("paper_id", "") for p in c.contradicting_papers])
            reason = self._escape_latex(c.likely_reason[:120])
            conflict_rows.append(
                f"    {self._escape_latex(c.claim_id)} & \\cite{{{sup_ids}}} & \\cite{{{contra_ids}}} & {reason} \\\\"
            )
        conflict_table_body = "\n".join(conflict_rows) if conflict_rows else "    None & None & None & No contradictions detected \\\\"

        # Build Empirical Benchmark Table if meta-analysis present
        benchmark_table = ""
        if report.meta_analysis and report.meta_analysis.records:
            bm_rows = []
            for r in report.meta_analysis.records[:6]:
                bm_rows.append(
                    f"    {self._escape_latex(r.model_name)} & {self._escape_latex(r.dataset)} & {self._escape_latex(r.metric_name)} & {r.metric_value:.1f}{r.unit} & \\cite{{{r.paper_id}}} \\\\"
                )
            bm_table_body = "\n".join(bm_rows)
            benchmark_table = f"""
\\section{{Empirical Meta-Analysis Matrix}}
Table~\\ref{{tab:empirical}} synthesizes quantitative evaluations across the reviewed benchmark corpus.

\\begin{{table}}[ht]
\\centering
\\caption{{Cross-Paper Empirical Performance & Discrepancy Matrix}}
\\label{{tab:empirical}}
\\begin{{tabular}}{{lllrl}}
\\toprule
\\textbf{{Model Architecture}} & \\textbf{{Dataset}} & \\textbf{{Metric}} & \\textbf{{Score}} & \\textbf{{Ref.}} \\\\
\\midrule
{bm_table_body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

        # Build LaTeX Document
        latex_template = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{booktabs}}
\\usepackage{{tabularx}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\usepackage{{cite}}
\\usepackage{{hyperref}}

\\geometry{{margin=1in}}
\\hypersetup{{colorlinks=true, linkcolor=blue, citecolor=teal, urlcolor=blue}}

\\title{{\\textbf{{EvidenceFirst-RAG: Scientific Literature Synthesis}}\\\\\\large Research Query: ``{safe_query}''}}
\\author{{Automated Scientific Verification Engine \\\\ \\small EvidenceFirst-RAG Intelligence System}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
This systematic synthesis investigates the scientific literature concerning: \\textit{{{safe_query}}}. Using atomic query decomposition, hybrid dense-sparse retrieval, cross-encoder neural reranking, natural language inference (NLI) entailment, and contradiction attribution, we analyze factual claims and cross-paper empirical findings. The resulting synthesis achieves an Evidence Coverage Score (ECS) of {report.citation_audit.evidence_coverage_score*100:.1f}\\% and a Citation Integrity Score (CIS) of {report.citation_audit.citation_integrity_score*100:.1f}\\%.
\\end{{abstract}}

\\section{{Executive Summary}}
{safe_summary}

\\section{{Query Decomposition & Sub-Inquiries}}
The primary research question was decomposed into atomic sub-questions:
\\begin{{itemize}}
"""
        for sq in report.decomposed_subqueries:
            latex_template += f"    \\item \\texttt{{{self._escape_latex(sq)}}}\n"

        latex_template += f"""\\end{{itemize}}

\\section{{Claim-Level Evidence & Verification}}
Each proposition extracted from the corpus was evaluated against indexed literature chunks via DeBERTa NLI:

{claims_str}

\\section{{Contradiction Detection \\& Conflict Attribution}}
Where research findings diverge, our framework isolates the methodological or distribution differences responsible for the discrepancy (Table~\\ref{{tab:conflicts}}).

\\begin{{table}}[ht]
\\centering
\\caption{{Contradiction Attribution: Root Causes of Academic Disagreement}}
\\label{{tab:conflicts}}
\\begin{{tabularx}}{{\\textwidth}}{{lXXp{{5.5cm}}}}
\\toprule
\\textbf{{Claim ID}} & \\textbf{{Supporting Studies}} & \\textbf{{Contradicting Studies}} & \\textbf{{Attributed Divergence Cause}} \\\\
\\midrule
{conflict_table_body}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}

{benchmark_table}

\\section{{Citation Faithfulness \\& Reliability Audit}}
To guarantee strict grounding and eliminate hallucinations:
\\begin{{itemize}}
    \\item \\textbf{{Evidence Coverage Score (ECS)}}: $\\text{{ECS}} = \\frac{{\\text{{Supported Claims}}}}{{\\text{{Total Claims}}}} = {report.citation_audit.supported_claims}/{report.citation_audit.total_claims} = {report.citation_audit.evidence_coverage_score*100:.1f}\\%$
    \\item \\textbf{{Citation Integrity Score (CIS)}}: $\\text{{CIS}} = {report.citation_audit.citation_integrity_score*100:.1f}\\%$
    \\item \\textbf{{Overall Reliability Index}}: {report.overall_confidence*100:.1f}\\%
\\end{{itemize}}

\\section{{Identified Research Gaps}}
\\begin{{enumerate}}
"""
        for gap in report.research_gaps:
            latex_template += f"    \\item {self._escape_latex(gap)}\n"

        latex_template += """\\end{enumerate}

\\bibliographystyle{plain}
\\bibliography{references}

\\end{document}
"""
        return latex_template

    def generate_bibtex(self, report: SynthesisReport) -> str:
        """Generate valid BibTeX entries for all references in the synthesis."""
        bib_entries = []
        seen_pids = set()

        # Primary references
        for ref in report.references:
            pid = ref.get("paper_id", "ref")
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            title = ref.get("title", "Untitled")
            year = ref.get("year", 2024)
            venue = ref.get("venue", "Scientific Proceedings")
            authors_list = ref.get("authors", ["Author, A."])
            if isinstance(authors_list, list):
                authors_str = " and ".join(authors_list)
            else:
                authors_str = str(authors_list)
            url = ref.get("url", "")

            entry = f"""@article{{{pid},
  title = {{{title}}},
  author = {{{authors_str}}},
  journal = {{{venue}}},
  year = {{{year}}},
  url = {{{url}}}
}}"""
            bib_entries.append(entry)

        # Fallback to evidence items from claims if references were not populated
        if not bib_entries:
            for claim in report.claims:
                for ev in claim.supporting_evidence + claim.contradicting_evidence:
                    if ev.paper_id not in seen_pids:
                        seen_pids.add(ev.paper_id)
                        bib_entries.append(f"""@article{{{ev.paper_id},
  title = {{{ev.paper_title}}},
  author = {{Research Team}},
  journal = {{Scientific Literature Review}},
  year = {{{ev.year}}},
  url = {{https://arxiv.org/abs/{ev.paper_id}}}
}}""")

        if not bib_entries:
            bib_entries.append("""@article{evidencefirst2025,
  title = {EvidenceFirst-RAG: Claim-Level Verification for Scientific Literature},
  author = {Vasudev et al.},
  journal = {arXiv preprint},
  year = {2025}
}""")

        return "\n\n".join(bib_entries)

    def _escape_latex(self, text: str) -> str:
        """Escape LaTeX special characters."""
        if not text:
            return ""
        s = str(text)
        specials = [
            ("&", "\\&"),
            ("%", "\\%"),
            ("$", "\\$"),
            ("#", "\\#"),
            ("_", "\\_"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        for char, replacement in specials:
            s = s.replace(char, replacement)
        return s
