"""Methodological Rigor and Scientific Evidence Hierarchy Analyzer."""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from src.models import (
    Paper,
    EvidenceItem,
    MethodologyAssessment,
    MethodologyRigorReport,
)

logger = logging.getLogger(__name__)


class MethodologyRigorAnalyzer:
    """Classifies paper scientific methodology, evaluates empirical rigor,

    and computes evidence hierarchy weight multipliers.
    """

    def __init__(self):
        # Academic keyword signatures for methodological classification
        self.benchmark_signatures = [
            "benchmark", "dataset", "map", "accuracy", "f1", "leaderboard",
            "cross-dataset", "comparative evaluation", "coco", "darkface", "exdark"
        ]
        self.ablation_signatures = [
            "ablation", "controlled experiment", "component analysis", "hyperparameter",
            "variant", "module removal", "sensitivity analysis"
        ]
        self.theory_signatures = [
            "theorem", "lemma", "proof", "convergence", "bound", "complexity",
            "mathematical", "asymptotic", "formal guarantee"
        ]
        self.case_study_signatures = [
            "case study", "observational", "field study", "user study", "pilot",
            "qualitative", "interview", "clinical trial"
        ]

    def assess_paper(self, paper: Paper) -> MethodologyAssessment:
        """Evaluate methodological rigor for a single paper."""
        full_text = (
            paper.abstract + " " +
            " ".join(paper.sections.values()) + " " +
            str(paper.metadata)
        ).lower()

        # Score signatures
        bench_score = sum(full_text.count(sig) for sig in self.benchmark_signatures)
        ablation_score = sum(full_text.count(sig) for sig in self.ablation_signatures)
        theory_score = sum(full_text.count(sig) for sig in self.theory_signatures)
        case_score = sum(full_text.count(sig) for sig in self.case_study_signatures)

        # Count tables and empirical anchors
        table_count = len(paper.tables)
        figures_count = len(paper.figures)

        # Classify methodology
        strengths: List[str] = []
        limitations: List[str] = []

        if bench_score >= 8 or table_count >= 2:
            methodology_type = "Large-Scale Empirical Benchmark"
            rigor_score = min(0.96, 0.85 + 0.02 * table_count)
            multiplier = 1.25
            strengths.append(f"Standardized multi-dataset quantitative benchmarks with {table_count} empirical tables.")
            strengths.append("High statistical reproducibility across standardized metrics.")
            limitations.append("May overfit to specific benchmark dataset distribution characteristics.")

        elif ablation_score >= 4:
            methodology_type = "Controlled Ablation Study"
            rigor_score = 0.90
            multiplier = 1.15
            strengths.append("Controlled isolation of architectural components and hyperparameters.")
            strengths.append("Clear causal attribution for performance deltas.")
            limitations.append("Computational cost may constrain ablation hyperparameter search space.")

        elif theory_score >= 3:
            methodology_type = "Theoretical / Algorithmic Proof"
            rigor_score = 0.88
            multiplier = 1.10
            strengths.append("Rigorous mathematical grounding with theoretical bounds and proofs.")
            limitations.append("Theoretical bounds may assume idealized distributions that diverge in practice.")

        elif case_score >= 3:
            methodology_type = "Observational / Case Study"
            rigor_score = 0.78
            multiplier = 0.95
            strengths.append("Granular real-world observational insights and qualitative domain nuance.")
            limitations.append("Smaller sample sizes; findings may not generalize across all hardware/scenarios.")

        else:
            methodology_type = "Qualitative / Conceptual Review"
            rigor_score = 0.70
            multiplier = 0.85
            strengths.append("Broad literature synthesis and taxonomy framing.")
            limitations.append("Lacks primary empirical ablation or novel standardized benchmark replication.")

        # Check dataset counts
        datasets_found = 0
        for ds in ["DarkFace", "ExDark", "COCO", "ImageNet", "Pascal VOC", "HotpotQA", "PubMed"]:
            if ds.lower() in full_text:
                datasets_found += 1
        datasets_found = max(datasets_found, 1)

        return MethodologyAssessment(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            year=paper.year,
            methodology_type=methodology_type,
            rigor_score=round(rigor_score, 3),
            evidence_weight_multiplier=multiplier,
            sample_size_or_datasets_count=datasets_found,
            strengths=strengths,
            limitations=limitations,
        )

    def analyze_corpus(self, papers: List[Paper]) -> MethodologyRigorReport:
        """Evaluate methodological distribution and rigor across the active paper corpus."""
        assessments: List[MethodologyAssessment] = []
        distribution: Dict[str, int] = {}

        for p in papers:
            assess = self.assess_paper(p)
            assessments.append(assess)
            distribution[assess.methodology_type] = distribution.get(assess.methodology_type, 0) + 1

        if not assessments:
            return MethodologyRigorReport(
                overall_corpus_rigor=0.0,
                high_rigor_paper_ratio=0.0,
                methodology_distribution={},
                assessments=[],
                recommendations=["Corpus is empty. Index research papers to evaluate methodological rigor."],
            )

        mean_rigor = sum(a.rigor_score for a in assessments) / len(assessments)
        high_rigor = sum(1 for a in assessments if a.rigor_score >= 0.85) / len(assessments)

        recommendations: List[str] = []
        if high_rigor >= 0.70:
            recommendations.append("Corpus demonstrates robust empirical validation backed by standardized benchmarks.")
        else:
            recommendations.append("Literature contains substantial qualitative/conceptual papers; prioritize empirical benchmark validations.")

        if distribution.get("Large-Scale Empirical Benchmark", 0) > 0 and distribution.get("Controlled Ablation Study", 0) > 0:
            recommendations.append("Balanced balance between macro benchmark comparisons and micro architectural ablations.")

        return MethodologyRigorReport(
            overall_corpus_rigor=round(mean_rigor, 3),
            high_rigor_paper_ratio=round(high_rigor, 3),
            methodology_distribution=distribution,
            assessments=assessments,
            recommendations=recommendations,
        )

    def apply_rigor_weighting(
        self,
        evidence_items: List[EvidenceItem],
        assessments: Dict[str, MethodologyAssessment],
    ) -> List[EvidenceItem]:
        """Adjust composite evidence scores by the paper's methodological rigor multiplier."""
        for ev in evidence_items:
            assess = assessments.get(ev.paper_id)
            if assess:
                # Modulate composite score by multiplier, clamped to [0.0, 1.0]
                ev.composite_score = min(1.0, max(0.0, round(ev.composite_score * assess.evidence_weight_multiplier, 3)))
        return evidence_items
