"""Multi-Domain Diagnostic Benchmark Runner across Computer Vision, NLP, and Biomedical Corpora.

Executes comparative evaluation across 5 RAG architectures on:
1. RAGChecker Tripartite Diagnostic Metrics (Precision, Recall, F1, CR, CP, CU, Faith, Hallu, NS)
2. Information Retrieval Metrics (MRR, nDCG@5, Precision@5, Recall@5)
3. Conformal Risk Coverage (95% Safety Bound)
"""

import sys
import os
import json
import logging

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import EvidenceFirstPipeline
from src.ingestion.paper_loader import PaperLoader
from src.evaluation.ragchecker import RAGCheckerEvaluator
from src.evaluation.benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiDomainBenchmark")


def run_multidomain_benchmark():
    print("\n" + "=" * 90)
    print("  🏆 EvidenceFirst-RAG: Multi-Domain Frontier Diagnostic Benchmark Suite")
    print("=" * 90 + "\n")

    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    print(f"📚 Loaded {len(papers)} multi-domain papers from 'data/sample_papers'.")

    domains = ["All Domains", "Computer Vision / Deep Learning", "NLP / Language Models", "Biomedical / Clinical"]
    all_results = {}

    pipeline = EvidenceFirstPipeline()
    benchmarker = BenchmarkRunner()
    evaluator = RAGCheckerEvaluator()

    for dom in domains:
        print(f"\n🔬 Evaluating Domain: [{dom}]...")
        dom_filter = dom if dom != "All Domains" else None
        pipeline.index_papers(papers, domain=dom_filter)

        query = (
            "What are the limitations of Vision Transformers for low-light object detection?"
            if "Vision" in dom or dom == "All Domains"
            else ("Retrieval augmented generation vs long context models in scientific QA"
                  if "NLP" in dom else "Adverse drug interactions and clinical contraindications in medical RAG")
        )

        report = pipeline.run(query)
        ragck = evaluator.evaluate_pipeline_output(report, target_name=f"EvidenceFirst-RAG [{dom}]")
        all_results[dom] = ragck.model_dump()

        p = ragck.proposed_system_metrics
        d = ragck.baseline_standard_dense_metrics
        print(f"   • Claim Precision:  {p.overall_precision*100:.1f}% vs Dense: {d.overall_precision*100:.1f}% (Gain: +{((p.overall_precision-d.overall_precision)/d.overall_precision)*100:.1f}%)")
        print(f"   • Context Util (CU): {p.context_utilization*100:.1f}% vs Dense: {d.context_utilization*100:.1f}% (Gain: +{((p.context_utilization-d.context_utilization)/d.context_utilization)*100:.1f}%)")
        print(f"   • Faithfulness:     {p.faithfulness*100:.1f}% vs Dense: {d.faithfulness*100:.1f}% (Gain: +{((p.faithfulness-d.faithfulness)/d.faithfulness)*100:.1f}%)")
        print(f"   • Hallucination:    {p.hallucination_rate*100:.1f}% vs Dense: {d.hallucination_rate*100:.1f}% (Reduction: -{ragck.relative_hallucination_reduction_percent:.1f}%)")

    # Save benchmark results
    out_path = "data/multidomain_ragchecker_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Saved full multi-domain benchmark results to: {out_path}\n")
    print("=" * 90)
    print("  ✅ Multi-Domain Benchmark Execution Completed Successfully!")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_multidomain_benchmark()
