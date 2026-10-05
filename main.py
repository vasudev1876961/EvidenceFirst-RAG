"""Command Line Interface for EvidenceFirst-RAG."""

from __future__ import annotations
import os
import sys

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
import argparse
import logging
from src.pipeline import EvidenceFirstPipeline
from src.ingestion.paper_loader import PaperLoader
from src.evaluation.benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvidenceFirst-RAG")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="EvidenceFirst-RAG: Claim-Level Evidence Verification and Contradiction-Aware RAG for Scientific Literature"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the limitations of Vision Transformers for low-light object detection?",
        help="Scientific research query to investigate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample_papers",
        help="Directory containing papers (.json, .pdf, .txt)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run ablation benchmark study across RAG baseline architectures",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--arxiv-search",
        type=str,
        default=None,
        help="Search ArXiv live API and index retrieved papers into active corpus",
    )
    parser.add_argument(
        "--export-latex",
        action="store_true",
        help="Export research synthesis to compilation-ready LaTeX and BibTeX files",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run Leave-One-Out counterfactual robustness stress-test on synthesis claims",
    )
    parser.add_argument(
        "--serve-api",
        action="store_true",
        help="Launch the EvidenceFirst-RAG FastAPI REST server on port 8000",
    )

    args = parser.parse_args()

    # Serve API Mode
    if args.serve_api:
        import uvicorn
        print("\n🚀 Starting EvidenceFirst-RAG FastAPI Server on http://127.0.0.1:8000 ...")
        print("📖 Interactive OpenAPI Documentation: http://127.0.0.1:8000/docs\n")
        uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
        return

    config = load_config(args.config)

    print("\n" + "=" * 80)
    print("  🔬 EvidenceFirst-RAG: Scientific Literature Synthesis & Verification")
    print("=" * 80 + "\n")

    # 1. Initialize Pipeline
    pipeline = EvidenceFirstPipeline(config=config)

    # 2. Ingest & Index Papers
    loader = PaperLoader()
    papers = loader.load_all_from_directory(args.data_dir)
    if not papers:
        logger.warning(f"No papers found in {args.data_dir}. Using sample seed papers...")
        papers = loader.load_all_from_directory("data/sample_papers")

    print(f"📚 Indexing {len(papers)} research papers from '{args.data_dir}'...")
    pipeline.index_papers(papers)

    # ArXiv Live Ingestion Flag
    if args.arxiv_search:
        print(f"\n🌐 Querying ArXiv Live API for: '{args.arxiv_search}'...")
        arxiv_papers = pipeline.fetch_and_index_arxiv(args.arxiv_search, max_results=3)
        print(f"📥 Successfully indexed {len(arxiv_papers)} papers from ArXiv:")
        for ap in arxiv_papers:
            print(f"   • [{ap.paper_id}] {ap.title[:55]}... ({ap.year})")
        print(f"📊 Active Corpus Size: {len(pipeline.indexed_papers)} papers.\n")

    # 3. Benchmark Ablation Mode
    if args.benchmark:
        print("\n📊 Executing Ablation Study (Basic RAG vs Hybrid vs Reranker vs Verification vs EvidenceFirst)...")
        benchmarker = BenchmarkRunner()
        results = benchmarker.run_ablation_study(pipeline)

        print("\n" + "-" * 88)
        print(f"{'System Configuration':<36} | {'ECS':<8} | {'CIS':<8} | {'Contra Det.':<12} | {'Halluc.':<8}")
        print("-" * 88)
        for r in results:
            print(
                f"{r['system']:<36} | {r['evidence_coverage']*100:>5.1f}%  | "
                f"{r['citation_accuracy']*100:>5.1f}%  | {r['conflict_detection_rate']*100:>10.1f}% | {r['hallucination_rate']*100:>6.1f}%"
            )
        print("-" * 88 + "\n")
        return

    # 4. Standard Synthesis Mode
    print(f"\n🔍 Processing Research Query: \"{args.query}\"\n")
    report = pipeline.run(args.query)

    # Print formatted markdown report
    markdown_output = pipeline.report_generator.format_markdown(report)
    print(markdown_output)

    # Save output to report.md
    output_path = "data/last_synthesis_report.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"\n💾 Saved full report to: {output_path}")

    # Robustness Analysis Output
    if args.robustness and report.robustness:
        print("\n" + "=" * 80)
        print("  🛡️ Leave-One-Out Robustness & Counterfactual Analysis (ERI)")
        print("=" * 80)
        print(f"Evidence Robustness Index (ERI): {report.robustness.evidence_robustness_index * 100:.1f}%")
        print(f"Consensus Divergence Score:     {report.robustness.consensus_divergence_score * 100:.1f}%\n")
        if report.robustness.critical_papers:
            print("🚨 Critical Pivot Papers (omission flips claim verification status):")
            for cp in report.robustness.critical_papers:
                print(f"   • [{cp['paper_id']}] {cp['title'][:48]}... (Impacts {cp['impacted_claims_count']} claims)")
        else:
            print("✅ Consensus is stable across all individual paper omissions.")
        print("=" * 80 + "\n")

    # LaTeX & BibTeX Export Flag
    if args.export_latex and report.latex_code and report.bibtex_code:
        tex_path = "data/synthesis_paper.tex"
        bib_path = "data/references.bib"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(report.latex_code)
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(report.bibtex_code)
        print(f"📄 LaTeX paper exported to: {tex_path}")
        print(f"📚 BibTeX bibliography exported to: {bib_path}\n")


if __name__ == "__main__":
    main()
