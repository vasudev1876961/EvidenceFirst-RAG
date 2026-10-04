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

    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
