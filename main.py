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
        "--domain",
        type=str,
        default=None,
        help="Filter corpus by scientific domain (e.g. 'Computer Vision', 'NLP', 'Biomedical')",
    )
    parser.add_argument(
        "--benchmark-ir",
        action="store_true",
        help="Run formal Information Retrieval evaluation suite (MRR, nDCG, Precision, Recall)",
    )
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="Export research synthesis to standalone interactive HTML report",
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
        "--iterative",
        action="store_true",
        help="Display agentic iterative verification trajectory across uncertain claims",
    )
    parser.add_argument(
        "--hypotheses",
        action="store_true",
        help="Synthesize and display automated testable scientific hypotheses from literature tensions",
    )
    parser.add_argument(
        "--rigor",
        action="store_true",
        help="Display methodological rigor and evidence hierarchy analysis across active corpus",
    )
    parser.add_argument(
        "--export-sankey",
        action="store_true",
        help="Export interactive Sankey verification flow diagram to standalone HTML file",
    )
    parser.add_argument(
        "--export-network",
        action="store_true",
        help="Export interactive Physics Network Graph to standalone HTML file",
    )
    parser.add_argument(
        "--peer-review",
        action="store_true",
        help="Conduct formal NeurIPS / ICLR conference-grade peer review and soundness audit",
    )
    parser.add_argument(
        "--debate-council",
        action="store_true",
        help="Convene 3-agent scientific consensus council to debate and resolve literature contradictions",
    )
    parser.add_argument(
        "--bibliometrics",
        action="store_true",
        help="Compute PageRank centrality, impact tiers, and co-citation network metrics",
    )
    parser.add_argument(
        "--export-slides",
        action="store_true",
        help="Export academic slide presentation to standalone HTML and Marp Markdown files",
    )
    parser.add_argument(
        "--causal-dag",
        action="store_true",
        help="Display Causal DAG nodes, confounding factors, and Do-Calculus intervention scores",
    )
    parser.add_argument(
        "--reproducibility",
        action="store_true",
        help="Conduct comprehensive Open Science and Statistical Reproducibility Audit (RRI)",
    )
    parser.add_argument(
        "--roadmap",
        action="store_true",
        help="Display Living Research Roadmap, paradigm decay, and 2-year projected milestones",
    )
    parser.add_argument(
        "--export-dossier",
        action="store_true",
        help="Export Autonomous Research Dossier & Press Briefing to standalone HTML and Markdown",
    )
    parser.add_argument(
        "--in-silico-sim",
        action="store_true",
        help="Run In Silico Laboratory Experiment Simulator & Monte Carlo statistical power analysis",
    )
    parser.add_argument(
        "--cross-domain",
        action="store_true",
        help="Discover isomorphic cross-domain analogies and architectural transfer blueprints",
    )
    parser.add_argument(
        "--novelty-sentinel",
        action="store_true",
        help="Conduct Academic Novelty, Prior Art Collision, and Terminology Semantic Drift Audit",
    )
    parser.add_argument(
        "--export-sim-lab",
        action="store_true",
        help="Export standalone interactive HTML In Silico Laboratory Simulator",
    )
    parser.add_argument(
        "--symbolic",
        action="store_true",
        help="Run SymPy automated theorem prover, mathematical bound derivations, and formal proof certification",
    )
    parser.add_argument(
        "--adversarial-test",
        action="store_true",
        help="Run Epistemic Red-Teaming against Simpson's paradox, cherry-picking, p-hacking, and citation poisoning",
    )
    parser.add_argument(
        "--prisma",
        action="store_true",
        help="Generate PRISMA 2020 systematic review flow, Cochrane Risk of Bias 2.0 matrix, and random-effects meta-analysis",
    )
    parser.add_argument(
        "--export-prisma",
        action="store_true",
        help="Export standalone interactive HTML PRISMA 2020 & Cochrane Forest Plot dashboard",
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Run Sequential Bayesian Belief Updating, Bayes Factor (BF10) calibration, and 95% HDI intervals",
    )
    parser.add_argument(
        "--preregister",
        action="store_true",
        help="Synthesize formal OSF / AsPredicted confirmatory preregistration protocols with SHA-256 hashes",
    )
    parser.add_argument(
        "--export-prereg",
        action="store_true",
        help="Export formal OSF Preregistration Dossier to standalone markdown document",
    )
    parser.add_argument(
        "--quantum-epistemic",
        action="store_true",
        help="Calculate Hilbert-space quantum epistemic state vectors, von Neumann entropy, and inter-study interference",
    )
    parser.add_argument(
        "--export-quantum",
        action="store_true",
        help="Export standalone interactive 3D HTML Quantum Epistemic Bloch Sphere Radar dashboard",
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

    # Domain filter
    if args.domain:
        print(f"🎯 Filtering corpus to domain: '{args.domain}'...")
        pipeline.filter_by_domain(args.domain)
        print(f"📊 Active Domain Corpus Size: {len(pipeline.indexed_papers)} papers ({len(pipeline.indexed_chunks)} chunks).\n")

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

    # IR Evaluation Mode
    if args.benchmark_ir:
        print("\n🎯 Running Formal IR Evaluation Suite (MRR, nDCG@5, Precision@5, Recall@5)...")
        benchmarker = BenchmarkRunner()
        ir_res = benchmarker.run_ir_benchmark(pipeline, domain=args.domain, top_k=5)
        print("\n" + "=" * 70)
        print(f"  Macro MRR:          {ir_res['macro_mrr']:.4f}")
        print(f"  Macro nDCG@5:       {ir_res['macro_ndcg_at_k']:.4f}")
        print(f"  Macro Precision@5:  {ir_res['macro_precision_at_k']*100:.1f}%")
        print(f"  Macro Recall@5:     {ir_res['macro_recall_at_k']*100:.1f}%")
        print(f"  Conflict Det Rate:  {ir_res['conflict_detection_rate']*100:.1f}%")
        print("=" * 70)
        print("\nPer-Query Breakdown:")
        for q in ir_res.get("per_query_metrics", []):
            print(f" • [{q['query_id']}] MRR: {q['mrr']:.2f} | nDCG: {q['ndcg_at_k']:.2f} | P@5: {q['precision_at_k']:.2f} | {q['query'][:48]}...")
        print("\n")
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

    # Agentic Verification Trajectory
    if args.iterative and report.verification_trajectory and report.verification_trajectory.steps:
        print("\n" + "=" * 80)
        print("  🧬 Agentic Iterative Verification Trajectory")
        print("=" * 80)
        print(f"Summary: {report.verification_trajectory.trajectory_summary}\n")
        for step in report.verification_trajectory.steps:
            print(f" • Step {step.step_number} [{step.claim_id}]: {step.initial_status} -> {step.updated_status} (ΔConf: {step.confidence_delta:+.2f})")
            print(f"   Targeted Query: '{step.query_generated}'")
            print(f"   Reasoning: {step.reasoning}")
        print("=" * 80 + "\n")

    # Methodological Rigor Analysis
    if args.rigor and report.rigor_report:
        print("\n" + "=" * 80)
        print("  ⚖️ Corpus Methodological Rigor & Evidence Hierarchy")
        print("=" * 80)
        print(f"Overall Corpus Rigor: {report.rigor_report.overall_corpus_rigor * 100:.1f}%")
        print(f"High-Rigor Ratio:     {report.rigor_report.high_rigor_paper_ratio * 100:.1f}%\n")
        for a in report.rigor_report.assessments:
            print(f" • [{a.paper_id}] {a.methodology_type} | Rigor: {a.rigor_score:.2f} | Multiplier: x{a.evidence_weight_multiplier:.2f}")
        print("=" * 80 + "\n")

    # Automated Scientific Hypotheses
    if args.hypotheses and report.hypotheses:
        print("\n" + "=" * 80)
        print("  💡 Automated Scientific Hypotheses & Proposed Experimental Protocols")
        print("=" * 80)
        for h in report.hypotheses:
            print(f"\n🔬 [{h.hypothesis_id}] {h.title}")
            print(f"   Statement:     \"{h.statement}\"")
            print(f"   Tension:       {h.tension_addressed}")
            print(f"   Rationale:     {h.rationale}")
            print(f"   Protocol:      {h.proposed_methodology}")
            print(f"   Predictions:   {', '.join(h.testable_predictions)}")
            print(f"   Falsification: {h.falsification_criteria}")
            print(f"   Feasibility:   {h.feasibility_tier} | Novelty: {h.novelty_impact_tier} | Impact: {h.estimated_impact_score * 100:.0f}%")
        print("\n" + "=" * 80 + "\n")

    # Interactive Sankey Export Flag
    if args.export_sankey:
        sankey_html = pipeline.graph_visualizer.generate_sankey_html(report)
        sankey_path = "data/synthesis_sankey.html"
        with open(sankey_path, "w", encoding="utf-8") as f:
            f.write(sankey_html)
        print(f"🔀 Interactive Sankey diagram exported to: {sankey_path}")

    # Interactive Network Export Flag
    if args.export_network:
        network_html = pipeline.graph_visualizer.generate_network_html(report)
        network_path = "data/synthesis_network.html"
        with open(network_path, "w", encoding="utf-8") as f:
            f.write(network_html)
        print(f"🌐 Interactive Physics Network Graph exported to: {network_path}\n")

    # Conference Peer Review & Soundness Audit
    if args.peer_review and report.peer_review:
        pr = report.peer_review
        sc = pr.score_card
        print("\n" + "=" * 80)
        print(f"  🎓 Conference Peer Review & Soundness Audit ({pr.conference_venue})")
        print("=" * 80)
        print(f"Recommendation: {sc.recommendation_label} ({sc.overall_recommendation}/10) | Confidence: {sc.confidence_score}/5")
        print(f"Scores: Soundness: {sc.soundness_score}/5 | Presentation: {sc.presentation_score}/5 | Contribution: {sc.contribution_score}/5\n")
        print("✅ Core Strengths:")
        for s in pr.strengths:
            print(f"   • {s}")
        print("\n⚠️ Identified Limitations & Weaknesses:")
        for w in pr.weaknesses:
            print(f"   • {w}")
        print("\n❓ Targeted Questions for Authors:")
        for q in pr.questions_for_authors:
            print(f"   • {q}")
        print("\n🔍 Missing Baseline Benchmarks:")
        for b in pr.missing_baselines:
            print(f"   • {b}")
        print(f"\n🛡️ Ethical & Societal Implications:\n   {pr.ethical_and_societal_implications}")
        print("=" * 80 + "\n")

    # Multi-Agent Consensus Council
    if args.debate_council and report.consensus_council:
        cc = report.consensus_council
        print("\n" + "=" * 80)
        print("  🏛️ Multi-Agent Scientific Consensus Council Deliberation")
        print("=" * 80)
        print(f"Topic: {cc.council_topic}\n")
        for r in cc.debate_rounds:
            print(f"--- Round {r.round_number}: {r.round_theme} ---")
            for t in r.turns:
                print(f"[{t.speaker_name} - {t.persona_title}]:")
                print(f"   {t.content}\n")
        print("=" * 80)
        print(f"🏆 COUNCIL VERDICT: {cc.verdict.verdict_type} (Confidence: {cc.verdict.confidence * 100:.0f}%)")
        print(f"📜 Golden Synthesis Principle:\n   \"{cc.verdict.golden_synthesis_principle}\"")
        print(f"⚙️ Operational Boundary Rule:\n   \"{cc.verdict.concrete_boundary_rule}\"")
        print("=" * 80 + "\n")

    # Bibliometrics & Co-Citation Network
    if args.bibliometrics and report.bibliometrics:
        bm = report.bibliometrics
        print("\n" + "=" * 80)
        print("  🌐 Bibliometric Co-Citation & PageRank Network Analysis")
        print("=" * 80)
        print(f"Papers Analyzed: {bm.total_papers_analyzed} | Graph Density: {bm.graph_density:.4f}")
        print(f"Foundational Anchors:   {', '.join(bm.foundational_anchors)}")
        print(f"Methodological Bridges: {', '.join(bm.methodological_bridges)}\n")
        print(f"{'Paper ID':<12} | {'PageRank':<10} | {'Degree':<8} | {'Impact Tier':<22} | {'Title'}")
        print("-" * 88)
        for pm in bm.paper_metrics:
            print(f"{pm.paper_id:<12} | {pm.pagerank_centrality:<10.4f} | {pm.citation_degree:<8} | {pm.impact_tier:<22} | {pm.paper_title[:32]}...")
        print("=" * 80 + "\n")

    # Academic Slide Deck Export Flag
    if args.export_slides and report.slide_deck:
        sd = report.slide_deck
        html_slide_path = "data/presentation_slides.html"
        marp_slide_path = "data/presentation_slides.marp.md"
        with open(html_slide_path, "w", encoding="utf-8") as f:
            f.write(sd.html_deck)
        with open(marp_slide_path, "w", encoding="utf-8") as f:
            f.write(sd.marp_markdown)
        print(f"🖥️ Interactive HTML Presentation Deck exported to: {html_slide_path}")
        print(f"📑 Marp Academic Markdown Slides exported to:    {marp_slide_path}\n")

    # Causal DAG Flag
    if args.causal_dag and report.causal_dag:
        cd = report.causal_dag
        print("\n" + "=" * 80)
        print("  🔮 Causal Directed Acyclic Graph (DAG) & Counterfactual Reasoning")
        print("=" * 80)
        print(f"Target Inquiry: {cd.target_query}")
        print(f"Causal Soundness: {cd.overall_causal_soundness_score * 100:.1f}% | Confounding Vulnerability: {cd.confounding_vulnerability_index * 100:.1f}%\n")
        print("Nodes Identified:")
        for n in cd.nodes:
            print(f"  • [{n.id}] {n.name} ({n.variable_type.upper()}) - {n.description}")
        print("\nCausal Relationships & Do-Calculus Interventions:")
        for e in cd.edges:
            print(f"  • {e.source} --({e.relation_type} | {e.effect_direction})--> {e.target} [P(Y|do(X)): {e.do_intervention_effect:+.2f}]")
        if cd.spurious_correlations_detected:
            print("\n⚠️ Spurious Correlations Debunked:")
            for sp in cd.spurious_correlations_detected:
                print(f"  ❌ Apparent: {sp['apparent_correlation']}")
                print(f"  ✅ Reality:  {sp['causal_reality']}\n")
        print("=" * 80 + "\n")

    # Reproducibility Audit Flag
    if args.reproducibility and report.reproducibility:
        rp = report.reproducibility
        print("\n" + "=" * 80)
        print("  🧪 Autonomous Open Science & Reproducibility Audit")
        print("=" * 80)
        print(f"Corpus Size: {rp.corpus_size} | Open Science Grade: {rp.overall_open_science_grade} | Mean Risk (RRI): {rp.mean_reproducibility_risk_index * 100:.1f}%")
        print(f"Open Code Ratio: {rp.open_code_ratio * 100:.1f}% | Open Data Ratio: {rp.open_data_ratio * 100:.1f}% | Statistical Rigor: {rp.statistical_rigor_ratio * 100:.1f}%\n")
        print(f"{'Paper ID':<12} | {'Grade':<6} | {'Risk (RRI)':<10} | {'Code':<6} | {'Data':<6} | {'Seeds/Error'}")
        print("-" * 88)
        for pa in rp.paper_audits:
            print(f"{pa.paper_id:<12} | {pa.open_science_grade:<6} | {pa.reproducibility_risk_score * 100:>5.1f}%     | {'YES' if pa.open_code_available else 'NO':<6} | {'YES' if pa.open_data_available else 'NO':<6} | {'YES' if pa.random_seeds_reported or pa.error_bars_present else 'NO'}")
        print("\nRemediation Protocol Patches:")
        for pt in rp.remediation_patch_protocol:
            print(f"  • {pt}")
        print("=" * 80 + "\n")

    # Living Research Roadmap Flag
    if args.roadmap and report.temporal_roadmap:
        rm = report.temporal_roadmap
        print("\n" + "=" * 80)
        print(f"  ⏳ Living Research Roadmap & Temporal Horizons ({rm.timespan_analyzed})")
        print("=" * 80)
        print("Paradigm Acceleration & Velocity:")
        for tr in rm.paradigm_trends:
            print(f"  • [{tr.temporal_status}] {tr.technology_or_paradigm} (Velocity: {tr.velocity_score * 100:.0f}%)")
            print(f"    Trend: {tr.empirical_support_trend}")
        print("\nProjected 2-Year Experimental Milestones:")
        for ms in rm.projected_milestones:
            print(f"  🚀 [{ms.milestone_id}] {ms.milestone_title} ({ms.target_timeframe})")
            print(f"     Target:  {ms.technical_objective}")
            print(f"     Metric:  {ms.success_metric}")
        print("=" * 80 + "\n")

    # Autonomous Research Dossier Export Flag
    if args.export_dossier and report.dossier:
        ds = report.dossier
        html_dossier_path = "data/research_dossier.html"
        md_dossier_path = "data/research_dossier.md"
        with open(html_dossier_path, "w", encoding="utf-8") as f:
            f.write(ds.html_dossier)
        with open(md_dossier_path, "w", encoding="utf-8") as f:
            f.write(ds.markdown_dossier)
        print(f"📰 Autonomous Research Dossier HTML exported to: {html_dossier_path}")
        print(f"📑 Autonomous Research Dossier MD exported to:   {md_dossier_path}\n")

    # In Silico Simulator Flag
    if args.in_silico_sim and report.in_silico_simulation:
        sim = report.in_silico_simulation
        print("\n" + "=" * 80)
        print(f"  🧪 In Silico Laboratory Experiment Simulation [{sim.simulation_id}]")
        print("=" * 80)
        print(f"Target Hypothesis: {sim.target_hypothesis_title}")
        print(f"Monte Carlo Trials: {sim.trials_count} | Projected Gain: +{sim.projected_net_gain_pct:.1f}%")
        print(f"Baseline μ: {sim.mean_baseline_performance:.1f} mAP | Proposed μ: {sim.mean_simulated_performance:.1f} mAP (95% CI: [{sim.confidence_interval_95[0]} - {sim.confidence_interval_95[1]}])")
        print(f"Statistical Power: {sim.power_analysis.statistical_power_achieved * 100:.1f}% (α = 0.05) | Cohen's d: {sim.power_analysis.cohens_d_effect_size:.2f} | Critical N: {sim.power_analysis.critical_sample_size_required}")
        print("\nMulti-Hardware Scaling Estimates:")
        for hw in sim.hardware_projections:
            print(f"  • {hw.target_hardware:<30} | {hw.vram_required_gb:>4.1f} GB VRAM | {hw.throughput_samples_sec:>5.1f} samp/s | {hw.training_time_hours:>4.1f} hrs | ${hw.estimated_cost_usd:>5.2f}")
        print("=" * 80 + "\n")

    # Cross-Domain Analogy Flag
    if args.cross_domain and report.cross_domain_transfer:
        tr = report.cross_domain_transfer
        print("\n" + "=" * 80)
        print(f"  🌐 Interdisciplinary Cross-Domain Analogies ({tr.primary_domain})")
        print("=" * 80)
        print(f"Isomorphism Density Score: {tr.isomorphism_density_score * 100:.0f}%")
        for ana in tr.analogies:
            print(f"\n  ⚡ [{ana.analogy_id}] {ana.source_domain} → {ana.target_domain}")
            print(f"     Isomorphic Challenge:   {ana.isomorphic_challenge}")
            print(f"     Mathematical Form:      {ana.shared_mathematical_form}")
            print(f"     Hybrid Architecture:    {ana.proposed_hybrid_architecture}")
            print(f"     Novelty: {ana.novelty_score * 100:.0f}% | Feasibility: {ana.feasibility_score * 100:.0f}% | Projected Gain: {ana.expected_empirical_gain}")
        print("=" * 80 + "\n")

    # Novelty Sentinel Flag
    if args.novelty_sentinel and report.novelty_sentinel:
        ns = report.novelty_sentinel
        print("\n" + "=" * 80)
        print("  🛡️ Academic Novelty, Prior Art & Semantic Drift Sentinel")
        print("=" * 80)
        print(f"Novelty Index (NI): {ns.mean_novelty_index * 100:.0f}% | Verdict: {ns.overall_sentinel_verdict}\n")
        print("Novelty Quadrant Distribution:")
        for quad, count in ns.novelty_quadrant_distribution.items():
            print(f"  • {quad:<55}: {count} items")
        print("\nPrior Art Collision Defense:")
        for col in ns.prior_art_collisions:
            print(f"  • [{col.collision_id}] vs '{col.overlapping_paper_title}' (Similarity: {col.similarity_score * 100:.0f}%, Risk: {col.infringement_or_redundancy_risk})")
            print(f"    Defense: {col.differentiation_argument}")
        print("\nTerminology Semantic Drift Audit:")
        for dr in ns.semantic_drift_records:
            print(f"  • '{dr.term}': {dr.verdict} (Dilution Score: {dr.dilution_score * 100:.0f}%)")
            print(f"    Recommendation: {dr.audit_recommendation}")
        print("=" * 80 + "\n")

    # Export In Silico Simulation Lab HTML Flag
    if args.export_sim_lab and report.in_silico_simulation:
        sim = report.in_silico_simulation
        out_html = "data/in_silico_simulation_lab.html"
        out_py = "data/in_silico_benchmark.py"
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(sim.interactive_lab_html)
        with open(out_py, "w", encoding="utf-8") as f:
            f.write(sim.executable_python_benchmark)
        print(f"🧪 In Silico Simulation Lab HTML exported to: {out_html}")
        print(f"🐍 Standalone PyTorch Benchmark Harness saved to: {out_py}\n")

    # Symbolic Proof Verifier Flag (v6.0)
    if args.symbolic and report.symbolic_proof_report:
        pr = report.symbolic_proof_report
        print("\n" + "=" * 80)
        print(f"  📐 Automated Theorem & Symbolic Formal Proof Verifier [{pr.proof_system_id}]")
        print("=" * 80)
        print(f"Theorems Evaluated: {pr.theorems_evaluated} | Consistency Score: {pr.formal_consistency_score * 100:.0f}%\n")
        for cert in pr.proof_certificates:
            print(f"  • [{cert.proof_id}] {cert.target_claim_or_theorem}")
            print(f"    Domain: {cert.domain_formalism} | Verdict: {cert.verdict}")
            print(f"    Asymptotic Bound: {cert.asymptotic_bound} | Lipschitz: {cert.lipschitz_constant_or_gradient_bound}")
            print(f"    SymPy: {cert.sympy_formulation}")
            if cert.unstated_assumptions:
                print(f"    Unstated Assumptions: {'; '.join(cert.unstated_assumptions)}")
            print()
        print("Critical Theoretical Breakdowns Identified:")
        for bd in pr.critical_theoretical_breakdowns:
            print(f"  ⚠ {bd}")
        print("=" * 80 + "\n")

    # Adversarial Red-Team Flag (v6.0)
    if args.adversarial_test and report.adversarial_stress_report:
        adv = report.adversarial_stress_report
        print("\n" + "=" * 80)
        print(f"  🎯 Epistemic Red-Teaming & Adversarial Hallucination Stress-Test [{adv.stress_test_id}]")
        print("=" * 80)
        print(f"Defense Rate: {adv.defense_rate_pct:.1f}% ({adv.attacks_successfully_defended}/{adv.total_attacks_simulated} Defended)")
        print(f"Epistemic Vulnerability Index (EVI): {adv.epistemic_vulnerability_index:.3f} | Resilience: {adv.adversarial_resilience_verdict}\n")
        print("Adversarial Attack Audits:")
        for atk in adv.attack_vectors:
            status = "✓ DEFENDED" if atk.pipeline_defended else "✗ FAILED"
            print(f"  • [{atk.attack_id}] {atk.attack_category:<35} [{status}] (Severity: {atk.vulnerability_severity})")
            print(f"    Target:  {atk.intended_deception_target}")
            print(f"    Defense: {atk.observed_failure_or_defense}")
            print(f"    Patch:   {atk.remediation_patch}\n")
        print("Automated Epistemic Defense Rules:")
        for r in adv.automated_defense_rules:
            print(f"  🛡️ {r}")
        print("=" * 80 + "\n")

    # PRISMA Systematic Review Flag (v6.0)
    if args.prisma and report.prisma_meta_analysis:
        pma = report.prisma_meta_analysis
        print("\n" + "=" * 80)
        print(f"  📊 PRISMA 2020 Systematic Review & Cochrane Meta-Analysis [{pma.prisma_id}]")
        print("=" * 80)
        print(f"Domain: {pma.inquiry_domain} | Strength: {pma.overall_synthesis_strength}")
        print(f"PRISMA Flow: Identified {pma.prisma_flow.identification_count} -> Screened {pma.prisma_flow.screening_count} -> Included {pma.prisma_flow.included_synthesis_count} Studies")
        print(f"Pooled Effect Size (Random-Effects): {pma.pooled_random_effects_estimate:+.3f} (95% CI: [{pma.pooled_confidence_interval_95[0]}, {pma.pooled_confidence_interval_95[1]}])")
        print(f"Heterogeneity I²: {pma.heterogeneity_i2_pct:.1f}% (Cochran's Q = {pma.cochran_q_statistic:.2f}, p = {pma.cochran_q_p_value:.4f}, τ² = {pma.tau_squared_variance})")
        print(f"Publication Bias (Egger's Test): p = {pma.eggers_p_value:.2f} (Intercept β₀ = {pma.eggers_intercept_beta0:.2f}, Detected: {pma.publication_bias_detected})\n")
        print("Cochrane Risk of Bias 2.0 (RoB 2) Summary:")
        for rob in pma.cochrane_risk_of_bias:
            print(f"  • {rob.study_id:<12}: Overall Risk = {rob.overall_risk_of_bias:<14} | Selection: {rob.domain_1_randomization_selection:<4} | Reporting: {rob.domain_5_selective_reporting}")
        print("=" * 80 + "\n")

    # Export PRISMA HTML Report Flag (v6.0)
    if args.export_prisma and report.prisma_meta_analysis:
        pma = report.prisma_meta_analysis
        out_prisma_html = "data/prisma_meta_analysis_report.html"
        with open(out_prisma_html, "w", encoding="utf-8") as f:
            f.write(pma.interactive_forest_html)
        print(f"📊 PRISMA 2020 & Cochrane Forest Plot HTML exported to: {out_prisma_html}\n")

    # Bayesian Belief Updating Flag (v7.0)
    if args.bayesian and report.bayesian_belief_report:
        bayes = report.bayesian_belief_report
        print("\n" + "=" * 80)
        print(f"  🧠 Bayesian Epistemic Belief Updating & Bayes Factor Calibration [{bayes.report_id}]")
        print("=" * 80)
        print(f"Claims Evaluated: {bayes.claims_evaluated} | Consensus Convergence Rate: {bayes.consensus_convergence_rate * 100:.1f}%")
        print(f"Epistemic Shannon Entropy: {bayes.overall_epistemic_entropy_prior:.3f} -> {bayes.overall_epistemic_entropy_posterior:.3f} nats (Info Gain: {bayes.mean_information_gain_nats:.3f} nats)")
        print(f"Calibration Quality Score (Brier): {bayes.calibration_quality_score * 100:.1f}%\n")
        print("Bayesian Claim Credence Shifts & Credible Intervals (95% HDI):")
        for b in bayes.claim_beliefs:
            print(f"  • [{b.claim_id}] {b.statement[:65]}...")
            print(f"    Prior P(H): {b.prior_credence:.2f} -> Posterior P(H|E): {b.posterior_credence:.2f} (Shift: {b.belief_shift_magnitude:+.2f})")
            print(f"    Aggregate BF10: {b.bayes_factor_aggregate_10:.2f} | Status: {b.epistemic_state_label}")
            print(f"    95% Credible Interval (HDI): [{b.credible_interval_95_hdi[0]:.2f}, {b.credible_interval_95_hdi[1]:.2f}] | Info Gain: {b.epistemic_information_gain_nats:.3f} nats\n")
        print("Epistemic Belief Insights:")
        for ins in bayes.epistemic_insights:
            print(f"  💡 {ins}")
        print("=" * 80 + "\n")

    # OSF Preregistration Flag (v7.0)
    if args.preregister and report.preregistration_dossier:
        dossier = report.preregistration_dossier
        print("\n" + "=" * 80)
        print(f"  📋 Open Science Framework (OSF) Confirmatory Preregistrations [{dossier.dossier_id}]")
        print("=" * 80)
        print(f"Total Protocols: {dossier.total_protocols_synthesized} | Open Science Compliance: {dossier.open_science_compliance_index * 100:.1f}%\n")
        for proto in dossier.protocols:
            print(f"  • [{proto.protocol_id}] {proto.title}")
            print(f"    Type: {proto.study_type} | Status: {proto.osf_badge_audit.preregistration_status}")
            print(f"    Cryptographic SHA-256 Hash: {proto.cryptographic_sha256_hash}")
            print(f"    Power: 1-β = {proto.sample_size_determination.get('power_target', 0.90)}, α = {proto.sample_size_determination.get('alpha_level', 0.01)}, N = {proto.sample_size_determination.get('min_required_test_samples', 150)}")
            print(f"    FDR Correction: {proto.fdr_correction_method}")
            print()
        print("Anti-P-Hacking Safeguards:")
        for sg in dossier.anti_phacking_safeguards:
            print(f"  🛡️ {sg}")
        print("=" * 80 + "\n")

    # Export OSF Preregistration Markdown Flag (v7.0)
    if args.export_prereg and report.preregistration_dossier:
        dossier = report.preregistration_dossier
        out_prereg_md = "data/osf_preregistration_dossier.md"
        with open(out_prereg_md, "w", encoding="utf-8") as f:
            f.write(dossier.preregistration_markdown_document)
        print(f"📋 OSF Preregistration Dossier Markdown exported to: {out_prereg_md}\n")

    # Quantum Epistemic Engine Flag (v7.0)
    if args.quantum_epistemic and report.quantum_epistemic_report:
        q_rep = report.quantum_epistemic_report
        print("\n" + "=" * 80)
        print(f"  ⚛️ Quantum-Inspired Epistemic State & Hilbert Space Superposition [{q_rep.report_id}]")
        print("=" * 80)
        print(f"Mean von Neumann Entropy: {q_rep.mean_von_neumann_entropy:.3f} nats | Claims in Superposition: {q_rep.total_claims_in_superposition}")
        print(f"Systemic Coherence Index: {q_rep.systemic_coherence_index:.3f}\n")
        print("Bloch Sphere Epistemic Coordinates & Quantum Density Matrices:")
        for qs in q_rep.quantum_states:
            coord = qs.bloch_coordinates
            print(f"  • [{qs.claim_id}] {qs.statement[:65]}...")
            print(f"    Regime: {qs.epistemic_regime}")
            print(f"    Amplitudes: |Support⟩={qs.support_amplitude_alpha:.2f}, |Contradict⟩={qs.contradiction_amplitude_beta:.2f}")
            print(f"    Bloch Vector: (x={coord.x:.2f}, y={coord.y:.2f}, z={coord.z:.2f}) [Lat: {coord.latitude_deg:.1f}°, Lon: {coord.longitude_deg:.1f}°]")
            print(f"    von Neumann S: {qs.von_neumann_entropy:.3f} | Coherence: {qs.quantum_coherence:.3f} | Purity: {qs.superposition_purity:.3f}")
            if qs.interfering_pairs:
                for pair in qs.interfering_pairs[:1]:
                    print(f"    Interference ({pair.interference_type}): Mag={pair.interference_magnitude:+.3f}, Shift={pair.phase_shift_deg:.1f}°")
            print()
        print("Quantum Epistemic Phase Insights:")
        for ins in q_rep.epistemic_phase_insights:
            print(f"  ⚛️ {ins}")
        print("=" * 80 + "\n")

    # Export Quantum Bloch Sphere HTML Flag (v7.0)
    if args.export_quantum and report.quantum_epistemic_report:
        q_rep = report.quantum_epistemic_report
        out_bloch_html = "data/quantum_epistemic_radar.html"
        with open(out_bloch_html, "w", encoding="utf-8") as f:
            f.write(q_rep.bloch_sphere_html)
        print(f"⚛️ Quantum Epistemic Bloch Sphere Radar HTML exported to: {out_bloch_html}\n")



if __name__ == "__main__":
    main()


