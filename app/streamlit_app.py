"""Streamlit Research Analysis Dashboard for EvidenceFirst-RAG."""

from __future__ import annotations
import os
import sys
import json
import yaml
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import EvidenceFirstPipeline
from src.ingestion.paper_loader import PaperLoader
from src.evaluation.benchmark import BenchmarkRunner

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="EvidenceFirst-RAG | Scientific Literature Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Academic Research Tool
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .claim-card {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .badge-strong {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-moderate {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-conflict {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-weak {
        background-color: #78350F;
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    config_path = "configs/config.yaml"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    pipe = EvidenceFirstPipeline(config=config)
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipe.index_papers(papers)
    return pipe


pipeline = get_pipeline()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/microscope.png", width=64)
    st.title("EvidenceFirst-RAG")
    st.caption("Contradiction-Aware Scientific Literature Synthesis")

    st.markdown("---")
    st.subheader("⚙️ Query Configuration")

    domain = st.selectbox(
        "Scientific Domain",
        ["All Domains", "Computer Vision / Deep Learning", "NLP / LLMs (ACL)", "Biomedical Literature (PubMed)"],
        index=1,
    )

    # Trigger re-indexing if domain filter changed
    if "active_domain" not in st.session_state or st.session_state["active_domain"] != domain:
        st.session_state["active_domain"] = domain
        pipeline.filter_by_domain(domain if domain != "All Domains" else None)

    year_range = st.slider("Publication Year Range", min_value=2018, max_value=2026, value=(2020, 2026))
    evidence_threshold = st.slider("Evidence Confidence Threshold", min_value=0.30, max_value=0.95, value=0.60, step=0.05)
    hybrid_alpha = st.slider("Retrieval Balance (α BM25 vs 1-α Dense)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    st.markdown("---")
    st.subheader("📚 Active Corpus Index")
    st.write(f"Indexed Papers: **{len(pipeline.indexed_papers)}**")
    st.write(f"Indexed Chunks: **{len(pipeline.indexed_chunks)}**")
    with st.expander("View Indexed Papers Catalog"):
        for p in pipeline.indexed_papers:
            st.caption(f"• **[{p.paper_id}]** {p.title[:38]}... ({p.year}) [{p.metadata.get('domain', 'CV')[:10]}]")


# ----------------- MAIN PANEL -----------------
st.markdown('<div class="main-header">EvidenceFirst-RAG: Scientific Literature Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Claim-Level Verification, Contradiction Detection, Empirical Meta-Analysis & Robustness Auditing</div>', unsafe_allow_html=True)

# Query selection / presets adapted to domain
if domain == "NLP / LLMs (ACL)":
    preset_queries = [
        "Compare modular RAG against native long-context windows for multi-hop scientific reasoning.",
        "How does claim-level verification impact hallucination and reasoning in scientific RAG?",
        "What are the risks of text chunking boundary fragmentation on multi-paragraph discourse?",
        "Custom Query..."
    ]
elif domain == "Biomedical Literature (PubMed)":
    preset_queries = [
        "What are the risks and efficacy of dense retrieval in clinical decision support and contraindication queries?",
        "Factuality and citation faithfulness in biomedical question answering with PubMedBERT.",
        "Why do dense embeddings struggle with negated drug-drug interactions?",
        "Custom Query..."
    ]
else:
    preset_queries = [
        "What are the limitations of Vision Transformers for low-light object detection?",
        "Compare CNNs and Vision Transformers for low-light object detection and identify their limitations.",
        "Does illumination enhancement preprocessing consistently improve object detection accuracy across dark datasets?",
        "Edge-device computational trade-offs between lightweight YOLO and Deformable DETR.",
        "Custom Query..."
    ]

selected_preset = st.selectbox("Select Research Query Template or Custom", preset_queries, index=0)

if selected_preset == "Custom Query...":
    user_query = st.text_input("Enter your research question:", value=preset_queries[0])
else:
    user_query = selected_preset

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_btn = st.button("🚀 Analyze Literature", type="primary", use_container_width=True)


# Check session state for report
if "report" not in st.session_state or run_btn:
    with st.spinner("Executing EvidenceFirst-RAG Pipeline (Decomposition → Hybrid Search → Rerank → Claim Verification → Contradiction Engine → Meta-Analysis)..."):
        pipeline.hybrid.alpha = hybrid_alpha
        report = pipeline.run(user_query)
        st.session_state["report"] = report

report = st.session_state["report"]

# Top Metrics Row
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Overall Confidence", f"{report.overall_confidence * 100:.0f}%", delta="Reliable" if report.overall_confidence >= 0.75 else "Moderate")
with m2:
    st.metric("Evidence Coverage (ECS)", f"{report.citation_audit.evidence_coverage_score * 100:.0f}%", help="Claims with verified supporting evidence / Total claims")
with m3:
    st.metric("Citation Integrity (CIS)", f"{report.citation_audit.citation_integrity_score * 100:.0f}%", help="Citations passing entailment check / Total citations")
with m4:
    eri_val = report.robustness.evidence_robustness_index if report.robustness else 1.0
    st.metric("Robustness Index (ERI)", f"{eri_val * 100:.0f}%", help="Leave-one-out claim stability score across all papers")
with m5:
    num_conflicts = len(report.conflicting_findings)
    st.metric("Conflicting Claims", num_conflicts, delta=f"{num_conflicts} Detected", delta_color="inverse" if num_conflicts > 0 else "normal")
with m6:
    st.metric("Verified Claims", report.citation_audit.total_claims)

st.markdown("---")

# Navigation Tabs
tab_syn, tab_claims, tab_conflicts, tab_hypo, tab_trajectory, tab_rigor, tab_sankey, tab_graph, tab_empirical, tab_robustness, tab_arxiv, tab_latex, tab_audit, tab_ablation = st.tabs([
    "📑 Executive Synthesis",
    "🔍 Claim-Level Evidence",
    "⚡ Contradiction Attribution",
    "💡 Scientific Hypotheses",
    "🧬 Agentic Trajectory",
    "⚖️ Methodological Rigor",
    "🔀 Sankey Flow",
    "🕸️ Knowledge Graph",
    "📊 Empirical Meta-Analysis",
    "🛡️ Robustness & Counterfactuals",
    "🌐 Live ArXiv Ingestion",
    "📄 LaTeX & BibTeX Export",
    "🛡️ Citation Audit",
    "📈 Ablation Benchmarks"
])


# ----------------- TAB 1: EXECUTIVE SYNTHESIS -----------------
with tab_syn:
    st.subheader("Executive Literature Summary")
    st.info(report.executive_summary)

    col_subq, col_gaps = st.columns(2)
    with col_subq:
        st.subheader("🔍 Query Decomposition Subqueries")
        for sq in report.decomposed_subqueries:
            st.markdown(f"- **Subquery**: `{sq}`")

    with col_gaps:
        st.subheader("🔬 Identified Research Gaps")
        for gap in report.research_gaps:
            st.markdown(f"- ⚠️ {gap}")

    st.subheader("Key Findings Summary")
    for f in report.key_findings:
        status = f["status"]
        badge_class = "badge-strong" if "Strong" in status else ("badge-conflict" if "Conflicting" in status else "badge-moderate")
        st.markdown(f"""
        <div class="claim-card">
            <span class="{badge_class}">{status.upper()}</span> &nbsp; <b>{f['aspect'].title()}</b> ({f['claim_id']})<br>
            <p style="margin-top: 6px; font-size: 1.05rem;">{f['statement']}</p>
            <small style="color: #94A3B8;">Confidence: <b>{f['confidence_score']:.2f}</b> | Supporting: <b>{f['supporting_count']}</b> papers | Contradicting: <b>{f['contradicting_count']}</b> papers</small>
        </div>
        """, unsafe_allow_html=True)


# ----------------- TAB 2: CLAIM-LEVEL EVIDENCE INSPECTOR -----------------
with tab_claims:
    st.subheader("Interactive Claim Inspector")
    st.caption("Inspect individual factual claims, entailment verification, and source snippets.")

    claim_options = {c.claim_id: f"{c.claim_id} [{c.evidence_status}] - {c.statement[:65]}..." for c in report.claims}
    selected_cid = st.selectbox("Select Claim to Inspect:", list(claim_options.keys()), format_func=lambda x: claim_options[x])

    selected_claim = next((c for c in report.claims if c.claim_id == selected_cid), report.claims[0])

    st.markdown(f"### Claim Statement ({selected_claim.claim_id})")
    st.markdown(f"**\"{selected_claim.statement}\"**")

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.markdown("#### ✅ Supporting Evidence Passages")
        if selected_claim.supporting_evidence:
            for ev in selected_claim.supporting_evidence:
                with st.expander(f"[{ev.paper_id}] {ev.paper_title} ({ev.section}, {ev.year}) - Score: {ev.composite_score:.2f}"):
                    st.write(f"**Text Snippet**:\n> \"{ev.text_snippet}\"")
                    st.caption(f"Entailment Prob: {ev.entailment_prob:.2f} | Semantic Sim: {ev.semantic_similarity:.2f}")
                    if ev.extracted_attributes:
                        st.json(ev.extracted_attributes)
        else:
            st.caption("No direct supporting evidence found.")

    with col_ev2:
        st.markdown("#### ❌ Contradicting Evidence Passages")
        if selected_claim.contradicting_evidence:
            for ev in selected_claim.contradicting_evidence:
                with st.expander(f"⚠️ [{ev.paper_id}] {ev.paper_title} ({ev.section}, {ev.year}) - Conflict Prob: {ev.contradiction_prob:.2f}"):
                    st.write(f"**Text Snippet**:\n> \"{ev.text_snippet}\"")
                    st.caption(f"Contradiction Prob: {ev.contradiction_prob:.2f} | Semantic Sim: {ev.semantic_similarity:.2f}")
                    if ev.extracted_attributes:
                        st.json(ev.extracted_attributes)
        else:
            st.caption("No contradictory passages detected.")


# ----------------- TAB 3: CONTRADICTION & CONFLICT ATTRIBUTION -----------------
with tab_conflicts:
    st.subheader("⚡ Contradiction Detection & Conflict Attribution Engine")
    st.caption("Identifying where scientific literature diverges and explaining the methodological root cause.")

    if report.conflicting_findings:
        for conflict in report.conflicting_findings:
            st.markdown(f"### ⚔️ Disagreement on Claim: `{conflict.claim_id}`")
            st.markdown(f"**\"{conflict.claim_statement}\"**")

            c_sup, c_contra = st.columns(2)
            with c_sup:
                st.success("**Supporting Studies Side**")
                for p in conflict.supporting_papers:
                    st.write(f"• **[{p.get('paper_id')}]** {p.get('title')} ({p.get('year')})")
                    st.caption(f"Dataset: `{p.get('dataset')}` | Metric: `{p.get('metric')}`")

            with c_contra:
                st.error("**Contradicting Studies Side**")
                for p in conflict.contradicting_papers:
                    st.write(f"• **[{p.get('paper_id')}]** {p.get('title')} ({p.get('year')})")
                    st.caption(f"Dataset: `{p.get('dataset')}` | Metric: `{p.get('metric')}`")

            st.markdown("#### 🔍 Root-Cause Methodological Attribution")
            st.warning(f"**Likely Reason for Disagreement**:\n{conflict.likely_reason}")

            if conflict.divergent_factors:
                st.markdown("**Empirical Comparison Table**:")
                df_factors = pd.DataFrame(conflict.divergent_factors)
                st.table(df_factors)
            st.markdown("---")
    else:
        st.success("No active scientific contradictions detected for the extracted claims in this query.")


# ----------------- TAB 4: SCIENTIFIC HYPOTHESES & PROTOCOLS -----------------
with tab_hypo:
    st.subheader("💡 Automated Scientific Hypotheses & Proposed Experimental Protocols")
    st.caption("AI-synthesized falsifiable research hypotheses designed to resolve literature tensions and empirical contradictions.")

    if report.hypotheses:
        for h in report.hypotheses:
            with st.container():
                st.markdown(f"### 🔬 [{h.hypothesis_id}] {h.title}")
                st.info(f"**Formal Hypothesis Statement:**\n\n\"{h.statement}\"")

                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown("**Underlying Tension Addressed:**")
                    st.write(h.tension_addressed)
                    st.markdown("**Theoretical Rationale:**")
                    st.write(h.rationale)

                with col_h2:
                    st.markdown("**Proposed Experimental Protocol:**")
                    st.write(h.proposed_methodology)
                    st.markdown("**Falsification Criteria:**")
                    st.error(h.falsification_criteria)

                if h.testable_predictions:
                    st.markdown("**🎯 Testable Quantitative Predictions:**")
                    for p in h.testable_predictions:
                        st.markdown(f"- • `{p}`")

                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Feasibility Tier", h.feasibility_tier)
                m_col2.metric("Novelty & Impact", h.novelty_impact_tier)
                m_col3.metric("Estimated Impact Score", f"{h.estimated_impact_score * 100:.0f}%")
                st.markdown("---")
    else:
        st.info("No hypothesis synthesis triggered for this query.")


# ----------------- TAB 5: AGENTIC VERIFICATION TRAJECTORY -----------------
with tab_trajectory:
    st.subheader("🧬 Agentic Multi-Hop Verification Trajectory")
    st.caption("Auditable trace of targeted counter-queries generated to resolve scientific ambiguities and under-supported claims.")

    if report.verification_trajectory and report.verification_trajectory.steps:
        st.success(f"**Trajectory Overview:** {report.verification_trajectory.trajectory_summary}")

        traj_rows = []
        for step in report.verification_trajectory.steps:
            traj_rows.append({
                "Step": step.step_number,
                "Claim ID": step.claim_id,
                "Targeted Query Generated": step.query_generated,
                "Chunks Retrieved": step.retrieved_chunks_count,
                "Initial Status": step.initial_status,
                "Updated Status": step.updated_status,
                "Confidence Delta": f"{step.confidence_delta:+.2f}",
                "Reasoning": step.reasoning,
            })
        st.dataframe(pd.DataFrame(traj_rows), use_container_width=True)

        for step in report.verification_trajectory.steps:
            with st.expander(f"Step {step.step_number}: [{step.claim_id}] {step.initial_status} ➔ {step.updated_status} (ΔConf: {step.confidence_delta:+.2f})"):
                st.write(f"**Target Claim**: {step.target_statement}")
                st.write(f"**Generated Search Query**: `{step.query_generated}`")
                st.write(f"**Targeted Chunks Retrieved**: `{step.retrieved_chunks_count}`")
                st.info(f"**Verification Reasoning**: {step.reasoning}")
    else:
        st.info("Initial retrieval satisfied high evidence confidence; no secondary agentic hops were required.")


# ----------------- TAB 6: METHODOLOGICAL RIGOR & EVIDENCE HIERARCHY -----------------
with tab_rigor:
    st.subheader("⚖️ Corpus Methodological Rigor & Evidence Hierarchy")
    st.caption("Evaluation of scientific methodology rigor and study designs, dynamically weighting empirical evidence.")

    if report.rigor_report:
        r_rep = report.rigor_report
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Overall Corpus Rigor", f"{r_rep.overall_corpus_rigor * 100:.1f}%")
        rc2.metric("High-Rigor Evidence Ratio", f"{r_rep.high_rigor_paper_ratio * 100:.1f}%")
        rc3.metric("Evaluated Studies", len(r_rep.assessments))

        if r_rep.methodology_distribution:
            df_dist = pd.DataFrame(list(r_rep.methodology_distribution.items()), columns=["Methodology Tier", "Paper Count"])
            fig_rigor = px.pie(
                df_dist,
                names="Methodology Tier",
                values="Paper Count",
                title="Scientific Methodology Distribution Across Corpus",
                hole=0.45,
                color_discrete_sequence=["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B", "#64748B"]
            )
            fig_rigor.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"))
            st.plotly_chart(fig_rigor, use_container_width=True)

        st.markdown("### 📋 Study Methodology Hierarchy & Weight Multipliers")
        rigor_table_data = [
            {
                "Paper ID": a.paper_id,
                "Title": a.paper_title[:45] + "...",
                "Year": a.year,
                "Methodology Type": a.methodology_type,
                "Rigor Score": f"{a.rigor_score:.2f}",
                "Weight Multiplier": f"x{a.evidence_weight_multiplier:.2f}",
                "Datasets Count": a.sample_size_or_datasets_count,
            }
            for a in r_rep.assessments
        ]
        st.dataframe(pd.DataFrame(rigor_table_data), use_container_width=True)
    else:
        st.info("Methodological rigor assessment not available.")


# ----------------- TAB 7: SANKEY EVIDENCE FLOW -----------------
with tab_sankey:
    st.subheader("🔀 Multi-Stage Literature Verification Flow (Sankey)")
    st.caption("Traceability flow: Research Query ➔ Extracted Claims ➔ Supporting/Contradicting Papers ➔ Final Verification Consensus.")

    if report.sankey_data:
        sdata = report.sankey_data
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=18,
                thickness=24,
                line=dict(color="#0F172A", width=1.5),
                label=sdata["nodes"],
                color=sdata["node_colors"]
            ),
            link=dict(
                source=sdata["sources"],
                target=sdata["targets"],
                value=sdata["values"],
                color=sdata["link_colors"]
            )
        )])
        fig_sankey.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(color="#F8FAFC", size=12),
            height=600,
            margin=dict(l=20, r=20, t=25, b=25)
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Sankey diagram data is not available.")



# ----------------- TAB 4: EVIDENCE KNOWLEDGE GRAPH -----------------
with tab_graph:
    st.subheader("🕸️ Scientific Evidence Knowledge Graph")
    st.caption("Directed graph illustrating connections between Query, Claims, Supporting/Contradicting Papers, and Datasets. Red dashed lines denote contradictions.")

    fig = pipeline.graph_builder.to_plotly_figure(report.evidence_graph)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Graph visualization requires plotly.")


# ----------------- TAB 5: EMPIRICAL META-ANALYSIS MATRIX -----------------
with tab_empirical:
    st.subheader("📊 Cross-Paper Empirical Meta-Analysis Matrix")
    st.caption("Synthesizing quantitative results across models, datasets, and computational metrics.")

    if report.meta_analysis:
        meta = report.meta_analysis

        st.markdown("### 🏆 Cross-Dataset Benchmark Matrix")
        if meta.dataset_comparison_table:
            df_comp = pd.DataFrame(meta.dataset_comparison_table)
            st.dataframe(df_comp, use_container_width=True)

        col_lb, col_var = st.columns([3, 2])
        with col_lb:
            st.markdown("### 🥇 Empirical Model Leaderboard")
            if meta.model_leaderboard:
                df_lb = pd.DataFrame(meta.model_leaderboard)
                st.dataframe(df_lb[["Rank", "Model", "Dataset", "Score", "FPS", "Notes"]], use_container_width=True)

        with col_var:
            st.markdown("### 📈 Cross-Dataset Benchmark Variance")
            if meta.cross_dataset_variance:
                df_var = pd.DataFrame([
                    {"Dataset": k, "Standard Deviation (σ)": v}
                    for k, v in meta.cross_dataset_variance.items()
                ])
                fig_var = px.bar(
                    df_var, x="Dataset", y="Standard Deviation (σ)",
                    title="Empirical Discrepancy by Benchmark Dataset",
                    color="Standard Deviation (σ)",
                    color_continuous_scale="Purples"
                )
                fig_var.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"))
                st.plotly_chart(fig_var, use_container_width=True)

        st.markdown("### 💡 Key Quantitative Findings")
        for ins in meta.key_empirical_insights:
            st.markdown(f"- 📌 {ins}")
    else:
        st.info("No empirical meta-analysis generated.")


# ----------------- TAB 6: ROBUSTNESS & COUNTERFACTUAL STUDIO -----------------
with tab_robustness:
    st.subheader("🛡️ Counterfactual & Leave-One-Out Robustness Studio")
    st.caption("Testing the fragility of synthesis conclusions when individual papers are ablated from the literature pool.")

    if report.robustness:
        rob = report.robustness
        r1, r2, r3 = st.columns(3)
        r1.metric("Evidence Robustness Index (ERI)", f"{rob.evidence_robustness_index * 100:.1f}%", help="Resilience score under single-paper removal")
        r2.metric("Consensus Divergence Score", f"{rob.consensus_divergence_score * 100:.1f}%", help="Ratio of conflicting evidence in corpus")
        r3.metric("Critical Pivot Papers", len(rob.critical_papers), help="Papers whose omission flips claim consensus")

        st.markdown("### 🚨 Critical Pivot Papers Analysis")
        if rob.critical_papers:
            st.warning("The following papers act as single points of truth or contradiction. Removing them directly reverses claim outcomes:")
            for cp in rob.critical_papers:
                with st.expander(f"📌 [{cp['paper_id']}] {cp['title']} (Impacts {cp['impacted_claims_count']} claims)"):
                    for imp in cp.get("impacts", []):
                        st.markdown(f"- **Claim `{imp['claim_id']}`**: {imp['description']}")
        else:
            st.success("✅ The current synthesis exhibits high consensus stability: no single paper's omission flips claim verification status.")

        st.markdown("### ⏱️ Temporal Drift Across Literature")
        for tf in rob.temporal_drift_findings:
            st.markdown(f"- 📅 {tf}")

        if rob.counterfactual_scenarios:
            st.markdown("### 🔬 Leave-One-Out Counterfactual Simulation Log")
            df_cf = pd.DataFrame(rob.counterfactual_scenarios)
            st.dataframe(df_cf[["omitted_paper_id", "affected_claim_id", "original_status", "counterfactual_status", "impact"]], use_container_width=True)
    else:
        st.info("Robustness analysis not available.")


# ----------------- TAB 7: LIVE ARXIV INGESTION -----------------
with tab_arxiv:
    st.subheader("🌐 Live ArXiv Scientific Retrieval & On-the-Fly Ingestion")
    st.caption("Search recent research from ArXiv and index papers into the active retrieval pipeline with one click.")

    col_arx1, col_arx2 = st.columns([3, 1])
    with col_arx1:
        arxiv_q = st.text_input("Search ArXiv Query or ArXiv ID (e.g. '2405.08912'):", value="spectral vision transformer low light")
    with col_arx2:
        arx_count = st.number_input("Max Papers", min_value=1, max_value=5, value=2)

    col_btn_arx, _ = st.columns([1, 4])
    with col_btn_arx:
        fetch_arx_btn = st.button("📥 Retrieve from ArXiv", type="secondary", use_container_width=True)

    if fetch_arx_btn:
        with st.spinner("Querying ArXiv API and parsing papers..."):
            found_papers = pipeline.fetch_and_index_arxiv(arxiv_q, max_results=arx_count)
            st.success(f"Indexed {len(found_papers)} papers into active corpus! Total papers now: {len(pipeline.indexed_papers)}")
            for fp in found_papers:
                with st.expander(f"📄 [{fp.paper_id}] {fp.title} ({fp.year})"):
                    st.write(f"**Authors**: {', '.join(fp.authors)}")
                    st.write(f"**Abstract**: {fp.abstract}")
                    st.write(f"**Venue**: {fp.venue}")
                    st.json(fp.metadata)


# ----------------- TAB 8: ACADEMIC LATEX & BIBTEX EXPORT -----------------
with tab_latex:
    st.subheader("📄 Publication-Grade Academic Export Studio")
    st.caption("Ready-to-compile LaTeX paper, BibTeX bibliography, standalone HTML report, and markdown.")

    report_html = pipeline.report_generator.format_html(report)
    report_md = pipeline.report_generator.format_markdown(report)

    c_tex_down, c_bib_down, c_html_down, c_md_down = st.columns(4)
    if report.latex_code:
        with c_tex_down:
            st.download_button(
                label="⬇️ Download synthesis.tex",
                data=report.latex_code,
                file_name="synthesis_paper.tex",
                mime="text/x-tex",
                use_container_width=True
            )
    if report.bibtex_code:
        with c_bib_down:
            st.download_button(
                label="⬇️ Download references.bib",
                data=report.bibtex_code,
                file_name="references.bib",
                mime="text/plain",
                use_container_width=True
            )
    with c_html_down:
        st.download_button(
            label="⬇️ Download report.html",
            data=report_html,
            file_name="synthesis_report.html",
            mime="text/html",
            use_container_width=True
        )
    with c_md_down:
        st.download_button(
            label="⬇️ Download report.md",
            data=report_md,
            file_name="synthesis_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with st.expander("🌐 Preview Standalone Publication HTML Report"):
        st.components.v1.html(report_html, height=450, scrolling=True)

    col_tex_view, col_bib_view = st.columns(2)
    with col_tex_view:
        st.markdown("#### LaTeX Article Preview (`.tex`)")
        st.code(report.latex_code or "% No LaTeX generated", language="latex", line_numbers=True)

    with col_bib_view:
        st.markdown("#### BibTeX Bibliography Preview (`.bib`)")
        st.code(report.bibtex_code or "% No BibTeX generated", language="bibtex", line_numbers=True)



# ----------------- TAB 9: CITATION AUDIT -----------------
with tab_audit:
    st.subheader("🛡️ Citation Correctness & Faithfulness Audit")
    st.caption("Auditing whether cited sections genuinely entail the generated propositions.")

    audit = report.citation_audit
    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("Evidence Coverage Score (ECS)", f"{audit.evidence_coverage_score * 100:.1f}%")
    ca2.metric("Citation Integrity Score (CIS)", f"{audit.citation_integrity_score * 100:.1f}%")
    ca3.metric("Citation Faithfulness", f"{audit.citation_faithfulness * 100:.1f}%")

    if audit.audit_details:
        st.markdown("### Detailed Audit Breakdown")
        audit_rows = []
        for d in audit.audit_details:
            audit_rows.append({
                "Claim ID": d["claim_id"],
                "Statement": d["statement"],
                "Verification Status": d["status"],
                "Supporting Evidence Count": d["supporting_count"],
                "Contradicting Count": d["contradicting_count"],
            })
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)


# ----------------- TAB 10: BENCHMARK & EVALUATION STUDIO -----------------
with tab_ablation:
    st.subheader("📈 Systematic Benchmark & Information Retrieval Studio")
    st.caption("Evaluate EvidenceFirst-RAG against standard RAG baselines and measure IR ranking metrics (MRR, nDCG, Precision, Recall).")

    eval_mode = st.radio("Select Benchmark Evaluation View:", ["Ablation Study (Section 20 Architecture Comparison)", "Quantitative IR Metrics (MRR, nDCG@K, Precision, Recall)"], horizontal=True)

    benchmarker = BenchmarkRunner()

    if "Ablation" in eval_mode:
        ablation_data = benchmarker.run_ablation_study(pipeline)
        df_ablation = pd.DataFrame(ablation_data)

        st.dataframe(
            df_ablation[[
                "system", "evidence_verification", "contradiction_detection",
                "citation_accuracy", "evidence_coverage", "hallucination_rate", "conflict_detection_rate"
            ]].style.format({
                "citation_accuracy": "{:.1%}",
                "evidence_coverage": "{:.1%}",
                "hallucination_rate": "{:.1%}",
                "conflict_detection_rate": "{:.1%}",
            }),
            use_container_width=True
        )

        fig_ablation = px.bar(
            df_ablation,
            x="system",
            y=["citation_accuracy", "evidence_coverage", "conflict_detection_rate"],
            barmode="group",
            title="Comparative Evaluation: Baseline RAG vs EvidenceFirst-RAG",
            labels={"value": "Score", "system": "RAG System Architecture", "variable": "Metric"},
            color_discrete_sequence=["#3B82F6", "#10B981", "#EF4444"]
        )
        fig_ablation.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"))
        st.plotly_chart(fig_ablation, use_container_width=True)

    else:
        st.markdown("### 🎯 Formal Information Retrieval Metrics on Ground-Truth Queries")
        st.caption("Evaluating hybrid retrieval quality against gold-standard scientific papers across domains.")

        run_ir_btn = st.button("🚀 Run IR Evaluation Suite", type="secondary")
        if run_ir_btn or "ir_results" in st.session_state:
            if run_ir_btn or "ir_results" not in st.session_state:
                with st.spinner("Computing MRR, nDCG@5, Precision@5, and Recall@5 across benchmark suite..."):
                    dom_filter = domain if domain != "All Domains" else None
                    ir_res = benchmarker.run_ir_benchmark(pipeline, domain=dom_filter, top_k=5)
                    st.session_state["ir_results"] = ir_res

            ir_res = st.session_state["ir_results"]

            ir1, ir2, ir3, ir4, ir5 = st.columns(5)
            ir1.metric("Macro MRR", f"{ir_res['macro_mrr']:.3f}", help="Mean Reciprocal Rank")
            ir2.metric("Macro nDCG@5", f"{ir_res['macro_ndcg_at_k']:.3f}", help="Normalized Discounted Cumulative Gain")
            ir3.metric("Macro Precision@5", f"{ir_res['macro_precision_at_k'] * 100:.1f}%")
            ir4.metric("Macro Recall@5", f"{ir_res['macro_recall_at_k'] * 100:.1f}%")
            ir5.metric("Conflict Det. Rate", f"{ir_res['conflict_detection_rate'] * 100:.1f}%")

            st.markdown("#### Per-Query Detailed IR Metrics Breakdown")
            df_ir = pd.DataFrame(ir_res["per_query_metrics"])
            st.dataframe(df_ir[["query_id", "domain", "query", "mrr", "ndcg_at_k", "precision_at_k", "recall_at_k", "has_known_conflict", "conflict_detected"]], use_container_width=True)

            fig_ir = px.bar(
                df_ir,
                x="query_id",
                y=["mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"],
                barmode="group",
                title="IR Metric Scores per Benchmark Query",
                labels={"value": "Score", "query_id": "Benchmark Query ID", "variable": "IR Metric"}
            )
            fig_ir.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"))
            st.plotly_chart(fig_ir, use_container_width=True)

