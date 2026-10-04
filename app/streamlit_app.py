"""Streamlit Research Analysis Dashboard for EvidenceFirst-RAG."""

from __future__ import annotations
import os
import sys
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
            config = yaml.safe_load(f)

    pipe = EvidenceFirstPipeline(config=config)
    loader = PaperLoader()
    papers = loader.load_all_from_directory("data/sample_papers")
    pipe.index_papers(papers)
    return pipe, papers


pipeline, papers_catalog = get_pipeline()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/microscope.png", width=64)
    st.title("EvidenceFirst-RAG")
    st.caption("Contradiction-Aware Scientific Literature Synthesis")

    st.markdown("---")
    st.subheader("⚙️ Query Configuration")

    domain = st.selectbox(
        "Scientific Domain",
        ["Computer Vision / Deep Learning", "NLP / LLMs (ACL)", "Biomedical Literature (PubMed)"],
        index=0,
    )

    year_range = st.slider("Publication Year Range", min_value=2018, max_value=2026, value=(2020, 2026))
    evidence_threshold = st.slider("Evidence Confidence Threshold", min_value=0.30, max_value=0.95, value=0.60, step=0.05)
    hybrid_alpha = st.slider("Retrieval Balance (α BM25 vs 1-α Dense)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    st.markdown("---")
    st.subheader("📚 Corpus Index")
    st.write(f"Indexed Papers: **{len(papers_catalog)}**")
    for p in papers_catalog:
        st.caption(f"• **[{p.paper_id}]** {p.title[:38]}... ({p.year})")


# ----------------- MAIN PANEL -----------------
st.markdown('<div class="main-header">EvidenceFirst-RAG: Scientific Literature Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Claim-Level Verification, Contradiction Detection, and Citation Faithfulness Auditing</div>', unsafe_allow_html=True)

# Query selection / presets
preset_queries = [
    "What are the limitations of Vision Transformers for low-light object detection?",
    "Compare CNNs and Vision Transformers for low-light object detection and identify their limitations.",
    "Does illumination enhancement preprocessing consistently improve object detection accuracy across dark datasets?",
    "Edge-device computational trade-offs between lightweight YOLO and Deformable DETR.",
    "Custom Query..."
]

selected_preset = st.selectbox("Select Research Query Template or Custom", preset_queries, index=0)

if selected_preset == "Custom Query...":
    user_query = st.text_input("Enter your research question:", value="What are the limitations of Vision Transformers for low-light object detection?")
else:
    user_query = selected_preset

col_btn, _ = st.columns([1, 4])
with col_btn:
    run_btn = st.button("🚀 Analyze Literature", type="primary", use_container_width=True)

# Check session state for report
if "report" not in st.session_state or run_btn:
    with st.spinner("Executing EvidenceFirst-RAG Pipeline (Decomposition → Hybrid Search → Rerank → Claim Verification → Contradiction Engine)..."):
        # Update pipeline alpha
        pipeline.hybrid.alpha = hybrid_alpha
        report = pipeline.run(user_query)
        st.session_state["report"] = report

report = st.session_state["report"]

# Top Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Overall Confidence", f"{report.overall_confidence * 100:.0f}%", delta="Reliable" if report.overall_confidence >= 0.75 else "Moderate")
with m2:
    st.metric("Evidence Coverage (ECS)", f"{report.citation_audit.evidence_coverage_score * 100:.0f}%", help="Claims with verified supporting evidence / Total claims")
with m3:
    st.metric("Citation Integrity (CIS)", f"{report.citation_audit.citation_integrity_score * 100:.0f}%", help="Citations passing entailment check / Total citations")
with m4:
    num_conflicts = len(report.conflicting_findings)
    st.metric("Conflicting Claims", num_conflicts, delta=f"{num_conflicts} Detected", delta_color="inverse" if num_conflicts > 0 else "normal")
with m5:
    st.metric("Evaluated Claims", report.citation_audit.total_claims)

st.markdown("---")

# Navigation Tabs
tab_syn, tab_claims, tab_conflicts, tab_graph, tab_audit, tab_ablation = st.tabs([
    "📑 Executive Synthesis",
    "🔍 Claim-Level Evidence",
    "⚡ Contradiction Attribution",
    "🕸️ Evidence Knowledge Graph",
    "🛡️ Citation Audit",
    "📊 Ablation Study & Benchmarks"
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


# ----------------- TAB 4: EVIDENCE KNOWLEDGE GRAPH -----------------
with tab_graph:
    st.subheader("🕸️ Scientific Evidence Knowledge Graph")
    st.caption("Directed graph illustrating connections between Query, Claims, Supporting/Contradicting Papers, and Datasets.")

    fig = pipeline.graph_builder.to_plotly_figure(report.evidence_graph)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Graph visualization requires plotly.")


# ----------------- TAB 5: CITATION AUDIT -----------------
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


# ----------------- TAB 6: ABLATION BENCHMARKS -----------------
with tab_ablation:
    st.subheader("📊 Systematic Ablation Study (Section 20)")
    st.caption("Evaluating the proposed EvidenceFirst-RAG architecture against traditional RAG baselines.")

    benchmarker = BenchmarkRunner()
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

    # Plotly comparison bar chart
    fig_ablation = px.bar(
        df_ablation,
        x="system",
        y=["citation_accuracy", "evidence_coverage", "conflict_detection_rate"],
        barmode="group",
        title="Comparative Evaluation: Baseline RAG vs EvidenceFirst-RAG",
        labels={"value": "Score", "system": "RAG System Architecture", "variable": "Metric"},
        color_discrete_sequence=["#3B82F6", "#10B981", "#EF4444"]
    )
    st.plotly_chart(fig_ablation, use_container_width=True)
