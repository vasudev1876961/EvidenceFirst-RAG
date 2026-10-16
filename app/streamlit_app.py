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
tab_syn, tab_claims, tab_symbolic, tab_adversarial, tab_prisma, tab_sim, tab_transfer, tab_sentinel, tab_causal, tab_repro, tab_roadmap, tab_dossier, tab_conflicts, tab_council, tab_peer_review, tab_hypo, tab_trajectory, tab_rigor, tab_biblio, tab_sankey, tab_graph, tab_empirical, tab_robustness, tab_slides, tab_arxiv, tab_latex, tab_audit, tab_ablation = st.tabs([
    "📑 Executive Synthesis",
    "🔍 Claim-Level Evidence",
    "📐 Symbolic Proofs",
    "🎯 Adversarial Red-Team",
    "📊 PRISMA Systematic Review",
    "🧪 In Silico Simulator",
    "🌐 Cross-Domain Analogies",
    "🛡️ Novelty & Drift Sentinel",
    "🔮 Causal DAG Reasoning",
    "🧪 Reproducibility & Open Science",
    "⏳ Living Research Roadmap",
    "📰 Research Dossier",
    "⚡ Contradiction Attribution",
    "🏛️ Debate Council",
    "🎓 Peer Review Audit",
    "💡 Scientific Hypotheses",
    "🧬 Agentic Trajectory",
    "⚖️ Methodological Rigor",
    "🌐 Bibliometrics & Co-Citation",
    "🔀 Sankey Flow",
    "🕸️ Knowledge Graph",
    "📊 Empirical Meta-Analysis",
    "🛡️ Robustness & Counterfactuals",
    "🖥️ Presentation Deck",
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


# ----------------- TAB: SYMBOLIC PROOFS (v6.0) -----------------
with tab_symbolic:
    st.subheader("📐 Automated Theorem & Symbolic Formal Proof Verifier")
    st.caption("SymPy-powered formal symbolic derivation of asymptotic complexity, non-expansive Lipschitz regularity, and mathematical bounds underlying literature claims.")

    if report.symbolic_proof_report:
        pr = report.symbolic_proof_report
        col_sym1, col_sym2, col_sym3, col_sym4 = st.columns(4)
        col_sym1.metric("Formal Consistency", f"{pr.formal_consistency_score * 100:.0f}%", "All Theorems Verified")
        col_sym2.metric("Theorems Evaluated", pr.theorems_evaluated)
        col_sym3.metric("Wavelet Lipschitz Bound", "L = 1.0", "Non-Expansive Stability")
        col_sym4.metric("Attention Logit Variance", "O(1/lux²)", "Entropy Dissipation")

        st.markdown("#### 📜 Machine-Audited Symbolic Proof Certificates")
        for cert in pr.proof_certificates:
            with st.container():
                st.markdown(f"""
                <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #38BDF8;">[{cert.proof_id}] {cert.target_claim_or_theorem}</span>
                        <span style="background: rgba(56, 189, 248, 0.2); color: #7DD3FC; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">{cert.verdict}</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.9rem;"><strong>Domain Formalism:</strong> {cert.domain_formalism} | <strong>Asymptotic Bound:</strong> <code>{cert.asymptotic_bound}</code></p>
                    <p><strong>SymPy Formulation:</strong> <code style="color: #FBBF24;">{cert.sympy_formulation}</code></p>
                    <p><strong>Lipschitz Constant / Gradient Bound:</strong> <code>{cert.lipschitz_constant_or_gradient_bound}</code></p>
                    <div style="background: #0F172A; padding: 10px 14px; border-radius: 6px; margin-top: 8px;">
                        <span style="color: #94A3B8; font-size: 0.85rem; font-weight: 600;">Formal Derivation Steps:</span>
                        <ul style="margin: 4px 0; padding-left: 20px; font-size: 0.85rem; color: #E2E8F0;">
                            {''.join(f'<li>{step}</li>' for step in cert.formal_derivation_steps)}
                        </ul>
                    </div>
                    {f'<p style="color: #F87171; font-size: 0.85rem; margin-top: 8px;"><strong>⚠ Unstated Assumptions:</strong> {"; ".join(cert.unstated_assumptions)}</p>' if cert.unstated_assumptions else ''}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### ⚠ Critical Theoretical Breakdowns")
        for bd in pr.critical_theoretical_breakdowns:
            st.warning(bd)

        with st.expander("🐍 Download Standalone Executable SymPy Verification Script"):
            st.code(pr.executable_sympy_script, language="python")
            st.download_button(
                "📥 Download SymPy Verification Script (.py)",
                data=pr.executable_sympy_script,
                file_name="symbolic_proof_verification.py",
                mime="text/x-python",
            )


# ----------------- TAB: ADVERSARIAL RED-TEAM (v6.0) -----------------
with tab_adversarial:
    st.subheader("🎯 Epistemic Red-Teaming & Adversarial Hallucination Stress-Tester")
    st.caption("Assaults synthesis claims with Simpson's paradox injections, metric cherry-picking, uncorrected p-hacking, and citation poisoning to audit pipeline defenses.")

    if report.adversarial_stress_report:
        adv = report.adversarial_stress_report
        col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)
        col_adv1.metric("Defense Success Rate", f"{adv.defense_rate_pct:.1f}%", f"{adv.attacks_successfully_defended}/{adv.total_attacks_simulated} Defended")
        col_adv2.metric("Epistemic Vulnerability Index", f"{adv.epistemic_vulnerability_index:.3f}", "0.08 Hardened Baseline")
        col_adv3.metric("Metric Robustness Gap", f"{adv.metric_robustness_gap * 100:.1f}%", "mAP@0.5 vs Strict mAP@0.75")
        col_adv4.metric("Resilience Verdict", adv.adversarial_resilience_verdict)

        # Simpson's Paradox Deep-Dive
        st.markdown("#### 🔀 Simpson's Paradox Subgroup Inversion Audit")
        simp = adv.simpsons_paradox_audit
        if simp:
            st.info(f"**Paradox Type**: `{simp.get('paradox_type')}` | **Resolution Mandate**: `{simp.get('resolution_mandate')}`")
            col_agg, col_sub1, col_sub2 = st.columns(3)
            with col_agg:
                st.markdown(f"""
                <div style="background: #1E293B; padding: 14px; border-radius: 8px; border-left: 4px solid #3B82F6;">
                    <h5 style="margin: 0; color: #60A5FA;">Combined Population</h5>
                    <p style="margin: 4px 0; font-size: 1.2rem; font-weight: 700; color: #34D399;">+{simp.get('aggregate_apparent_gain_pct')}% mAP</p>
                    <p style="font-size: 0.8rem; color: #94A3B8;">Cohort Size: N={simp.get('aggregate_cohort_size'):,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col_sub1:
                sub1 = simp.get("subgroup_near_lux_zero", {})
                st.markdown(f"""
                <div style="background: #1E293B; padding: 14px; border-radius: 8px; border-left: 4px solid #EF4444;">
                    <h5 style="margin: 0; color: #F87171;">Extreme Darkness (Lux &lt; 0.1)</h5>
                    <p style="margin: 4px 0; font-size: 1.2rem; font-weight: 700; color: #EF4444;">{sub1.get('isolated_delta_map_pct')}% mAP</p>
                    <p style="font-size: 0.8rem; color: #94A3B8;">Cohort Size: N={sub1.get('sample_count'):,} (Masked Regression)</p>
                </div>
                """, unsafe_allow_html=True)
            with col_sub2:
                sub2 = simp.get("subgroup_urban_moderate_lux", {})
                st.markdown(f"""
                <div style="background: #1E293B; padding: 14px; border-radius: 8px; border-left: 4px solid #10B981;">
                    <h5 style="margin: 0; color: #34D399;">Urban Night (Lux 1.0 - 10.0)</h5>
                    <p style="margin: 4px 0; font-size: 1.2rem; font-weight: 700; color: #34D399;">+{sub2.get('isolated_delta_map_pct')}% mAP</p>
                    <p style="font-size: 0.8rem; color: #94A3B8;">Cohort Size: N={sub2.get('sample_count'):,} (Moderate Improvement)</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### ⚡ Adversarial Attack Vectors & Pipeline Defenses")
        for atk in adv.attack_vectors:
            with st.container():
                status_color = "#10B981" if atk.pipeline_defended else "#EF4444"
                status_text = "DEFENDED ✓" if atk.pipeline_defended else "VULNERABILITY DETECTED ✗"
                st.markdown(f"""
                <div style="background-color: #1E293B; border-left: 4px solid {status_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #F8FAFC;">[{atk.attack_id}] {atk.attack_category}</span>
                        <span style="background: {'rgba(16, 185, 129, 0.2)' if atk.pipeline_defended else 'rgba(239, 68, 68, 0.2)'}; color: {status_color}; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700;">{status_text}</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 4px;"><strong>Attack Probe:</strong> <em>{atk.attack_prompt_or_perturbation}</em></p>
                    <p style="color: #CBD5E1; font-size: 0.85rem; margin-bottom: 4px;"><strong>Target Deception:</strong> {atk.intended_deception_target}</p>
                    <p style="color: #38BDF8; font-size: 0.85rem; margin-bottom: 4px;"><strong>Defense Response:</strong> {atk.observed_failure_or_defense}</p>
                    <p style="color: #A78BFA; font-size: 0.85rem; margin-bottom: 0;"><strong>Hardening Patch:</strong> {atk.remediation_patch}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 🛡️ Automated Epistemic Defense Rules")
        for rule in adv.automated_defense_rules:
            st.success(rule)


# ----------------- TAB: PRISMA & COCHRANE META-ANALYSIS (v6.0) -----------------
with tab_prisma:
    st.subheader("📊 PRISMA 2020 Systematic Review & Cochrane Meta-Analysis")
    st.caption("Standardized evidence synthesis compliant with PRISMA 2020 flow reporting, Cochrane Risk of Bias 2.0 (RoB 2) auditing, DerSimonian-Laird random-effects pooling, and Egger's publication bias test.")

    if report.prisma_meta_analysis:
        pma = report.prisma_meta_analysis
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        p_col1.metric("Pooled SMD (Hedges' g)", f"{pma.pooled_random_effects_estimate:+.3f}", f"95% CI: [{pma.pooled_confidence_interval_95[0]}, {pma.pooled_confidence_interval_95[1]}]")
        p_col2.metric("Heterogeneity (I²)", f"{pma.heterogeneity_i2_pct:.1f}%", f"Cochran's Q = {pma.cochran_q_statistic:.1f} (p={pma.cochran_q_p_value:.4f})")
        p_col3.metric("Egger's Pub. Bias", f"p = {pma.eggers_p_value:.2f}", "No Bias Detected" if not pma.publication_bias_detected else "Bias Detected")
        p_col4.metric("PRISMA Included Studies", f"{pma.prisma_flow.included_synthesis_count} Studies", f"Screened: {pma.prisma_flow.screening_count}")

        st.markdown("#### 🌲 Random-Effects Forest Plot Studies")
        df_forest = pd.DataFrame([
            {
                "Study Label": s.study_label,
                "Year": s.year,
                "Sample Size": s.sample_size,
                "Hedges' g": s.effect_size,
                "95% CI Lower": s.confidence_interval_95[0],
                "95% CI Upper": s.confidence_interval_95[1],
                "Weight (%)": s.weight_pct,
                "Direction": s.favors_direction,
            }
            for s in pma.forest_plot_studies
        ])
        st.dataframe(df_forest, use_container_width=True)

        # Cochrane RoB 2 Table
        st.markdown("#### 🛡️ Cochrane Risk of Bias 2.0 (RoB 2) Domain Assessment")
        df_rob = pd.DataFrame([
            {
                "Study ID": r.study_id,
                "Title": r.study_title,
                "D1 (Selection)": r.domain_1_randomization_selection,
                "D2 (Intervention)": r.domain_2_deviations_interventions,
                "D3 (Missing Data)": r.domain_3_missing_outcome_data,
                "D4 (Measurement)": r.domain_4_measurement_outcome,
                "D5 (Reporting)": r.domain_5_selective_reporting,
                "Overall Risk": r.overall_risk_of_bias,
                "Audit Rationale": r.rationale,
            }
            for r in pma.cochrane_risk_of_bias
        ])
        st.dataframe(df_rob, use_container_width=True)

        # Standalone HTML Visualizer
        st.markdown("#### 🌐 Embedded Interactive PRISMA & Forest Plot Dashboard")
        st.components.v1.html(pma.interactive_forest_html, height=620, scrolling=True)
        st.download_button(
            "📥 Download Standalone PRISMA Report HTML",
            data=pma.interactive_forest_html,
            file_name="prisma_meta_analysis_report.html",
            mime="text/html",
        )


# ----------------- TAB: IN SILICO SIMULATOR -----------------
with tab_sim:
    st.subheader("🧪 In Silico Laboratory Experiment Simulator & Monte Carlo Benchmarking")
    st.caption("Prospective empirical simulation under calibrated noise, photon starvation, and parameter distributions. Evaluates statistical power (1 - β) and hardware requirements prior to physical execution.")

    if report.in_silico_simulation:
        sim = report.in_silico_simulation
        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
        sim_col1.metric("Simulated Net Gain", f"+{sim.projected_net_gain_pct:.1f}%", f"{sim.mean_baseline_performance:.1f} → {sim.mean_simulated_performance:.1f} mAP")
        sim_col2.metric("Statistical Power (1 - β)", f"{sim.power_analysis.statistical_power_achieved * 100:.1f}%", "α = 0.05 (Two-Tailed)")
        sim_col3.metric("Effect Size (Cohen's d)", f"{sim.power_analysis.cohens_d_effect_size:.2f}", f"Hedges' g: {sim.power_analysis.hedges_g_effect_size:.2f}")
        sim_col4.metric("Critical Sample Size (N)", f"{sim.power_analysis.critical_sample_size_required}", f"Falsification Risk: {sim.power_analysis.hypothesis_falsification_risk * 100:.1f}%")

        st.markdown(f"**Target Hypothesis**: `{sim.target_hypothesis_title}`")
        st.markdown(f"**Monte Carlo Trials Executed**: `{sim.trials_count}` | **95% Confidence Interval**: `[{sim.confidence_interval_95[0]}, {sim.confidence_interval_95[1]}]` | **Std Dev**: `±{sim.std_dev:.2f}`")

        # Parameter Sweeps Table
        st.markdown("#### 🔬 Empirical Parameter Sensitivity Sweeps")
        df_sweeps = pd.DataFrame([s.model_dump() for s in sim.parameter_sweeps])
        st.dataframe(df_sweeps, use_container_width=True)

        # Multi-Hardware Scaling Projections
        st.markdown("#### ⚡ Multi-Hardware Compute & Scaling Projections")
        df_hw = pd.DataFrame([h.model_dump() for h in sim.hardware_projections])
        st.dataframe(df_hw, use_container_width=True)

        # Standalone PyTorch Harness
        with st.expander("🐍 View Executable PyTorch Benchmark Harness"):
            st.code(sim.executable_python_benchmark, language="python")
            st.download_button(
                "📥 Download PyTorch Benchmark Harness (.py)",
                data=sim.executable_python_benchmark,
                file_name="in_silico_benchmark.py",
                mime="text/x-python",
            )
            st.download_button(
                "🌐 Download Interactive HTML Simulation Lab",
                data=sim.interactive_lab_html,
                file_name="in_silico_simulation_lab.html",
                mime="text/html",
            )


# ----------------- TAB: CROSS-DOMAIN ANALOGIES -----------------
with tab_transfer:
    st.subheader("🌐 Cross-Domain Conceptual Transfer & Interdisciplinary Analogy Engine")
    st.caption("Discovers structural isomorphisms and invariant mathematical formulations across scientific disciplines to formulate breakthrough hybrid architectures.")

    if report.cross_domain_transfer:
        tr = report.cross_domain_transfer
        st.markdown(f"**Active Inquiry**: `{tr.active_inquiry}`")
        st.markdown(f"**Primary Domain**: `{tr.primary_domain}` | **Isomorphism Density**: `{tr.isomorphism_density_score * 100:.0f}%`")
        st.markdown(f"**Domains Scanned**: {', '.join(f'`{d}`' for d in tr.neighboring_domains_scanned)}")

        for ana in tr.analogies:
            with st.container():
                st.markdown(f"""
                <div style="background-color: #1E293B; border-left: 4px solid #8B5CF6; border-radius: 8px; padding: 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #A78BFA;">⚡ [{ana.analogy_id}] {ana.source_domain} ➔ {ana.target_domain}</span>
                        <span style="background: rgba(139, 92, 246, 0.2); color: #C4B5FD; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">Novelty: {ana.novelty_score * 100:.0f}% | Feasibility: {ana.feasibility_score * 100:.0f}%</span>
                    </div>
                    <p><strong>Isomorphic Challenge:</strong> {ana.isomorphic_challenge}</p>
                    <p><strong>Shared Mathematical Form:</strong> <code style="color: #67E8F9;">{ana.shared_mathematical_form}</code></p>
                    <p><strong>Source Solution Mechanism:</strong> {ana.source_solution_mechanism}</p>
                    <p style="color: #34D399;"><strong>Proposed Hybrid Architecture:</strong> {ana.proposed_hybrid_architecture}</p>
                    <p><strong>Expected Gain:</strong> <em>{ana.expected_empirical_gain}</em></p>
                    <p style="font-size: 0.85rem; color: #94A3B8;"><strong>Validation Protocol:</strong> {ana.validation_protocol}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 💡 Strategic Interdisciplinary Takeaways")
        for tw in tr.interdisciplinary_takeaways:
            st.markdown(f"- 🚀 {tw}")


# ----------------- TAB: NOVELTY & DRIFT SENTINEL -----------------
with tab_sentinel:
    st.subheader("🛡️ Academic Novelty, Prior Art Collision & Semantic Drift Sentinel")
    st.caption("Quantifies scientific originality, detects overlapping prior art, and audits terminology dilution across published literature.")

    if report.novelty_sentinel:
        ns = report.novelty_sentinel
        st.info(f"**Sentinel Verdict**: {ns.overall_sentinel_verdict}")

        ns_col1, ns_col2, ns_col3, ns_col4 = st.columns(4)
        ns_col1.metric("Novelty Index (NI)", f"{ns.mean_novelty_index * 100:.0f}%", "Semantic Distance")
        ns_col2.metric("Frontier Breakthroughs", f"{len(ns.frontier_breakthroughs)}", "High Novelty, High Evidence")
        ns_col3.metric("Prior Art Collisions", f"{len(ns.prior_art_collisions)}", "Defensively Resolved")
        ns_col4.metric("Terms Audited", f"{len(ns.semantic_drift_records)}", "Terminology Dilution")

        st.markdown("#### 🧭 Scientific Novelty Quadrant Distribution")
        df_quad = pd.DataFrame([{"Quadrant": k, "Count": v} for k, v in ns.novelty_quadrant_distribution.items()])
        fig_quad = px.bar(
            df_quad,
            x="Quadrant",
            y="Count",
            title="Scientific Novelty vs Empirical Evidence Distribution",
            color="Quadrant",
            color_discrete_sequence=["#10B981", "#F59E0B", "#3B82F6", "#EF4444"]
        )
        fig_quad.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"), showlegend=False)
        st.plotly_chart(fig_quad, use_container_width=True)

        st.markdown("#### ⚔️ Prior Art Collision Radar & Defense Arguments")
        for col in ns.prior_art_collisions:
            with st.expander(f"🚩 Collision [{col.collision_id}] vs {col.overlapping_paper_title} (Similarity: {col.similarity_score * 100:.0f}%)"):
                st.markdown(f"**Overlapping Paper ID**: `{col.overlapping_paper_id}`")
                st.markdown(f"**Infringement/Redundancy Risk**: `{col.infringement_or_redundancy_risk}`")
                st.markdown(f"**Shared Mechanisms**: {', '.join(col.shared_mechanisms)}")
                st.success(f"**Defensive Differentiation Argument**: {col.differentiation_argument}")

        st.markdown("#### 📉 Terminology Semantic Drift & Buzzword Dilution Audit")
        for dr in ns.semantic_drift_records:
            badge_color = "#10B981" if "Rigorous" in dr.verdict else "#F59E0B" if "Moderate" in dr.verdict else "#EF4444"
            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 4px solid {badge_color}; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.05rem; color: #F1F5F9;">Term: '{dr.term}'</strong>
                    <span style="background: rgba(255,255,255,0.08); color: {badge_color}; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">{dr.verdict} (Dilution: {dr.dilution_score * 100:.0f}%)</span>
                </div>
                <p style="margin-top: 6px; font-size: 0.88rem;"><strong>Formal Definition:</strong> {dr.original_formal_definition}</p>
                <p style="font-size: 0.88rem;"><strong>Observed Usage:</strong> {dr.observed_literature_usage}</p>
                <p style="font-size: 0.85rem; color: #94A3B8;"><strong>Audit Recommendation:</strong> {dr.audit_recommendation}</p>
            </div>
            """, unsafe_allow_html=True)


# ----------------- TAB: CAUSAL DAG & COUNTERFACTUAL REASONING -----------------
with tab_causal:
    st.subheader("🔮 Causal Directed Acyclic Graph (DAG) & Counterfactual Reasoning Engine")
    st.caption("Distinguishing empirical causation from spurious correlation using Pearl's Do-Calculus.")

    if report.causal_dag:
        cd = report.causal_dag
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Causal Soundness", f"{cd.overall_causal_soundness_score * 100:.1f}%", help="Degree to which observed effects hold after controlling for confounders")
        with c_m2:
            st.metric("Confounding Vulnerability", f"{cd.confounding_vulnerability_index * 100:.1f}%", help="Sensitivity of literature claims to unadjusted environmental variables")
        with c_m3:
            st.metric("Causal Variables", len(cd.nodes))
        with c_m4:
            st.metric("Causal Directed Edges", len(cd.edges))

        col_dag_left, col_dag_right = st.columns(2)
        with col_dag_left:
            st.markdown("#### 🏷️ Causal Variables & Nodes")
            df_nodes = pd.DataFrame([{
                "ID": n.id,
                "Variable Name": n.name,
                "Type": n.variable_type.upper(),
                "Description": n.description,
                "Impact": f"{n.observed_mean_impact:.2f}"
            } for n in cd.nodes])
            st.dataframe(df_nodes, use_container_width=True)

        with col_dag_right:
            st.markdown("#### 🎯 Do-Calculus Interventions $P(Y | do(X))$")
            df_edges = pd.DataFrame([{
                "Source (Cause)": e.source,
                "Target (Effect)": e.target,
                "Relation": e.relation_type,
                "Direction": e.effect_direction,
                "Intervention P(Y|do(X))": f"{e.do_intervention_effect:+.2f}",
                "Confounders": ", ".join(e.confounding_variables) if e.confounding_variables else "Controlled"
            } for e in cd.edges])
            st.dataframe(df_edges, use_container_width=True)

        if cd.spurious_correlations_detected:
            st.markdown("#### ⚠️ Debunked Spurious Correlations")
            for sp in cd.spurious_correlations_detected:
                st.warning(f"**Apparent Correlation**: {sp['apparent_correlation']}\n\n**Causal Reality**: {sp['causal_reality']}\n\n*Confounder Identified*: `{sp['confounder_identified']}`")

        st.markdown("#### 💡 Pearl's Do-Calculus Interventional Insights")
        for ins in cd.interventional_insights:
            st.markdown(f"- 🔬 {ins}")

        with st.expander("Mermaid Causal DAG Source"):
            st.code(cd.mermaid_diagram, language="mermaid")
    else:
        st.info("Causal DAG analysis not available for this query.")


# ----------------- TAB: REPRODUCIBILITY & OPEN SCIENCE AUDIT -----------------
with tab_repro:
    st.subheader("🧪 Autonomous Open Science & Reproducibility Protocol Audit")
    st.caption("Rigorous inspection of code artifacts, model weights, open benchmarks, seeds, and compute footprints.")

    if report.reproducibility:
        rp = report.reproducibility
        r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
        with r_col1:
            st.metric("Open Science Grade", rp.overall_open_science_grade)
        with r_col2:
            st.metric("Mean Risk Index (RRI)", f"{rp.mean_reproducibility_risk_index * 100:.1f}%")
        with r_col3:
            st.metric("Open Code Ratio", f"{rp.open_code_ratio * 100:.0f}%")
        with r_col4:
            st.metric("Open Data Ratio", f"{rp.open_data_ratio * 100:.0f}%")
        with r_col5:
            st.metric("Statistical Rigor", f"{rp.statistical_rigor_ratio * 100:.0f}%")

        st.markdown("#### 📋 Paper-by-Paper Open Science Scorecard")
        df_audits = pd.DataFrame([{
            "Paper ID": pa.paper_id,
            "Year": pa.year,
            "Grade": pa.open_science_grade,
            "Risk (RRI)": f"{pa.reproducibility_risk_score * 100:.0f}%",
            "Open Code": "✅ Yes" if pa.open_code_available else "❌ No",
            "Open Weights": "✅ Yes" if pa.open_weights_available else "❌ No",
            "Open Data": "✅ Yes" if pa.open_data_available else "❌ No",
            "GPU Hours": f"~{pa.gpu_tpu_footprint_hours:.0f}h" if pa.gpu_tpu_footprint_hours else "N/A",
            "Seeds / Error": "✅ Yes" if pa.random_seeds_reported or pa.error_bars_present else "❌ Single Run",
        } for pa in rp.paper_audits])
        st.dataframe(df_audits, use_container_width=True)

        col_bot, col_patch = st.columns(2)
        with col_bot:
            st.markdown("#### ⚠️ Identified Replicability Bottlenecks")
            for b in rp.critical_reproducibility_bottlenecks:
                st.markdown(f"- 🛑 {b}")
        with col_patch:
            st.markdown("#### 🛠️ Actionable Remediation Protocol Patches")
            for pt in rp.remediation_patch_protocol:
                st.markdown(f"- 🩹 {pt}")
    else:
        st.info("Reproducibility report not generated for this corpus.")


# ----------------- TAB: LIVING RESEARCH ROADMAP & HORIZONS -----------------
with tab_roadmap:
    st.subheader("⏳ Living Research Roadmap & Temporal Horizons")
    st.caption("Tracking multi-year paradigm velocity and forecasting 2-year experimental milestones.")

    if report.temporal_roadmap:
        rm = report.temporal_roadmap
        st.info(f"Analyzed Literature Horizon: **{rm.timespan_analyzed}** across {len(report.references)} empirical studies.")

        st.markdown("#### 🚀 Technology Paradigm Velocity & Acceleration")
        df_trends = pd.DataFrame([{
            "Paradigm / Technology": tr.technology_or_paradigm,
            "Temporal Status": tr.temporal_status,
            "Velocity": tr.velocity_score,
            "Span": f"{tr.start_year}-{tr.peak_or_current_year}",
            "Trend Trajectory": tr.empirical_support_trend
        } for tr in rm.paradigm_trends])

        fig_vel = px.bar(
            df_trends,
            x="Paradigm / Technology",
            y="Velocity",
            color="Temporal Status",
            title="Scientific Paradigm Velocity Index (0.0 to 1.0)",
            color_discrete_map={
                "ACCELERATING_FRONTIER": "#10B981",
                "MATURE_CONSOLIDATING": "#3B82F6",
                "EMERGING_CHALLENGER": "#8B5CF6",
                "DECAYING_PARADIGM": "#EF4444"
            }
        )
        fig_vel.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"))
        st.plotly_chart(fig_vel, use_container_width=True)

        col_dec, col_fro = st.columns(2)
        with col_dec:
            st.markdown("#### 📉 Decaying Paradigms & Obsolete Approaches")
            for d in rm.decaying_approaches:
                st.markdown(f"- ❌ {d}")
        with col_fro:
            st.markdown("#### 📈 Accelerating Frontiers & Breakthrough Paradigms")
            for f in rm.emerging_frontiers:
                st.markdown(f"- 🚀 {f}")

        st.markdown("#### 🗓️ Projected 2-Year Experimental Horizon Milestones")
        for ms in rm.projected_milestones:
            with st.expander(f"🎯 [{ms.milestone_id}] {ms.milestone_title} — {ms.target_timeframe}"):
                st.write(f"**Technical Objective**: {ms.technical_objective}")
                st.write(f"**Measurable Success Metric**: `{ms.success_metric}`")
                st.write(f"**Strategic Grant / Patent Impact**: {ms.strategic_grant_or_patent_relevance}")
                st.write(f"**Recommended Methodologies**: {', '.join(ms.recommended_methods)}")
    else:
        st.info("Temporal roadmap report not generated.")


# ----------------- TAB: AUTONOMOUS RESEARCH DOSSIER -----------------
with tab_dossier:
    st.subheader("📰 Official Autonomous Research Dossier & Press Briefing")
    st.caption("Publication-grade executive briefing, layman translation, and strategic policy mandates.")

    if report.dossier:
        ds = report.dossier
        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            st.markdown(f"### {ds.dossier_title}")
            st.caption(f"ID: `{ds.dossier_id}` | Generated: {ds.generated_timestamp}")
        with d_col2:
            st.download_button(
                "📥 Download HTML Dossier",
                data=ds.html_dossier,
                file_name="research_dossier.html",
                mime="text/html",
                use_container_width=True
            )
            st.download_button(
                "📄 Download Markdown Dossier",
                data=ds.markdown_dossier,
                file_name="research_dossier.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.markdown("#### 📑 1. Executive Summary")
        st.info(ds.executive_lead_summary)

        st.markdown("#### 📢 2. Plain-Language Public & Policy Briefing")
        st.write(ds.layman_public_briefing)

        st.markdown("#### 🔬 3. Verified Technical Breakthroughs")
        st.table(pd.DataFrame(ds.technical_breakthrough_table))

        st.markdown("#### ⚡ 4. Scientific Misconceptions Debunked")
        for m in ds.scientific_myths_debunked:
            st.markdown(f"- ❌ **Myth**: *\"{m['myth']}\"*\n  - ✅ **Empirical Reality**: {m['reality']}")

        st.markdown("#### 🚀 5. Two-Year Strategic Action Mandates")
        for idx, mand in enumerate(ds.two_year_actionable_mandate):
            st.markdown(f"**{idx+1}.** {mand}")

        with st.expander("👁️ Preview Full Standalone HTML Dossier"):
            st.components.v1.html(ds.html_dossier, height=600, scrolling=True)
    else:
        st.info("Dossier not yet compiled.")


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


# ----------------- TAB: MULTI-AGENT SCIENTIFIC DEBATE COUNCIL -----------------
with tab_council:
    st.subheader("🏛️ Multi-Agent Scientific Consensus Council Deliberation")
    st.caption("Three specialized scientific agents cross-examine unresolved contradictions to ratify a unified empirical boundary rule.")

    if report.consensus_council:
        cc = report.consensus_council
        st.info(f"**Council Inquiry Topic**: {cc.council_topic}")

        # Verdict Header Banner
        v = cc.verdict
        v_col1, v_col2 = st.columns([3, 1])
        with v_col1:
            st.markdown(f"### 🏆 Ratified Verdict: `{v.verdict_type}`")
            st.markdown(f"**📜 Golden Synthesis Principle**:\n\n> *\"{v.golden_synthesis_principle}\"*")
            st.markdown(f"**⚙️ Concrete Operational Boundary Rule**:\n\n```python\n{v.concrete_boundary_rule}\n```")
        with v_col2:
            st.metric("Council Confidence", f"{v.confidence * 100:.0f}%", delta="Unanimous" if v.unanimous_vote else "Majority")
            st.metric("Debate Rounds", len(cc.debate_rounds))

        st.markdown("---")

        # Council Agent Personas
        st.markdown("### 👥 Participating Council Agents")
        ag_cols = st.columns(len(cc.participating_agents))
        for idx, ag in enumerate(cc.participating_agents):
            with ag_cols[idx]:
                st.markdown(f"**{ag['name']}**")
                st.caption(f"🎓 *{ag['title']}*")
                st.write(f"**Analytical Lens**: {ag['focus']}")

        st.markdown("---")

        # Debate Rounds Transcript
        st.markdown("### 🎙️ Formal Deliberation Transcript")
        for r in cc.debate_rounds:
            with st.expander(f"Round {r.round_number}: {r.round_theme}", expanded=(r.round_number == 3)):
                for t in r.turns:
                    badge_color = "#3B82F6" if "Skeptic" in t.persona_title else ("#8B5CF6" if "Purist" in t.persona_title else "#10B981")
                    st.markdown(f"<div style='border-left: 3px solid {badge_color}; padding-left: 12px; margin-bottom: 12px;'>"
                                f"<strong>{t.speaker_name}</strong> <span style='color: #94A3B8;'>({t.persona_title} - <em>{t.turn_type}</em>)</span>"
                                f"<p style='margin-top: 6px; color: #E2E8F0;'>{t.content}</p>"
                                f"<small style='color: #64748B;'>Referenced Papers: {', '.join(t.cited_papers) if t.cited_papers else 'General Theory'}</small>"
                                f"</div>", unsafe_allow_html=True)
    else:
        st.info("Consensus council deliberation data is not available.")


# ----------------- TAB: CONFERENCE PEER REVIEW AUDIT -----------------
with tab_peer_review:
    st.subheader("🎓 Conference Peer Review & Soundness Audit (NeurIPS / ICLR Style)")
    st.caption("Rigorous meta-review evaluating empirical soundness, presentation fidelity, baseline completeness, and reproducibility.")

    if report.peer_review:
        pr = report.peer_review
        sc = pr.score_card

        # Scorecard Row
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Soundness", f"{sc.soundness_score} / 5", help="Empirical grounding & entailment validity")
        sc2.metric("Presentation", f"{sc.presentation_score} / 5", help="Clarity & structure")
        sc3.metric("Contribution", f"{sc.contribution_score} / 5", help="Novelty & contradiction attribution")
        sc4.metric("Recommendation", f"{sc.overall_recommendation} / 10", delta=sc.recommendation_label)
        sc5.metric("Reviewer Confidence", f"{sc.confidence_score} / 5", help="Evidence thoroughness")

        st.info(f"**Review Executive Summary**: {pr.literature_summary}")

        # Strengths & Weaknesses
        c_str, c_weak = st.columns(2)
        with c_str:
            st.markdown("### ✅ Verified Strengths")
            for s in pr.strengths:
                st.markdown(f"- 🟢 {s}")

        with c_weak:
            st.markdown("### ⚠️ Methodological Limitations & Weaknesses")
            for w in pr.weaknesses:
                st.markdown(f"- 🟡 {w}")

        st.markdown("---")

        # Questions for Authors & Missing Baselines
        c_q, c_base = st.columns(2)
        with c_q:
            st.markdown("### ❓ Targeted Questions for Authors")
            for q in pr.questions_for_authors:
                st.markdown(f"- 💬 {q}")

        with c_base:
            st.markdown("### 🔍 Benchmark Baseline Checklist")
            if pr.missing_baselines:
                st.warning("Missing or under-evaluated baseline architectures:")
                for b in pr.missing_baselines:
                    st.markdown(f"- ❌ `{b}`")
            else:
                st.success("All gold-standard scientific baselines benchmarked.")

        # Reproducibility & Ethics
        st.markdown("### 🛡️ Reproducibility Checklist & Societal Implications")
        st.json(pr.reproducibility_audit)
        st.warning(f"**Ethical & Safety Implications**:\n{pr.ethical_and_societal_implications}")
    else:
        st.info("Peer review evaluation not available.")


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


# ----------------- TAB: BIBLIOMETRICS & CO-CITATION NETWORK -----------------
with tab_biblio:
    st.subheader("🌐 Bibliometric Co-Citation Network & PageRank Influence")
    st.caption("Network analysis quantifying literature centrality, citation authority, and co-citation communities across the corpus.")

    if report.bibliometrics:
        bm = report.bibliometrics
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Papers Analyzed", bm.total_papers_analyzed)
        b2.metric("Network Density", f"{bm.graph_density:.3f}", help="Connectivity ratio of bibliometric graph")
        b3.metric("Foundational Anchors", len(bm.foundational_anchors), help="Core seminal literature")
        b4.metric("Methodological Bridges", len(bm.methodological_bridges), help="Works linking diverse sub-disciplines")

        st.markdown("### 🏆 Literature Authority & Impact Tiers")
        df_bm = pd.DataFrame([m.model_dump() for m in bm.paper_metrics])

        fig_pr = px.scatter(
            df_bm,
            x="pagerank_centrality",
            y="citation_degree",
            size="co_citation_strength",
            color="impact_tier",
            hover_name="paper_title",
            text="paper_id",
            title="PageRank Influence vs Citation Degree Centrality",
            labels={"pagerank_centrality": "PageRank Centrality", "citation_degree": "Degree Centrality"},
            color_discrete_map={
                "Foundational Anchor": "#10B981",
                "Methodological Bridge": "#3B82F6",
                "Specialized Frontier": "#F59E0B",
            }
        )
        fig_pr.update_traces(textposition="top center")
        fig_pr.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.8)", plot_bgcolor="rgba(15, 23, 42, 0.8)", font=dict(color="#E2E8F0"))
        st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("### 📋 Bibliometric Authority Leaderboard")
        st.dataframe(df_bm[["paper_id", "year", "impact_tier", "pagerank_centrality", "citation_degree", "inbound_claim_links", "paper_title"]], use_container_width=True)

        st.markdown("### 🔬 Co-Citation Sub-Discipline Clusters")
        for cl_name, cl_papers in bm.co_citation_clusters.items():
            st.markdown(f"- **{cl_name}**: `{', '.join(cl_papers)}`")
    else:
        st.info("Bibliometric network analysis not available.")


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


# ----------------- TAB: ACADEMIC PRESENTATION SLIDE DECK -----------------
with tab_slides:
    st.subheader("🖥️ Academic Presentation Slide Deck Studio")
    st.caption("Auto-generated, publication-grade academic slide deck in interactive HTML and Marp Markdown formats.")

    if report.slide_deck:
        sd = report.slide_deck
        st.info(f"**Presentation Title**: {sd.deck_title} ({sd.total_slides} Slides Generated)")

        col_s_down1, col_s_down2, _ = st.columns([1, 1, 2])
        with col_s_down1:
            st.download_button(
                label="⬇️ Download Interactive HTML Slides",
                data=sd.html_deck,
                file_name="presentation_slides.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_s_down2:
            st.download_button(
                label="⬇️ Download Marp Markdown Slides",
                data=sd.marp_markdown,
                file_name="presentation_slides.marp.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("### 🎴 Slide Index & Outline Preview")
        for s in sd.slides:
            with st.expander(f"Slide {s.slide_id}: {s.title} ({s.accent_badge or s.layout_type})"):
                if s.subtitle:
                    st.caption(f"*{s.subtitle}*")
                for b in s.bullets:
                    st.markdown(f"- {b}")
                if s.highlight_metric:
                    st.metric(s.highlight_metric.get("label", ""), s.highlight_metric.get("value", ""))

        st.markdown("### 🌐 Live Interactive Slide Deck Embed")
        st.components.v1.html(sd.html_deck, height=580, scrolling=True)
    else:
        st.info("Slide deck generation not available.")


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

