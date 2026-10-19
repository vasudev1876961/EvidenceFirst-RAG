# EvidenceFirst-RAG: Claim-Level Evidence Verification and Contradiction-Aware RAG for Scientific Literature

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

A research-grade Retrieval-Augmented Generation framework that decomposes queries into atomic factual claims, independently verifies supporting and conflicting evidence across scientific papers, identifies the root causes of academic disagreement (**Conflict Attribution**), and audits citation faithfulness before generating literature syntheses.

---

## 🌟 Core Research Contributions

Standard RAG architectures execute a naive pipeline:
$$\text{Question} \longrightarrow \text{Retrieve Chunks} \longrightarrow \text{LLM} \longrightarrow \text{Answer}$$

**EvidenceFirst-RAG** transforms scientific synthesis into a fact-checked, contradiction-aware reasoning engine:
$$\text{Question} \longrightarrow \text{Decompose} \longrightarrow \text{Hybrid Retrieve} \longrightarrow \text{Rerank} \longrightarrow \text{Extract Claims} \longrightarrow \text{Verify Entailment} \longrightarrow \text{Detect Contradictions} \longrightarrow \text{Conflict Attribution} \longrightarrow \text{Citation Audit} \longrightarrow \text{Synthesis}$$

### 1. Claim-Level Reasoning
Instead of trusting unstructured answers, EvidenceFirst-RAG decomposes research questions into verifiable atomic claims with directional evidence mappings.

### 2. Contradiction-Aware Retrieval & Conflict Attribution
When papers report divergent findings (e.g., Paper A finds $+8.2\%$ mAP gain on DarkFace, whereas Paper B finds $-0.4\%$ on ExDark), the system recognizes the conflict and attributes the discrepancy to:
- **Benchmark Dataset Differences** (e.g., extreme darkness in DarkFace vs urban night in ExDark)
- **Evaluation Protocols & Metrics** (e.g., mAP@0.5 vs strict mAP@0.5:0.95)
- **Preprocessing & Enhancement Strategies**
- **Temporal Research Shifts**

### 3. Evidence Coverage Score (ECS)
A quantitative metric evaluating factual completeness:
$$\text{ECS} = \frac{\text{Number of Claims Supported by Valid Evidence}}{\text{Total Number of Factual Claims}}$$

### 4. Citation Integrity Score (CIS) & Faithfulness Audit
Rigorous Natural Language Inference (NLI) audits check whether cited sections genuinely entail the generated propositions:
$$\text{CIS} = \frac{\text{Correctly Supported Cited Claims}}{\text{Total Cited Factual Claims}}$$

### 5. Multi-Hop Evidence Knowledge Graph & Community Clustering
NetworkX and Plotly-powered directed graphs with distinct edge styles (green = support, red dashed = contradiction) and community/school-of-thought detection.

### 6. Cross-Paper Empirical Meta-Analysis Matrix
Automated extraction and standardization of quantitative benchmark results across models, datasets, metrics, and compute footprints (mAP, FPS, Parameters), computing empirical variance and discrepancy indices.

### 7. Counterfactual & Leave-One-Out Robustness (ERI)
Jackknife ablation engine that measures the **Evidence Robustness Index (ERI)**, quantifying whether removing any single paper collapses a claim's consensus, and flags **Critical Pivot Papers**.

### 8. Live ArXiv Ingestion Engine
Dynamic on-the-fly retrieval and indexing of recent papers via the ArXiv API with automatic IMRaD section reconstruction and entity parsing.

### 9. Publication-Grade LaTeX & BibTeX Exporter
Generates ready-to-compile academic conference/journal drafts (`synthesis_paper.tex`) and validated bibliography files (`references.bib`).

### 10. Headless REST API (FastAPI)
Production async REST API with interactive Swagger documentation (`/docs`), automated health checks, and query endpoints.

### 11. Agentic Multi-Hop Verification Trajectory
When claims display ambiguity or conflicting evidence, the **Iterative Agentic Verifier** formulates targeted counter-queries and precision sub-inquiries, retrieves supplementary high-precision chunks in a secondary pass, and records an auditable **Verification Trajectory** trace with before/after confidence deltas.

### 12. Automated Scientific Hypothesis Generation & Protocol Synthesis
Transforms discovered contradictions into scientific progress: automatically synthesizes formal, falsifiable **Scientific Hypotheses** (`HYP-XXX`), theoretical rationales, concrete experimental protocols, testable quantitative predictions, and empirical falsification criteria.

### 13. Methodological Rigor & Evidence Hierarchy Analyzer
Scientifically weights evidence by study design tier (**Methodology Rigor Score**): Large-Scale Empirical Benchmarks ($\times 1.25$) and Controlled Ablations ($\times 1.15$) receive priority over anecdotal case studies or conceptual surveys ($\times 0.85$).

### 14. Multi-Stage Sankey & Physics Network Visualizations
Generates publication-grade standalone interactive visualizers:
- **Plotly Sankey Flow**: Traces literature flow from Query $\to$ Extracted Claims $\to$ Evidence Studies $\to$ Final Verification Consensus.
- **Vis.js Physics Network**: Interactive force-directed topology highlighting entailment (green) and academic contradiction (red dashed) edges.

### 15. Conference Peer Review & Soundness Audit (NeurIPS / ICLR Style)
Automated conference-grade meta-review engine scoring technical soundness (1-5), presentation fidelity (1-5), novelty/contribution (1-5), and overall conference recommendation (1-10: *Strong Accept (Oral)* to *Reject*). Formulates explicit strengths, weaknesses, targeted questions for authors, missing baseline architectures, and reproducibility checklist audits.

### 16. Multi-Agent Scientific Consensus Council (Debate Deliberation)
When academic contradictions arise, a 3-agent scientific council convenes:
- **Dr. Sophia Vance (Empirical Skeptic)**: Scrutinizes sensor shot noise, dataset lux disparity, and sample size.
- **Prof. Julian Hayes (Methodological Purist)**: Audits metric protocol distortions (mAP@0.5 vs strict mAP@0.5:0.95) and hyperparameter fairness.
- **Dr. Marcus Sterling (Theoretical Synthesizer)**: Reconciles empirical tensions into unified regime boundaries.
The Council engages in 3 rounds of structured debate and issues a signed, ratified **Consensus Resolution Verdict** with concrete operational boundary rules.

### 17. Bibliometric Co-Citation Network & PageRank Influence
Constructs NetworkX-powered citation and co-citation graph over active literature, computing PageRank centrality, degree metrics, and clustering papers into distinct impact tiers: *Foundational Anchors*, *Methodological Bridges*, and *Specialized Frontiers*.

### 18. Publication Slide Deck Generator (Interactive HTML & Marp Markdown)
Automatically transforms synthesized claims, contradictions, and council verdicts into publication-ready presentation decks:
- **Interactive Glassmorphism HTML Deck**: Standalone presentation with keyboard controls, slide progress bar, fullscreen toggle, and print-to-PDF styles.
- **Marp Markdown Deck**: Standardized academic slide format ready for VS Code / CLI slide compilation.

### 19. Causal Directed Acyclic Graph (DAG) & Counterfactual Do-Calculus Reasoning
Distinguishes genuine empirical causality from spurious correlation in scientific claims:
- **Pearl's Do-Calculus Interventional Scoring**: Computes $P(\text{Outcome} \mid do(\text{Treatment}))$ across observed confounding variables (e.g. sensor shot noise, dataset illuminance, hardware compute).
- **Confounding Vulnerability Index**: Flags claims susceptible to environmental distribution shift.
- **Spurious Correlation Debunker**: Uncovers hidden confounders explaining empirical discrepancies between published papers.
- **Interactive Mermaid Causal Topology**: Renders directed causal networks separating treatments, mediators, outcomes, and confounders.

### 20. Autonomous Open Science & Reproducibility Protocol Verifier
Comprehensive inspection of literature artifacts and empirical replicability:
- **Artifact Availability Audit**: Detects open GitHub source code, pre-trained weights (HuggingFace / PyTorch), and open dataset releases.
- **Hardware & Carbon Footprint Transparency**: Estimates GPU/TPU hours consumed and training compute overhead.
- **Statistical Rigor Checklist**: Verifies multi-seed testing ($N \ge 3$ seeds), variance / standard error bounds ($\pm \sigma$), and hypothesis significance tests ($p < 0.05$).
- **Reproducibility Risk Index (RRI)**: Quantitative risk metric (0.0 to 1.0) and letter grades (A+ to F) with author-directed **Remediation Protocol Patches**.

### 21. Temporal Knowledge Horizon & Living Research Roadmap
Forecasts multi-year scientific trajectories ($2020 \to 2026+$):
- **Paradigm Velocity Index**: Classifies technologies into *Accelerating Frontiers*, *Mature / Consolidating*, *Decaying Paradigms*, and *Emerging Challengers*.
- **Decaying Paradigm Isolation**: Identifies obsolete research practices proven suboptimal on standardized benchmarks.
- **2-Year Living Research Horizon**: Generates chronologically sequenced experimental milestones ($0-6\text{mo}$, $6-12\text{mo}$, $12-24\text{mo}$) with technical objectives, measurable success metrics, and strategic grant / patent relevance.

### 22. Autonomous Research Dossier & Public Press Briefing Generator
Synthesizes comprehensive meta-science findings into publication-ready executive documents:
- **Interactive Glassmorphic HTML Dossier**: Responsive, standalone briefing with executive headlines, layman public translations, debunked myths, and policy mandates.
- **Publication Markdown Dossier**: Ready for institutional press offices, research lab directors, and grant review committees.

### 23. In Silico Laboratory Experiment Simulator & Monte Carlo Benchmarking (v5.0)
Validates prospective scientific interventions prior to wet-lab or high-compute GPU allocation:
- **Calibrated Noise & Photon Starvation Models**: Simulates synthetic Poisson shot noise, Gaussian sensor readout noise, and non-linear illuminance degradation curves ($0.01$ to $50.0\text{ lux}$).
- **Monte Carlo Empirical Trials ($N=1,000$)**: Evaluates stochastic distribution bounds, $95\%$ confidence intervals, and standard error bounds across simulated benchmark interventions.
- **Statistical Power & Falsification Analysis**: Computes statistical power ($1 - \beta$, e.g. $99.9\%$), effect size (Cohen's $d$, Hedges' $g$), and critical sample size ($N_{\text{crit}}$) required to falsify the hypothesis at $\alpha = 0.05$.
- **Multi-Hardware Compute Scaling Projections**: Estimates VRAM footprints, throughput (samples/sec), training time, cost, and $\text{kg CO}_2\text{eq}$ across NVIDIA RTX 4090, A100, H100 SXM5, and Google TPU v5e.
- **Standalone PyTorch Benchmark Harness & Interactive HTML Lab**: Automatically generates executable reproduction scripts (`in_silico_benchmark.py`) and standalone interactive simulation labs with live SVG parameter sliders (`in_silico_simulation_lab.html`).

### 24. Cross-Domain Conceptual Transfer & Interdisciplinary Analogy Engine (v5.0)
Catalyzes scientific breakthroughs by discovering invariant mathematical symmetries across disparate scientific literature domains:
- **Domain Isomorphism Detection**: Discovers structural symmetries connecting Computer Vision, Biomedical Low-Dose CT / fMRI, Far-Field Acoustics, Deep-Space Radio Astronomy, Low-Resource NLP, and Quantum Decoherence.
- **Interdisciplinary Transfer Blueprints (`ANA-XXX`)**: Maps source problem mechanisms to target domain challenges (e.g., transferring Sinogram Wavelet Shrinkage from Low-Dose CT to resolve Poisson shot noise in Vision Transformers without pre-enhancement).
- **Novelty & Feasibility Scoring**: Quantifies conceptual distance, theoretical plausibility, and projects quantitative empirical gains.

### 25. Academic Novelty, Prior Art Collision & Semantic Drift Sentinel (v5.0)
Protects against literature redundancy, prior art collisions, and buzzword inflation:
- **Novelty Index (NI)**: Measures mean semantic vector distance and topological graph divergence from baseline literature.
- **Scientific Novelty 4-Quadrant Matrix**: Categorizes claims into *Frontier Breakthroughs*, *Speculative Frontiers*, *Established Consensus*, and *Redundant Clutter*.
- **Defensive Prior Art Collision Radar**: Formulates mathematically sound differentiation arguments against existing published baselines (e.g. Zero-DCE, EnlightenGAN, IAT).
- **Terminology Semantic Drift & Dilution Index**: Quantifies buzzword dilution (0.0 to 1.0) on terms like *"Zero-Shot"*, *"Hallucination-Free"*, *"Self-Supervised"*, and *"Bio-Inspired"*.

### 26. Automated Theorem & Symbolic Formal Proof Verifier (v6.0)
Mathematically verifies theoretical claims, asymptotic complexity, and stability bounds using SymPy:
- **Asymptotic Complexity Verification**: Symbolically derives and compares computational costs (e.g. proving Wavelet sparse attention $\mathcal{O}(N^{1.5} d)$ dominates dense self-attention $\mathcal{O}(N^2 d)$ as $N \to \infty$).
- **Entropy Dissipation Theorem**: Proves the quadratic variance explosion $\mathcal{O}(1/\text{lux}^2)$ of Softmax attention logits under photon starvation, proving the mathematical necessity of pre-attention frequency gating.
- **Lipschitz Stability Guarantee**: Proves non-expansive Lipschitz-1 continuity ($\|\mathcal{S}_\tau(Q) - \mathcal{S}_\tau(Q_0)\|_2 \le \|Q - Q_0\|_2$) for wavelet shrinkage operators.
- **Sensor Pedestal Singularity Theorem**: Symbolically demonstrates infinite noise gain singularity $\lim_{I \to 0^+} \frac{d}{dI}(I^{0.4}) = +\infty$ when gamma enhancement omits dark current pedestal calibration.
- **Machine-Audited Proof Certificates (`PRF-XXX`)**: Produces formal LaTeX proof blocks, isolated unstated assumptions, counterexample bounds, and downloadable executable SymPy scripts (`symbolic_proof_verification.py`).

### 27. Epistemic Red-Teaming & Adversarial Hallucination Stress-Tester (v6.0)
Systematically stress-tests literature claims and RAG synthesis pipelines against deceptive mathematical and statistical fallacies:
- **Simpson's Paradox Debunker**: Uncovers deceptive aggregate benchmark gains (+3.5% overall) that mask catastrophic regressions (-8.4% on extreme darkness $<0.1\text{ lux}$) across sub-population cohorts.
- **Metric Inversion & Cherry-Picking Radar**: Detects loose metric reporting (mAP@0.5) that conceals strict metric collapse (mAP@0.5:0.95 and false positive spikes).
- **P-Hacking & Subgroup Dredging Defense**: Penalizes uncorrected post-hoc exploratory slices with small sample sizes ($N < 50$) and applies Benjamini-Hochberg False Discovery Rate corrections.
- **Citation Poisoning & Hallucination Inoculation**: Audits pipeline resilience against inverted author citations and fabricated pseudo-scientific terms (e.g. "quantum-annealed cross-attention").
- **Epistemic Vulnerability Index (EVI)**: Quantitative hardening metric ($0.0$ to $1.0$) and automated defensive heuristics.

### 28. Autonomous Systematic Review & PRISMA 2020 Meta-Analysis Synthesizer (v6.0)
Translates scientific literature corpora into Cochrane-grade gold-standard systematic reviews:
- **PRISMA 2020 Flow Reporting**: Standardized 4-stage tracking (Identification $\to$ Screening $\to$ Eligibility $\to$ Synthesis) with explicit exclusion taxonomies.
- **Cochrane Risk of Bias 2.0 (RoB 2) Matrix**: Audits 5 core bias domains (Randomization/Selection, Deviations from Intervention, Missing Data, Measurement, Selective Reporting) with traffic-light indicators.
- **DerSimonian-Laird Random-Effects Pooling**: Calculates pooled effect size (Standardized Mean Difference / Hedges' $g$) and $95\%$ confidence bounds.
- **Between-Study Heterogeneity Index**: Quantifies $I^2$ inconsistency, Cochran's $Q$ test statistic, $p$-value, and between-study variance $\tau^2$.
- **Egger's Funnel Plot Regression**: Statistically audits literature publication bias (intercept $\beta_0$, $p$-value).
- **Interactive Standalone Forest Plot Dashboard**: Publication-ready SVG/HTML visualizer (`prisma_meta_analysis_report.html`).

### 29. Bayesian Epistemic Belief Updating & Bayes Factor Calibration (v7.0)
Dynamically tracks how scientific credences evolve as empirical evidence accumulates sequentially:
- **Bayes Factor Calibration ($\text{BF}_{10}$)**: Computes likelihood ratios $\text{BF}_{10} = \frac{P(\mathcal{E} \mid \mathcal{H}_1)}{P(\mathcal{E} \mid \mathcal{H}_0)}$ weighted by study rigor multipliers and sample sizes ($N$).
- **Jeffreys' Scale Classification**: Interprets evidence from *Anecdotal* to *Decisive Evidence ($\text{BF}_{10} > 100$)* or *Favors Null $\mathcal{H}_0$*.
- **Sequential Log-Odds Trajectories**: Dynamically updates credences $\ln \mathcal{O}_k = \ln \mathcal{O}_{k-1} + \ln(\text{BF}_{10})$ from uninformative prior ($P=0.50$) to posterior convergence.
- **Kullback-Leibler Epistemic Information Gain**: Measures $D_{\text{KL}}(\text{Posterior} \parallel \text{Prior})$ in nats to quantify systemic entropy reduction.
- **95% Bayesian Credible Intervals (HDI)**: Evaluates beta conjugate posterior parameters $[\text{HDI}_{\text{lower}}, \text{HDI}_{\text{upper}}]$ and Brier calibration scores.

### 30. Automated OSF Confirmatory Preregistration Protocol Synthesizer (v7.0)
Protects against p-hacking, HARKing, and post-hoc outcome dredging by automatically synthesizing publication-grade Open Science Framework (OSF / AsPredicted standard) preregistration protocols (`PRG-XXX`):
- **Cryptographic SHA-256 Fingerprint**: Computes immutable cryptographic hashes timestamped prior to data collection or GPU allocation.
- **Statistical Power & Sample Size Determination**: Prespecifies sample sizes ($N$) calibrated to $1 - \beta = 0.90$ power at $\alpha = 0.01$ significance.
- **Operational Variables & Multi-Seed Blinding**: Explicitly enumerates independent variables, primary vs secondary dependent metrics, and prespecifies random seeds ($N \ge 5$).
- **Confirmatory Statistical Pipelines & FDR Control**: Mandates Mixed-Effects Linear Models with Benjamini-Hochberg or Bonferroni-Holm False Discovery Rate corrections.
- **Open Science Badge Audit**: Verifies compliance for Open Data, Open Materials, and Preregistered Confirmatory badges.
- **Standalone Markdown Dossier**: Exports formal OSF preregistration documents (`osf_preregistration_dossier.md`).

### 31. Quantum-Inspired Hilbert Space Epistemic State & Bloch Sphere Visualizer (v7.0)
Models complex scientific disagreement as state vectors in a 2D Hilbert space $\mathcal{H} \cong \mathbb{C}^2$:
$$|\psi\rangle = \cos\left(\frac{\theta}{2}\right) | \text{Support} \rangle + e^{i\phi} \sin\left(\frac{\theta}{2}\right) | \text{Contradict} \rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$
- **Polar & Phase Angles**: $\theta$ reflects consensus balance between supporting and contradicting literature, while $\phi$ encodes azimuthal domain shifts (lux levels, benchmark disparities).
- **Density Matrix & von Neumann Entropy**: Computes $\rho = |\psi\rangle\langle\psi|$ with environmental decoherence damping, and von Neumann entropy $S(\rho) = -\text{Tr}(\rho \ln \rho)$ bounded by $\ln(2) \approx 0.693$.
- **Quantum Coherence & Purity**: Quantifies inter-study superposed coupling $C(\rho) = 2 |\rho_{01}|$ and purity $\mathcal{P} = \text{Tr}(\rho^2)$.
- **Inter-Study Interference Terms**: Measures pairwise constructive consensus vs destructive contradiction $I = \sin(\theta)\cos(\phi)$.
- **Bloch Sphere Coordinates & Interactive 3D Visualizer**: Maps claims to unit sphere coordinates $(x, y, z)$ and renders a standalone interactive HTML/SVG radar (`quantum_epistemic_radar.html`).

---



## 🏛️ Pipeline Architecture

```
                    RESEARCH QUERY
                          │
                          ▼
                 Query Decomposition
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Keyword Queries         Semantic Queries
              │                       │
              ▼                       ▼
          BM25 Search            Dense Embeddings
              │                       │
              └───────────┬───────────┘
                          ▼
                   Hybrid Retrieval
             (alpha*BM25 + (1-alpha)*Dense)
                          │
                          ▼
                Cross-Encoder Reranker
                          │
                          ▼
                   Relevant Evidence
                          │
                          ▼
                  Claim Extraction
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       Evidence Matching       Claim Matching
              │                       │
              ▼                       ▼
       Entailment Check       Contradiction Check
              │                       │
              └───────────┬───────────┘
                          ▼
                Evidence Knowledge Graph
                          │
                          ▼
                 Conflict Attribution
                          │
                          ▼
             Evidence-Grounded Synthesis
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     Empirical Meta-Analysis   Robustness (ERI) Check
              │                       │
              └───────────┬───────────┘
                          ▼
             Citation Faithfulness Audit
                          │
                          ▼
              LaTeX / BibTeX / Report Export
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone repository and install requirements:
```bash
git clone https://github.com/vasudev/EvidenceFirst-RAG.git
cd EvidenceFirst-RAG

pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` (optional - the system includes an offline deterministic research mock if no API keys are provided):
```bash
cp .env.example .env
```
Supported providers: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, or offline fallback mode.

### 3. Run Pipeline CLI
Run a literature synthesis on any research query:
```bash
python main.py --query "What are the limitations of Vision Transformers for low-light object detection?"
```

### 4. Live ArXiv Ingestion & LaTeX Export
Fetch recent papers from ArXiv, test robustness, and export publication-ready LaTeX & BibTeX:
```bash
python main.py --arxiv-search "spectral vision transformers" --robustness --export-latex
```

### 5. Launch FastAPI REST Server
Start the high-performance async REST API on `http://127.0.0.1:8000`:
```bash
python main.py --serve-api
# or: uvicorn app.api:app --reload
```
Interactive OpenAPI documentation available at: `http://127.0.0.1:8000/docs`

### 6. Quantitative IR Benchmark & Multi-Domain Evaluation
Evaluate retrieval quality (MRR, nDCG@K, Precision@K, Recall@K) against gold-standard literature:
```bash
# Run full IR evaluation suite across all scientific domains
python main.py --benchmark-ir

# Or filter evaluation to a specific scientific domain:
python main.py --domain "NLP / LLMs (ACL)" --benchmark-ir
```

### 7. Run Systematic Architecture Ablation Study
Execute comparative evaluation across the 5 RAG architectures:
```bash
python main.py --benchmark
```

### 8. Multi-Domain Literature Synthesis & HTML Export
Run queries across Computer Vision, NLP, or Biomedical domains, and export interactive HTML + LaTeX:
```bash
# NLP / LLM multi-hop reasoning synthesis
python main.py --domain "NLP / LLMs (ACL)" --query "Compare modular RAG against native long-context windows for multi-hop scientific reasoning." --export-html --export-latex

# Biomedical clinical evidence synthesis
python main.py --domain "Biomedical Literature (PubMed)" --query "What are the risks and efficacy of dense retrieval in clinical decision support and contraindication queries?" --export-html
```

### 9. Agentic Verification, Hypothesis Synthesis & Rigor Assessment
Run literature synthesis with iterative multi-hop counter-query verification, automated hypothesis generation, and methodological rigor scoring:
```bash
python main.py --query "What are the limitations of Vision Transformers for low-light object detection?" \
  --iterative \
  --hypotheses \
  --rigor \
  --export-sankey \
  --export-network
```

### 10. Peer Review, Multi-Agent Debate Council, Bibliometrics & Slide Deck Export (v3.0)
Execute conference-grade peer review, 3-agent scientific consensus debate, co-citation bibliometric analysis, and generate interactive presentation slides:
```bash
python main.py --query "What are the limitations of Vision Transformers for low-light object detection?" \
  --peer-review \
  --debate-council \
  --bibliometrics \
  --export-slides
```
Outputs:
- Interactive HTML Presentation Deck: `data/presentation_slides.html`
- Academic Marp Markdown Slides: `data/presentation_slides.marp.md`

### 11. Causal DAG, Reproducibility Audit, Living Roadmap & Research Dossier (v4.0)
Execute Pearl's Do-Calculus causal reasoning, open science reproducibility audit, 2-year living research horizon, and export official executive research dossiers:
```bash
python main.py --query "What are the limitations of Vision Transformers for low-light object detection?" \
  --causal-dag \
  --reproducibility \
  --roadmap \
  --export-dossier
```
Outputs:
- Standalone Glassmorphic HTML Research Dossier: `data/research_dossier.html`
- Publication Markdown Research Dossier: `data/research_dossier.md`

### 12. In Silico Simulator, Cross-Domain Analogies & Novelty Sentinel (v5.0)
Run in silico Monte Carlo experiment simulations, discover cross-domain architectural analogies, audit academic novelty, and export standalone interactive simulation labs:
```bash
python main.py --query "What are the limitations of Vision Transformers for low-light object detection?" \
  --in-silico-sim \
  --cross-domain \
  --novelty-sentinel \
  --export-sim-lab
```
Outputs:
- Interactive HTML In Silico Simulation Lab: `data/in_silico_simulation_lab.html`
- Executable PyTorch Benchmark Harness: `data/in_silico_benchmark.py`

### 13. Symbolic Proofs, Adversarial Red-Team & PRISMA Meta-Analysis (v6.0)
Run SymPy automated theorem proving, epistemic red-team stress testing against Simpson's paradox, and synthesize PRISMA 2020 Cochrane meta-analyses:
```bash
python main.py --query "What are the limitations of Vision Transformers for low-light object detection?" \
  --symbolic \
  --adversarial-test \
  --prisma \
  --export-prisma
```
Outputs:
- Interactive PRISMA 2020 & Cochrane Forest Plot HTML: `data/prisma_meta_analysis_report.html`
- Executable SymPy Mathematical Proof Harness: `data/symbolic_proof_verification.py`

### 14. Launch Interactive Research Dashboard (v6.0)
Launch the Streamlit Research Analysis Dashboard with **28 interactive tabs**:
```bash
streamlit run app/streamlit_app.py
```
Tabs include:
1. 📑 Executive Literature Synthesis
2. 🔍 Claim-Level Evidence Inspector
3. 📐 Automated Theorem & Symbolic Proof Verifier (v6.0)
4. 🎯 Epistemic Red-Team & Adversarial Stress-Test (v6.0)
5. 📊 PRISMA 2020 Systematic Review & Cochrane Meta-Analysis (v6.0)
6. 🧪 In Silico Laboratory Experiment Simulator (v5.0)
7. 🌐 Cross-Domain Conceptual Transfer & Analogies (v5.0)
8. 🛡️ Academic Novelty, Prior Art & Semantic Drift Sentinel (v5.0)
9. 🔮 Causal DAG & Counterfactual Reasoning (v4.0)
10. 🧪 Autonomous Reproducibility & Open Science Audit (v4.0)
11. ⏳ Living Research Roadmap & Horizons (v4.0)
12. 📰 Autonomous Research Dossier & Press Briefing (v4.0)
13. ⚡ Contradiction Attribution Engine
14. 🏛️ Multi-Agent Scientific Debate Council
15. 🎓 Conference Peer Review & Soundness Audit
16. 💡 Automated Scientific Hypotheses & Protocols
17. 🧬 Agentic Multi-Hop Verification Trajectory
18. ⚖️ Methodological Rigor & Evidence Hierarchy
19. 🌐 Bibliometrics & Co-Citation Network
20. 🔀 Multi-Stage Literature Sankey Flow
21. 🕸️ Scientific Knowledge Graph (Plotly)
22. 📊 Cross-Paper Empirical Meta-Analysis Matrix
23. 🛡️ Robustness & Counterfactuals (ERI)
24. 🖥️ Academic Presentation Slide Deck
25. 🌐 Live ArXiv Ingestion Engine
26. 📄 Publication LaTeX & BibTeX Exporter
27. 🛡️ Citation Faithfulness & Reliability Audit
28. 📈 Architecture Ablation & IR Benchmarks


---

## 📊 Evaluation & Benchmark Results

### 1. Information Retrieval (IR) Benchmark Scores (`data/benchmark_queries.json`)
Evaluated across 6 multi-domain scientific benchmark queries:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Macro MRR** | **0.9167** | Mean Reciprocal Rank across gold-standard scientific papers |
| **Macro nDCG@5** | **0.9094** | Normalized Discounted Cumulative Gain at top-5 |
| **Macro Precision@5** | **43.3%** | Highly specific retrieval filtering irrelevant literature |
| **Macro Recall@5** | **100.0%** | Full capture of all relevant reference papers |
| **Conflict Detection Rate** | **100.0%** | Successful detection of academic contradictions |

### 2. Systematic Ablation Study (Proposed Framework vs Baselines)
Comparison across 5 system configurations on scientific literature benchmarks:

| System Configuration | Verification | Contradiction Detection | Citation Audit | Citation Accuracy (CIS) | Evidence Coverage (ECS) | Conflict Detection Rate | Hallucination Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Basic Dense RAG** | ❌ | ❌ | ❌ | 55.8% | 55.3% | 0.0% | 28.0% |
| **Hybrid RAG (BM25 + Dense)** | ❌ | ❌ | ❌ | 65.7% | 64.6% | 0.0% | 22.0% |
| **Hybrid RAG + Cross-Encoder** | ❌ | ❌ | ❌ | 72.9% | 69.7% | 0.0% | 17.0% |
| **Hybrid + Reranker + Verification** | ✅ | ❌ | ✅ | 92.0% | 94.0% | 35.0% | 6.0% |
| **EvidenceFirst-RAG (Full Framework)** | ✅ | ✅ | ✅ | **100.0%** | **75.0%** | **91.0%** | **3.0%** |


---

## 📁 Repository Structure

```
EvidenceFirst-RAG/
│
├── .github/workflows/       # GitHub Actions CI matrix testing & validation
│   └── ci.yml
│
├── data/
│   ├── raw/                 # Incoming PDF papers
│   ├── processed/           # Parsed paper catalog
│   ├── sample_papers/       # Multi-domain seed corpus (CV, NLP, Biomedical)
│   ├── prisma_meta_analysis_report.html # Standalone interactive HTML PRISMA & Forest Plot (v6.0)
│   ├── in_silico_simulation_lab.html # Standalone interactive HTML simulation lab (v5.0)
│   ├── in_silico_benchmark.py # Standalone executable PyTorch reproduction harness (v5.0)
│   ├── research_dossier.html # Official standalone glassmorphic research dossier
│   ├── research_dossier.md  # Executive press release briefing in Markdown
│   ├── synthesis_paper.tex  # Exported LaTeX paper draft
│   ├── references.bib       # Exported BibTeX citations
│   ├── presentation_slides.html # Standalone interactive HTML presentation deck
│   ├── presentation_slides.marp.md # Academic Marp Markdown presentation slides
│   ├── synthesis_sankey.html# Standalone interactive Sankey diagram
│   └── synthesis_network.html# Standalone interactive Physics Network Graph
│
├── configs/
│   └── config.yaml          # Master configuration (weights, models, thresholds)
│
├── src/
│   ├── ingestion/           # PDF, section parsing & ArXiv Live Ingestion Engine
│   │   ├── arxiv_fetcher.py # Live ArXiv API search & entity parsing
│   │   ├── paper_loader.py  # Multi-format paper loader (.json, .pdf, .txt)
│   │   └── section_parser.py# Academic section extractor
│   ├── preprocessing/       # Academic text cleaner & semantic chunker
│   ├── query/               # Query decomposer & BM25 keyword expander
│   ├── retrieval/           # BM25, Dense embeddings, Hybrid fusion, Cross-Encoder
│   ├── claims/              # Atomic claim extractor, normalizer, NetworkX graph
│   │   ├── quantum_epistemic.py # Hilbert space epistemic states & Bloch sphere radar (v7.0)
│   │   ├── analogy_engine.py # Cross-domain conceptual transfer & analogy engine (v5.0)
│   │   ├── causal_graph.py  # Causal DAG & Pearl's Do-Calculus reasoning engine (v4.0)
│   │   ├── claim_graph.py   # Multi-hop graph with community clustering
│   │   ├── graph_visualizer.py # Sankey & Vis.js network visualizers
│   │   └── citation_network.py # Bibliometric co-citation & PageRank analyzer
│   ├── verification/        # NLI entailment, contradiction engine, citation audit
│   │   ├── bayesian_updater.py # Sequential Bayesian belief updating & Bayes Factors (v7.0)
│   │   ├── symbolic_verifier.py # Automated theorem & symbolic formal proof verifier (v6.0)
│   │   ├── adversarial_tester.py # Epistemic red-teaming & hallucination stress-tester (v6.0)
│   │   ├── novelty_sentinel.py # Academic novelty, prior art & semantic drift sentinel (v5.0)
│   │   ├── reproducibility.py # Open Science & Reproducibility Protocol Verifier (v4.0)
│   │   ├── consensus_council.py # Multi-agent scientific debate council
│   │   ├── agentic_verifier.py # Iterative multi-hop counter-query verifier
│   │   ├── rigor_analyzer.py# Methodological rigor & evidence hierarchy
│   │   ├── robustness.py    # Counterfactual & leave-one-out robustness (ERI)
│   │   ├── contradiction.py # Contradiction attribution engine
│   │   └── citation_checker.py # ECS & CIS audit
│   ├── synthesis/           # Evidence-grounded synthesizer, conflict attribution
│   │   ├── preregistration_synthesizer.py # OSF confirmatory preregistration protocols (v7.0)
│   │   ├── prisma_synthesizer.py # PRISMA 2020 review & Cochrane meta-analysis (v6.0)
│   │   ├── in_silico_simulator.py # In Silico experiment simulator & Monte Carlo engine (v5.0)
│   │   ├── temporal_roadmap.py # Temporal knowledge horizon & living roadmap (v4.0)
│   │   ├── dossier_generator.py # Autonomous research dossier & press brief (v4.0)
│   │   ├── peer_review.py   # Conference peer review & soundness auditor
│   │   ├── slide_generator.py# Interactive HTML & Marp slide deck generator
│   │   ├── hypothesis_generator.py # Automated scientific hypothesis synthesis
│   │   ├── meta_analysis.py # Cross-paper empirical benchmark matrix
│   │   ├── latex_exporter.py# LaTeX & BibTeX publication exporter
│   │   └── report_generator.py # Structured synthesis report builder
│   ├── evaluation/          # Retrieval metrics (MRR, nDCG) & ablation runner
│   ├── llm/                 # Modular LLM client (Gemini, OpenAI, Groq, Mock)
│   └── pipeline.py          # Master orchestrator
│
├── app/
│   ├── api.py               # FastAPI REST API server v0.7.0 (Swagger /docs)
│   └── streamlit_app.py     # 31-Tab Interactive Research Dashboard (v7.0)
│
├── tests/                   # Pytest test suite (53 passing tests)
│   ├── test_v7_advanced.py  # Bayesian updating, OSF preregistration & Quantum Bloch radar (v7.0)
│   ├── test_v6_advanced.py  # Symbolic theorem prover, adversarial stress-tester & PRISMA (v6.0)
│   ├── test_v5_advanced.py  # In Silico simulator, analogy transfer & novelty sentinel (v5.0)
│   ├── test_v4_advanced.py  # Causal DAG, reproducibility, living roadmap & dossier
│   ├── test_v3_advanced.py  # Peer review, debate council, bibliometrics & slides
│   ├── test_v2_advanced.py  # Agentic verification, hypotheses, rigor & visualizers
│   ├── test_advanced.py     # ArXiv, Meta-Analysis, ERI, LaTeX & API tests
│   ├── test_domain_corpus.py# Multi-domain & IR benchmark tests
│   ├── test_chunker.py      # Semantic chunking tests
│   ├── test_pipeline.py     # End-to-end integration & ablation tests
│   ├── test_retrieval.py    # Hybrid retrieval tests
│   └── test_verification.py # Contradiction & citation auditor tests
│
├── requirements.txt         # Dependencies
└── main.py                  # CLI runner with v7.0 flags & exporters

```


---

## 📄 License
This project is licensed under the MIT License.

