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

### 12. Launch Interactive Research Dashboard
Launch the Streamlit Research Analysis Dashboard with **22 interactive tabs**:
```bash
streamlit run app/streamlit_app.py
```
Tabs include:
1. 📑 Executive Literature Synthesis
2. 🔍 Claim-Level Evidence Inspector
3. 🔮 Causal DAG & Counterfactual Reasoning (v4.0)
4. 🧪 Autonomous Reproducibility & Open Science Audit (v4.0)
5. ⏳ Living Research Roadmap & Horizons (v4.0)
6. 📰 Autonomous Research Dossier & Press Briefing (v4.0)
7. ⚡ Contradiction Attribution Engine
8. 🏛️ Multi-Agent Scientific Debate Council
9. 🎓 Conference Peer Review & Soundness Audit
10. 💡 Automated Scientific Hypotheses & Protocols
11. 🧬 Agentic Multi-Hop Verification Trajectory
12. ⚖️ Methodological Rigor & Evidence Hierarchy
13. 🌐 Bibliometrics & Co-Citation Network
14. 🔀 Multi-Stage Literature Sankey Flow
15. 🕸️ Scientific Knowledge Graph (Plotly)
16. 📊 Cross-Paper Empirical Meta-Analysis Matrix
17. 🛡️ Robustness & Counterfactuals (ERI)
18. 🖥️ Academic Presentation Slide Deck
19. 🌐 Live ArXiv Ingestion Engine
20. 📄 Publication LaTeX & BibTeX Exporter
21. 🛡️ Citation Faithfulness & Reliability Audit
22. 📈 Architecture Ablation & IR Benchmarks


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
│   │   ├── causal_graph.py  # Causal DAG & Pearl's Do-Calculus reasoning engine (v4.0)
│   │   ├── claim_graph.py   # Multi-hop graph with community clustering
│   │   ├── graph_visualizer.py # Sankey & Vis.js network visualizers
│   │   └── citation_network.py # Bibliometric co-citation & PageRank analyzer
│   ├── verification/        # NLI entailment, contradiction engine, citation audit
│   │   ├── reproducibility.py # Open Science & Reproducibility Protocol Verifier (v4.0)
│   │   ├── consensus_council.py # Multi-agent scientific debate council
│   │   ├── agentic_verifier.py # Iterative multi-hop counter-query verifier
│   │   ├── rigor_analyzer.py# Methodological rigor & evidence hierarchy
│   │   ├── robustness.py    # Counterfactual & leave-one-out robustness (ERI)
│   │   ├── contradiction.py # Contradiction attribution engine
│   │   └── citation_checker.py # ECS & CIS audit
│   ├── synthesis/           # Evidence-grounded synthesizer, conflict attribution
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
│   ├── api.py               # FastAPI REST API server v0.4.0 (Swagger /docs)
│   └── streamlit_app.py     # 22-Tab Interactive Research Dashboard
│
├── tests/                   # Pytest test suite (35 passing tests)
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
└── main.py                  # CLI runner with v4.0 flags & exporters
```


---

## 📄 License
This project is licensed under the MIT License.

