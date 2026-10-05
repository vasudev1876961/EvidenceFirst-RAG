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

### 6. Run Benchmark & Ablation Study
Execute comparative evaluation across the 5 RAG architectures:
```bash
python main.py --benchmark
```

### 7. Launch Interactive Research Dashboard
Launch the Streamlit Research Analysis Dashboard with 10 interactive tabs:
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Systematic Ablation Study Results

Comparison across 5 system configurations on Computer Vision benchmarks:

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
├── data/
│   ├── raw/                 # Incoming PDF papers
│   ├── processed/           # Parsed paper catalog
│   ├── sample_papers/       # Seed CV corpus (DarkFace, ExDark, ViT, YOLO)
│   ├── synthesis_paper.tex  # Exported LaTeX paper draft
│   └── references.bib       # Exported BibTeX citations
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
│   │   └── claim_graph.py   # Multi-hop graph with community clustering
│   ├── verification/        # NLI entailment, contradiction engine, citation audit
│   │   ├── robustness.py    # Counterfactual & leave-one-out robustness (ERI)
│   │   ├── contradiction.py # Contradiction attribution engine
│   │   └── citation_checker.py # ECS & CIS audit
│   ├── synthesis/           # Evidence-grounded synthesizer, conflict attribution
│   │   ├── meta_analysis.py # Cross-paper empirical benchmark matrix
│   │   ├── latex_exporter.py# LaTeX & BibTeX publication exporter
│   │   └── report_generator.py # Structured synthesis report builder
│   ├── evaluation/          # Retrieval metrics (MRR, nDCG) & ablation runner
│   ├── llm/                 # Modular LLM client (Gemini, OpenAI, Groq, Mock)
│   └── pipeline.py          # Master orchestrator
│
├── app/
│   ├── api.py               # FastAPI REST API server
│   └── streamlit_app.py     # 10-Tab Interactive Research Dashboard
│
├── tests/                   # Pytest test suite (13 passing tests)
│   ├── test_advanced.py     # ArXiv, Meta-Analysis, ERI, LaTeX & API tests
│   ├── test_pipeline.py     # End-to-end integration & ablation tests
│   ├── test_retrieval.py    # Hybrid retrieval tests
│   └── test_verification.py # Contradiction & citation auditor tests
│
├── requirements.txt         # Dependencies
└── main.py                  # CLI runner with ArXiv, LaTeX, Robustness & API flags
```

---

## 📄 License
This project is licensed under the MIT License.

