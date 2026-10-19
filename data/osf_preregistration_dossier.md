# Formal Open Science Preregistration Dossier (OSF / AsPredicted Standard)

> **Open Science Compliance**: Fully Preregistered Confirmatory Protocol Suite
> **Generated Timestamp**: 2026-10-19 04:45:35 UTC

## 🛡️ Anti-P-Hacking & Rigor Safeguards

- **Mandatory prespecification of primary vs exploratory outcome measures to prevent outcome switching.**: Mandatory prespecification of primary vs exploratory outcome measures to prevent outcome switching.
- **Immutable cryptographic SHA-256 fingerprint timestamped prior to data collection or GPU allocation.**: Immutable cryptographic SHA-256 fingerprint timestamped prior to data collection or GPU allocation.
- **Benjamini-Hochberg False Discovery Rate (FDR) corrections prespecified for all post-hoc subgroup slices.**: Benjamini-Hochberg False Discovery Rate (FDR) corrections prespecified for all post-hoc subgroup slices.
- **Standardized outlier rejection criteria (3-sigma z-score bounds) fixed ex-ante to eliminate selective pruning.**: Standardized outlier rejection criteria (3-sigma z-score bounds) fixed ex-ante to eliminate selective pruning.
- **Blind evaluation protocol with automated multi-seed (N >= 5) verification passes.**: Blind evaluation protocol with automated multi-seed (N >= 5) verification passes.

---

## 📋 Preregistered Confirmatory Protocols

### Protocol PRG-2026-HYP-001: Preregistered Confirmatory Validation of Hypothesis: Frequency-Decoupled Dynamic Attention for Extreme Sensor-Noise Mitigation

- **Study Type**: `Factorial Ablation Experiment`
- **Cryptographic SHA-256 Fingerprint**: `e32e1b86b44b4417fb957d49cba0bfcf931e0bbc57b23b301b7abb5a6d4f5bba`
- **Primary Hypothesis (H1)**: *Under extreme darkness (< 0.1 lux), decoupling high-frequency Poisson-Gaussian sensor noise from low-frequency structural contours prior to self-attention eliminates the 8.6% mAP regression on DarkFace while preserving global relational context on ExDark.*
- **Null Hypothesis (H0)**: *The proposed method does not demonstrate superior performance compared to standard baselines (effect size d <= 0.20, p >= 0.01).*

#### Prespecified Experimental Variables
| Variable Category | Name | Specified Levels / Target |
| :--- | :--- | :--- |
| Independent (IV) | Methodology Variant | Proposed (Implement a dual-branch Wavele), Baseline Architectures |
| Independent (IV) | Evaluation Benchmark | Standardized Multi-Dataset Suite (N=3) |
| Dependent (DV) | Primary Benchmark Score | Statistically significant improvement (p < 0.01) (%) |
| Dependent (DV) | Falsification Threshold | If wavelet-guided token separation yields less tha (Criterion) |

#### Sample Size & Statistical Power
- **Statistical Power**: $1 - \beta = 0.9$
- **Significance Alpha**: $\alpha = 0.01$
- **Minimum Samples ($N$)**: 200
- **Prespecified Seeds**: `[101, 202, 303, 404, 505]`

#### Confirmatory Analysis Pipeline & FDR Control
- 1. Two-way repeated measures ANOVA across benchmark datasets.
- 2. Bonferroni-Holm step-down procedure for family-wise error rate control.
- 3. Bootstrap 95% confidence intervals (B = 2,000 resamples).
- **FDR Correction**: `Bonferroni-Holm Step-Down (alpha=0.01)`

---

### Protocol PRG-2026-HYP-002: Preregistered Confirmatory Validation of Hypothesis: Luminance-Aware Hybrid Preprocessing Gating Mechanism

- **Study Type**: `Factorial Ablation Experiment`
- **Cryptographic SHA-256 Fingerprint**: `a777123f76a34d10aaea6423fded1c862671e03aa3533004cbcebbf06a456b7c`
- **Primary Hypothesis (H1)**: *An adaptive illumination gating network that bypasses neural enhancement filters when ambient lux exceeds 1.0 lux resolves the negative transfer artifact (-0.4% mAP) observed on natural low-light images.*
- **Null Hypothesis (H0)**: *The proposed method does not demonstrate superior performance compared to standard baselines (effect size d <= 0.20, p >= 0.01).*

#### Prespecified Experimental Variables
| Variable Category | Name | Specified Levels / Target |
| :--- | :--- | :--- |
| Independent (IV) | Methodology Variant | Proposed (Train an end-to-end differenti), Baseline Architectures |
| Independent (IV) | Evaluation Benchmark | Standardized Multi-Dataset Suite (N=3) |
| Dependent (DV) | Primary Benchmark Score | Statistically significant improvement (p < 0.01) (%) |
| Dependent (DV) | Falsification Threshold | If a single static enhancement model matches or ex (Criterion) |

#### Sample Size & Statistical Power
- **Statistical Power**: $1 - \beta = 0.9$
- **Significance Alpha**: $\alpha = 0.01$
- **Minimum Samples ($N$)**: 200
- **Prespecified Seeds**: `[101, 202, 303, 404, 505]`

#### Confirmatory Analysis Pipeline & FDR Control
- 1. Two-way repeated measures ANOVA across benchmark datasets.
- 2. Bonferroni-Holm step-down procedure for family-wise error rate control.
- 3. Bootstrap 95% confidence intervals (B = 2,000 resamples).
- **FDR Correction**: `Bonferroni-Holm Step-Down (alpha=0.01)`

---
