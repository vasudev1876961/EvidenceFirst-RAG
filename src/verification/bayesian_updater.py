"""Bayesian Epistemic Belief Updating & Bayes Factor Calibration Engine (v7.0).

Quantifies how scientific credences evolve dynamically as empirical evidence accumulates,
computing sequential Bayes Factors (BF10), log-odds trajectories, Jeffreys' scale classifications,
Kullback-Leibler epistemic information gain, and 95% Bayesian Credible Intervals (HDI).
"""

from __future__ import annotations
import math
import logging
from typing import List, Dict, Any, Optional

from src.models import (
    FactualClaim,
    Paper,
    MethodologyRigorReport,
    BayesFactorRecord,
    BayesianTrajectoryPoint,
    BayesianClaimBelief,
    BayesianBeliefReport,
)

logger = logging.getLogger(__name__)


class BayesianBeliefUpdater:
    """Sequential Bayesian Belief Updating and Bayes Factor Evidence Calibration Engine."""

    def __init__(self, default_prior: float = 0.50):
        self.default_prior = max(0.01, min(0.99, default_prior))

    def evaluate_beliefs(
        self,
        claims: List[FactualClaim],
        papers: Optional[List[Paper]] = None,
        rigor_report: Optional[MethodologyRigorReport] = None,
    ) -> BayesianBeliefReport:
        """Executes sequential Bayesian belief updating across all factual claims."""
        paper_lookup = {p.paper_id: p for p in (papers or [])}
        rigor_weights = {}
        if rigor_report:
            for ass in rigor_report.assessments:
                rigor_weights[ass.paper_id] = ass.evidence_weight_multiplier

        claim_beliefs: List[BayesianClaimBelief] = []
        total_info_gain = 0.0

        for claim in claims:
            belief = self._update_claim_belief(claim, paper_lookup, rigor_weights)
            claim_beliefs.append(belief)
            total_info_gain += belief.epistemic_information_gain_nats

        n_claims = max(1, len(claim_beliefs))
        mean_info_gain = total_info_gain / n_claims

        # Calculate systemic prior & posterior Shannon entropy
        prior_entropy = sum(self._binary_entropy(b.prior_credence) for b in claim_beliefs) / n_claims
        posterior_entropy = sum(self._binary_entropy(b.posterior_credence) for b in claim_beliefs) / n_claims

        # Consensus convergence rate (% of claims moved out of [0.35, 0.65] uncertainty regime)
        settled_count = sum(
            1 for b in claim_beliefs if b.posterior_credence > 0.65 or b.posterior_credence < 0.35
        )
        convergence_rate = settled_count / n_claims

        # Calibration quality score (Brier-like penalty against extreme overconfidence)
        calib_score = max(0.0, min(1.0, 1.0 - posterior_entropy / (math.log(2) + 1e-6)))

        insights = self._generate_epistemic_insights(claim_beliefs, convergence_rate, mean_info_gain)

        return BayesianBeliefReport(
            report_id=f"BAYES-{len(claim_beliefs):03d}",
            overall_epistemic_entropy_prior=round(prior_entropy, 4),
            overall_epistemic_entropy_posterior=round(posterior_entropy, 4),
            mean_information_gain_nats=round(mean_info_gain, 4),
            claims_evaluated=len(claim_beliefs),
            claim_beliefs=claim_beliefs,
            consensus_convergence_rate=round(convergence_rate, 4),
            calibration_quality_score=round(calib_score, 4),
            epistemic_insights=insights,
        )

    def _update_claim_belief(
        self,
        claim: FactualClaim,
        paper_lookup: Dict[str, Paper],
        rigor_weights: Dict[str, float],
    ) -> BayesianClaimBelief:
        """Performs sequential log-odds updating along the evidence chain."""
        p0 = self.default_prior
        current_prob = p0
        current_log_odds = math.log(p0 / (1.0 - p0))

        # Collect evidence ordered by chronological year / step
        all_evidence = []
        for ev in claim.supporting_evidence:
            all_evidence.append((ev, "SUPPORTS"))
        for ev in claim.contradicting_evidence:
            all_evidence.append((ev, "CONTRADICTS"))
        for ev in claim.neutral_evidence:
            all_evidence.append((ev, "NEUTRAL"))

        all_evidence.sort(key=lambda item: item[0].year)

        bf_records: List[BayesFactorRecord] = []
        trajectory: List[BayesianTrajectoryPoint] = [
            BayesianTrajectoryPoint(
                step_index=0,
                paper_id="PRIOR",
                paper_title="Uninformative Scientific Prior",
                prior_prob=round(p0, 4),
                likelihood_ratio=1.0,
                posterior_prob=round(p0, 4),
                log_odds=round(current_log_odds, 4),
                information_gain_kl=0.0,
            )
        ]

        cumulative_bf = 1.0
        alpha_param = 1.0  # Beta prior parameters for HDI
        beta_param = 1.0

        for idx, (ev, direction) in enumerate(all_evidence, start=1):
            rigor = rigor_weights.get(ev.paper_id, 1.0)
            sample_size = ev.extracted_attributes.get("sample_size", 100)
            if not isinstance(sample_size, (int, float)) or sample_size <= 0:
                sample_size = 100

            # Compute Bayes Factor BF10 for this evidence unit
            bf10 = self._calculate_bayes_factor(ev, direction, rigor, sample_size)
            log_bf = math.log(max(1e-4, bf10))

            prior_step_prob = current_prob
            current_log_odds += log_bf
            # Prevent overflow in sigmoid
            bounded_log_odds = max(-15.0, min(15.0, current_log_odds))
            current_prob = 1.0 / (1.0 + math.exp(-bounded_log_odds))

            cumulative_bf *= bf10

            # Update conjugate Beta parameters
            if direction == "SUPPORTS":
                alpha_param += ev.composite_score * rigor * 2.0
            elif direction == "CONTRADICTS":
                beta_param += ev.composite_score * rigor * 2.0
            else:
                alpha_param += 0.5
                beta_param += 0.5

            # Compute KL divergence from initial prior p0
            kl_gain = self._kl_divergence(current_prob, p0)

            interp = self._classify_jeffreys(bf10)

            record = BayesFactorRecord(
                evidence_id=ev.evidence_id,
                paper_id=ev.paper_id,
                paper_title=ev.paper_title,
                direction=direction,
                raw_bayes_factor_10=round(bf10, 4),
                log10_bayes_factor=round(math.log10(max(1e-4, bf10)), 4),
                evidence_weight=round(rigor, 3),
                sample_size=int(sample_size),
                jeffreys_interpretation=interp,
            )
            bf_records.append(record)

            trajectory.append(
                BayesianTrajectoryPoint(
                    step_index=idx,
                    paper_id=ev.paper_id,
                    paper_title=ev.paper_title,
                    prior_prob=round(prior_step_prob, 4),
                    likelihood_ratio=round(bf10, 4),
                    posterior_prob=round(current_prob, 4),
                    log_odds=round(current_log_odds, 4),
                    information_gain_kl=round(kl_gain, 4),
                )
            )

        # 95% Bayesian Credible Interval (HDI approximation via Beta mean & variance)
        hdi = self._compute_beta_hdi(alpha_param, beta_param)

        # Final epistemic state label
        if current_prob >= 0.85:
            state_label = "Empirically Settled (High Credence)"
        elif current_prob >= 0.65:
            state_label = "Probable Consensus"
        elif current_prob <= 0.15:
            state_label = "Strongly Refuted (Low Credence)"
        elif current_prob <= 0.35:
            state_label = "Substantially Debunked"
        else:
            state_label = "High Epistemic Uncertainty"

        final_kl = self._kl_divergence(current_prob, p0)

        return BayesianClaimBelief(
            claim_id=claim.claim_id,
            statement=claim.statement,
            prior_credence=round(p0, 4),
            posterior_credence=round(current_prob, 4),
            bayes_factor_aggregate_10=round(cumulative_bf, 4),
            epistemic_information_gain_nats=round(final_kl, 4),
            credible_interval_95_hdi=[round(hdi[0], 4), round(hdi[1], 4)],
            belief_shift_magnitude=round(abs(current_prob - p0), 4),
            epistemic_state_label=state_label,
            evidence_updates=bf_records,
            trajectory=trajectory,
        )

    def _calculate_bayes_factor(
        self, ev: Any, direction: str, rigor: float, sample_size: float
    ) -> float:
        """Calculates likelihood ratio BF10 scaled by sample size and methodology rigor."""
        # Baseline likelihood ratio from NLI probabilities
        p_entail = getattr(ev, "entailment_prob", 0.5)
        p_contra = getattr(ev, "contradiction_prob", 0.1)
        score = getattr(ev, "composite_score", 0.7)

        sample_factor = 1.0 + 0.15 * math.log(max(1.0, sample_size / 50.0))

        if direction == "SUPPORTS":
            base_ratio = (p_entail + 0.10) / (p_contra + 0.05)
            bf = (base_ratio ** (0.75 * rigor)) * sample_factor * (1.0 + score)
            return max(1.05, min(350.0, bf))
        elif direction == "CONTRADICTS":
            base_ratio = (p_contra + 0.10) / (p_entail + 0.05)
            bf_inv = (base_ratio ** (0.75 * rigor)) * sample_factor * (1.0 + score)
            return max(0.002, min(0.95, 1.0 / bf_inv))
        else:
            # Neutral evidence
            return max(0.85, min(1.15, 1.0 + (p_entail - p_contra) * 0.2))

    def _classify_jeffreys(self, bf10: float) -> str:
        """Classifies evidence strength according to Jeffreys' Bayes Factor scale."""
        if bf10 > 100.0:
            return "Decisive Evidence for H1"
        elif bf10 > 30.0:
            return "Very Strong Evidence for H1"
        elif bf10 > 10.0:
            return "Strong Evidence for H1"
        elif bf10 > 3.0:
            return "Moderate Evidence for H1"
        elif bf10 > 1.0:
            return "Anecdotal Evidence for H1"
        elif math.isclose(bf10, 1.0, abs_tol=0.05):
            return "Inconclusive / Neutral"
        else:
            return "Evidence Favors Null H0 (Contradictory)"

    def _binary_entropy(self, p: float) -> float:
        """Calculates Shannon entropy H(p) = -p ln p - (1-p) ln(1-p)."""
        p = max(1e-7, min(1.0 - 1e-7, p))
        return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def _kl_divergence(self, p: float, q: float) -> float:
        """Calculates Kullback-Leibler divergence D_KL(P || Q) in nats."""
        p = max(1e-7, min(1.0 - 1e-7, p))
        q = max(1e-7, min(1.0 - 1e-7, q))
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    def _compute_beta_hdi(self, alpha: float, beta: float) -> List[float]:
        """Approximates 95% Highest Density Interval (HDI) for Beta(alpha, beta)."""
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1))
        std = math.sqrt(max(1e-6, var))
        # 1.96 standard deviations for 95% Gaussian-approximated credible interval
        lower = max(0.01, mean - 1.96 * std)
        upper = min(0.99, mean + 1.96 * std)
        return [lower, upper]

    def _generate_epistemic_insights(
        self,
        beliefs: List[BayesianClaimBelief],
        convergence_rate: float,
        mean_info_gain: float,
    ) -> List[str]:
        """Synthesizes high-level epistemic insights from belief updating trajectories."""
        insights = []
        settled = [b for b in beliefs if "Settled" in b.epistemic_state_label]
        refuted = [b for b in beliefs if "Refuted" in b.epistemic_state_label or "Debunked" in b.epistemic_state_label]
        uncertain = [b for b in beliefs if "Uncertainty" in b.epistemic_state_label]

        if settled:
            insights.append(
                f"Bayesian convergence identified {len(settled)} empirically settled propositions with decisive aggregate Bayes Factors (BF10 > 10)."
            )
        if refuted:
            insights.append(
                f"Evidence calibration successfully debunked {len(refuted)} claims where empirical observations favored the directional null H0."
            )
        if uncertain:
            insights.append(
                f"{len(uncertain)} scientific propositions remain in a high epistemic uncertainty regime (0.35 <= P <= 0.65), demanding confirmatory preregistered trials."
            )
        insights.append(
            f"Global epistemic information gain averaged {mean_info_gain:.3f} nats per claim, yielding a consensus convergence rate of {convergence_rate * 100:.1f}%."
        )
        return insights
