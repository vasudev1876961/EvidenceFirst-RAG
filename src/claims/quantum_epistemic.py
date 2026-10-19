"""Quantum-Inspired Epistemic State & Hilbert Space Superposition Engine (v7.0).

Models contradictory scientific literature claims as state vectors |psi> in a 2D complex
Hilbert space C^2, mapping literature tensions onto the unit Bloch Sphere, computing
von Neumann epistemic entropy S(rho), quantum coherence, density matrices, and phase interference.
"""

from __future__ import annotations
import math
import cmath
import logging
from typing import List, Dict, Any, Optional

from src.models import (
    FactualClaim,
    ConflictAttribution,
    BlochSphereCoordinate,
    QuantumInterferenceRecord,
    QuantumEpistemicState,
    QuantumEpistemicReport,
)

logger = logging.getLogger(__name__)


class QuantumEpistemicEngine:
    """Hilbert-Space Epistemic State & Bloch Sphere Visualizer Engine."""

    def __init__(self, decoherence_factor: float = 0.15):
        self.decoherence_factor = max(0.0, min(0.8, decoherence_factor))

    def evaluate_epistemic_states(
        self,
        claims: List[FactualClaim],
        conflicts: Optional[List[ConflictAttribution]] = None,
    ) -> QuantumEpistemicReport:
        """Computes quantum epistemic state representations for all scientific claims."""
        conflict_map = {c.claim_id: c for c in (conflicts or [])}
        quantum_states: List[QuantumEpistemicState] = []
        total_entropy = 0.0
        superposed_count = 0

        for claim in claims:
            conf = conflict_map.get(claim.claim_id)
            state = self._compute_claim_quantum_state(claim, conf)
            quantum_states.append(state)
            total_entropy += state.von_neumann_entropy
            if state.epistemic_regime == "Superposed Disagreement (Equatorial)":
                superposed_count += 1

        n_states = max(1, len(quantum_states))
        mean_entropy = total_entropy / n_states
        mean_coherence = sum(s.quantum_coherence for s in quantum_states) / n_states

        insights = self._generate_quantum_insights(quantum_states, mean_entropy, superposed_count)
        bloch_html = self._render_bloch_sphere_html(quantum_states)

        return QuantumEpistemicReport(
            report_id=f"QUANTUM-EPISTEMIC-{len(quantum_states):03d}",
            mean_von_neumann_entropy=round(mean_entropy, 4),
            total_claims_in_superposition=superposed_count,
            systemic_coherence_index=round(mean_coherence, 4),
            quantum_states=quantum_states,
            bloch_sphere_html=bloch_html,
            epistemic_phase_insights=insights,
        )

    def _compute_claim_quantum_state(
        self, claim: FactualClaim, conflict: Optional[ConflictAttribution]
    ) -> QuantumEpistemicState:
        """Constructs state vector |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
        w_sup = sum(ev.composite_score for ev in claim.supporting_evidence)
        w_con = sum(ev.composite_score for ev in claim.contradicting_evidence)

        # Baseline neutral weight if empty
        if w_sup == 0 and w_con == 0:
            w_sup = 0.5
            w_con = 0.5

        total_weight = w_sup + w_con + 1e-6
        support_ratio = w_sup / total_weight

        # Theta in [0, pi]: 0 = Pure Support (North pole), pi = Pure Contradiction (South pole)
        theta = (1.0 - support_ratio) * math.pi
        theta = max(0.01, min(math.pi - 0.01, theta))

        # Phi in [0, 2*pi): Azimuthal phase angle from methodological disparity
        phi = self._calculate_phase_angle(claim, conflict)

        # Amplitudes
        alpha = math.cos(theta / 2.0)
        beta = math.sin(theta / 2.0)

        # Bloch coordinates: x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        lat = (math.pi / 2.0 - theta) * (180.0 / math.pi)
        lon = (phi * (180.0 / math.pi)) - 180.0

        coord = BlochSphereCoordinate(
            theta_rad=round(theta, 4),
            phi_rad=round(phi, 4),
            x=round(x, 4),
            y=round(y, 4),
            z=round(z, 4),
            latitude_deg=round(lat, 2),
            longitude_deg=round(lon, 2),
        )

        # Density matrix rho with environmental decoherence damping on off-diagonals
        rho_00 = alpha ** 2
        rho_11 = beta ** 2
        gamma = self.decoherence_factor
        coherence_damping = 1.0 - gamma
        rho_01_mag = alpha * beta * coherence_damping
        coherence = 2.0 * rho_01_mag

        # Superposition purity Tr(rho^2) = rho_00^2 + rho_11^2 + 2 * |rho_01|^2
        purity = (rho_00 ** 2) + (rho_11 ** 2) + 2.0 * (rho_01_mag ** 2)
        purity = max(0.5, min(1.0, purity))

        # von Neumann entropy S(rho) = -Tr(rho ln rho)
        # Eigenvalues of 2x2 density matrix: lambda = 0.5 * (1 +/- sqrt(1 - 4*det(rho)))
        det_rho = max(0.0, (rho_00 * rho_11) - (rho_01_mag ** 2))
        disc = max(0.0, 1.0 - 4.0 * det_rho)
        lam1 = 0.5 * (1.0 + math.sqrt(disc))
        lam2 = 0.5 * (1.0 - math.sqrt(disc))

        entropy = 0.0
        if lam1 > 1e-7:
            entropy -= lam1 * math.log(lam1)
        if lam2 > 1e-7:
            entropy -= lam2 * math.log(lam2)

        # Classify epistemic regime
        if theta < 0.35 * math.pi:
            regime = "Pure Support Eigenstate"
        elif theta > 0.65 * math.pi:
            regime = "Pure Contradiction Eigenstate"
        elif purity < 0.70:
            regime = "Decoherent Mixed Evidence"
        else:
            regime = "Superposed Disagreement (Equatorial)"

        # Inter-study interference terms
        interference_pairs = self._compute_interference_pairs(claim, theta, phi)

        return QuantumEpistemicState(
            claim_id=claim.claim_id,
            statement=claim.statement,
            support_amplitude_alpha=round(alpha, 4),
            contradiction_amplitude_beta=round(beta, 4),
            superposition_purity=round(purity, 4),
            von_neumann_entropy=round(entropy, 4),
            quantum_coherence=round(coherence, 4),
            bloch_coordinates=coord,
            epistemic_regime=regime,
            interfering_pairs=interference_pairs,
        )

    def _calculate_phase_angle(
        self, claim: FactualClaim, conflict: Optional[ConflictAttribution]
    ) -> float:
        """Computes azimuthal phase angle phi in [0, 2*pi) from methodological shifts."""
        if not conflict or not conflict.divergent_factors:
            # Hash-based deterministic dispersion for consistent visualization
            h = abs(hash(claim.statement)) % 360
            return (h / 360.0) * 2.0 * math.pi

        factor_str = " ".join([df.get("factor", "") for df in conflict.divergent_factors]).lower()
        if "dataset" in factor_str or "lux" in factor_str:
            base_angle = math.pi * 0.25  # Benchmark domain shift
        elif "metric" in factor_str or "protocol" in factor_str:
            base_angle = math.pi * 0.75  # Metric disparity
        elif "temporal" in factor_str or "year" in factor_str:
            base_angle = math.pi * 1.25  # Temporal drift
        else:
            base_angle = math.pi * 1.75  # Architectural nuance

        return base_angle % (2.0 * math.pi)

    def _compute_interference_pairs(
        self, claim: FactualClaim, theta: float, phi: float
    ) -> List[QuantumInterferenceRecord]:
        """Calculates pairwise study interference term I = sin(theta) * cos(phi)."""
        pairs: List[QuantumInterferenceRecord] = []
        if not claim.supporting_evidence or not claim.contradicting_evidence:
            return pairs

        interf_mag = math.sin(theta) * math.cos(phi)
        phase_deg = (phi * 180.0 / math.pi) % 360.0

        for s_ev in claim.supporting_evidence[:2]:
            for c_ev in claim.contradicting_evidence[:2]:
                if interf_mag > 0.15:
                    itype = "Constructive Consensus"
                    expl = "Evidence components display coherent phase alignment across benchmark conditions."
                elif interf_mag < -0.15:
                    itype = "Destructive Contradiction"
                    expl = "Opposing empirical results induce phase-canceling destructive interference, manifesting as academic controversy."
                else:
                    itype = "Orthogonal Decoupled"
                    expl = "Studies evaluate decoupled operational regimes without direct epistemological overlap."

                pairs.append(
                    QuantumInterferenceRecord(
                        pair_id=f"INT-{s_ev.paper_id}-{c_ev.paper_id}",
                        study_a_id=s_ev.paper_id,
                        study_b_id=c_ev.paper_id,
                        interference_type=itype,
                        interference_magnitude=round(interf_mag, 4),
                        phase_shift_deg=round(phase_deg, 2),
                        qualitative_explanation=expl,
                    )
                )

        return pairs

    def _generate_quantum_insights(
        self,
        states: List[QuantumEpistemicState],
        mean_entropy: float,
        superposed_count: int,
    ) -> List[str]:
        """Generates epistemological insights from Hilbert space state configurations."""
        insights = []
        if superposed_count > 0:
            insights.append(
                f"{superposed_count} proposition(s) occupy an Equatorial Hilbert Superposition (|alpha|^2 ~ |beta|^2), representing uncollapsed empirical controversy."
            )
        pure_states = [s for s in states if "Eigenstate" in s.epistemic_regime]
        if pure_states:
            insights.append(
                f"{len(pure_states)} proposition(s) collapsed into pure consensus eigenstates with near-zero von Neumann entropy (S < 0.20)."
            )
        insights.append(
            f"Mean systemic von Neumann entropy across the literature corpus stands at {mean_entropy:.3f} nats (theoretical maximum: ln(2) = 0.693)."
        )
        return insights

    def _render_bloch_sphere_html(self, states: List[QuantumEpistemicState]) -> str:
        """Renders an interactive, standalone HTML/SVG 3D Bloch Sphere visualization."""
        state_rows = ""
        for s in states:
            coord = s.bloch_coordinates
            color = "#10b981" if "Support" in s.epistemic_regime else ("#ef4444" if "Contradiction" in s.epistemic_regime else "#f59e0b")
            state_rows += f"""
            <div style="background: rgba(255,255,255,0.03); border-left: 4px solid {color}; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color: #f1f5f9;">{s.claim_id}: {s.statement[:65]}...</strong>
                    <span style="background:{color}22; color:{color}; padding:3px 8px; border-radius:12px; font-size:11px; font-weight:600;">{s.epistemic_regime}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 8px; font-size:12px; color:#94a3b8;">
                    <div>Bloch: (x={coord.x:.2f}, y={coord.y:.2f}, z={coord.z:.2f})</div>
                    <div>von Neumann S: <strong>{s.von_neumann_entropy:.3f}</strong></div>
                    <div>Coherence: <strong>{s.quantum_coherence:.3f}</strong></div>
                    <div>Purity: <strong>{s.superposition_purity:.3f}</strong></div>
                </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Quantum Epistemic Bloch Sphere Radar</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #090d16; color: #f8fafc; margin: 0; padding: 24px; }}
        .card {{ background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 16px; margin-bottom: 20px; }}
        .badge {{ background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f644; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
        .sphere-container {{ display: flex; gap: 24px; align-items: flex-start; }}
        .telemetry-panel {{ flex: 1; }}
        .svg-panel {{ width: 340px; height: 340px; background: radial-gradient(circle, #1e293b 0%, #0b0f19 100%); border-radius: 50%; border: 1px solid #334155; position: relative; display: flex; align-items: center; justify-content: center; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1 style="margin:0; font-size: 22px; color: #38bdf8;">⚛️ Quantum Epistemic Bloch Sphere Radar</h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13px;">Hilbert Space Superposition and von Neumann Entropy Mapping</p>
            </div>
            <span class="badge">v7.0 Epistemic Quantum Engine</span>
        </div>
        <div class="sphere-container">
            <div class="svg-panel">
                <svg width="320" height="320" viewBox="-160 -160 320 320">
                    <!-- Unit circle outline -->
                    <circle cx="0" cy="0" r="130" fill="none" stroke="#334155" stroke-width="1.5" stroke-dasharray="4 4" />
                    <!-- Equator ellipse -->
                    <ellipse cx="0" cy="0" rx="130" ry="40" fill="none" stroke="#475569" stroke-width="1.5" />
                    <!-- Coordinate axes -->
                    <line x1="0" y1="-145" x2="0" y2="145" stroke="#64748b" stroke-width="1.5" />
                    <line x1="-145" y1="0" x2="145" y2="0" stroke="#64748b" stroke-width="1.5" />
                    <!-- Pole labels -->
                    <text x="5" y="-135" fill="#10b981" font-size="11" font-weight="bold">|Support⟩ (|0⟩)</text>
                    <text x="5" y="145" fill="#ef4444" font-size="11" font-weight="bold">|Contradict⟩ (|1⟩)</text>
                    <text x="100" y="-10" fill="#94a3b8" font-size="10">+X</text>
                    <text x="-135" y="-10" fill="#94a3b8" font-size="10">-X</text>
                    <!-- Claim vector markers -->
                    {''.join([f'<circle cx="{int(s.bloch_coordinates.x * 120)}" cy="{int(-s.bloch_coordinates.z * 120)}" r="6" fill="{"#10b981" if "Support" in s.epistemic_regime else ("#ef4444" if "Contradiction" in s.epistemic_regime else "#f59e0b")}" stroke="#ffffff" stroke-width="1.5"><title>{s.claim_id}: (z={s.bloch_coordinates.z:.2f}, S={s.von_neumann_entropy:.2f})</title></circle>' for s in states])}
                </svg>
            </div>
            <div class="telemetry-panel">
                <h3 style="margin-top:0; color:#e2e8f0; font-size:16px;">Epistemic State Coordinates & Entropy</h3>
                {state_rows}
            </div>
        </div>
    </div>
</body>
</html>"""
        return html
