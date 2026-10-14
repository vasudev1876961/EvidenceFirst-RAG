"""EvidenceFirst-RAG: Claim-Level Evidence Verification and Contradiction-Aware RAG for Scientific Literature."""

__version__ = "0.5.0"

from src.pipeline import EvidenceFirstPipeline
from src.models import (
    Paper,
    SemanticChunk,
    FactualClaim,
    EvidenceItem,
    ConflictAttribution,
    CitationAudit,
    SynthesisReport,
    CausalDAGReport,
    CorpusReproducibilityReport,
    TemporalRoadmapReport,
    AutonomousResearchDossier,
    InSilicoExperimentReport,
    InterdisciplinaryTransferReport,
    NoveltySentinelReport,
)

__all__ = [
    "EvidenceFirstPipeline",
    "Paper",
    "SemanticChunk",
    "FactualClaim",
    "EvidenceItem",
    "ConflictAttribution",
    "CitationAudit",
    "SynthesisReport",
    "CausalDAGReport",
    "CorpusReproducibilityReport",
    "TemporalRoadmapReport",
    "AutonomousResearchDossier",
    "InSilicoExperimentReport",
    "InterdisciplinaryTransferReport",
    "NoveltySentinelReport",
]
