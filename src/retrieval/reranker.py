"""Cross-Encoder Reranker for precision filtering of retrieved evidence."""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional
from src.models import RetrievalCandidate, SemanticChunk

logger = logging.getLogger(__name__)

# Empirical section weights in scientific papers (Results/Experiments contain harder evidence)
SECTION_PRIORITY = {
    "results": 1.25,
    "experiments": 1.20,
    "experimental results": 1.25,
    "experimental setup": 1.10,
    "methodology": 1.15,
    "method": 1.15,
    "discussion": 1.10,
    "conclusion": 1.00,
    "abstract": 1.05,
    "introduction": 0.90,
    "related work": 0.85,
}


class CrossEncoderReranker:
    """Reranks candidate chunks using a neural Cross-Encoder or contextual score adjustment."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._use_fallback = False

        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded CrossEncoder reranker: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder ({e}). Using section-aware heuristic reranker fallback.")
            self._use_fallback = True

    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_n: int = 12) -> List[RetrievalCandidate]:
        """Rerank candidates using CrossEncoder scores combined with section weighting."""
        if not candidates:
            return []

        pairs = [[query, f"[{c.chunk.section}] {c.chunk.text}"] for c in candidates]

        if not self._use_fallback and self._model is not None:
            try:
                raw_scores = self._model.predict(pairs)
                # Sigmoid normalization
                import numpy as np
                norm_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
            except Exception as e:
                logger.warning(f"CrossEncoder prediction error: {e}. Falling back to heuristic reranking.")
                norm_scores = [c.hybrid_score for c in candidates]
        else:
            norm_scores = [c.hybrid_score for c in candidates]

        for i, cand in enumerate(candidates):
            base_score = float(norm_scores[i])
            sec_weight = SECTION_PRIORITY.get(cand.chunk.section.lower(), 1.0)
            cand.rerank_score = min(1.0, base_score * sec_weight)

        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_n]
