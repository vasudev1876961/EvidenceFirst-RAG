"""Dense semantic retrieval module using SentenceTransformers with fallback."""

from __future__ import annotations
import math
import logging
from typing import List, Tuple, Optional
import numpy as np
from src.models import SemanticChunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense vector retriever using embedding models with cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.chunks: List[SemanticChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._model = None
        self._use_fallback = False

        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded SentenceTransformer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using semantic TF-IDF dense fallback.")
            self._use_fallback = True

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Lightweight bag-of-ngrams semantic embedding fallback."""
        vocab: dict = {}
        for t in texts:
            words = t.lower().split()
            for w in words:
                if w not in vocab:
                    vocab[w] = len(vocab)

        matrix = np.zeros((len(texts), max(len(vocab), 1)), dtype=np.float32)
        for i, t in enumerate(texts):
            words = t.lower().split()
            for w in words:
                if w in vocab:
                    matrix[i, vocab[w]] += 1.0

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def index_chunks(self, chunks: List[SemanticChunk]):
        """Encode and index semantic chunks."""
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [f"{c.paper_title}. Section: {c.section}. {c.text}" for c in chunks]

        if not self._use_fallback and self._model is not None:
            try:
                self.embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            except Exception as e:
                logger.warning(f"Dense encoding failed: {e}. Falling back to lightweight encoder.")
                self.embeddings = self._fallback_embed(texts)
        else:
            self.embeddings = self._fallback_embed(texts)

    def search(self, query: str, top_k: int = 25) -> List[Tuple[SemanticChunk, float]]:
        """Dense similarity search."""
        if not self.chunks or self.embeddings is None:
            return []

        if not self._use_fallback and self._model is not None:
            try:
                q_vec = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            except Exception:
                q_vec = self._fallback_embed([query])
        else:
            q_vec = self._fallback_embed([query])

        # Cosine similarity
        scores = np.dot(self.embeddings, q_vec.T).squeeze()
        if np.ndim(scores) == 0:
            scores = np.array([float(scores)])

        # Rank
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score_val = float(scores[idx])
            # Normalize to 0..1
            norm_score = max(0.0, min(1.0, (score_val + 1.0) / 2.0 if score_val < 0 else score_val))
            if norm_score > 0.05:
                results.append((self.chunks[idx], norm_score))

        return results
