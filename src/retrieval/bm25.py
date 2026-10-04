"""BM25 keyword retrieval engine with native zero-dependency fallback."""

from __future__ import annotations
import math
import re
from typing import List, Tuple, Dict
from src.models import SemanticChunk, RetrievalCandidate


class PurePythonBM25:
    """Zero-dependency BM25Okapi implementation for robust execution."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        # Count frequencies
        df: Dict[str, int] = {}
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies: Dict[str, int] = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] = df.get(word, 0) + 1

        # Calculate IDFs
        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Compute BM25 scores for all documents given query tokens."""
        scores = [0.0] * self.corpus_size
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(word, 0)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1)
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl)) if self.avgdl > 0 else 1
                scores[idx] += idf_val * (num / denom)
        return scores


class BM25Retriever:
    """BM25 search over semantic chunks."""

    def __init__(self, chunks: List[SemanticChunk] = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[SemanticChunk] = []
        self.bm25_model = None
        self._use_native = False

        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation-cleaned tokenization."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def index_chunks(self, chunks: List[SemanticChunk]):
        """Index a list of SemanticChunk objects."""
        self.chunks = chunks
        if not chunks:
            return

        corpus = [self._tokenize(c.text + " " + c.section + " " + (c.dataset or "") + " " + (c.model_architecture or "")) for c in chunks]

        try:
            from rank_bm25 import BM25Okapi
            self.bm25_model = BM25Okapi(corpus, k1=self.k1, b=self.b)
            self._use_native = False
        except ImportError:
            self.bm25_model = PurePythonBM25(corpus, k1=self.k1, b=self.b)
            self._use_native = True

    def search(self, query: str, top_k: int = 25) -> List[Tuple[SemanticChunk, float]]:
        """Search chunks and return top_k with scores."""
        if not self.chunks or self.bm25_model is None:
            return []

        tokens = self._tokenize(query)
        raw_scores = self.bm25_model.get_scores(tokens)

        # Normalize scores to [0, 1] range
        if len(raw_scores) > 0:
            max_s = float(max(raw_scores))
            max_score = max_s if max_s > 0 else 1.0
            normalized_scores = [float(s) / max_score for s in raw_scores]
        else:
            normalized_scores = []

        # Rank
        ranked_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            if normalized_scores[idx] > 0.01:
                results.append((self.chunks[idx], float(normalized_scores[idx])))

        return results
