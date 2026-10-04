"""Hybrid retrieval engine combining BM25 keyword matching and dense vector similarity."""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Optional
from src.models import SemanticChunk, RetrievalCandidate
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 and Dense embeddings using weighted fusion:
    Score_hybrid = alpha * Score_BM25 + (1 - alpha) * Score_Dense
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        alpha: float = 0.5,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha

    def retrieve(self, query: str, top_candidates: int = 35) -> List[RetrievalCandidate]:
        """Perform hybrid retrieval for a single query."""
        bm25_results = self.bm25.search(query, top_k=top_candidates)
        dense_results = self.dense.search(query, top_k=top_candidates)

        candidate_map: Dict[str, RetrievalCandidate] = {}

        # Process BM25 candidates
        for chunk, score in bm25_results:
            candidate_map[chunk.chunk_id] = RetrievalCandidate(
                chunk=chunk,
                bm25_score=score,
                dense_score=0.0,
                hybrid_score=self.alpha * score,
            )

        # Process Dense candidates
        for chunk, score in dense_results:
            cid = chunk.chunk_id
            if cid in candidate_map:
                candidate = candidate_map[cid]
                candidate.dense_score = score
                candidate.hybrid_score = (self.alpha * candidate.bm25_score) + ((1.0 - self.alpha) * score)
            else:
                candidate_map[cid] = RetrievalCandidate(
                    chunk=chunk,
                    bm25_score=0.0,
                    dense_score=score,
                    hybrid_score=(1.0 - self.alpha) * score,
                )

        ranked = sorted(candidate_map.values(), key=lambda c: c.hybrid_score, reverse=True)
        return ranked[:top_candidates]

    def retrieve_multi(self, subqueries: List[str], top_candidates_per_query: int = 20) -> List[RetrievalCandidate]:
        """Multi-hop retrieval across decomposed subqueries."""
        aggregated_candidates: Dict[str, RetrievalCandidate] = {}

        for sq in subqueries:
            results = self.retrieve(sq, top_candidates=top_candidates_per_query)
            for cand in results:
                cid = cand.chunk.chunk_id
                if cid not in aggregated_candidates:
                    aggregated_candidates[cid] = cand
                else:
                    # Update with best hybrid score seen across subqueries
                    if cand.hybrid_score > aggregated_candidates[cid].hybrid_score:
                        aggregated_candidates[cid] = cand

        ranked = sorted(aggregated_candidates.values(), key=lambda c: c.hybrid_score, reverse=True)
        return ranked
