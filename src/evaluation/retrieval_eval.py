"""Retrieval evaluation metrics: Recall@K, Precision@K, MRR, and nDCG."""

from __future__ import annotations
import math
from typing import List, Set, Dict


class RetrievalEvaluator:
    """Evaluates retrieval quality against ground-truth relevant chunk/paper IDs."""

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for rid in retrieved_k if rid in relevant_ids)
        return round(hits / k, 4)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for rid in retrieved_k if rid in relevant_ids)
        return round(hits / len(relevant_ids), 4)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return round(1.0 / rank, 4)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k], 1):
            rel = 1.0 if rid in relevant_ids else 0.0
            dcg += rel / math.log2(i + 1)

        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
        return round(dcg / idcg, 4) if idcg > 0 else 0.0
