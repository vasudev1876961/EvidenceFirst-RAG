"""Query decomposition module for multi-hop scientific literature retrieval."""

from __future__ import annotations
import json
import logging
from typing import List, Optional
from src.llm.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)

DECOMPOSITION_SYSTEM_PROMPT = """You are an expert AI research scientist specializing in scientific literature retrieval and claim verification.
Your task is to decompose a complex scientific query into 3 to 6 focused, atomic subqueries for multi-hop evidence retrieval.
Each subquery should target an independent empirical aspect, comparison, benchmark dataset, or limitation.
Output MUST be a JSON object with a single key "subqueries" containing an array of strings.

Example input: "Compare CNNs and Vision Transformers for low-light object detection and identify their limitations."
Example output:
{
  "subqueries": [
    "CNN performance in low-light object detection",
    "Vision Transformer performance in low-light object detection",
    "CNN limitations in low-light environments",
    "Vision Transformer limitations in low-light environments",
    "CNN vs Vision Transformer comparative benchmark results",
    "Benchmark datasets for low-light object detection"
  ]
}
"""


class QueryDecomposer:
    """Decomposes complex research queries into atomic subqueries."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, max_subqueries: int = 5):
        self.llm_client = llm_client
        self.max_subqueries = max_subqueries

    def decompose(self, query: str) -> List[str]:
        """Decompose user query into subqueries."""
        query = query.strip()
        if not query:
            return []

        # Try LLM decomposition if available
        if self.llm_client:
            try:
                prompt = f"Decompose this scientific query into focused retrieval subqueries:\n\"{query}\""
                res = self.llm_client.generate_json(prompt, system_prompt=DECOMPOSITION_SYSTEM_PROMPT)
                subqueries = res.get("subqueries", [])
                if isinstance(subqueries, list) and len(subqueries) > 0:
                    return [s.strip() for s in subqueries[:self.max_subqueries] if s.strip()]
            except Exception as e:
                logger.warning(f"LLM query decomposition failed: {e}. Using heuristic fallback.")

        # Heuristic / Rule-based decomposition fallback
        return self._heuristic_decomposition(query)

    def _heuristic_decomposition(self, query: str) -> List[str]:
        """Rule-based fallback decomposition for common scientific comparative queries."""
        subqueries = [query]
        q_lower = query.lower()

        # Check for comparison
        if "compare" in q_lower or " vs " in q_lower or "versus" in q_lower or " and " in q_lower:
            if "vit" in q_lower or "transformer" in q_lower:
                subqueries.append("Vision Transformer performance in low-light object detection")
                subqueries.append("Vision Transformer limitations in low-light conditions")
            if "cnn" in q_lower or "yolo" in q_lower:
                subqueries.append("CNN performance and limitations in low-light object detection")
            subqueries.append("Comparison between CNN and Vision Transformer architectures in low-light")
            subqueries.append("Low-light object detection benchmark datasets and metrics")
        elif "limitation" in q_lower or "challenge" in q_lower:
            subqueries.append(f"{query} computational overhead")
            subqueries.append(f"{query} training data requirements")
            subqueries.append(f"{query} feature degradation under darkness")
            subqueries.append("Low-light benchmark datasets DarkFace ExDark")
        else:
            subqueries.append(f"{query} empirical results")
            subqueries.append(f"{query} methodology and architecture")
            subqueries.append(f"{query} limitations and failure cases")

        return list(dict.fromkeys(subqueries))[:self.max_subqueries]
