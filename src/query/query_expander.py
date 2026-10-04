"""Query expander: generates keyword and semantic variations for hybrid retrieval."""

from __future__ import annotations
import re
from typing import List, Dict

# Domain synonym and acronym mappings for Computer Vision
CV_SYNONYMS = {
    "vit": ["vision transformer", "vit", "patch attention"],
    "cnn": ["convolutional neural network", "cnn", "convnet"],
    "low-light": ["low illumination", "low-light", "dark imagery", "nighttime", "underexposed"],
    "detection": ["object detection", "bounding box", "mAP", "AP50"],
    "yolo": ["yolo", "you only look once", "single stage detector"],
    "detr": ["detection transformer", "detr", "deformable detr"],
    "enhancement": ["illumination enhancement", "low light enhancement", "exposure correction", "gamma correction"],
}


class QueryExpander:
    """Expands queries with domain-specific terminology and synonyms."""

    def __init__(self, synonym_dict: Dict[str, List[str]] = None):
        self.synonyms = synonym_dict or CV_SYNONYMS

    def expand_for_bm25(self, query: str) -> List[str]:
        """Generate expanded keyword tokens for BM25 retrieval."""
        tokens = query.lower().split()
        expanded_tokens = list(tokens)

        for key, expansions in self.synonyms.items():
            if re.search(rf"\b{re.escape(key)}\b", query, re.IGNORECASE):
                for exp in expansions:
                    expanded_tokens.extend(exp.split())

        # Deduplicate while preserving order
        return list(dict.fromkeys(expanded_tokens))

    def get_expanded_query_string(self, query: str) -> str:
        """Returns string representation of expanded query."""
        tokens = self.expand_for_bm25(query)
        return " ".join(tokens)
