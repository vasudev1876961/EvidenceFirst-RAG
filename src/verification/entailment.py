"""Natural Language Inference (NLI) engine for claim entailment and contradiction checking."""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional
from src.llm.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class NLIEntailmentChecker:
    """NLI verifier to evaluate whether evidence entails, contradicts, or is neutral towards a claim."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        device: str = "cpu",
        llm_client: Optional[BaseLLMClient] = None,
        use_fallback: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.llm_client = llm_client
        self._pipe = None
        self._use_fallback = use_fallback

        if not self._use_fallback and model_name.lower() != "fallback":
            self._init_nli_model()

    def _init_nli_model(self):
        try:
            from transformers import pipeline
            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                framework="pt",
                device=-1 if self.device == "cpu" else 0,
                top_k=None,
            )
            logger.info(f"Loaded NLI pipeline: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load NLI pipeline ({e}). Using semantic verification fallback.")
            self._use_fallback = True

    def check(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Evaluate relationship between premise (evidence text) and hypothesis (claim statement).
        Returns dict with:
        {
            "entailment": float,
            "contradiction": float,
            "neutral": float,
            "prediction": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL"
        }
        """
        if not premise or not hypothesis:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0, "prediction": "NEUTRAL"}

        if not self._use_fallback and self._pipe is not None:
            try:
                # Format for cross-encoder NLI
                text_input = f"{premise} [SEP] {hypothesis}"
                results = self._pipe(text_input)[0]
                label_map = {item["label"].lower(): item["score"] for item in results}

                ent = label_map.get("entailment", 0.0)
                contra = label_map.get("contradiction", 0.0)
                neutral = label_map.get("neutral", 0.0)

                top_label = max(label_map.items(), key=lambda x: x[1])[0].upper()
                return {
                    "entailment": ent,
                    "contradiction": contra,
                    "neutral": neutral,
                    "prediction": top_label,
                }
            except Exception as e:
                logger.warning(f"NLI model inference error: {e}. Falling back.")

        # Heuristic / semantic NLI fallback
        return self._heuristic_check(premise, hypothesis)

    def _heuristic_check(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Heuristic semantic polarity and overlap checker."""
        p_lower = premise.lower()
        h_lower = hypothesis.lower()

        # Negation / contradiction cues
        contradiction_cues = [
            "no significant improvement", "fails to improve", "fail to improve",
            "does not outperform", "do not outperform", "not outperform", "fails to",
            "fail to", "little benefit", "degrades performance", "worse than", "inferior",
            "contrary to previous findings", "no noticeable advantage", "drop in map", "fails", "no improvement"
        ]
        support_cues = [
            "outperforms", "outperform", "improves", "improve", "surpasses", "superior", "achieves higher",
            "significant gain", "boosts map", "demonstrates effectiveness", "state-of-the-art"
        ]

        has_contra = any(c in p_lower for c in contradiction_cues)
        has_support = any(s in p_lower for s in support_cues)

        # Token overlap
        import re
        p_words = set(re.findall(r"\b\w+\b", p_lower))
        h_words = set(re.findall(r"\b\w+\b", h_lower))
        overlap = len(p_words.intersection(h_words)) / max(len(h_words), 1)

        if has_contra and ("outperform" in h_lower or "improve" in h_lower or "superior" in h_lower):
            return {"entailment": 0.10, "contradiction": 0.85, "neutral": 0.05, "prediction": "CONTRADICTION"}
        elif has_support and overlap > 0.30:
            return {"entailment": 0.88, "contradiction": 0.04, "neutral": 0.08, "prediction": "ENTAILMENT"}
        elif overlap > 0.40:
            return {"entailment": 0.65, "contradiction": 0.10, "neutral": 0.25, "prediction": "ENTAILMENT"}
        else:
            return {"entailment": 0.20, "contradiction": 0.15, "neutral": 0.65, "prediction": "NEUTRAL"}
