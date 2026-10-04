"""Claim normalizer to standardize scientific propositions."""

from __future__ import annotations
import re
from src.models import FactualClaim


class ClaimNormalizer:
    """Standardizes grammatical form and removes rhetorical preamble from claims."""

    REDUNDANT_PREAMBLE = [
        r"^the authors claim that\s*",
        r"^it is demonstrated that\s*",
        r"^our results show that\s*",
        r"^studies suggest that\s*",
        r"^recent research indicates that\s*",
        r"^in this paper, we show that\s*",
    ]

    def normalize(self, claim: FactualClaim) -> FactualClaim:
        """Clean and normalize a claim statement."""
        text = claim.statement.strip()

        for pat in self.REDUNDANT_PREAMBLE:
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

        if text and not text.endswith("."):
            text += "."

        # Capitalize first character
        if text:
            text = text[0].upper() + text[1:]

        claim.statement = text
        return claim
