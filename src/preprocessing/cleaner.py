"""Text cleaner for scientific literature."""

from __future__ import annotations
import re


def clean_academic_text(text: str) -> str:
    """Cleans academic paper text from PDF artifacts, hyphens, and noise."""
    if not text:
        return ""

    # Fix de-hyphenation across line breaks (e.g. "con- \nnected" -> "connected")
    cleaned = re.sub(r"(\b\w+)-\s*\n\s*(\w+\b)", r"\1\2", text)

    # Normalize multiple whitespace and newlines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)

    # Remove isolated page numbers or footer lines (e.g. "Page 5", "\n4\n")
    cleaned = re.sub(r"\n\s*\d+\s*\n", "\n", cleaned)

    # Normalize unicode quotation marks and dashes
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    cleaned = cleaned.replace("—", "-").replace("–", "-")

    return cleaned.strip()
