"""Section parser for scientific papers. Identifies common IMRaD sections and structures."""

from __future__ import annotations
import re
from typing import Dict, List, Tuple


COMMON_SECTIONS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "method",
    "proposed method",
    "architecture",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "references",
]

SECTION_PATTERN = re.compile(
    r"^(?:(?:\d+\.?\s*)?(?:I|II|III|IV|V|VI|VII|VIII|IX|X|\d+)\.?\s*)?"
    r"(abstract|introduction|related work|background|methodology|method|proposed method|"
    r"experiments|experimental results|experimental setup|results|discussion|conclusion|limitations|references)"
    r"(?:\s*[:\n]|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


class SectionParser:
    """Parses full paper text into structured sections."""

    def __init__(self, section_names: List[str] = None):
        self.section_names = section_names or COMMON_SECTIONS

    def parse(self, text: str) -> Dict[str, str]:
        """Split document text into a dictionary of {section_name: section_content}."""
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {}
        current_section = "Introduction"
        sections[current_section] = []

        # Find section markers
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = SECTION_PATTERN.match(stripped)
            if match and len(stripped) < 60:
                header = match.group(1).title()
                current_section = header
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line)

        # Merge lines into clean text blocks
        result: Dict[str, str] = {}
        for sec, content_lines in sections.items():
            content = "\n".join(content_lines).strip()
            if content:
                result[sec] = content

        return result
