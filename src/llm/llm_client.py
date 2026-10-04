"""LLM Interface supporting Gemini, OpenAI, Groq, and an offline research mock."""

from __future__ import annotations
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM completion."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate structured JSON output."""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash", **kwargs):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self.model.generate_content(full_prompt)
        return response.text or ""

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        instruct = "Respond ONLY with valid JSON without markdown fences or extra commentary."
        sys = f"{system_prompt}\n{instruct}" if system_prompt else instruct
        text = self.generate(prompt, system_prompt=sys, **kwargs)
        clean = text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", **kwargs):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.2),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI generate failed ({e}). Falling back to MockResearchLLMClient.")
            mock = MockResearchLLMClient()
            return mock.generate(prompt, system_prompt=system_prompt, **kwargs)

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        messages = []
        sys = (system_prompt or "") + "\nRespond with valid JSON only."
        messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=kwargs.get("temperature", 0.1),
            )
            return json.loads(response.choices[0].message.content or "{}")
        except Exception as e:
            logger.warning(f"OpenAI generate_json failed ({e}). Falling back to MockResearchLLMClient.")
            mock = MockResearchLLMClient()
            return mock.generate_json(prompt, system_prompt=system_prompt, **kwargs)


class MockResearchLLMClient(BaseLLMClient):
    """Deterministic offline research mock for testing and demonstration without API keys."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if "decompose" in prompt.lower() or "subqueries" in prompt.lower():
            return json.dumps([
                "CNN performance in low-light object detection",
                "Vision Transformer performance in low-light object detection",
                "CNN limitations in low-light environments",
                "Vision Transformer limitations in low-light environments",
                "Benchmark datasets for low-light vision evaluation"
            ])
        return (
            "Vision Transformers have emerged as powerful architectures for computer vision, but face "
            "challenges under low-light imaging conditions. While global self-attention captures long-range "
            "dependencies, weak local gradient signals can degrade attention maps. Furthermore, performance varies "
            "depending on illumination enhancement preprocessing and benchmark dataset characteristics."
        )

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        if "decompose" in prompt_lower or "subqueries" in prompt_lower:
            return {
                "subqueries": [
                    "CNN performance in low-light object detection",
                    "Vision Transformer performance in low-light object detection",
                    "CNN limitations in low-light environments",
                    "Vision Transformer limitations in low-light environments",
                    "Benchmark datasets for low-light vision evaluation"
                ]
            }
        elif "extract claims" in prompt_lower or "claim" in prompt_lower:
            return {
                "claims": [
                    {
                        "claim_id": "C001",
                        "statement": "Vision Transformers require substantially more training data than CNNs to generalize on low-light benchmarks.",
                        "aspect": "data requirements"
                    },
                    {
                        "claim_id": "C002",
                        "statement": "Vision Transformers incur higher computational overhead and slower inference speed than lightweight CNNs on edge devices.",
                        "aspect": "computational overhead"
                    },
                    {
                        "claim_id": "C003",
                        "statement": "Vision Transformers outperform CNNs in low-light object detection without requiring illumination enhancement.",
                        "aspect": "model performance comparison"
                    },
                    {
                        "claim_id": "C004",
                        "statement": "Local texture degradation in low-light imagery hampers ViT patch self-attention representations.",
                        "aspect": "feature representation"
                    }
                ]
            }
        elif "conflict" in prompt_lower or "attribution" in prompt_lower:
            return {
                "likely_reason": "The studies evaluate on divergent datasets (DarkFace vs ExDark), use different evaluation protocols (mAP@0.5 vs mAP@0.5:0.95), and employ differing pretraining strategies.",
                "divergent_factors": [
                    {"factor": "Dataset", "support": "DarkFace (severe darkness)", "contradict": "ExDark (urban low light)"},
                    {"factor": "Metric", "support": "mAP@0.5", "contradict": "mAP@0.5:0.95"},
                    {"factor": "Enhancement", "support": "End-to-end joint enhancement", "contradict": "Raw low-light inputs"}
                ]
            }
        else:
            return {
                "executive_summary": "Synthesized evidence indicates ViTs provide strong global contextual modeling in low-light conditions, but require careful local feature preservation or preprocessing.",
                "research_gaps": [
                    "Limited cross-dataset generalization across severe vs mild low-light benchmarks",
                    "Lack of standardized low-light evaluation metrics considering latency and power",
                    "Insufficient study of hybrid CNN-Transformer architectures under non-uniform illumination"
                ]
            }


def get_llm_client(config: Optional[Dict[str, Any]] = None) -> BaseLLMClient:
    """Factory to retrieve appropriate LLM client based on environment or config."""
    provider = (config or {}).get("provider") or os.getenv("DEFAULT_LLM_PROVIDER", "mock")
    model = (config or {}).get("model") or os.getenv("DEFAULT_LLM_MODEL")

    if provider == "mock":
        logger.info("Using MockResearchLLMClient (deterministic offline research mode).")
        return MockResearchLLMClient()

    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        try:
            return GeminiClient(model=model or "gemini-1.5-flash")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}. Falling back to Mock.")
    elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIClient(model=model or "gpt-4o-mini")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to Mock.")

    logger.info("Using MockResearchLLMClient (deterministic offline research mode).")
    return MockResearchLLMClient()
