from __future__ import annotations
from typing import Protocol, Optional


class LLMGeneratorProtocol(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:  # pragma: no cover - protocol
        ...


def get_llm_generator(settings) -> LLMGeneratorProtocol:
    """
    Factory that returns an LLM generator based on settings.
    Backwards compatible with existing use_ollama flags.
    """
    provider = getattr(settings, "llm_provider", "").strip().lower()
    if provider == "bedrock":
        from models.llm_bedrock import BedrockGenerator
        model_id = getattr(settings, "bedrock_model_id", None) or getattr(settings, "llm_model", None)
        region = getattr(settings, "bedrock_region", None)
        profile = getattr(settings, "aws_profile", None)
        return BedrockGenerator(model_id=model_id, region=region, profile=profile)
    # fallback to existing behavior
    if getattr(settings, "use_ollama", False):
        from models.llm_ollama import OllamaGenerator
        return OllamaGenerator(model=getattr(settings, "ollama_model", "gemma3:270m"))
    from models.llm_gemma import GemmaGenerator
    return GemmaGenerator(model_name=getattr(settings, "llm_model", None))


