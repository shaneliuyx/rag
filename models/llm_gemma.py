from __future__ import annotations
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
from config.settings import settings


@lru_cache(maxsize=1)
def _get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(model_name)


class GemmaGenerator:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.llm_model
        self.tokenizer = _get_tokenizer(self.model_name)
        self.model = _get_model(self.model_name)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Return only the completion after the prompt
        return text[len(prompt):].strip()
