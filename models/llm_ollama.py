from __future__ import annotations
from typing import Optional
from ollama import Client


class OllamaGenerator:
    def __init__(self, model: str = "gemma3:270m", host: Optional[str] = None) -> None:
        self.model = model
        self.client = Client(host=host) if host else Client()

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        stream = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens
            },
            stream=False,
        )
        return stream["response"].strip()
