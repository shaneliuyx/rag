from __future__ import annotations
from typing import List, Dict, Any
import json
from models.llm_ollama import OllamaGenerator
from models.llm_gemma import GemmaGenerator


DECOMPOSE_PROMPT = """
You are a planning assistant. Decompose the user query into 2-6 atomic sub-queries that can be answered from documents.
Return strict JSON array where each item has: id (q1,q2,...), text (string), depends_on (list of ids), type ("lookup"|"synthesis").
Only return JSON. Do not include explanations.

Query:
{query}
"""


class LLMDecomposer:
    def __init__(self) -> None:
        from config.settings import settings
        self.settings = settings
        self.gen = None

    def _generator(self):
        if self.gen is not None:
            return self.gen
        if getattr(self.settings, "use_ollama", False):
            self.gen = OllamaGenerator(model=getattr(self.settings, "ollama_model", "gemma3:270m"))
        else:
            self.gen = GemmaGenerator()
        return self.gen

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        # If generation disabled, fallback will be handled by caller
        if not self.settings.enable_generation:
            return []
        prompt = DECOMPOSE_PROMPT.format(query=query)
        out = self._generator().generate(prompt, max_new_tokens=256, temperature=0.2)
        # Try to extract JSON
        try:
            start = out.find('[')
            end = out.rfind(']')
            if start != -1 and end != -1:
                js = out[start:end+1]
                items = json.loads(js)
                # Validate minimal shape
                cleaned = []
                for i, it in enumerate(items, start=1):
                    cleaned.append({
                        "id": it.get("id", f"q{i}"),
                        "text": it.get("text", "").strip() or query,
                        "depends_on": it.get("depends_on", []),
                        "type": it.get("type", "lookup"),
                    })
                return cleaned
        except Exception:
            pass
        return []
