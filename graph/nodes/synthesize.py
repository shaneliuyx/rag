from __future__ import annotations
from typing import List, Dict, Any
import re
from models.llm_gemma import GemmaGenerator
from models.llm_ollama import OllamaGenerator


ANSWER_PROMPT = """You are a helpful assistant. Using only the provided context, answer the user question.
Rules:
- Write 3-5 bullet points.
- Each bullet must end with a citation in the form [#{{i}}] where i is the passage index.
- Prefer quoting short phrases (in quotes) from the supporting passage.
- Do NOT restate the question. Be concise and factual.
- If evidence is insufficient, add one final bullet: "Insufficient evidence [#{{i}}]" citing the closest passage.

Question:
{question}

Context passages:
{contexts}

Answer (bullets with cites):
"""


def _format_context(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        # include a short quote to encourage attribution
        text = h['text']
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        quote = lines[0][:200] if lines else text[:200]
        parts.append(f"[{i}] {text[:800]}\n> \"{quote}\"")
    return "\n\n".join(parts)


def _extract_bullets(text: str, limit: int) -> List[str]:
    bullets: List[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("-") or s.startswith("•") or s.startswith("*"):
            bullets.append(s.lstrip("-•* "))
        if len(bullets) >= limit:
            break
    return bullets


class SynthesizeNode:
    def __init__(self) -> None:
        self.gen = None  # lazy-load to avoid gated model downloads when disabled

    def __call__(self, question: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        from config.settings import settings
        # Optional extractive fallback for small models
        if getattr(settings, "use_extractive_summary", True):
            bullets = []
            kmin = getattr(settings, "summary_bullets_min", 3)
            for i, h in enumerate(hits[: getattr(settings, "summary_bullets_max", 5)], start=1):
                text = h["text"].strip().split("\n")[:3]
                snippet = " ".join(s.strip() for s in text if s.strip())[:240]
                bullets.append(f"- {snippet} [#{i}]")
            if len(bullets) >= kmin and not settings.enable_generation:
                return {"answer": "\n".join(bullets)}
        if not settings.enable_generation:
            return {"answer": "[generation disabled; set RAG_MCP_ENABLE_GENERATION=1 to enable]"}
        if self.gen is None:
            if getattr(settings, "use_ollama", False):
                self.gen = OllamaGenerator(model=getattr(settings, "ollama_model", "gemma3:270m"))
            else:
                self.gen = GemmaGenerator()
        prompt = ANSWER_PROMPT.format(question=question, contexts=_format_context(hits))
        text = self.gen.generate(prompt)
        # Post-process: enforce bullets and attach cites/quotes
        raw_bullets = _extract_bullets(text, getattr(settings, "summary_bullets_max", 5))
        bullets = [f"- {b}" for b in raw_bullets] if raw_bullets else [ln if ln.startswith("-") else f"- {ln}" for ln in (ln for ln in text.splitlines() if ln.strip())]
        out = []
        vocab = set(_keywords(question)) if callable(globals().get('_keywords')) else set()
        for idx, b in enumerate(bullets[: getattr(settings, "summary_bullets_max", 5)], start=1):
            # attach cite if missing
            # normalize malformed cites like [#{1}] -> [#1]
            b = re.sub(r"\[#\{(\d+)\}\]", r"[#\1]", b)
            has_proper_cite = re.search(r"\[#(\d+)\]", b) is not None
            if not has_proper_cite:
                # pick a short quote from the corresponding hit if available
                q = hits[idx - 1]["text"].strip().split("\n")[0][:120] if idx - 1 < len(hits) else ""
                if q:
                    b = f"{b} — \"{q}\" [#{idx}]"
                else:
                    b = f"{b} [#{idx}]"
            # drop bullet if it shares no keywords with its supporting passage to reduce drift
            if vocab:
                passage = hits[idx - 1]["text"] if idx - 1 < len(hits) else ""
                if not any(tok in passage.lower() for tok in vocab):
                    continue
            out.append(b)
        if not out:
            # force bullets with best passages
            for i, h in enumerate(hits[: getattr(settings, "summary_bullets_max", 5)], start=1):
                first_line = h["text"].strip().split("\n")[0][:200]
                out.append(f"- {first_line} [#{i}]")
        return {"answer": "\n".join(out)}
