from __future__ import annotations
from typing import Dict, Any, List
from models.llm_ollama import OllamaGenerator
from models.llm_gemma import GemmaGenerator
from config.settings import settings


def _get_llm():
    if getattr(settings, "use_ollama", False):
        return OllamaGenerator(model=getattr(settings, "ollama_model", "gemma3:270m"))
    return GemmaGenerator()


def _yn(text: str) -> str:
    s = (text or "").strip().lower()
    if "yes" in s and "no" not in s:
        return "yes"
    if "no" in s and "yes" not in s:
        return "no"
    return "no"


def grade_doc_relevance(question: str, document: str) -> Dict[str, Any]:
    prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "Answer strictly with 'yes' or 'no'.\n\n"
        f"Question: {question}\n\nDocument:\n{document[:2000]}\n\nRelevance (yes/no):"
    )
    llm = _get_llm()
    out = llm.generate(prompt, max_new_tokens=4, temperature=0.0)
    yn = _yn(out)
    return {"binary_score": yn, "raw": out}


def grade_generation_grounded(question: str, documents: List[str], generation: str) -> Dict[str, Any]:
    joined = "\n\n".join(d[:1200] for d in documents[:6])
    prompt = (
        "You are a grader assessing whether an LLM generation is grounded in the set of retrieved facts.\n"
        "Answer strictly with 'yes' or 'no'.\n\n"
        f"Question: {question}\n\nFacts:\n{joined}\n\nGeneration:\n{generation[:2000]}\n\nGrounded (yes/no):"
    )
    llm = _get_llm()
    out = llm.generate(prompt, max_new_tokens=4, temperature=0.0)
    yn = _yn(out)
    return {"binary_score": yn, "raw": out}


def grade_answer_addresses(question: str, generation: str) -> Dict[str, Any]:
    prompt = (
        "You are a grader assessing whether an answer resolves the user question.\n"
        "Answer strictly with 'yes' or 'no'.\n\n"
        f"Question: {question}\n\nAnswer:\n{generation[:2000]}\n\nAddresses (yes/no):"
    )
    llm = _get_llm()
    out = llm.generate(prompt, max_new_tokens=4, temperature=0.0)
    yn = _yn(out)
    return {"binary_score": yn, "raw": out}


