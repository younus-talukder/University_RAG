from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .config import LLM_MODEL

try:
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover
    pipeline = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


SYSTEM_PROMPT = """You are a university information assistant.

Answer the student's question using ONLY the provided university document context.
Do not use outside knowledge to create university-specific facts, policies, fees, deadlines, requirements, or rules.
If the answer cannot be found in the provided context, say that the information could not be found in the available university documents.

Course-specific rule:
- If the question mentions a specific course code or course name (for example CSE 101, MTH 101, English), answer only from that course's context.
- Do not mix facts from other courses or give generic course outcomes when the question refers to one specific course.
- Ignore unrelated course content even if it is semantically similar.

Language rule:
- If the student's question is in Bangla, answer in Bangla.
- If the student's question is in Banglish, answer in Banglish.
- If the student's question is in English, answer in English.
- Preserve the same meaning as the context; do not invent information.
- If the context is in English but the question is Bangla, still answer in Bangla by translating only the retrieved facts into Bangla without adding anything new.

Student Question:
{question}

University Document Context:
{context}
"""


def detect_question_language(question: str) -> str:
    if not question:
        return "English"
    text = question.strip()
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "Bangla"
    banglish_markers = ["ki", "kora", "kono", "kothay", "kobe", "kivabe", "ebong", "er", "er", "course-er"]
    lowered = text.lower()
    if any(marker in lowered for marker in banglish_markers):
        return "Banglish"
    return "English"


def build_prompt(question: str, context: Sequence[str]) -> str:
    joined_context = "\n\n---\n\n".join(str(item) for item in context if str(item).strip())
    language = detect_question_language(question)
    language_note = (
        "Answer entirely in Bangla. "
        if language == "Bangla"
        else "Answer entirely in Banglish. "
        if language == "Banglish"
        else "Answer entirely in English. "
    )
    return SYSTEM_PROMPT.format(question=question, context=joined_context) + "\nLanguage requirement: " + language_note


def generate_answer(question: str, context: Sequence[str], model_name: str = LLM_MODEL, max_new_tokens: int = 256) -> str:
    if pipeline is None:
        raise ImportError(
            "transformers is required for local Qwen generation. Install the project requirements first."
        ) from _IMPORT_ERROR

    prompt = build_prompt(question, context)
    try:
        import accelerate  # noqa: F401
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device_map="auto",
        )
    except ImportError:
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    output = generator(prompt, truncation=True)
    generated_text = output[0]["generated_text"]
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


__all__ = ["build_prompt", "generate_answer"]
