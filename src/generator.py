from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from .config import LLM_MODEL
from .language_detector import Language, detect_language

try:
    from transformers import pipeline
except Exception as exc:  # pragma: no cover
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
- Detected language: {language}
- If detected language is bangla, answer in Bengali Unicode.
- If detected language is banglish, answer in Latin-character Banglish.
- If detected language is english, answer in English.
- Preserve the same meaning as the context; do not invent information.
- If the context is in English but the question is Bangla, still answer in Bangla by translating only the retrieved facts into Bangla without adding anything new.
- Never convert Banglish to Bangla.
- Never convert Bangla to Banglish.

Student Question:
{question}

University Document Context:
{context}
"""


def _language_instruction(language: Language) -> str:
    if language == "bangla":
        return "Answer entirely in Bengali Unicode."
    if language == "banglish":
        return "Answer entirely in Latin-character Banglish. Do not use Bengali script."
    return "Answer entirely in English."


def build_prompt(question: str, context: Sequence[str], language: Language | None = None) -> str:
    joined_context = "\n\n---\n\n".join(str(item) for item in context if str(item).strip())
    detected_language = language or detect_language(question)
    return (
        SYSTEM_PROMPT.format(question=question, context=joined_context, language=detected_language)
        + "\nLanguage requirement: "
        + _language_instruction(detected_language)
    )


def build_language_retry_prompt(
    question: str,
    context: Sequence[str],
    language: Language,
    previous_answer: str,
) -> str:
    return (
        build_prompt(question, context, language=language)
        + "\n\nPrevious answer used the wrong language/style:\n"
        + previous_answer
        + "\n\nRegenerate the answer using the same retrieved context, and obey the detected language exactly."
    )


@lru_cache(maxsize=1)
def _get_generator(model_name: str, max_new_tokens: int):
    if pipeline is None:
        raise ImportError(
            "transformers is required for local Qwen generation. "
            f"Install the project requirements first. Original import error: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR

    try:
        import accelerate  # noqa: F401
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device_map="auto",
            model_kwargs={"use_safetensors": True},
        )
    except ImportError:
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            model_kwargs={"use_safetensors": True},
        )


def generate_answer(
    question: str,
    context: Sequence[str],
    language: Language | None = None,
    model_name: str = LLM_MODEL,
    max_new_tokens: int = 256,
) -> str:
    prompt = build_prompt(question, context, language=language)
    generator = _get_generator(model_name, max_new_tokens)
    output = generator(prompt, truncation=True, return_full_text=False)
    generated_text = output[0]["generated_text"]
    return generated_text.strip()


def regenerate_answer_for_language(
    question: str,
    context: Sequence[str],
    language: Language,
    previous_answer: str,
    model_name: str = LLM_MODEL,
    max_new_tokens: int = 256,
) -> str:
    prompt = build_language_retry_prompt(question, context, language, previous_answer)
    generator = _get_generator(model_name, max_new_tokens)
    output = generator(prompt, truncation=True, return_full_text=False)
    generated_text = output[0]["generated_text"]
    return generated_text.strip()


__all__ = ["build_prompt", "generate_answer", "regenerate_answer_for_language"]
