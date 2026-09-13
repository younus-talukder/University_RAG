from __future__ import annotations

import os
import re
import time
from typing import Sequence

from .config import LLM_MODEL
from .language_detector import Language, detect_language

try:
    from llama_cpp import Llama
except Exception as exc:  # pragma: no cover
    Llama = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

GENERATOR_CACHE: dict[str, object] = {}
FIXED_N_CTX = 8192
FIXED_N_THREADS = max(1, os.cpu_count() or 1)


def _context_text(context: Sequence[str]) -> str:
    return "\n\n---\n\n".join(str(item).strip() for item in context if str(item).strip())


def _ensure_context_within_n_ctx(context: Sequence[str], n_ctx: int = FIXED_N_CTX) -> list[str]:
    cleaned = [str(item).strip() for item in context if str(item).strip()]
    if not cleaned:
        return []

    prompt_budget = max(512, n_ctx - 256)
    total_text = _context_text(cleaned)
    if len(total_text.split()) * 2 <= prompt_budget:
        return cleaned

    truncated: list[str] = []
    remaining_tokens = prompt_budget
    for item in cleaned:
        item_words = item.split()
        token_count = len(item_words) * 2
        if token_count <= remaining_tokens:
            truncated.append(item)
            remaining_tokens -= token_count
        else:
            keep_words = max(1, remaining_tokens // 2)
            truncated.append(" ".join(item_words[:keep_words]))
            break

    if not truncated:
        raise ValueError(
            f"Context exceeds fixed n_ctx={n_ctx}. "
            "Reduce the number of retrieved chunks or shorten each chunk before generation."
        )

    return truncated


def _language_instruction(language: Language) -> str:
    if language == "bangla":
        return "Respond in Bangla only, as a short factual answer."
    if language == "banglish":
        return "Respond in Banglish only, as a short factual answer."
    return "Respond in English only, as a short factual answer."


def _build_chat_messages(
    question: str,
    context: Sequence[str],
    language: Language | None = None,
    requested_entity: str | None = None,
    requested_field: str | None = None,
) -> list[dict[str, str]]:
    joined_context = _context_text(context)
    detected_language = language or detect_language(question)
    system_message = (
        "You are a university information assistant. "
        "Answer using ONLY the provided university document context. "
        "Do not use outside knowledge or mix facts from different courses. "
        "If a question mentions a specific course, answer only from that course's context. "
        f"{_language_instruction(detected_language)} "
        "Do not repeat or rephrase the question. "
        "If the answer is a list of topics, use a short heading followed by bullet points. "
        "If the answer cannot be found in the context, say that it could not be found in the available university documents."
    )
    user_message = (
        f"Detected question language for final post-processing: {detected_language}\n\n"
        f"Requested entity: {requested_entity or 'none explicitly stated'}\n"
        f"Requested field: {requested_field or 'general'}\n\n"
        f"Question:\n{question}\n\n"
        f"Verified supporting excerpts:\n{joined_context}"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _translate_answer_to_language(text: str, target_language: Language) -> str:
    from .fast_answer import format_answer_for_language

    return format_answer_for_language(text, target_language)


def build_prompt(question: str, context: Sequence[str], language: Language | None = None) -> str:
    messages = _build_chat_messages(question, context, language=language)
    return "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def _clean_generated_answer(question: str, answer: str) -> str:
    cleaned = str(answer).strip()
    if not cleaned:
        return ""

    compact_question = re.sub(r"\s+", " ", question).strip(" ?.।")
    compact_answer = re.sub(r"\s+", " ", cleaned).strip()
    if compact_question and compact_answer.lower().startswith(compact_question.lower()):
        cleaned = compact_answer[len(compact_question):].lstrip(" ?:।-")

    question_words = {
        token
        for token in re.findall(r"[A-Za-z]+", question.lower())
        if len(token) >= 4
    }
    if ":" in cleaned and question_words:
        prefix, suffix = cleaned.split(":", 1)
        prefix_words = set(re.findall(r"[A-Za-z]+", prefix.lower()))
        if len(question_words & prefix_words) >= min(2, len(question_words)):
            cleaned = suffix.strip()

    return cleaned.strip()


def _infer_topic_subject(question: str) -> str:
    stripped = " ".join(str(question).split()).strip(" ?।")
    patterns = [
        r"^(.+?)-er\s+under-e\b",
        r"^(.+?)-এর\s+অধীনে(?=\s|$)",
        r"\bunder\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" :")
    return ""


def _prepare_answer_for_formatting(question: str, answer: str) -> str:
    cleaned = _clean_generated_answer(question, answer)
    if not cleaned:
        return ""

    asks_for_topics = re.search(
        r"\btopics?\b|\binclude(?:d|s)?\b|কোন\s+কোন\s+বিষয়|বিষয়\s+অন্তর্ভুক্ত",
        question,
        flags=re.IGNORECASE,
    )
    if not asks_for_topics:
        return cleaned

    subject = _infer_topic_subject(question)
    if subject and not re.search(r"\b(?:under|topics?\s+include|includes|included)\b", cleaned, flags=re.IGNORECASE):
        return f"Under {subject}, topics include {cleaned}"
    return cleaned


def _get_generator(model_name: str, n_ctx: int = FIXED_N_CTX, n_threads: int = FIXED_N_THREADS):
    global GENERATOR_CACHE

    if model_name in GENERATOR_CACHE:
        return GENERATOR_CACHE[model_name]

    if Llama is None:
        raise ImportError(
            "llama-cpp-python is required for local GGUF generation. "
            f"Install the project requirements first. Original import error: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR

    model_path = str(model_name).strip()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GGUF model not found at '{model_path}'. Download the Qwen2.5-1.5B-Instruct-Q4_K_M.gguf file "
            "and point LLM_MODEL to its local path."
        )

    load_start = time.perf_counter()
    generator = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=512,
        n_gpu_layers=0,
        use_mlock=False,
        verbose=False,
    )
    load_seconds = time.perf_counter() - load_start
    print(f"[generator] model load: {load_seconds:.3f}s | model={model_path} | n_ctx={n_ctx} | n_threads={n_threads}")

    GENERATOR_CACHE[model_name] = generator
    return generator


def generate_answer(
    question: str,
    context: Sequence[str],
    language: Language | None = None,
    model_name: str = LLM_MODEL,
    max_new_tokens: int = 180,
    requested_entity: str | None = None,
    requested_field: str | None = None,
) -> str:
    detected_language = language or detect_language(question)
    safe_context = _ensure_context_within_n_ctx(context)
    messages = _build_chat_messages(
        question,
        safe_context,
        language=detected_language,
        requested_entity=requested_entity,
        requested_field=requested_field,
    )
    generator = _get_generator(model_name, FIXED_N_CTX, FIXED_N_THREADS)

    generation_start = time.perf_counter()
    response = generator.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stream=False,
    )
    generation_seconds = time.perf_counter() - generation_start
    print(f"[generator] generation: {generation_seconds:.3f}s | max_new_tokens={max_new_tokens}")

    english_answer = _prepare_answer_for_formatting(question, response["choices"][0]["message"]["content"])
    return _translate_answer_to_language(english_answer, detected_language)


def regenerate_answer_for_language(
    question: str,
    context: Sequence[str],
    language: Language,
    previous_answer: str,
    model_name: str = LLM_MODEL,
    max_new_tokens: int = 180,
) -> str:
    detected_language = language or detect_language(question)
    safe_context = _ensure_context_within_n_ctx(context)
    retry_messages = [
        {
            "role": "system",
            "content": (
                "You are a university information assistant. "
                "Answer using ONLY the supplied context. "
                "Do not invent facts. "
                f"{_language_instruction(detected_language)} "
                "Keep the answer grounded in the provided context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context:\n{_context_text(safe_context)}\n\n"
                f"Previous answer (wrong language/style):\n{previous_answer}"
            ),
        },
    ]
    generator = _get_generator(model_name, FIXED_N_CTX, FIXED_N_THREADS)

    generation_start = time.perf_counter()
    response = generator.create_chat_completion(
        messages=retry_messages,
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stream=False,
    )
    generation_seconds = time.perf_counter() - generation_start
    print(f"[generator] retried generation: {generation_seconds:.3f}s | language={detected_language}")

    english_answer = _prepare_answer_for_formatting(question, response["choices"][0]["message"]["content"])
    return _translate_answer_to_language(english_answer, detected_language)


__all__ = ["build_prompt", "generate_answer", "regenerate_answer_for_language"]
