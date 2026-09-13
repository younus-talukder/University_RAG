from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.config import VECTOR_DB_DIR

try:
    from src.pipeline import answer_question
except Exception as exc:  # pragma: no cover - UI presentation only
    answer_question = None
    PIPELINE_IMPORT_ERROR = exc
else:
    PIPELINE_IMPORT_ERROR = None


st.set_page_config(page_title="University RAG Chatbot", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner=False)
def get_index_status() -> bool:
    return (VECTOR_DB_DIR / "index.faiss").exists() and (VECTOR_DB_DIR / "metadata.pkl").exists()


def render_sources(sources):
    if not sources:
        st.info("No source metadata was returned for this answer.")
        return

    st.subheader("Supporting evidence")
    for i, item in enumerate(sources, start=1):
        source = item.get("relative_path") or item.get("source") or "Unknown source"
        page = item.get("page")
        score = item.get("score")
        label = f"Source {i}"
        if page is not None:
            label += f" • Page {page}"
        if score is not None:
            label += f" • Score {score:.4f}"
        st.markdown(f"**{label}**")
        st.caption(source)
        excerpt = item.get("supporting_excerpt")
        if excerpt:
            st.write(excerpt)


def render_debug_context(chunks):
    if not chunks:
        return

    with st.expander("Retrieved context / debugging"):
        for i, item in enumerate(chunks, start=1):
            st.markdown(
                f"**Chunk {i}:** {item.get('relative_path') or item.get('source', 'unknown')} | "
                f"page {item.get('page', 'unknown')} | score {item.get('score', 0.0):.4f}"
            )
            st.write(item.get("text", ""))


st.title("University Information Chatbot")
st.caption("RAG-based assistant that answers questions using the university PDF documents.")

with st.sidebar:
    st.header("Status")
    index_ready = get_index_status()
    if index_ready:
        st.success("Vector index is ready.")
    else:
        st.warning("Vector index is missing. Build it before using the chatbot.")
    if PIPELINE_IMPORT_ERROR is not None:
        st.error("Backend import problem detected.")
        st.caption(str(PIPELINE_IMPORT_ERROR))
    st.markdown(
        """
        Supported question styles:
        - Bangla
        - English
        - Banglish
        """
    )
    use_generation = st.checkbox("Use local Qwen generation", value=False)

question = st.text_input(
    "Ask a question about the university",
    placeholder="Example: CSE 101 কোর্সের প্রধান উদ্দেশ্য কী? / What is CSE 101? / CSE 101 course-er main objectives ki?",
    key="user_question",
)

if st.button("Ask"):
    if PIPELINE_IMPORT_ERROR is not None:
        st.error(
            "The chatbot backend could not initialize because of a dependency/runtime problem: "
            f"{PIPELINE_IMPORT_ERROR}.\n\nPlease fix the Python environment and restart the app."
        )
    elif not question or not question.strip():
        st.warning("Please enter a valid question.")
    elif not index_ready:
        st.error("The FAISS vector index is missing. Please build it first using the project scripts.")
    else:
        started_at = time.perf_counter()
        status = st.status("Starting chatbot pipeline...", expanded=True)

        def update_status(message: str) -> None:
            elapsed = time.perf_counter() - started_at
            status.update(label=f"{message} ({elapsed:.1f}s elapsed)", state="running")

        try:
            response = answer_question(
                question,
                top_k=3,
                status_callback=update_status,
                use_generation=use_generation,
            )
        except Exception as exc:  # pragma: no cover - UI presentation only
            status.update(label="Chatbot pipeline failed.", state="error")
            st.error(f"The chatbot could not generate an answer: {exc}")
        else:
            elapsed = time.perf_counter() - started_at
            status.update(label=f"Answer generated in {elapsed:.1f}s.", state="complete", expanded=False)
            st.subheader("Answer")
            st.caption(f"Detected Language: {response.get('detected_language', 'unknown')}")
            st.caption(f"Mode: {response.get('answer_mode', response.get('generation_mode', 'unknown'))}")
            st.caption(f"Evidence status: {response.get('support_status', 'unknown')}")
            st.write(response.get("answer", "No answer was generated."))
            render_sources(response.get("sources", []))
            render_debug_context(response.get("retrieved_context", []))
