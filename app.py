from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.config import VECTOR_DB_DIR
from src.pipeline import answer_question


st.set_page_config(page_title="University RAG Chatbot", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner=False)
def get_index_status() -> bool:
    return (VECTOR_DB_DIR / "index.faiss").exists() and (VECTOR_DB_DIR / "metadata.pkl").exists()


def render_sources(sources):
    if not sources:
        st.info("No source metadata was returned for this answer.")
        return

    st.subheader("Retrieved sources")
    for i, item in enumerate(sources, start=1):
        source = item.get("source") or "Unknown source"
        page = item.get("page")
        score = item.get("score")
        label = f"Source {i}"
        if page is not None:
            label += f" • Page {page}"
        if score is not None:
            label += f" • Score {score:.4f}"
        st.markdown(f"**{label}**")
        st.caption(source)


st.title("University Information Chatbot")
st.caption("RAG-based assistant that answers questions using the university PDF documents.")

with st.sidebar:
    st.header("Status")
    index_ready = get_index_status()
    if index_ready:
        st.success("Vector index is ready.")
    else:
        st.warning("Vector index is missing. Build it before using the chatbot.")
    st.markdown(
        """
        Supported question styles:
        - Bangla
        - English
        - Banglish
        """
    )

question = st.text_input(
    "Ask a question about the university",
    placeholder="Example: CSE 101 কোর্সের প্রধান উদ্দেশ্য কী? / What is CSE 101? / CSE 101 course-er main objectives ki?",
    key="user_question",
)

if st.button("Ask"):
    if not question or not question.strip():
        st.warning("Please enter a valid question.")
    elif not index_ready:
        st.error("The FAISS vector index is missing. Please build it first using the project scripts.")
    else:
        with st.spinner("Retrieving relevant university context and generating an answer..."):
            try:
                response = answer_question(question, top_k=3)
            except Exception as exc:  # pragma: no cover - UI presentation only
                st.error(f"The chatbot could not generate an answer: {exc}")
            else:
                st.subheader("Answer")
                st.write(response.get("answer", "No answer was generated."))
                render_sources(response.get("sources", []))
