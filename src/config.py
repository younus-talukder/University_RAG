from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENT_DIR = DATA_DIR / "documents"
QUESTIONS_DIR = DATA_DIR / "questions"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
RESULTS_DIR = ROOT_DIR / "results"

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TOP_K = 3
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf"}


def ensure_directories() -> None:
    for directory in (DATA_DIR, DOCUMENT_DIR, QUESTIONS_DIR, VECTOR_DB_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "DOCUMENT_DIR",
    "QUESTIONS_DIR",
    "VECTOR_DB_DIR",
    "RESULTS_DIR",
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "ensure_directories",
]
