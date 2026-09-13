import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENT_DIR = DATA_DIR / "documents"
QUESTIONS_DIR = DATA_DIR / "questions"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"
RESULTS_DIR = ROOT_DIR / "results"
INGESTION_REPORT_PATH = RESULTS_DIR / "ingestion_report.json"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = os.environ.get(
    "LLM_MODEL_PATH",
    str(MODEL_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
)

TOP_K = 3
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
STRUCTURED_CHUNK_MAX_WORDS = 260
STRUCTURED_CHUNK_OVERLAP = 40
STRUCTURED_CHUNK_MIN_WORDS = 8
CHUNKER_SCHEMA_VERSION = "structure-aware-v1.1"

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
    "INGESTION_REPORT_PATH",
    "EMBEDDING_MODEL",
    "LLM_MODEL",
    "TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "STRUCTURED_CHUNK_MAX_WORDS",
    "STRUCTURED_CHUNK_OVERLAP",
    "STRUCTURED_CHUNK_MIN_WORDS",
    "CHUNKER_SCHEMA_VERSION",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "ensure_directories",
]
