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

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off.")


def _env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
# Immutable Hugging Face commit already present as a complete local PyTorch snapshot.
EMBEDDING_REVISION = os.environ.get(
    "EMBEDDING_REVISION",
    "5617a9f61b028005a4858fdac845db406aefb181",
)
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = _env_positive_int("EMBEDDING_BATCH_SIZE", 16)
EMBEDDING_DIMENSION = _env_positive_int("EMBEDDING_DIMENSION", 1024)
EMBEDDING_NORMALIZE = _env_bool("EMBEDDING_NORMALIZE", True)
EMBEDDING_LOCAL_ONLY = _env_bool("EMBEDDING_LOCAL_ONLY", True)
# The pinned complete snapshot contains pytorch_model.bin. A separate cached
# revision contains only safetensors and must not be mixed with this revision.
EMBEDDING_USE_SAFETENSORS = _env_bool("EMBEDDING_USE_SAFETENSORS", False)
EMBEDDING_DTYPE = os.environ.get("EMBEDDING_DTYPE", "float32")
EMBEDDING_NORM_TOLERANCE = float(os.environ.get("EMBEDDING_NORM_TOLERANCE", "0.001"))

SPARSE_TOKENIZER_SCHEMA = "unicode-university-v1.2"
BM25_IMPLEMENTATION = "local-okapi-bm25-v1"
BM25_K1 = float(os.environ.get("BM25_K1", "1.5"))
BM25_B = float(os.environ.get("BM25_B", "0.75"))
DENSE_CANDIDATE_K = _env_positive_int("DENSE_CANDIDATE_K", 30)
SPARSE_CANDIDATE_K = _env_positive_int("SPARSE_CANDIDATE_K", 30)
METADATA_CANDIDATE_K = _env_positive_int("METADATA_CANDIDATE_K", 20)
RRF_K = _env_positive_int("RRF_K", 60)
RRF_DENSE_WEIGHT = float(os.environ.get("RRF_DENSE_WEIGHT", "1.0"))
RRF_SPARSE_WEIGHT = float(os.environ.get("RRF_SPARSE_WEIGHT", "1.0"))
RRF_METADATA_WEIGHT = float(os.environ.get("RRF_METADATA_WEIGHT", "0.8"))
DENSE_QUERY_STRATEGY = os.environ.get("DENSE_QUERY_STRATEGY", "original").strip().casefold()
if DENSE_QUERY_STRATEGY not in {"original", "normalized", "original_plus_normalized"}:
    raise ValueError("DENSE_QUERY_STRATEGY must be original, normalized, or original_plus_normalized.")
RERANKER_ENABLED = _env_bool("RERANKER_ENABLED", False)
APPROVED_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
APPROVED_RERANKER_REVISION = "b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", APPROVED_RERANKER_MODEL).strip()
RERANKER_REVISION = os.environ.get(
    "RERANKER_REVISION",
    APPROVED_RERANKER_REVISION,
).strip()
if RERANKER_MODEL != APPROVED_RERANKER_MODEL or RERANKER_REVISION != APPROVED_RERANKER_REVISION:
    raise ValueError(
        "Only the approved pinned reranker is permitted: "
        f"{APPROVED_RERANKER_MODEL}@{APPROVED_RERANKER_REVISION}."
    )
# Least harmful of the evaluated K values; reranking itself remains disabled by default.
RERANKER_CANDIDATE_K = _env_positive_int("RERANKER_CANDIDATE_K", 10)
RERANKER_LOCAL_ONLY = _env_bool("RERANKER_LOCAL_ONLY", True)
RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", "cpu").strip().casefold()
RERANKER_BATCH_SIZE = _env_positive_int("RERANKER_BATCH_SIZE", 1)
RERANKER_MAX_LENGTH = _env_positive_int("RERANKER_MAX_LENGTH", 512)
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
    "EMBEDDING_REVISION",
    "EMBEDDING_DEVICE",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_NORMALIZE",
    "EMBEDDING_LOCAL_ONLY",
    "EMBEDDING_USE_SAFETENSORS",
    "EMBEDDING_DTYPE",
    "EMBEDDING_NORM_TOLERANCE",
    "SPARSE_TOKENIZER_SCHEMA",
    "BM25_IMPLEMENTATION",
    "BM25_K1",
    "BM25_B",
    "DENSE_CANDIDATE_K",
    "SPARSE_CANDIDATE_K",
    "METADATA_CANDIDATE_K",
    "RRF_K",
    "RRF_DENSE_WEIGHT",
    "RRF_SPARSE_WEIGHT",
    "RRF_METADATA_WEIGHT",
    "DENSE_QUERY_STRATEGY",
    "APPROVED_RERANKER_MODEL",
    "APPROVED_RERANKER_REVISION",
    "RERANKER_ENABLED",
    "RERANKER_MODEL",
    "RERANKER_REVISION",
    "RERANKER_CANDIDATE_K",
    "RERANKER_LOCAL_ONLY",
    "RERANKER_DEVICE",
    "RERANKER_BATCH_SIZE",
    "RERANKER_MAX_LENGTH",
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
