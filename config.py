
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

# Load local .env file (local development)
load_dotenv()

# Load Streamlit Cloud Secrets into environment variables
# This keeps the existing os.getenv() configuration working.
try:
    import streamlit as st

    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))

except Exception:
    # Streamlit may not be installed or secrets may not be configured.
    # Local development through .env continues to work.
    pass


def _get_int(name: str, default: int) -> int:
    """Read an integer setting from the environment."""
    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(
            f"[config] '{name}={raw_value}' is not a number. "
            f"Using the value -> {default}"
        )
        return default


def _get_str(name: str, default: str) -> str:
    """Read a text setting from the environment."""
    value = os.getenv(name, "").strip()
    return value or default


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

_active_root: ContextVar[Optional[Path]] = ContextVar(
    "active_root", default=None
)

_WORKSPACE_FOLDERS = {
    "DOCUMENTS_DIR": ("data", "documents"),
    "EXTRACTED_IMAGES_DIR": ("data", "extracted_images"),
    "EVALUATION_DIR": ("data", "evaluation"),
    "VECTORSTORE_DIR": ("vectorstore",),
    "FAISS_DIR": ("vectorstore", "faiss"),
    "DOC_STORE_PATH": ("vectorstore", "doc_store.json"),
    "IMAGE_CACHE_PATH": ("vectorstore", "image_descriptions.json"),
}


def set_workspace(root: Optional[Path]) -> None:
    """
    Choose which folder tree the current request reads and writes.

    Each visitor can use a separate workspace.
    """
    _active_root.set(root)


def active_root() -> Path:
    """Return the active workspace or project root."""
    return _active_root.get() or PROJECT_ROOT


def __getattr__(name: str) -> Path:
    """Resolve folder paths dynamically for the active workspace."""

    if name in _WORKSPACE_FOLDERS:
        return active_root().joinpath(*_WORKSPACE_FOLDERS[name])

    raise AttributeError(
        f"module 'config' has no attribute '{name}'"
    )


# ---------------------------------------------------------------------------
# NVIDIA NIM
# ---------------------------------------------------------------------------

NVIDIA_API_KEY = _get_str("NVIDIA_API_KEY", "")

NVIDIA_BASE_URL = _get_str(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)

NVIDIA_CHAT_MODEL = _get_str(
    "NVIDIA_CHAT_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b"
)

NVIDIA_VISION_MODEL = _get_str(
    "NVIDIA_VISION_MODEL",
    "nvidia/nemotron-3.5-super-vl-120b-a12b"
)

NVIDIA_EMBEDDING_MODEL = _get_str(
    "NVIDIA_EMBEDDING_MODEL",
    "nvidia/nemotron-3.5-embed-1b"
)

NVIDIA_RERANK_MODEL = _get_str(
    "NVIDIA_RERANK_MODEL",
    "nvidia/nemotron-3.5-rerank-8b"
)


# ---------------------------------------------------------------------------
# Groq - Backup chat model
# ---------------------------------------------------------------------------

GROQ_API_KEY = _get_str("GROQ_API_KEY", "")

GROQ_BASE_URL = _get_str(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

GROQ_CHAT_MODEL = _get_str(
    "GROQ_CHAT_MODEL",
    "openai/gpt-oss-20b"
)

LLM_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Multimodal settings
# ---------------------------------------------------------------------------

MAX_IMAGES_PER_ANSWER = _get_int(
    "MAX_IMAGES_PER_ANSWER", 3
)


# ---------------------------------------------------------------------------
# Visitor workspaces
# ---------------------------------------------------------------------------

PER_SESSION_WORKSPACES = (
    _get_str("PER_SESSION_WORKSPACES", "false").lower() == "true"
)

WORKSPACE_IDLE_MINUTES = _get_int(
    "WORKSPACE_IDLE_MINUTES", 45
)

MAX_UPLOAD_FILES = _get_int(
    "MAX_UPLOAD_FILES", 5
)

MAX_UPLOAD_MB = _get_int(
    "MAX_UPLOAD_MB", 20
)


# ---------------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------------

VECTOR_DB = _get_str("VECTOR_DB", "faiss").lower()

PINECONE_API_KEY = _get_str("PINECONE_API_KEY", "")

PINECONE_INDEX_NAME = _get_str(
    "PINECONE_INDEX_NAME", "advanced-rag"
)

PINECONE_NAMESPACE = _get_str(
    "PINECONE_NAMESPACE", "default"
)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

PARENT_CHUNK_SIZE = _get_int(
    "PARENT_CHUNK_SIZE", 1600
)

PARENT_CHUNK_OVERLAP = _get_int(
    "PARENT_CHUNK_OVERLAP", 200
)

CHILD_CHUNK_SIZE = _get_int(
    "CHILD_CHUNK_SIZE", 400
)

CHILD_CHUNK_OVERLAP = _get_int(
    "CHILD_CHUNK_OVERLAP", 60
)


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------

NUM_QUERY_VARIANTS = _get_int(
    "NUM_QUERY_VARIANTS", 3
)

# Backwards compatibility
NUM_QUERY_VARIENTS = NUM_QUERY_VARIANTS

VECTOR_TOP_K = _get_int(
    "VECTOR_TOP_K", 6
)

BM25_TOP_K = _get_int(
    "BM25_TOP_K", 6
)

HYBRID_TOP_K = _get_int(
    "HYBRID_TOP_K", 10
)

RERANK_TOP_K = _get_int(
    "RERANK_TOP_K", 4
)

MIN_RERANK_SCORE = float(
    os.getenv("MIN_RERANK_SCORE", "-4.0")
)

MAX_CONTEXT_CHARS = _get_int(
    "MAX_CONTEXT_CHARS", 8000
)

MEMORY_WINDOW = _get_int(
    "MEMORY_WINDOW", 4
)


# ---------------------------------------------------------------------------
# Reranker configuration
# ---------------------------------------------------------------------------

# Options:
# "nvidia"        -> NVIDIA reranker API
# "cross-encoder" -> Local sentence-transformers model
# "simple"        -> Keyword overlap

RERANKER = _get_str(
    "RERANKER", "nvidia"
).lower()

CROSS_ENCODER_MODEL = _get_str(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------------------------------------------------
# Validation and display helpers
# ---------------------------------------------------------------------------

def check_llm_keys() -> None:
    """Ensure the required NVIDIA API key is configured."""

    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is not set. "
            "Add it to your .env file locally or configure "
            "it in Streamlit Cloud Secrets. "
            "Get a key at build.nvidia.com."
        )


def groq_backup_enabled() -> bool:
    """Check whether Groq backup is configured."""
    return bool(GROQ_API_KEY)


def check_pinecone_config() -> None:
    """Validate Pinecone configuration when Pinecone is enabled."""

    if not PINECONE_API_KEY:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_API_KEY is empty."
        )

    if not PINECONE_INDEX_NAME:
        raise ValueError(
            "VECTOR_DB=pinecone but PINECONE_INDEX_NAME is empty."
        )


def describe_settings() -> dict:
    """Summarise active settings for the Streamlit sidebar."""

    return {
        "Chat Model": NVIDIA_CHAT_MODEL,
        "Backup Chat": (
            GROQ_CHAT_MODEL
            if groq_backup_enabled()
            else "off (no GROQ_API_KEY)"
        ),
        "Vision Model": NVIDIA_VISION_MODEL,
        "Embedding Model": NVIDIA_EMBEDDING_MODEL,
        "VectorDB": VECTOR_DB,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE,
    }
