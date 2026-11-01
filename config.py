import os
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    """
    Read an integer setting from the environment.

    Everything in a `.env` file is text, so "400" arrives as the string "400".
    This helper converts it to a real integer and falls back to the default if
    the variable is missing or is not a valid number.

    Args:
        name: Environment variable name, for example "CHILD_CHUNK_SIZE".
        default: Value to use when the variable is missing or invalid.

    Returns:
        The setting as an integer.
    """
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        print(f"[config] '{name}={raw_value}' is not a number. Using the value -> {default}")
        return default


def _get_str(name: str, default: str) -> str:
    """
    Read a text setting from the environment, falling back to a default.

    Args:
        name: Environment variable name.
        default: Value used when the variable is empty or missing.

    Returns:
        The stripped value, or the default.
    """
    value = os.getenv(name, "").strip()
    return value or default


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# Every folder below is resolved at the moment it is used, not when this file
# is imported. Code keeps writing `config.DOCUMENTS_DIR`, `config.FAISS_DIR` and
# so on, and each visitor gets their own folders when a workspace is active
# (see `set_workspace`). Without a workspace, the project's own `data/` and
# `vectorstore/` folders are used, as before.
PROJECT_ROOT = Path(__file__).parent

_active_root: ContextVar[Optional[Path]] = ContextVar("active_root", default=None)

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

    Streamlit runs each visitor's script in its own thread. Calling this at the
    start of a run makes that run use the visitor's folders, without affecting
    other visitors.

    Args:
        root: The workspace folder, or None to go back to the project folders.
    """
    _active_root.set(root)


def active_root() -> Path:
    """
    Return the folder tree in use for the current request.

    Returns:
        The active workspace folder, or PROJECT_ROOT if none is set.
    """
    return _active_root.get() or PROJECT_ROOT


def __getattr__(name: str) -> Path:
    """
    Resolve folder names such as `config.FAISS_DIR` for the active workspace.

    Python calls this only when a name is not defined in this module, so the
    folder names above are never stored as fixed values.

    Args:
        name: The attribute being read, for example "DOCUMENTS_DIR".

    Returns:
        The path for that folder under the active workspace.

    Raises:
        AttributeError: If the name is not a known folder.
    """
    if name in _WORKSPACE_FOLDERS:
        return active_root().joinpath(*_WORKSPACE_FOLDERS[name])
    raise AttributeError(f"module 'config' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# NVIDIA NIM (primary provider: chat, vision, embeddings, reranker)
# ---------------------------------------------------------------------------
# NVIDIA's endpoints are OpenAI-compatible, so llm_client.py can use the
# OpenAI client classes with this base URL.
NVIDIA_API_KEY = _get_str("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = _get_str("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_CHAT_MODEL = _get_str("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_VISION_MODEL = _get_str("NVIDIA_VISION_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
NVIDIA_EMBEDDING_MODEL = _get_str("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b")
NVIDIA_RERANK_MODEL = _get_str("NVIDIA_RERANK_MODEL", "nvidia/nemotron-3.5-rerank-8b")


# ---------------------------------------------------------------------------
# Groq (backup chat model only, used when the NVIDIA chat call fails)
# ---------------------------------------------------------------------------
GROQ_API_KEY = _get_str("GROQ_API_KEY", "")
GROQ_BASE_URL = _get_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_CHAT_MODEL = _get_str("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")

LLM_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Multimodal settings
# ---------------------------------------------------------------------------
# Upper limit on how many retrieved images are sent to the vision model in one
# answer. Images are billed as tokens, so a small cap protects free-tier limits.
MAX_IMAGES_PER_ANSWER = _get_int("MAX_IMAGES_PER_ANSWER", 3)


# ---------------------------------------------------------------------------
# Visitor workspaces (used by hosted demos)
# ---------------------------------------------------------------------------
# On a hosted demo, each visitor gets a private folder. It is deleted after
# this many minutes without activity.
PER_SESSION_WORKSPACES = _get_str("PER_SESSION_WORKSPACES", "false").lower() == "true"
WORKSPACE_IDLE_MINUTES = _get_int("WORKSPACE_IDLE_MINUTES", 45)

# Upload limits for each workspace, so the shared free API quota is protected.
MAX_UPLOAD_FILES = _get_int("MAX_UPLOAD_FILES", 5)
MAX_UPLOAD_MB = _get_int("MAX_UPLOAD_MB", 20)


# ---------------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------------
VECTOR_DB = _get_str("VECTOR_DB", "faiss").lower()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = _get_str("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = _get_str("PINECONE_NAMESPACE", "default")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# ---------------------------------------------------------------------------
# Retrieval pipeline
# ---------------------------------------------------------------------------
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)
NUM_QUERY_VARIENTS = NUM_QUERY_VARIANTS  # alias for backwards compatibility

VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)
RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)

# "nvidia"        -> NVIDIA reranker API (default, no PyTorch needed)
# "cross-encoder" -> local model (needs sentence-transformers + PyTorch)
# "simple"        -> keyword overlap (no extra packages)
RERANKER = _get_str("RERANKER", "nvidia").lower()
CROSS_ENCODER_MODEL = _get_str("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


# ---------------------------------------------------------------------------
# Validation and display helpers
# ---------------------------------------------------------------------------
def check_llm_keys() -> None:
    """
    Make sure the NVIDIA key is set, because NVIDIA is the required provider.

    NVIDIA handles chat, vision and embeddings, so the app cannot run without
    it. The Groq key is optional: without it, there is simply no backup chat
    model.

    Raises:
        ValueError: If NVIDIA_API_KEY is empty.
    """
    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to your .env file. "
            "Get a free key at build.nvidia.com."
        )


def groq_backup_enabled() -> bool:
    """
    Tell whether the Groq backup chat model can be used.

    Returns:
        True if GROQ_API_KEY is set, False otherwise.
    """
    return bool(GROQ_API_KEY)


def check_pinecone_config() -> None:
    """
    Make sure the Pinecone settings are present when VECTOR_DB=pinecone.

    Raises:
        ValueError: If the API key or index name is missing.
    """
    if not PINECONE_API_KEY:
        raise ValueError("VECTOR_DB=pinecone but PINECONE_API_KEY is empty.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("VECTOR_DB=pinecone but PINECONE_INDEX_NAME is empty.")


def describe_settings() -> dict:
    """
    Summarise the active settings for the Streamlit sidebar.

    Returns:
        A dictionary of label -> value.
    """
    return {
        "Chat Model": NVIDIA_CHAT_MODEL,
        "Backup Chat": GROQ_CHAT_MODEL if groq_backup_enabled() else "off (no GROQ_API_KEY)",
        "Vision Model": NVIDIA_VISION_MODEL,
        "Embedding Model": NVIDIA_EMBEDDING_MODEL,
        "VectorDB": VECTOR_DB,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE,
    }
