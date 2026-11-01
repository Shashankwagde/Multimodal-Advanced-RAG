"""
Model factory for the whole project.

Every module that needs a language model, a vision model or an embedding
model gets it from this file. Nothing else talks to a provider directly, so
switching a model or a provider only ever means editing config.py / .env.

Providers used:
    NVIDIA NIM - primary for chat, vision and embeddings (free endpoints).
    Groq       - backup chat model, used only when the NVIDIA chat call fails.

Both providers expose OpenAI-compatible APIs, so we reuse the OpenAI client
classes from `langchain-openai` and only change the base URL and the key.
"""

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI

import config

# NVIDIA's embedding endpoints need to know whether the text is a question
# ("query") or a document chunk ("passage"). Sending the wrong type lowers
# retrieval quality, so we set it explicitly.
EMBEDDING_BATCH_SIZE = 64


class NVIDIAEmbeddings(Embeddings):
    """
    Embedding model backed by NVIDIA NIM.

    LangChain's built-in OpenAI embedding class does not send NVIDIA's
    required `input_type` field, so this small wrapper adds it. It implements
    the two methods FAISS and BM25 call: one for chunks, one for queries.
    """

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        """
        Store the connection details and create the HTTP client.

        Args:
            model: NVIDIA embedding model ID, for example
                "nvidia/nemotron-3.5-embed-1b".
            api_key: Your NVIDIA API key.
            base_url: NVIDIA's OpenAI-compatible base URL.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """
        Call the embedding endpoint in batches.

        Args:
            texts: Strings to embed.
            input_type: "passage" for document chunks, "query" for questions.

        Returns:
            One vector per input string, in the same order.
        """
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start:start + EMBEDDING_BATCH_SIZE]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed document chunks (used when building the index).

        Args:
            texts: The chunk texts.

        Returns:
            One vector per chunk.
        """
        return self._embed(texts, input_type="passage")

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a user question (used when searching).

        Args:
            text: The question text.

        Returns:
            A single vector.
        """
        return self._embed([text], input_type="query")[0]


def _nvidia_chat(model: str) -> ChatOpenAI:
    """
    Build a chat model that talks to NVIDIA NIM.

    Args:
        model: NVIDIA model ID.

    Returns:
        A LangChain chat model.
    """
    return ChatOpenAI(
        model=model,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Return the chat model used for every text answer and query rewrite.

    NVIDIA is the primary model. If the Groq key is set, Groq is attached as a
    backup: LangChain calls it only when the NVIDIA call raises an error
    (outage, quota, invalid model). Without a Groq key, NVIDIA alone is used.

    Returns:
        A LangChain runnable chat model.

    Raises:
        ValueError: If NVIDIA_API_KEY is missing.
    """
    config.check_llm_keys()

    primary = _nvidia_chat(config.NVIDIA_CHAT_MODEL)

    if not config.groq_backup_enabled():
        return primary

    backup = ChatOpenAI(
        model=config.GROQ_CHAT_MODEL,
        api_key=config.GROQ_API_KEY,
        base_url=config.GROQ_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=60,
        max_retries=1,
    )
    return primary.with_fallbacks([backup])


@lru_cache(maxsize=1)
def get_vision_model() -> ChatOpenAI:
    """
    Return the vision-capable model used to describe images and to answer
    questions that include images.

    Vision has no backup: Groq's free vision options are too small for this
    workload, so if NVIDIA is down the image step reports the error instead.

    Returns:
        A LangChain chat model that accepts image content parts.

    Raises:
        ValueError: If NVIDIA_API_KEY is missing.
    """
    config.check_llm_keys()
    return _nvidia_chat(config.NVIDIA_VISION_MODEL)


@lru_cache(maxsize=1)
def get_embedding_model() -> NVIDIAEmbeddings:
    """
    Return the embedding model used to build and search the vector index.

    Returns:
        An NVIDIAEmbeddings instance.

    Raises:
        ValueError: If NVIDIA_API_KEY is missing.
    """
    config.check_llm_keys()
    return NVIDIAEmbeddings(
        model=config.NVIDIA_EMBEDDING_MODEL,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
    )
