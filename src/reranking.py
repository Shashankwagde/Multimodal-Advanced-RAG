"""
Step 5 of the pipeline: reorder the candidates with a reranker.

Why do we need this at all? Retrieval compares two vectors that were created
*separately* - one for the question, one for the chunk. It is fast, but it is
only an approximation of relevance.

A reranker reads the question and each candidate TOGETHER and scores how well
the candidate answers the question. That is more accurate, but too slow to
run over thousands of chunks. So we use the two-stage setup:

    fast retrieval  -> ~10 candidates  -> accurate reranker -> best 4

Three reranker options, chosen with RERANKER in .env:

    nvidia         (default) NVIDIA's hosted reranker. No PyTorch needed,
                   runs on the same free NVIDIA key as chat and embeddings.
    cross-encoder  A local model. Needs sentence-transformers and PyTorch.
    simple         Keyword overlap. No extra packages, least accurate.

If the chosen reranker fails for one request (network error, rate limit,
missing package), this module logs a warning and falls back to the keyword
reranker for that request, so the app keeps answering.
"""

import logging
import re
from collections import Counter
from functools import lru_cache
from typing import List

import requests
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)

# NVIDIA serves rerankers from a retrieval endpoint, not from the chat
# /v1/models list.
NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/{model}/reranking"
NVIDIA_RERANK_TIMEOUT = 30

# Why the most recent request fell back to the keyword reranker, if it did.
# Shown in the sidebar so the user can see what is really running.
_last_fallback_reason = ""

# Words that appear in almost every sentence and so say nothing about
# relevance. Removing them stops "the" and "is" from inflating the score.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int | None = None,
) -> List[Document]:
    """
    Score every candidate against the query and keep the best ones.

    Each document gets a `rerank_score` in its metadata so the refinement step
    and the debug panel can see how confident the reranker was. Image chunks
    are scored on their description, so they compete fairly with text.

    Args:
        query: The user's original question.
        documents: Candidate documents from hybrid retrieval.
        top_k: How many documents to keep. Defaults to config.RERANK_TOP_K.

    Returns:
        The documents sorted from most to least relevant, trimmed to `top_k`.
    """
    global _last_fallback_reason

    if not documents:
        return []

    keep = top_k or config.RERANK_TOP_K

    if config.RERANKER == "simple":
        _last_fallback_reason = ""
        return keyword_rerank(query, documents, keep)

    if config.RERANKER == "cross-encoder":
        return cross_encoder_rerank(query, documents, keep)

    # Default: NVIDIA's reranker.
    try:
        scores = nvidia_rerank_scores(query, documents)
    except Exception as error:
        _last_fallback_reason = str(error).split("\n")[0][:120]
        logger.warning(
            "NVIDIA reranker failed (%s). Using keyword reranking for this question.",
            _last_fallback_reason,
        )
        return keyword_rerank(query, documents, keep)

    _last_fallback_reason = ""
    for document, score in zip(documents, scores):
        document.metadata["rerank_score"] = float(score)
    return sort_by_score(documents, keep)


# ---------------------------------------------------------------------------
# Option 1: NVIDIA reranker (default)
# ---------------------------------------------------------------------------

def nvidia_rerank_scores(query: str, documents: List[Document]) -> List[float]:
    """
    Ask NVIDIA's reranker to score each document against the query.

    The API returns a ranking: a list of {"index", "logit"} objects, where
    "index" points back to the passage and "logit" is its relevance score.
    We map those scores back onto the original document order.

    Args:
        query: The user's question.
        documents: Candidate documents. Their page_content is sent as passages.

    Returns:
        One score per document, in the same order as `documents`.

    Raises:
        requests.HTTPError: If NVIDIA returns an error status.
        ValueError: If the reply does not contain a complete ranking.
    """
    url = NVIDIA_RERANK_URL.format(model=config.NVIDIA_RERANK_MODEL)
    payload = {
        "model": config.NVIDIA_RERANK_MODEL,
        "query": {"text": query},
        "passages": [{"text": document.page_content} for document in documents],
        # Long passages are cut at the end instead of failing the request.
        "truncate": "END",
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
            "Accept": "application/json",
        },
        json=payload,
        timeout=NVIDIA_RERANK_TIMEOUT,
    )
    response.raise_for_status()

    rankings = response.json().get("rankings", [])
    if len(rankings) != len(documents):
        raise ValueError(
            f"Reranker returned {len(rankings)} scores for {len(documents)} passages."
        )

    scores = [0.0] * len(documents)
    for item in rankings:
        scores[int(item["index"])] = float(item["logit"])
    return scores


# ---------------------------------------------------------------------------
# Option 2: local cross-encoder (needs PyTorch)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_cross_encoder():
    """
    Load the local cross-encoder model once per application run.

    The model is about 90 MB and is downloaded the first time it is used.

    Returns:
        A sentence-transformers CrossEncoder model.

    Raises:
        ImportError: If `sentence-transformers` is not installed.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is not installed "
            "(pip install sentence-transformers)"
        ) from error

    logger.info("Loading cross-encoder: %s", config.CROSS_ENCODER_MODEL)
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def cross_encoder_rerank(query: str, documents: List[Document], keep: int) -> List[Document]:
    """
    Rerank with the local cross-encoder, falling back to keywords on failure.

    Args:
        query: The user's question.
        documents: Candidate documents.
        keep: How many documents to return.

    Returns:
        The best `keep` documents, best first.
    """
    global _last_fallback_reason

    try:
        model = get_cross_encoder()
        pairs = [(query, document.page_content) for document in documents]
        scores = model.predict(pairs)
    except Exception as error:
        # Missing package, a Windows DLL error or a failed download all end
        # the same way: keep the app running with the keyword reranker.
        _last_fallback_reason = str(error).split("\n")[0][:120]
        logger.warning(
            "Cross-encoder unavailable (%s). Using keyword reranking.",
            _last_fallback_reason,
        )
        return keyword_rerank(query, documents, keep)

    _last_fallback_reason = ""
    for document, score in zip(documents, scores):
        document.metadata["rerank_score"] = float(score)
    return sort_by_score(documents, keep)


# ---------------------------------------------------------------------------
# Option 3: keyword reranker (no extra packages)
# ---------------------------------------------------------------------------

def keyword_rerank(query: str, documents: List[Document], keep: int) -> List[Document]:
    """
    Rerank by counting how many of the question's words appear in the chunk.

    This is the no-dependency fallback. It understands words, not meaning, but
    it keeps the pipeline shape intact so the rest of the project behaves the
    same way.

    Args:
        query: The user's original question.
        documents: Candidate documents from hybrid retrieval.
        keep: How many documents to return.

    Returns:
        The documents sorted by keyword overlap, trimmed to `keep`.
    """
    for document in documents:
        document.metadata["rerank_score"] = keyword_score(query, document.page_content)

    return sort_by_score(documents, keep)


def keyword_score(query: str, text: str) -> float:
    """
    Score one chunk by the share of question words it contains.

    The result is mapped onto a rough scale similar to the other rerankers
    (about -5 for junk, about +5 for a strong match), so refinement and
    MIN_RERANK_SCORE keep working.

    Args:
        query: The user's question.
        text: The chunk text.

    Returns:
        A score between about -5.0 and +5.8.
    """
    query_words = tokenize(query)
    if not query_words:
        return 0.0

    # Coverage decides the score; repeats only break ties.
    found = Counter(
        word for word in re.findall(r"[a-z0-9]+", text.lower()) if word in query_words
    )
    coverage = len(found) / len(query_words)
    repeats = min(sum(found.values()) - len(found), 10) * 0.08

    return 10.0 * coverage - 5.0 + repeats


def tokenize(text: str) -> set:
    """
    Split text into a set of meaningful lowercase words.

    Args:
        text: Any piece of text.

    Returns:
        A set of words, without punctuation, stop words or single characters.

    Example:
        "What is the refund policy?" -> {"refund", "policy"}
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if len(word) > 1 and word not in STOP_WORDS}


def sort_by_score(documents: List[Document], keep: int) -> List[Document]:
    """
    Sort documents by their `rerank_score`, highest first.

    Args:
        documents: Documents that already have a `rerank_score` in metadata.
        keep: How many of the best documents to return.

    Returns:
        The top `keep` documents, best first.
    """
    ranked = sorted(
        documents,
        key=lambda document: document.metadata.get("rerank_score", 0.0),
        reverse=True,
    )
    return ranked[:keep]


def reranker_status() -> str:
    """
    Describe which reranker is actually being used right now.

    The Streamlit sidebar shows this so the user can tell whether the hosted
    reranker is working or the app has fallen back.

    Returns:
        A short status string.
    """
    if config.RERANKER == "simple":
        return "simple keyword (set by RERANKER=simple)"
    if config.RERANKER == "cross-encoder":
        return f"cross-encoder ({config.CROSS_ENCODER_MODEL})"
    if _last_fallback_reason:
        return f"NVIDIA {config.NVIDIA_RERANK_MODEL} (last request fell back: {_last_fallback_reason[:60]})"
    return f"NVIDIA ({config.NVIDIA_RERANK_MODEL})"
