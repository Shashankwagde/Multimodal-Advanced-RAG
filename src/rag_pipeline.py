import base64
import io
import logging
from typing import Dict, List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import Image

import config
from llm_client import get_chat_model, get_vision_model
from src.chunking import build_parent_lookup
from src.indexing import load_doc_store, load_vector_store
from src.memory import format_history, get_recent_messages
from src.refinement import RefinedContext, refine_context
from src.reranking import rerank_documents
from src.retrieval import (
    build_keyword_retriever,
    generate_query_variants,
    hybrid_retrieve,
    recover_parents,
)

logger = logging.getLogger(__name__)

# Images are shrunk before they are sent with the question, to keep the
# token cost low on free tiers.
ANSWER_IMAGE_MAX_SIDE = 1024

# The rules the model follows. "Only use the context" is still the most
# important line: it stops the model answering from general knowledge.
ANSWER_RULES = (
    "You are a helpful assistant answering questions about internal "
    "company documents.\n\n"
    "Rules:\n"
    "- Answer ONLY using the context below and the attached figures, if any.\n"
    "- If the context does not contain the answer, say so plainly instead of guessing.\n"
    "- Cite text sources with their numbers, like [1] or [2].\n"
    "- Cite figures with their label, like [Figure 1].\n"
    "- Figures are images attached to the question. Read their content, labels and "
    "numbers carefully when they help answer the question.\n"
    "- Keep the answer clear and concise."
)


def load_pipeline() -> Dict[str, object]:
    """
    Load everything the pipeline needs to answer questions.

    Returns:
        A dictionary with three keys:
            "vector_store"      - FAISS or Pinecone, for semantic search.
            "keyword_retriever" - BM25, for exact word matching.
            "parent_lookup"     - parent_id -> parent Document.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
    parents, children = load_doc_store()
    if not children:
        raise FileNotFoundError(
            "No indexed documents found. Build the vector database first."
        )

    return {
        "vector_store": load_vector_store(),
        "keyword_retriever": build_keyword_retriever(children),
        "parent_lookup": build_parent_lookup(parents),
    }


def answer_question(
    question: str,
    pipeline: Dict[str, object],
    memory: InMemoryChatMessageHistory,
) -> Dict[str, object]:
    """
    Run one question through the complete multimodal RAG pipeline.

    Args:
        question: What the user typed.
        pipeline: The objects returned by `load_pipeline()`.
        memory: The conversation history for this session.

    Returns:
        A dictionary with:
            "answer"  - the generated answer text.
            "sources" - citation labels for text and figures.
            "images"  - file paths of the figures shown with the answer.
            "debug"   - what happened at each step.

    Raises:
        ValueError: If the question is empty.
    """
    if not question or not question.strip():
        raise ValueError("Please type a question first.")

    # Step 1: rewrite the question a few ways, using the conversation so far.
    history_text = format_history(memory)
    queries = generate_query_variants(question, history_text)

    # Step 2: hybrid retrieval over text and figure-description chunks.
    candidate_children = hybrid_retrieve(
        queries,
        pipeline["vector_store"],
        pipeline["keyword_retriever"],
    )

    if not candidate_children:
        return {
            "answer": "I could not find anything related to that in the indexed documents.",
            "sources": [],
            "images": [],
            "debug": {
                "original_query": question,
                "generated_queries": queries,
                "retrieved": [],
                "reranked": [],
                "final_context": [],
            },
        }

    # Step 3: swap each small child for its larger parent.
    candidate_parents = recover_parents(candidate_children, pipeline["parent_lookup"])

    # Step 4: rerank the parents against the question.
    reranked = rerank_documents(question, candidate_parents)

    # Step 5: drop weak chunks, respect the budget, split text from figures.
    refined = refine_context(reranked)

    # Step 6: ask the model, attaching the figure images.
    answer = generate_answer(question, refined, memory)

    return {
        "answer": answer,
        "sources": refined.citations,
        "images": refined.image_paths,
        "debug": {
            "original_query": question,
            "generated_queries": queries,
            "retrieved": summarise_documents(candidate_children),
            "reranked": summarise_documents(reranked),
            "final_context": summarise_documents(refined.selected),
        },
    }


def generate_answer(
    question: str,
    refined: RefinedContext,
    memory: InMemoryChatMessageHistory,
) -> str:
    """
    Answer the question from the text context and any attached figures.

    When figures were retrieved, the vision model gets the question, the
    images and the context. If that call fails, the answer is produced from
    the text context alone, using the chat model (with its Groq backup).

    Args:
        question: The user's original question.
        refined: The refined context from `refine_context`.
        memory: Conversation history, for natural follow-up answers.

    Returns:
        The answer text, or a readable error message if every call fails.
    """
    history = get_recent_messages(memory)
    system_text = f"{ANSWER_RULES}\n\nContext:\n{refined.text_context or '(no context)'}"

    if refined.image_paths:
        try:
            messages: List[BaseMessage] = [SystemMessage(content=system_text), *history]
            messages.append(HumanMessage(content=build_multimodal_content(question, refined.image_paths)))
            return content_to_text(get_vision_model().invoke(messages).content)
        except Exception as error:
            logger.warning(
                "Vision answer failed (%s). Answering from text context only.", error
            )
            system_text += (
                "\n\nNote: the figures could not be shown to you in this answer. "
                "Use only the figure descriptions in the context."
            )

    try:
        messages = [SystemMessage(content=system_text), *history, HumanMessage(content=question)]
        return content_to_text(get_chat_model().invoke(messages).content)
    except Exception as error:
        logger.error("Answer generation failed: %s", error)
        return (
            "Sorry, the model request failed. Check your NVIDIA_API_KEY, your "
            "internet connection and your usage limits.\n\n"
            f"Details: {error}"
        )


def build_multimodal_content(question: str, image_paths: List[str]) -> List[dict]:
    """
    Build the content list for a message that has text and images.

    Args:
        question: The user's question.
        image_paths: Files of the figures to attach, in ranked order.

    Returns:
        A list of content parts: one text part followed by one image part per
        figure, each image as a base64 data URL.
    """
    parts: List[dict] = [{"type": "text", "text": question}]
    for path in image_paths:
        parts.append({"type": "image_url", "image_url": {"url": encode_image(path)}})
    return parts


def encode_image(path: str) -> str:
    """
    Read an image file, shrink it and return it as a base64 JPEG data URL.

    Args:
        path: Path to a PNG, JPG, JPEG or WEBP file.

    Returns:
        A string such as "data:image/jpeg;base64,...".
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((ANSWER_IMAGE_MAX_SIDE, ANSWER_IMAGE_MAX_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def content_to_text(content) -> str:
    """
    Turn a model reply into plain text.

    Some providers return a string; others return a list of parts. This keeps
    only the text.

    Args:
        content: The `content` field of a model reply.

    Returns:
        The reply as a single string.
    """
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        ).strip()
    return str(content).strip()


def summarise_documents(documents: List[Document], preview_chars: int = 300) -> List[dict]:
    """
    Shrink documents into small dictionaries for the debug panel.

    Args:
        documents: Documents from any pipeline step.
        preview_chars: How much of the text to show.

    Returns:
        A list of dictionaries with "source", "score" and "preview" keys.
    """
    from src.refinement import describe_figure, describe_source, is_image

    summaries = []
    for position, document in enumerate(documents, start=1):
        if is_image(document):
            source = describe_figure(document, position)
        else:
            source = describe_source(document)

        score = document.metadata.get("rerank_score")
        summaries.append(
            {
                "source": source,
                "score": round(score, 3) if score is not None else None,
                "preview": document.page_content[:preview_chars].strip(),
                "image_path": document.metadata.get("image_path"),
            }
        )
    return summaries
