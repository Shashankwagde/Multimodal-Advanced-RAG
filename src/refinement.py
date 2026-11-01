"""
Step 6 of the pipeline: decide what actually gets sent to the LLM.

Reranking gave us an ordered list. Context refinement answers the follow-up
question: "should all of these really go into the prompt?"

We do four things:

1. Drop documents the reranker scored as clearly irrelevant.
2. Stop once we hit a character budget, so the prompt stays small and cheap.
3. Split the survivors into text chunks and figures. Figures also contribute
   their image file, up to MAX_IMAGES_PER_ANSWER.
4. Format everything with citation labels: [1], [2] for text and
   [Figure 1: page 3 of report.pdf] for figures.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


@dataclass
class RefinedContext:
    """
    Everything the answer step needs, split by type.

    Attributes:
        selected: The documents that survived filtering and budgeting, best
            first. Used for the citation list and the debug panel.
        text_context: The formatted text block (text chunks and figure
            descriptions) that goes into the prompt.
        image_paths: Files of the figures to send to the vision model, in the
            order they were ranked. Capped at MAX_IMAGES_PER_ANSWER.
        citations: One label per selected document, in the same order as
            `selected`. Used under the answer in the UI.
    """

    selected: List[Document]
    text_context: str
    image_paths: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


def describe_source(document: Document) -> str:
    """
    Build a short, human-readable label describing where a chunk came from.

    Args:
        document: A chunk with metadata from the ingestion step.

    Returns:
        A string such as "handbook.pdf - Page 4" or
        "handbook.pdf - Section: Password Policy". When neither a page nor a
        section is known, just the filename.
    """
    filename = document.metadata.get("filename", "unknown file")

    page_number = document.metadata.get("page_number")
    if page_number:
        return f"{filename} - Page {page_number}"

    section = document.metadata.get("section")
    if section:
        return f"{filename} - Section: {section}"

    return filename


def describe_figure(document: Document, figure_number: int) -> str:
    """
    Build the citation label for a figure.

    Args:
        document: An image chunk with `filename` and optionally `page_number`.
        figure_number: The figure's position among the figures in this answer.

    Returns:
        A label such as "Figure 1: page 3 of report.pdf", or
        "Figure 1: image.png" for a standalone image.
    """
    filename = document.metadata.get("filename", "unknown file")
    page_number = document.metadata.get("page_number")
    if page_number:
        return f"Figure {figure_number}: page {page_number} of {filename}"
    return f"Figure {figure_number}: {filename}"


def is_image(document: Document) -> bool:
    """
    Tell whether a chunk describes a figure rather than text.

    Args:
        document: Any chunk from the index.

    Returns:
        True for image chunks.
    """
    return document.metadata.get("type") == "image"


def drop_irrelevant(documents: List[Document]) -> List[Document]:
    """
    Remove chunks the reranker scored below the relevance threshold.

    If every chunk is below the threshold we keep the single best one, so the
    LLM can still say "the documents do not cover this" based on real text.

    Args:
        documents: Reranked documents carrying a `rerank_score`.

    Returns:
        Only the documents worth reading.
    """
    relevant = [
        document
        for document in documents
        if document.metadata.get("rerank_score", 0.0) >= config.MIN_RERANK_SCORE
    ]

    if not relevant and documents:
        logger.info("All chunks scored below the threshold; keeping the best one.")
        return documents[:1]

    return relevant


def apply_context_budget(documents: List[Document]) -> List[Document]:
    """
    Keep adding documents until the character budget runs out.

    Long prompts cost more and make answers worse. Because the list is already
    sorted best-first, cutting from the end removes the least useful text.

    Args:
        documents: Relevant documents, best first.

    Returns:
        The documents that fit inside config.MAX_CONTEXT_CHARS.
    """
    selected: List[Document] = []
    used_characters = 0

    for document in documents:
        length = len(document.page_content)
        if used_characters + length > config.MAX_CONTEXT_CHARS:
            continue
        selected.append(document)
        used_characters += length

    # Never return nothing: if even the first document is larger than the
    # whole budget, send a trimmed version of it rather than empty context.
    if not selected and documents:
        first = documents[0]
        selected.append(
            Document(
                page_content=first.page_content[: config.MAX_CONTEXT_CHARS],
                metadata=first.metadata,
            )
        )

    return selected


def refine_context(documents: List[Document]) -> RefinedContext:
    """
    Run the full refinement step: filter, budget, split, then format.

    Args:
        documents: Reranked documents, text and images mixed.

    Returns:
        A RefinedContext with the text block for the prompt, the figure files
        to attach, and one citation label per selected document.
    """
    relevant = drop_irrelevant(documents)
    selected = apply_context_budget(relevant)

    blocks: List[str] = []
    citations: List[str] = []
    image_paths: List[str] = []

    text_number = 0
    figure_number = 0

    for document in selected:
        if is_image(document):
            figure_number += 1
            label = describe_figure(document, figure_number)
            citations.append(label)
            blocks.append(
                f"[{label}]\n{document.page_content}"
            )

            image_path = document.metadata.get("image_path")
            if (
                image_path
                and image_path not in image_paths
                and len(image_paths) < config.MAX_IMAGES_PER_ANSWER
            ):
                image_paths.append(image_path)
        else:
            text_number += 1
            label = describe_source(document)
            citations.append(label)
            blocks.append(
                f"[{text_number}] Source: {label}\n{document.page_content}"
            )

    return RefinedContext(
        selected=selected,
        text_context="\n\n".join(blocks),
        image_paths=image_paths,
        citations=citations,
    )
