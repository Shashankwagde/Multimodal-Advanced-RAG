"""
Step 2 of the pipeline: split documents into parent and child chunks.

Text documents are split the same way as before: large parent chunks for the
LLM, small child chunks for searching.

Image documents (figures from PDFs and standalone images) have no text, so the
vision model describes each one. That description is stored as a single
chunk: it is searched like text, and it points back to the image file through
`image_path`. The answer step later sends the image itself to the model.

Descriptions are cached in `vectorstore/image_descriptions.json`, keyed by a
hash of the image bytes and the model name. Rebuilding the index does not call
the vision model again for images that have not changed.
"""

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except OSError as e:
    raise ImportError(f"Error while import in chunking module: {e}")

import config
from llm_client import get_vision_model
from src.ingestion import guess_section

logger = logging.getLogger(__name__)

IMAGE_DESCRIPTION_PROMPT = (
    "Describe this image so that someone could find it by searching for its "
    "content. Include: the main subject, every visible text label or heading "
    "(copy it exactly), numbers and units from charts or tables, the relation "
    "between parts of a diagram, and any conclusion the image shows. Be "
    "factual. Do not guess at details you cannot see. Keep it under 200 words."
)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Vision description with a disk cache
# ---------------------------------------------------------------------------

def _load_image_cache() -> Dict[str, str]:
    """
    Read the saved image descriptions from disk.

    Returns:
        A dictionary of cache key -> description. Empty if no cache exists yet
        or the file cannot be read.
    """
    cache_path = config.IMAGE_CACHE_PATH
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Image description cache is unreadable, starting fresh: %s", error)
        return {}


def _save_image_cache(cache: Dict[str, str]) -> None:
    """
    Write the image description cache to disk.

    Args:
        cache: Cache key -> description mapping.
    """
    cache_path = config.IMAGE_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def describe_image(image_path: str) -> str:
    """
    Ask the vision model for a searchable description of one image.

    The result is cached. The cache key includes the image bytes, the model
    name and the prompt, so changing the image, NVIDIA_VISION_MODEL or the
    prompt produces a fresh description instead of a stale one.

    Args:
        image_path: Path to a PNG, JPG, JPEG or WEBP file.

    Returns:
        The description text.

    Raises:
        ValueError: If the file type is not supported.
        Exception: Any error from the vision model call. The caller decides
            whether to skip the image.
    """
    path = Path(image_path)
    mime_type = MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image type for {path.name}")

    image_bytes = path.read_bytes()
    model_name = config.NVIDIA_VISION_MODEL
    cache_key = hashlib.sha256(
        image_bytes + model_name.encode("utf-8") + IMAGE_DESCRIPTION_PROMPT.encode("utf-8")
    ).hexdigest()

    cache = _load_image_cache()
    if cache_key in cache:
        return cache[cache_key]

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    response = get_vision_model().invoke([message])

    content = response.content
    if isinstance(content, list):
        # Some providers return a list of parts instead of a plain string.
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    description = str(content).strip()

    if description:
        cache[cache_key] = description
        _save_image_cache(cache)

    return description


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter that tries to cut text at meaningful boundaries.

    "Meaning-aware" splitting here means the splitter prefers to break at a
    paragraph break, then at a line break, then at the end of a sentence, and
    only chops mid-word as a last resort. That keeps ideas intact inside a
    chunk instead of slicing a sentence in half.

    Args:
        chunk_size: Target chunk length in characters.
        chunk_overlap: How many characters neighbouring chunks share, so a
            sentence sitting on a boundary is not lost.

    Returns:
        A configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
        length_function=len,
    )


def _add_image_chunk(
    document: Document,
    parents: List[Document],
    children: List[Document],
    parents_per_file: Dict[str, int],
) -> None:
    """
    Turn one image Document into a parent and a child chunk.

    The image description is both the searchable child and the parent the LLM
    reads. Images that the vision model cannot describe are skipped with a
    warning, so one bad file does not stop the whole build.

    Args:
        document: An image Document from `src.ingestion`.
        parents: List to append the parent chunk to.
        children: List to append the child chunk to.
        parents_per_file: Running parent counter per file, shared with text.
    """
    image_path = document.metadata.get("image_path")
    filename = document.metadata.get("filename", "unknown")
    if not image_path:
        logger.warning("Image document from %s has no image_path. Skipping.", filename)
        return

    try:
        description = describe_image(image_path)
    except Exception as error:
        logger.warning("Could not describe %s (%s). Skipping this image.", Path(image_path).name, error)
        return

    if not description:
        logger.warning("Empty description for %s. Skipping this image.", Path(image_path).name)
        return

    parent_number = parents_per_file.get(filename, 0)
    parents_per_file[filename] = parent_number + 1
    parent_id = f"{filename}::{parent_number}"

    parent_metadata = dict(document.metadata)
    parent_metadata["parent_id"] = parent_id
    parent_metadata["section"] = ""

    parents.append(Document(page_content=description, metadata=parent_metadata))

    child_metadata = dict(parent_metadata)
    child_metadata["chunk_id"] = f"{parent_id}::0"
    children.append(Document(page_content=description, metadata=child_metadata))


def split_documents(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    """
    Turn loaded documents into large parent chunks and small child chunks.

    This is the core of parent-child retrieval. Each child carries a
    `parent_id` in its metadata, which is the link we follow later to recover
    the bigger context. Image documents get one vision description each.

    Args:
        documents: Documents from `src.ingestion`, text and image.

    Returns:
        A tuple `(parents, children)`:
            parents  - large chunks, used as the context given to the LLM.
            children - small chunks, the ones we actually embed and search.

    Example:
        A 5-page handbook may produce 12 parents and 48 children. Searching
        finds child #37; the LLM reads parent "handbook.pdf::8" instead.
    """
    parent_splitter = make_splitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
    child_splitter = make_splitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP)

    parents: List[Document] = []
    children: List[Document] = []

    # We count parents separately for each file. Numbering per file (rather
    # than across the whole batch) means a chunk always gets the same id no
    # matter which other documents were processed alongside it - which is what
    # lets `update_index` recognise and skip a file it has already indexed.
    parents_per_file: Dict[str, int] = {}

    for document in documents:
        if document.metadata.get("type") == "image":
            _add_image_chunk(document, parents, children, parents_per_file)
            continue

        filename = document.metadata.get("filename", "unknown")

        for parent_text in parent_splitter.split_text(document.page_content):
            parent_number = parents_per_file.get(filename, 0)
            parents_per_file[filename] = parent_number + 1
            parent_id = f"{filename}::{parent_number}"

            parent_metadata = dict(document.metadata)
            parent_metadata["parent_id"] = parent_id
            parent_metadata["section"] = guess_section(parent_text)

            parents.append(Document(page_content=parent_text, metadata=parent_metadata))

            for child_index, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_metadata = dict(parent_metadata)
                child_metadata["chunk_id"] = f"{parent_id}::{child_index}"
                children.append(Document(page_content=child_text, metadata=child_metadata))

    return parents, children


def build_parent_lookup(parents: List[Document]) -> Dict[str, Document]:
    """
    Build a dictionary that maps a parent_id to its parent Document.

    Retrieval gives us child chunks; we need an instant way to fetch the
    matching parent. A dictionary lookup is O(1), which is as simple and fast
    as this problem needs to be.

    Args:
        parents: The list of parent chunks.

    Returns:
        A dictionary like {"handbook.pdf::3": Document(...), ...}.
    """
    return {parent.metadata["parent_id"]: parent for parent in parents}
