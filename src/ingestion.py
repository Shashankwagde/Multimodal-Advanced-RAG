import logging
import re
from pathlib import Path
from typing import List

import pymupdf as fitz  # PyMuPDF: reads PDFs and extracts embedded images
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from PIL import Image

import config

logger = logging.getLogger(__name__)

# Text-bearing files are read into text chunks.
SUPPORTED_TEXT_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Image files become image documents. Their meaning is captured later by the
# vision model (step 7), not here.
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

# Images smaller than this (in pixels, on either side) are usually logos,
# bullets or decorations. Skipping them saves vision-model calls and keeps
# the index free of noise.
MIN_IMAGE_SIDE = 80


def load_document(file_path: str | Path) -> List[Document]:
    """
    Load a single file into LangChain Documents.

    Text files (PDF, TXT, DOCX) return text Documents. Images embedded in a
    PDF are extracted to `data/extracted_images/` and returned as image
    Documents. Standalone image files (PNG, JPG, JPEG, WEBP) return one image
    Document each.

    Image Documents have an empty `page_content`. The vision model fills in
    the description later, in `src/chunking.py`.

    Args:
        file_path: Path to the file on disk.

    Returns:
        A list of Documents. A PDF returns one text Document per page plus one
        image Document per extracted figure.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    extension = path.suffix.lower()  # "security_policy.pdf" -> .pdf

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not path.exists():
        raise FileNotFoundError(f"File not found here: {path}")

    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return load_standalone_image(path)

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    cleaned_documents: List[Document] = []
    for doc in loader.load():
        doc.page_content = clean_text(doc.page_content)
        if doc.page_content.strip():
            cleaned_documents.append(doc)

    text_documents = add_metadata(cleaned_documents, path)

    if extension == ".pdf":
        image_documents = extract_pdf_images(path)
    else:
        image_documents = []

    if not text_documents and not image_documents:
        logger.warning("No readable text or images found in %s. Skipping it.", path.name)
        return []

    if not text_documents:
        logger.info("%s has no text layer, only images.", path.name)

    return text_documents + image_documents


def load_standalone_image(path: Path) -> List[Document]:
    """
    Turn a standalone image file (PNG, JPG, JPEG, WEBP) into an image Document.

    The file is checked with Pillow first, so a corrupt or renamed file is
    skipped with a warning instead of breaking the whole upload.

    Args:
        path: Path to the image file.

    Returns:
        A list with one image Document, or an empty list if the image is
        unreadable or too small to be useful.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as error:
        logger.warning("Could not read image %s: %s", path.name, error)
        return []

    if min(width, height) < MIN_IMAGE_SIDE:
        logger.info("Skipping small image %s (%dx%d).", path.name, width, height)
        return []

    document = Document(
        page_content="",
        metadata={
            "type": "image",
            "image_path": str(path),
            "source": str(path),
            "filename": path.name,
            "source_doc": path.name,
            "doc_type": path.suffix.lower().lstrip("."),
        },
    )
    return [document]


def extract_pdf_images(path: Path) -> List[Document]:
    """
    Save every sizeable image inside a PDF as a PNG file and describe it.

    Each image is saved as `data/extracted_images/<pdf-name>_p<page>_img<n>.png`.
    The name is built from the PDF name and the page number, so re-running the
    extraction overwrites the same files instead of creating duplicates.

    Args:
        path: Path to the PDF file.

    Returns:
        One image Document per extracted figure. Each has `image_path`,
        `source_doc`, `filename`, `page_number` and `type="image"` in its
        metadata. Returns an empty list if the PDF cannot be opened.
    """
    try:
        pdf = fitz.open(str(path))
    except Exception as error:
        logger.warning("Could not open %s for image extraction: %s", path.name, error)
        return []

    config.EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_documents: List[Document] = []
    seen_xrefs = set()

    with pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]

                # The same picture can appear on several pages; keep the first copy.
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    pixmap = fitz.Pixmap(pdf, xref)
                except Exception as error:
                    logger.warning(
                        "Skipping an image on page %d of %s: %s",
                        page_number, path.name, error,
                    )
                    continue

                if min(pixmap.width, pixmap.height) < MIN_IMAGE_SIDE:
                    continue

                # PNG needs RGB or grayscale. CMYK and other colour spaces are
                # converted to RGB first, so the file opens everywhere.
                if pixmap.n - pixmap.alpha >= 4:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

                output_path = config.EXTRACTED_IMAGES_DIR / (
                    f"{path.stem}_p{page_number}_img{image_index}.png"
                )
                pixmap.save(str(output_path))

                image_documents.append(
                    Document(
                        page_content="",
                        metadata={
                            "type": "image",
                            "image_path": str(output_path),
                            "source": str(path),
                            "filename": path.name,
                            "source_doc": path.name,
                            "doc_type": "pdf",
                            "page_number": page_number,
                        },
                    )
                )

    if image_documents:
        logger.info("Extracted %d image(s) from %s", len(image_documents), path.name)
    return image_documents


def load_documents_from_folder(folder_path: str | Path) -> List[Document]:
    """
    Load every supported document inside a folder.

    This is what the "Build Vector DB" button and the build script use to read
    the whole `data/documents` folder in one go.

    Args:
        folder_path: Folder to scan (not recursive, one level only).

    Returns:
        A combined list of Documents from all readable files. Files that fail
        to load are skipped with a warning instead of stopping the whole run.
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning("Document folder does not exist: %s", folder)
        return []

    all_documents: List[Document] = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            all_documents.extend(load_document(file_path))
        except Exception as error:
            logger.warning("Could not load %s: %s", file_path.name, error)

    logger.info("Loaded %d documents from %s", len(all_documents), folder)
    return all_documents


def clean_text(text: str) -> str:
    """
    Remove the noise that PDF and DOCX extraction usually leaves behind.

    Raw extracted text is full of stray line breaks, repeated spaces and
    hyphenated words split across lines. Cleaning it first means the chunks we
    embed are closer to real sentences, which improves retrieval quality.

    Args:
        text: Raw text straight out of a document loader.

    Returns:
        The same text, tidied up.
    """
    if not text:
        return ""

    # Ex: "secu-\nrity" -> "security"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def add_metadata(documents: List[Document], path: Path) -> List[Document]:
    """
    Attach source information to each text Document so we can cite it later.

    Retrieval returns a piece of text with no memory of where it came from.
    By storing the filename, file type and page number now, the final answer
    can say "technova_employee_handbook.pdf - Page 4".

    Args:
        documents: Text Documents produced by a loader.
        path: Path of the file they were loaded from.

    Returns:
        The same documents, with extra metadata keys filled in.
    """
    for document in documents:
        document.metadata["type"] = "text"
        document.metadata["source"] = str(path)
        document.metadata["filename"] = path.name
        document.metadata["doc_type"] = path.suffix.lower().lstrip(".")

        if "page" in document.metadata:
            document.metadata["page_number"] = int(document.metadata["page"]) + 1

    return documents


def guess_section(text: str) -> str:
    """
    Guess the section heading a piece of text belongs to.

    Policy documents are written as "Password Policy", "Refund Policy" and so
    on. Capturing that heading lets a citation say "Section: Password Policy",
    which is far more useful to a reader than a filename alone.

    Args:
        text: A chunk of document text.

    Returns:
        The heading we found, or an empty string if the text has no obvious
        heading. We deliberately keep this simple: a wrong guess only affects
        the citation label, never the answer itself.
    """
    for raw_line in text.split("\n")[:15]:
        line = raw_line.strip()

        # A heading is short, has a handful of words and does not end like a
        # sentence. "4. PASSWORD POLICY" passes; "We reset passwords." fails.
        if not (3 < len(line) <= 70) or line.endswith((".", ",", ";", ":")):
            continue
        if len(line.split()) > 8:
            continue

        # Does this line start like a numbered list item?
        # Examples: "1. Hello", "2) World", "10.3.4 item"
        looks_numbered = re.match(r"^\d+(\.\d+)*[.)]?\s+\S", line) is not None
        if line.isupper() or line.istitle() or looks_numbered:
            # Strip a leading number so "4. Password Policy" becomes
            # "Password Policy".
            return re.sub(r"^\d+(\.\d+)*[.)]?\s+", "", line).strip()

    return ""
