"""
Offline tests for the Multimodal RAG pipeline.

None of these tests call NVIDIA or Groq, so they need no API keys and no
internet. They check the parts of the pipeline that can break silently:
file loading, figure handling, citations, budgeting and fallbacks.

Run from the project root:
    python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import chunking, ingestion, refinement, reranking  # noqa: E402


def text_doc(text: str, filename: str = "a.txt", score: float | None = None) -> Document:
    """Build a text Document the way ingestion does."""
    metadata = {"type": "text", "filename": filename, "source": filename, "doc_type": "txt"}
    if score is not None:
        metadata["rerank_score"] = score
    return Document(page_content=text, metadata=metadata)


def image_doc(image_path: Path, filename: str = "report.pdf", page: int = 2,
              score: float | None = None) -> Document:
    """Build an image Document the way ingestion and chunking do."""
    metadata = {
        "type": "image",
        "image_path": str(image_path),
        "filename": filename,
        "source_doc": filename,
        "page_number": page,
    }
    if score is not None:
        metadata["rerank_score"] = score
    return Document(page_content="A bar chart of refunds by month.", metadata=metadata)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_clean_text_repairs_hyphenated_line_breaks():
    assert ingestion.clean_text("secu-\nrity   policy") == "security policy"


def test_text_file_is_loaded_as_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Refunds are accepted within 30 days.", encoding="utf-8")

    documents = ingestion.load_document(path)

    assert len(documents) == 1
    assert documents[0].metadata["type"] == "text"
    assert documents[0].metadata["filename"] == "notes.txt"


def test_standalone_image_becomes_one_image_document(tmp_path):
    path = tmp_path / "chart.png"
    Image.new("RGB", (400, 300), "white").save(path)

    documents = ingestion.load_document(path)

    assert len(documents) == 1
    assert documents[0].metadata["type"] == "image"
    assert documents[0].metadata["image_path"] == str(path)


def test_tiny_image_is_skipped(tmp_path):
    path = tmp_path / "bullet.png"
    Image.new("RGB", (20, 20), "black").save(path)

    assert ingestion.load_document(path) == []


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError):
        ingestion.load_document(path)


# ---------------------------------------------------------------------------
# Chunking (text only, so no vision calls are made)
# ---------------------------------------------------------------------------

def test_every_child_points_to_a_parent():
    document = text_doc("Refund Policy\n\n" + "Refunds are issued within thirty days. " * 120)

    parents, children = chunking.split_documents([document])

    parent_ids = {parent.metadata["parent_id"] for parent in parents}
    assert parents and children
    assert all(child.metadata["parent_id"] in parent_ids for child in children)
    assert all("chunk_id" in child.metadata for child in children)


# ---------------------------------------------------------------------------
# Reranking (fallback behaviour, no network)
# ---------------------------------------------------------------------------

def test_keyword_score_prefers_word_overlap():
    strong = reranking.keyword_score("refund policy", "The refund policy allows refunds.")
    weak = reranking.keyword_score("refund policy", "The office parking lot is on level two.")
    assert strong > weak


def test_failed_nvidia_reranker_falls_back_to_keywords(monkeypatch):
    def broken(query, documents):
        raise RuntimeError("network down")

    monkeypatch.setattr(config, "RERANKER", "nvidia")
    monkeypatch.setattr(reranking, "nvidia_rerank_scores", broken)

    documents = [
        text_doc("Parking is on level two.", filename="p.txt"),
        text_doc("Refund policy: refunds within 30 days.", filename="r.txt"),
    ]
    ranked = reranking.rerank_documents("refund policy", documents, top_k=2)

    assert ranked[0].metadata["filename"] == "r.txt"
    assert "network down" in reranking.reranker_status()


def test_simple_mode_never_calls_the_api(monkeypatch):
    def should_not_run(query, documents):
        raise AssertionError("the API must not be called in simple mode")

    monkeypatch.setattr(config, "RERANKER", "simple")
    monkeypatch.setattr(reranking, "nvidia_rerank_scores", should_not_run)

    ranked = reranking.rerank_documents(
        "refund policy", [text_doc("refund policy details")], top_k=1
    )
    assert len(ranked) == 1


# ---------------------------------------------------------------------------
# Refinement: text vs figures, citations and the figure cap
# ---------------------------------------------------------------------------

def test_refinement_separates_text_and_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGES_PER_ANSWER", 1)
    monkeypatch.setattr(config, "MIN_RERANK_SCORE", -4.0)

    first_figure = tmp_path / "a.png"
    second_figure = tmp_path / "b.png"
    documents = [
        text_doc("Refunds are accepted within 30 days.", score=2.0),
        image_doc(first_figure, score=1.5),
        image_doc(second_figure, score=1.0),
    ]

    refined = refinement.refine_context(documents)

    assert refined.image_paths == [str(first_figure)]
    assert refined.citations[0] == "a.txt"
    assert refined.citations[1] == "Figure 1: page 2 of report.pdf"
    assert "[1] Source: a.txt" in refined.text_context
    assert "[Figure 1: page 2 of report.pdf]" in refined.text_context


def test_refinement_drops_duplicate_figure_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGES_PER_ANSWER", 3)
    figure = tmp_path / "same.png"

    refined = refinement.refine_context([
        image_doc(figure, score=2.0),
        image_doc(figure, score=1.0),
    ])

    assert refined.image_paths == [str(figure)]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_missing_nvidia_key_is_reported(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        config.check_llm_keys()


def test_groq_backup_is_off_without_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")

    assert config.groq_backup_enabled() is False
