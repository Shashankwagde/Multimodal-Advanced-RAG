"""
Streamlit user interface for the Multimodal Advanced RAG project.

Run it with:
    streamlit run app.py

This file is only about the interface: showing buttons, collecting input and
printing results. All the RAG logic lives in the `src/` package.

Streamlit reruns this whole script from top to bottom on every click, which is
why anything we want to keep (the chat history, the loaded pipeline) is stored
in `st.session_state`.
"""

import logging
import os

import streamlit as st

import config
from src.indexing import (
    build_index,
    clear_index,
    index_exists,
    index_summary,
    update_index,
)
from src.ingestion import SUPPORTED_EXTENSIONS, load_documents_from_folder
from src.memory import add_exchange, clear_memory, create_memory
from src.paths import resolve_image_path
from src.rag_pipeline import answer_question, load_pipeline
from src.reranking import reranker_status
from src.workspace import activate_workspace, cleanup_idle_workspaces, create_workspace

# Show our own INFO messages, but keep third-party libraries at WARNING so the
# terminal stays readable.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

st.set_page_config(page_title="Multimodal RAG Assistant", page_icon="📚", layout="wide")

# Width in pixels for figures shown in the answer and the debug panel.
FIGURE_WIDTH = 420
THUMBNAIL_WIDTH = 160


def activate_visitor_workspace() -> None:
    """
    Give each visitor a private workspace on a hosted demo.

    The first time a visitor opens the page, they get an empty workspace
    (documents, figures, index). Idle workspaces are deleted first. On every
    run, this function makes that visitor's workspace the active one.

    Does nothing unless PER_SESSION_WORKSPACES=true, so on your own computer
    the project's own data folders are used.
    """
    if not config.PER_SESSION_WORKSPACES:
        return

    if "workspace_id" not in st.session_state:
        cleanup_idle_workspaces()
        st.session_state.workspace_id = create_workspace()
        st.session_state.pipeline = None
        st.session_state.messages = []
        st.session_state.memory = create_memory()

    activate_workspace(st.session_state.workspace_id)


def init_session_state() -> None:
    """
    Create the session values the app needs, the first time it runs.

    Streamlit reruns the script constantly, so without this guard we would
    wipe the chat history on every single click.
    """
    if "memory" not in st.session_state:
        st.session_state.memory = create_memory()
    if "messages" not in st.session_state:
        # The on-screen transcript. `memory` is what the LLM sees; `messages`
        # is what the user sees (it also holds sources and figures).
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def get_pipeline():
    """
    Load the retrieval pipeline once and reuse it for every question.

    Returns:
        The pipeline dictionary from `load_pipeline()`, or None if the index
        is missing or could not be opened.
    """
    if st.session_state.pipeline is None:
        try:
            with st.spinner("Loading the index..."):
                st.session_state.pipeline = load_pipeline()
        except Exception as error:
            st.error(f"Could not load the index: {error}")
            return None

    return st.session_state.pipeline


def check_upload_limits(uploaded_files) -> str | None:
    """
    Check a batch of uploads against the demo limits.

    The limits protect the shared free NVIDIA quota and keep the demo fast:
    at most config.MAX_UPLOAD_FILES files in total, and at most
    config.MAX_UPLOAD_MB in total. Files already saved for this workspace count
    toward both limits.

    Args:
        uploaded_files: Files returned by `st.file_uploader`.

    Returns:
        An error message if a limit would be exceeded, otherwise None.
    """
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in config.DOCUMENTS_DIR.iterdir() if path.is_file()]

    total_files = len(existing) + len(uploaded_files)
    if total_files > config.MAX_UPLOAD_FILES:
        return (
            f"Limit is {config.MAX_UPLOAD_FILES} files. You have {len(existing)} saved "
            f"and are adding {len(uploaded_files)}. Remove some files and try again."
        )

    existing_bytes = sum(path.stat().st_size for path in existing)
    new_bytes = sum(uploaded.size for uploaded in uploaded_files)
    if existing_bytes + new_bytes > config.MAX_UPLOAD_MB * 1024 * 1024:
        return (
            f"Total upload size is limited to {config.MAX_UPLOAD_MB} MB. "
            "Choose smaller files or fewer files."
        )

    return None


def save_uploaded_files(uploaded_files) -> list:
    """
    Write files uploaded in the browser into the documents folder.

    The loaders read from disk, so an uploaded file is saved before it can be
    processed. Text files, PDFs and images all go to the same folder.

    Args:
        uploaded_files: Files returned by `st.file_uploader`.

    Returns:
        A list of the saved file paths.
    """
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        destination = config.DOCUMENTS_DIR / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def render_sidebar() -> bool:
    """
    Draw the sidebar: configuration, document management and chat controls.

    Returns:
        True if the user switched on "Show Retrieval Details".
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Changing this only affects the running app. To make it permanent
        # edit VECTOR_DB in your .env file.
        options = ["faiss", "pinecone"]
        chosen_db = st.selectbox(
            "Vector database",
            options,
            index=options.index(config.VECTOR_DB) if config.VECTOR_DB in options else 0,
            help="FAISS runs locally. Pinecone needs an API key in your .env file.",
        )
        if chosen_db != config.VECTOR_DB:
            config.VECTOR_DB = chosen_db
            st.session_state.pipeline = None

        for name, value in config.describe_settings().items():
            st.caption(f"**{name}:** {value}")

        # Shows the active reranker, and whether the last question fell back
        # to the keyword reranker.
        st.caption(f"**Reranker:** {reranker_status()}")

        st.divider()
        render_document_management()

        st.divider()
        st.header("💬 Chat controls")

        if st.button("Clear conversation", width="stretch"):
            clear_memory(st.session_state.memory)
            st.session_state.messages = []
            st.rerun()

        show_debug = st.toggle(
            "Show Retrieval Details",
            value=False,
            help="See the queries, retrieved chunks, reranked chunks and final context.",
        )

    return show_debug


def render_document_management() -> None:
    """
    Draw the document upload and index-building part of the sidebar.

    The three buttons map onto `src/indexing.py`: build (from scratch), update
    (add new files) and clear.
    """
    st.header("📄 Documents")

    if index_exists():
        summary = index_summary()
        st.success(
            f"Index ready: {summary['children']} child chunks "
            f"from {len(summary['files'])} file(s)."
        )
        with st.expander("Indexed files"):
            for filename in summary["files"]:
                st.write(f"- {filename}")
    else:
        st.info("No index yet. Build one to start asking questions.")

    uploaded_files = st.file_uploader(
        "Upload documents or images",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        help=(
            f"Up to {config.MAX_UPLOAD_FILES} files and {config.MAX_UPLOAD_MB} MB in total. "
            "PDF, DOCX and TXT are read as text. PDF figures are extracted automatically. "
            "PNG, JPG and WEBP are indexed as images."
        ),
    )

    if uploaded_files and st.button("Save uploaded files", width="stretch"):
        problem = check_upload_limits(uploaded_files)
        if problem:
            st.error(problem)
        else:
            saved = save_uploaded_files(uploaded_files)
            st.success(f"Saved {len(saved)} file(s). Now build or update the index.")

    if st.button("🔨 Build Vector DB", width="stretch"):
        run_build()

    if st.button("➕ Update Vector DB", width="stretch"):
        run_update()

    if st.button("🗑️ Clear / Rebuild", width="stretch"):
        clear_index()
        st.session_state.pipeline = None
        st.success("Index cleared. Click 'Build Vector DB' to start again.")


def run_build() -> None:
    """
    Rebuild the whole index from everything in the documents folder.

    Shows a spinner while text is embedded and images are described by the
    vision model. Both use the NVIDIA API.
    """
    try:
        with st.spinner("Reading documents, describing images and embedding chunks..."):
            documents = load_documents_from_folder(config.DOCUMENTS_DIR)
            stats = build_index(documents)

        st.session_state.pipeline = None
        st.success(
            f"Indexed {stats['documents']} items -> {stats['parents']} parent chunks "
            f"-> {stats['children']} child chunks."
        )
    except Exception as error:
        st.error(f"Build failed: {error}")


def run_update() -> None:
    """
    Add any new documents to the existing index without rebuilding it.

    Chunks that were already indexed are detected by their id and skipped, so
    clicking this twice does not create duplicates.
    """
    try:
        with st.spinner("Adding new documents to the index..."):
            documents = load_documents_from_folder(config.DOCUMENTS_DIR)
            stats = update_index(documents)

        st.session_state.pipeline = None
        if "added_children" in stats:
            st.success(
                f"Added {stats['added_children']} new chunks "
                f"(skipped {stats['skipped_children']} already indexed)."
            )
        else:
            st.success(f"Built a new index with {stats['children']} chunks.")
    except Exception as error:
        st.error(f"Update failed: {error}")


def render_figures(image_paths: list) -> None:
    """
    Show the figures used for an answer, one under another.

    Args:
        image_paths: File paths of the figures. Missing files are skipped.
    """
    for number, image_path in enumerate(image_paths, start=1):
        local_path = resolve_image_path(image_path)
        if local_path is None:
            continue
        st.image(local_path, caption=f"Figure {number}", width=FIGURE_WIDTH)


def render_debug_panel(debug: dict) -> None:
    """
    Show what happened inside the pipeline for this question.

    Args:
        debug: The "debug" dictionary returned by `answer_question`.
    """
    with st.expander("🔍 Retrieval details", expanded=False):
        st.markdown("**1. Original query**")
        st.write(debug["original_query"])

        st.markdown("**2. Generated queries (multi-query retrieval)**")
        for query in debug["generated_queries"]:
            st.write(f"- {query}")

        st.markdown(
            f"**3. Retrieved candidates after hybrid search "
            f"({len(debug['retrieved'])} chunks)**"
        )
        show_chunks(debug["retrieved"])

        st.markdown(f"**4. After reranking ({len(debug['reranked'])} chunks)**")
        show_chunks(debug["reranked"])

        st.markdown(
            f"**5. Final context sent to the model ({len(debug['final_context'])} chunks)**"
        )
        show_chunks(debug["final_context"])


def show_chunks(chunks: list, limit: int = 5) -> None:
    """
    Print a short preview of a list of chunks in the debug panel.

    Figure chunks also show a small thumbnail of the image.

    Args:
        chunks: Summary dictionaries with "source", "score", "preview" and
            optionally "image_path".
        limit: Maximum number of chunks to print, to keep the panel readable.
    """
    if not chunks:
        st.caption("_(none)_")
        return

    for number, chunk in enumerate(chunks[:limit], start=1):
        score = f" · score {chunk['score']}" if chunk["score"] is not None else ""
        st.caption(f"{number}. **{chunk['source']}**{score}")

        local_path = resolve_image_path(chunk.get("image_path"))
        if local_path:
            st.image(local_path, width=THUMBNAIL_WIDTH)

        st.text(chunk["preview"][:300])

    if len(chunks) > limit:
        st.caption(f"_...and {len(chunks) - limit} more_")


def render_sources(sources: list) -> None:
    """
    Print the citation list underneath an answer.

    Args:
        sources: Citation strings such as "handbook.pdf - Page 4" or
            "Figure 1: page 3 of report.pdf".
    """
    if not sources:
        return

    lines = [f"{number}. {source}" for number, source in enumerate(sources, start=1)]
    st.markdown("**Sources:**\n\n" + "\n".join(lines))


def main() -> None:
    """
    Draw the page and handle one chat turn per rerun.
    """
    init_session_state()
    activate_visitor_workspace()

    st.title("📚 Multimodal RAG Assistant")
    st.caption(
        "Hybrid retrieval · Multi-query · Parent-child chunks · Figure search · "
        "Reranking · Context refinement · NVIDIA NIM"
    )

    show_debug = render_sidebar()

    if not config.NVIDIA_API_KEY:
        st.warning(
            "No NVIDIA API key found. Copy `.env.example` to `.env`, add your "
            "NVIDIA_API_KEY (free at build.nvidia.com), then restart the app."
        )
        return

    if not index_exists():
        st.info(
            "Nothing is indexed yet. Upload your documents or images in the sidebar, "
            "save them, then click **Build Vector DB**."
        )
        return

    # Redraw the conversation so far. Streamlit forgets the screen on every
    # rerun, so we repaint it from st.session_state.messages each time.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_figures(message.get("images", []))
                render_sources(message.get("sources", []))
                if show_debug and message.get("debug"):
                    render_debug_panel(message["debug"])

    question = st.chat_input("Ask a question about your documents or figures...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    pipeline = get_pipeline()
    if pipeline is None:
        return

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and writing an answer..."):
            try:
                result = answer_question(question, pipeline, st.session_state.memory)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                return

        st.markdown(result["answer"])
        render_figures(result["images"])
        render_sources(result["sources"])
        if show_debug:
            render_debug_panel(result["debug"])

    # Save the turn in both places: the LLM's memory and the on-screen log.
    add_exchange(st.session_state.memory, question, result["answer"])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "images": result["images"],
            "debug": result["debug"],
        }
    )


if __name__ == "__main__":
    main()