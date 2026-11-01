Multimodal RAG Assistant

Upload your own documents (PDF, DOCX, TXT) and images (PNG, JPG, JPEG, WEBP), then ask questions about them. Answers cite their sources, and the figures they relied on are shown next to the answer.

The app is built as a learning project and a working demo. Every AI service it uses has a free tier, so it can be run and hosted at no cost.

Features

• Document Q&A over PDF, DOCX and TXT files.
• Figure understanding. Charts, diagrams and images inside PDFs are extracted, described by a vision model, and searchable like text. Figures are shown with the answer.
• Hybrid retrieval. Semantic search (embeddings + FAISS) and keyword search (BM25) run together, and results are merged.
• Multi-query retrieval. The question is rewritten a few ways so that different wording still finds the right text.
• Parent-child chunks. Small chunks are searched for precision; their larger parent chunks are given to the model for context.
• Reranking. Candidates are re-scored for relevance before the answer is written.
• Citations. Text sources are cited as [1], figures as [Figure 1: page 3 of report.pdf].
• Conversation memory for follow-up questions.
• Debug view that shows each retrieval step, so you can see how an answer was reached.

How it works

Upload -> Load -> Chunk -> Index -> Retrieve -> Rerank -> Refine -> Answer
  (text + (parent/ (FAISS + (multi-query (NVIDIA (budget, (text +
  figures) child, BM25) + hybrid) reranker) figures) figures)
  images ->
  descriptions)

Ingestion extracts text and figures. Chunking turns figures into text descriptions with a vision model, so the rest of the pipeline treats them like any other chunk. Refinement splits the final context into text and figures, and the answer step sends the figure images to the vision model along with the text.

Tech stack

| Part | Tool |
|---|---|
| Chat, vision, embeddings, reranking | NVIDIA NIM (free endpoints) |
| Backup chat model (optional) | Groq (free tier) |
| Vector search | FAISS (local) |
| Keyword search | BM25 (rank-bm25) |
| PDF and image extraction | PyMuPDF, Pillow |
| Orchestration | LangChain |
| Interface | Streamlit |

Project structure

app.py Streamlit interface
config.py Settings, read from .env
llm_client.py Chat, vision and embedding models (NVIDIA, Groq backup)
scripts/check_models.py Checks that the model IDs and keys work
src/
  ingestion.py Loads PDF, DOCX, TXT and images; extracts figures
  chunking.py Parent-child chunks; figure descriptions
  indexing.py Builds and updates the FAISS index and document store
  retrieval.py Multi-query and hybrid retrieval, parent recovery
  reranking.py NVIDIA reranker, with fallbacks
  refinement.py Filtering, context budget, text/figure split, citations
  rag_pipeline.py Runs the full pipeline and writes the answer
  memory.py Conversation history
  paths.py Finds figure files on any machine
  workspace.py Per-visitor folders for hosted demos
tests/ Offline tests (no API calls)

Setup

Requirements

• Python 3.10 or newer
• A free NVIDIA API key from build.nvidia.com (https://build.nvidia.com/settings)
• Optional: a free Groq API key from console.groq.com (https://console.groq.com)

Install

git clone <your-repo-url>
cd <your-repo-folder>
python -m venv myenv
source myenv/bin/activate # Windows: myenv\Scripts\activate
pip install -r requirements.txt

Configure

Copy .env.example to .env and add your keys:

NVIDIA_API_KEY=your-key-here
GROQ_API_KEY= # optional; leave empty to turn the backup off

Other settings (model names, chunk sizes, reranker, upload limits) are documented in .env.example.

Check your setup

python scripts/check_models.py

This confirms that each model ID exists and answers. Each check uses only a few tokens.

Run

streamlit run app.py

Then:

1. Upload your files in the sidebar and click Save uploaded files.
2. Click Build Vector DB. Figures are described by the vision model, which takes a little time the first time.
3. Ask a question in the chat box.

Turn on Show Retrieval Details in the sidebar to see how the answer was found.

Tests

python -m pytest tests -q

The tests run offline and need no API keys.

Hosting (free)

The app runs on a free Hugging Face Space with the Streamlit SDK.

• Add NVIDIA_API_KEY as a secret.
• Set PER_SESSION_WORKSPACES=true as a variable. Each visitor then gets a private workspace. Workspaces are deleted after 45 minutes of inactivity.

Uploads are limited to 5 files and 20 MB per visitor by default, to protect the shared free API quota.

Limitations

• Free API tiers have rate limits. Heavy use can trigger fallbacks, such as keyword reranking.
• Image descriptions depend on the vision model. Small text in figures may be missed or misread.
• Hosted workspaces are cleared after inactivity, so they are not a permanent store.
• FAISS is stored locally. For large collections, a hosted vector database such as Pinecone is more suitable.

License

Add your license here (for example MIT).