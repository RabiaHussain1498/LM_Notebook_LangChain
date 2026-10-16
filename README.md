# Notebook: NotebookLM-Style RAG Workspace (LangChain)

A lightweight, local, full-featured **NotebookLM-style Grounded RAG Application** built with **FastAPI**, **LangChain**, embedded **Qdrant**, **SQLite**, and a responsive single-page web interface.

Upload documents (PDF, TXT, Markdown), select sources, and ask questions. Every claim is strictly grounded with interactive citations `[n]` that allow you to inspect the source excerpts directly on the left-side viewer.

---

## Features

- **Multi-Document Ingestion**: Extract, chunk, and index PDFs, text files, and markdown notes.
- **Pure LangChain Architecture**:
  - `RecursiveCharacterTextSplitter` configured for fine-grained, sentence-aware atomic chunking.
  - `OpenAIEmbeddings` (`text-embedding-3-small`) for dense vector representations.
  - `QdrantVectorStore` (embedded mode) for instant local vector search without separate server setups.
  - `ChatOpenAI` (`gpt-6-luna` / `gpt-4o-mini`) via LCEL chains for grounded question answering.
- **Precise In-Document Citations**:
  - Assistant answers cite claims with numbered references (e.g. `[1]`, `[2]`).
  - Hovering over citations displays an instant tooltip preview.
  - Clicking any citation opens the document viewer in the **left panel** and highlights the exact cited excerpt.
- **Interactive Sidebar & Source Management**:
  - One-click **"Select All"** toggle to enable or disable all sources simultaneously.
  - Real-time chunk counting and per-source embedding cost estimation.
  - Inspect any source document and browse its chunks.
- **Web Search Fallback**: When documents don't contain the answer, the assistant offers web search results with URL citations.
- **Conversation History & Multi-Session**: Persistent chat history per notebook with support for multiple chat sessions.
- **Ask About This**: Select any text in the chat or document viewer to instantly ask a follow-up question about it.
- **Multi-Layer Guardrails**:
  - Self-harm and violence detection with supportive crisis resources.
  - System-prompt leak and jailbreak probing prevention.
  - OpenAI moderation check prior to LLM execution.
- **Dual-Model Cost Optimization**: Uses `gpt-6-luna` for answers and `gpt-4o-mini` for cheap tasks (query analysis, guardrails, web search) to minimize API costs.
- **Token & Cost Tracking**: Live USD cost calculation across embeddings, input tokens, and output tokens.

---

## How It Works

1. **Upload** a document → text is extracted, split into small chunks, embedded via OpenAI, and stored in Qdrant.
2. **Ask a question** → the query is analyzed (rephrased if it's a follow-up), then the top matching chunks are retrieved via vector similarity search.
3. **Get a grounded answer** → the LLM answers strictly from the retrieved excerpts, citing them with `[n]` markers.
4. **Click a citation** → the left panel opens the original document and highlights the exact cited passage.

If no sources contain the answer, the assistant offers to provide web search results instead.

---

## Project Architecture

```
LM_Notebook_LangChain/
├── app/
│   ├── __init__.py
│   ├── db.py           # SQLite database schema (notebooks, sources, chunks, sessions, messages)
│   ├── guard.py        # Safety guardrails (self-harm, violence, prompt injection, content moderation)
│   ├── main.py         # FastAPI application, REST endpoints, static file serving
│   └── rag.py          # LangChain RAG pipeline (chunking, embedding, Qdrant store, retrieval chain)
├── data/               # Auto-created at runtime (gitignored)
│   ├── app.db          # SQLite database
│   └── qdrant/         # Qdrant embedded vector storage
├── static/
│   └── index.html      # Single-page web interface
├── .env.example        
├── .gitignore          
├── requirements.txt    
└── README.md           
```

---

## Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/RabiaHussain1498/LM_Notebook_LangChain.git
cd LM_Notebook_LangChain

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

Open `.env` and insert your OpenAI API key and preferred chat model:

```env
OPENAI_API_KEY=your-openai-api-key-here
CHAT_MODEL=gpt-6-luna
```

### 4. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> **Note:** Do not pass the `--reload` flag in development. Embedded Qdrant operates directly within the Python process and locks local storage files; running multiple reloader worker processes can cause file lock conflicts.

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.
