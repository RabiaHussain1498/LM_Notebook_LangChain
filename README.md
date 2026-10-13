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
- **Multi-Layer Guardrails**:
  - Self-harm detection with supportive crisis resources.
  - System-prompt leak and jailbreak probing prevention.
  - OpenAI moderation check prior to LLM execution.
- **Token & Cost Tracking**: Live USD cost calculation across embeddings, input tokens, and output tokens.

---

## Project Architecture

```
LM_Notebook_LangChain/
├── app/
│   ├── __init__.py
│   ├── db.py           # SQLite database schema (notebooks, sources, chunks, sessions, messages)
│   ├── guard.py        # Safety guardrails (self-harm, prompt injection, content moderation)
│   ├── main.py         # FastAPI application, REST endpoints, static file serving
│   └── rag.py          # LangChain RAG pipeline (chunking, embedding, Qdrant store, retrieval chain)
├── static/
│   └── index.html      
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

