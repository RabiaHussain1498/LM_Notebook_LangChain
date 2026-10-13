# Notebook: a NotebookLM-style RAG app

Stack: FastAPI + LangChain (OpenAIEmbeddings, ChatOpenAI, QdrantVectorStore, RecursiveCharacterTextSplitter) + embedded Qdrant + SQLite + one static HTML page.

## Run
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your OPENAI_API_KEY into .env
uvicorn app.main:app --port 8000      # no --reload: embedded Qdrant allows one process
```
Open http://127.0.0.1:8000. Data is stored in `./data/` (delete it to reset).

## Phases (where to look in the code)
1. Upload, chunk, embed, chunk count per source: `rag.ingest`, `rag.chunk_text`
2. Chat grounded in checked sources with [n] citations: `rag.retrieve`, `rag.answer`
3. Hover tooltip, click-to-highlight viewer: `static/index.html` (`.cite`, `openViewer`)
4. Highlight-and-ask, sessions, cost tracking, guardrails: `askSel`, `sessions` routes, `rag.PRICE`, `guard.py`

## Notes
- Prices in `rag.PRICE` are per-token USD; check them against OpenAI's pricing page.
- Only the free OpenAI moderation call in `guard.py` uses the OpenAI SDK directly; LangChain has no moderation primitive in the packages used here.
- Guardrail order: self-harm (supportive reply) -> system-prompt probing -> OpenAI moderation (free).
- Scanned PDFs have no text layer and are rejected (would need OCR).
