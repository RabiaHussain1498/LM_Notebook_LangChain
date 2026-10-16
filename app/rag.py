"""RAG pipeline built with LangChain:
   PDF text -> RecursiveCharacterTextSplitter -> OpenAIEmbeddings -> QdrantVectorStore
   question -> retriever (filtered to checked sources) -> ChatPromptTemplate | ChatOpenAI -> cited answer"""
import csv, io, json, os, re
import tiktoken
from pypdf import PdfReader
from pptx import Presentation
from docx import Document as DocxDocument
from openpyxl import load_workbook
from qdrant_client import QdrantClient, models
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import db

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-6-luna")
# USD per token. Verify against https://openai.com/api/pricing before quoting numbers.
PRICE = {"embed": 0.02 / 1e6, "in": 0.10 / 1e6, "out": 0.50 / 1e6}

RAG_SYSTEM = (
    "You are a strict, grounded AI notebook assistant.\n"
    "You answer questions using ONLY the provided source excerpts.\n\n"
    "STRICT GROUNDING RULES:\n"
    "1. STRICT GROUNDING: Answer the question using ONLY facts and information directly stated in the source excerpts. Do NOT guess, extrapolate, or use outside training data to invent answers.\n"
    "2. OUT OF CONTEXT / MISSING INFO: If the provided source excerpts do not contain the answer to the question, state plainly:\n"
    "\"I do not have any resource in your notebook about this. Would you like me to provide web search results?\"\n"
    "3. CITATION PLACEMENT: For direct answers, provide only 1 primary citation [1] placed cleanly at the end of your answer. Do not insert multiple citations or scatter them mid-sentence.\n"
    "4. NO IRRELEVANT CITATIONS: Only cite an excerpt if that excerpt directly proves the factual claim. If an excerpt does not contain the answer, NEVER cite it.\n"
    "5. DO NOT SUMMARIZE UNRELATED EXCERPTS: Never describe or summarize excerpts that do not directly answer the question.\n"
    "6. NO UNGROUNDED SPECULATION: Never state that an excerpt raises a question without answering it, and never attempt to supply answers from external knowledge that the excerpts leave unanswered."
)
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "Source excerpts:\n{context}\n\nQuestion: {question}"),
])

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent, friendly AI notebook assistant.\n\n"
        "STRICT NOTEBOOK SCOPE RULES:\n"
        "1. GREETINGS, IDENTITY & SIMPLE ARITHMETIC:\n"
        "   - For greetings ('hi', 'hello', 'good morning'), identity questions ('who are you', 'what is your name'), or basic math ('2+2 = 4', '3+3 = 6'), respond directly, helpfully, and concisely.\n"
        "   - Do NOT mention documents or offer web search for basic greetings or math.\n"
        "2. FACTUAL / KNOWLEDGE / TOPICAL QUESTIONS:\n"
        "   - For any factual, technical, political, historical, or domain question (e.g., 'what is deep learning', 'who is the prime minister of Pakistan', 'how does backpropagation work'):\n"
        "   - Since no notebook source documents are selected or available to answer this:\n"
        "   - Do NOT answer the question directly from memory.\n"
        "   - Respond with this exact pattern:\n"
        "     'I do not have any resource in your notebook about {{topic}}. Would you like me to provide web search results?'"
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

WEB_SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI research assistant with comprehensive web search capabilities.\n"
        "The user previously asked a question that was not in their notebook sources, and has now agreed to receive web search results.\n"
        "Answer the user's pending question concisely and accurately.\n\n"
        "CITATION RULES:\n"
        "1. Provide ONLY 1 primary, authoritative citation marker [1] placed cleanly at the very END of your answer text (do NOT include multiple citations like [2] or [3], and do not sprinkle citations mid-sentence).\n"
        "2. At the very end of your response, provide the single source under a 'Sources:' section in this exact format:\n"
        "[1] Source Title: https://...\n"
        "Use a single reputable, real URL (e.g., official government portal, Wikipedia, official documentation)."
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI query analyzer and rephraser for a document notebook assistant.\n"
        "Given the conversation history and the latest user message:\n"
        "1. If the user message is agreeing to a web search offer (e.g. 'yes', 'sure', 'please do', 'yes please', 'search the web', 'ok', 'okay') following a message that offered web search results, respond with: DO_WEB_SEARCH\n"
        "2. If the message is purely a greeting, pleasantry, small talk, or simple math / arithmetic calculation (e.g. \"hi\", \"hello\", \"3+3\", \"what is 5*5\"), respond with: NO_SEARCH\n"
        "3. If the user is asking about the conversation history itself (e.g. \"what did I just ask?\", \"repeat that\"), respond with: NO_SEARCH\n"
        "4. If the user message is a follow-up or pronoun reference referring to previous topics or documents, rephrase it into a clear, standalone search query.\n"
        "5. Otherwise, for any topical, technical, factual, or document-related question, formulate a clear search query.\n"
        "Respond with ONLY the standalone search query, \"DO_WEB_SEARCH\", or \"NO_SEARCH\"."
    )),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])
PROMPT = RAG_PROMPT
# LangChain text splitter tuned for atomic, high-precision retrieval:
# 280 chars (~40 words) splits cleanly on sentence boundaries so each chunk is a concise, relevant excerpt
splitter = RecursiveCharacterTextSplitter(
    chunk_size=280,
    chunk_overlap=35,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)

# ---- lazily-built LangChain components (so the app starts even before a key is set)
_emb = _llm = _fast_llm = _vs = None
def _need_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env)")

def embeddings():
    global _emb
    if _emb is None:
        _need_key(); _emb = OpenAIEmbeddings(model=EMBED_MODEL)
    return _emb

def llm():
    global _llm
    if _llm is None:
        _need_key(); _llm = ChatOpenAI(model=CHAT_MODEL)
    return _llm

def fast_llm():
    global _fast_llm
    if _fast_llm is None:
        _need_key(); _fast_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _fast_llm

def store():
    global _vs
    if _vs is None:
        client = QdrantClient(path=os.path.join(db.DATA_DIR, "qdrant"))  # embedded mode, no server needed
        if not client.collection_exists("chunks"):
            client.create_collection("chunks", vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
        _vs = QdrantVectorStore(client=client, collection_name="chunks", embedding=embeddings())
    return _vs

_enc = None
def count_tokens(text):
    """LangChain embeddings don't report usage, so we count tokens ourselves for cost.
    tiktoken downloads its vocab on first use; if that fails (offline), fall back to ~4 chars/token."""
    global _enc
    try:
        if _enc is None:
            _enc = tiktoken.get_encoding("cl100k_base")
        return len(_enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

SUPPORTED_FORMATS = {".pdf", ".txt", ".md", ".pptx", ".docx", ".xlsx", ".csv"}

# ---- Phase 1: ingestion
def extract(name, data):
    lower = name.lower()
    ext = os.path.splitext(lower)[1]
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload: {', '.join(sorted(SUPPORTED_FORMATS))}")
    if ext == ".pdf":
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(data)).pages)]
    if ext == ".pptx":
        prs = Presentation(io.BytesIO(data))
        return [(i + 1, "\n".join(sh.text for sh in slide.shapes if sh.has_text_frame))
                for i, slide in enumerate(prs.slides)]
    if ext == ".docx":
        doc = DocxDocument(io.BytesIO(data))
        return [(1, "\n".join(p.text for p in doc.paragraphs))]
    if ext == ".xlsx":
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        pages = []
        for i, ws in enumerate(wb.worksheets):
            rows = []
            for row in ws.iter_rows(values_only=True):
                rows.append("\t".join(str(c) if c is not None else "" for c in row))
            pages.append((i + 1, "\n".join(rows)))
        wb.close()
        return pages
    if ext == ".csv":
        text = data.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(text))
        return [(1, "\n".join("\t".join(row) for row in reader))]
    return [(1, data.decode("utf-8", errors="ignore"))]

def ingest(notebook_id, name, data):
    pages = extract(name, data)
    docs = splitter.create_documents([t for _, t in pages], metadatas=[{"page": p} for p, _ in pages])
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("No extractable text (a scanned PDF needs OCR).")
    cost = sum(count_tokens(d.page_content) for d in docs) * PRICE["embed"]
    pages_json = json.dumps([{"page": p, "text": t} for p, t in pages if t.strip()])
    sid = db.ins("INSERT INTO sources(notebook_id,name,n_chunks,cost,full_text) VALUES(?,?,?,?,?)",
                 (notebook_id, name, len(docs), cost, pages_json))
    ids = []
    for idx, d in enumerate(docs):
        cid = db.ins("INSERT INTO chunks(source_id,idx,page,text) VALUES(?,?,?,?)", (sid, idx, d.metadata["page"], d.page_content))
        d.metadata.update(source_id=sid, chunk_id=cid, source=name)
        ids.append(cid)
    store().add_documents(docs, ids=ids)  # embeds with OpenAIEmbeddings and upserts into Qdrant
    return sid

def delete_source(sid):
    ids = [r["id"] for r in db.q("SELECT id FROM chunks WHERE source_id=?", (sid,))]
    if ids:
        store().delete(ids=ids)
    db.q("DELETE FROM chunks WHERE source_id=?", (sid,))
    db.q("DELETE FROM sources WHERE id=?", (sid,))

def delete_notebook(nid: int):
    # 1. Delete all sources and chunks
    sources = db.q("SELECT id FROM sources WHERE notebook_id=?", (nid,))
    for s in sources:
        delete_source(s["id"])

    # 2. Delete all chat messages and sessions
    sessions = db.q("SELECT id FROM sessions WHERE notebook_id=?", (nid,))
    for sess in sessions:
        db.q("DELETE FROM messages WHERE session_id=?", (sess["id"],))
    db.q("DELETE FROM sessions WHERE notebook_id=?", (nid,))

    # 3. Delete notebook record
    db.q("DELETE FROM notebooks WHERE id=?", (nid,))

# ---- LangChain Chat Message History per Notebook
# ---- LangChain Chat Message History per Notebook & Session
class NotebookChatMessageHistory(BaseChatMessageHistory):
    """LangChain BaseChatMessageHistory that persists and loads messages per notebook/session from SQLite."""
    def __init__(self, notebook_id: int, session_id: int | None = None):
        self.notebook_id = notebook_id
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        if self.session_id:
            rows = db.q(
                "SELECT m.id, m.role, m.content, m.citations, m.cost, m.tokens "
                "FROM messages m WHERE m.session_id = ? ORDER BY m.id",
                (self.session_id,)
            )
        else:
            rows = db.q(
                "SELECT m.id, m.role, m.content, m.citations, m.cost, m.tokens "
                "FROM messages m JOIN sessions s ON s.id = m.session_id "
                "WHERE s.notebook_id = ? ORDER BY m.id",
                (self.notebook_id,)
            )
        msg_list: list[BaseMessage] = []
        for r in rows:
            cites = r["citations"]
            if isinstance(cites, str):
                try: cites = json.loads(cites)
                except Exception: cites = []
            elif not cites:
                cites = []
            extra = {
                "id": r["id"],
                "citations": cites,
                "cost": float(r["cost"] or 0.0),
                "tokens": int(r["tokens"] or 0)
            }
            if r["role"] == "user":
                msg_list.append(HumanMessage(content=r["content"], additional_kwargs=extra))
            else:
                msg_list.append(AIMessage(content=r["content"], additional_kwargs=extra))
        return msg_list

    def add_message(self, message: BaseMessage) -> None:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        sid = self.session_id
        if not sid:
            s = db.q("SELECT id FROM sessions WHERE notebook_id=? ORDER BY id DESC LIMIT 1", (self.notebook_id,), one=True)
            if s:
                sid = s["id"]
            else:
                sid = db.ins("INSERT INTO sessions(notebook_id, title) VALUES(?, ?)", (self.notebook_id, "Notebook Chat"))
                self.session_id = sid

        cites = json.dumps(message.additional_kwargs.get("citations", []))
        cost = float(message.additional_kwargs.get("cost", 0.0) or 0.0)
        tokens = int(message.additional_kwargs.get("tokens", 0) or 0)

        mid = db.ins(
            "INSERT INTO messages(session_id, role, content, citations, cost, tokens) VALUES(?,?,?,?,?,?)",
            (sid, role, message.content, cites, cost, tokens)
        )
        message.additional_kwargs["id"] = mid

    def clear(self) -> None:
        if self.session_id:
            db.q("DELETE FROM messages WHERE session_id=?", (self.session_id,))
        else:
            s_ids = [r["id"] for r in db.q("SELECT id FROM sessions WHERE notebook_id=?", (self.notebook_id,))]
            if s_ids:
                placeholders = ",".join(["?"] * len(s_ids))
                db.q(f"DELETE FROM messages WHERE session_id IN ({placeholders})", tuple(s_ids))

def get_notebook_history(notebook_id: int, session_id: int | None = None) -> NotebookChatMessageHistory:
    return NotebookChatMessageHistory(notebook_id, session_id)

# ---- Phase 2: retrieval + grounded answer
def retrieve(question, source_ids, k=2):
    flt = models.Filter(must=[models.FieldCondition(key="metadata.source_id", match=models.MatchAny(any=source_ids))])
    return store().as_retriever(search_kwargs={"k": k, "filter": flt}).invoke(question)

def extract_concerned_patch(chunk_text, question="", claim=""):
    if not chunk_text:
        return ""
    lines = [ln for ln in chunk_text.split("\n") if ln.strip()]
    text = chunk_text
    if len(lines) > 1:
        first = lines[0].strip()
        # Drop leading overlap line if it starts lowercase/conjunction and ends with punctuation
        if first and (first[0].islower() or re.match(r"^(and|or|but|the|with|in|to|of|so)\b", first, re.I)) and first.endswith((".", "!", "?", ";", ":")):
            rest = "\n".join(lines[1:]).strip()
            if rest:
                text = rest

    stop = {"what", "when", "where", "which", "who", "whom", "this", "that", "with", "from", "have", "more", "than", "does", "each", "the", "and", "for", "are", "was", "were", "you", "can"}
    keywords = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", (question + " " + claim).lower())) - stop
    if not keywords:
        return text

    split_lines = text.split("\n")
    scored = []
    for idx, ln in enumerate(split_lines):
        w = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", ln.lower())) - stop
        scored.append((idx, len(w & keywords), ln))

    matched = [s for s in scored if s[1] > 0 and s[2].strip()]
    if matched:
        start_idx = matched[0][0]
        end_idx = matched[-1][0]
        while end_idx + 1 < len(split_lines) and split_lines[end_idx + 1].strip() and not re.match(r"^(Q\.|Question|\d+\.)", split_lines[end_idx + 1].strip()):
            end_idx += 1
        patch = "\n".join(split_lines[start_idx:end_idx+1]).strip()
        if patch:
            return patch
    return text

def process_web_answer_and_citations(text: str) -> tuple[str, list[dict]]:
    """Clean the text to remove raw Sources: lines, ensure [1] is at the end, and extract citation metadata."""
    matches = re.findall(r"\[(\d+)\]\s*(.*?)(?::|-)?\s*(https?://[^\s\)]+)", text)
    cites = []
    seen = set()
    for num_str, title, url in matches:
        clean_url = url.strip().rstrip(".)")
        if clean_url not in seen:
            seen.add(clean_url)
            clean_title = re.sub(r"^[-:\s]+|[-:\s]+$", "", (title or "Web Source").strip())
            clean_title = re.sub(r"^(Sources?:?|\d+\.?)\s*", "", clean_title, flags=re.IGNORECASE).strip()
            cites.append({
                "n": 1,
                "source": clean_title if clean_title else "Web Source",
                "url": clean_url,
                "patch": f"Web Source: {clean_title} ({clean_url})"
            })

    if not cites:
        any_url = re.search(r"https?://[^\s\)]+", text)
        if any_url:
            clean_u = any_url.group(0).strip().rstrip(".)")
            cites.append({
                "n": 1,
                "source": "Web Source",
                "url": clean_u,
                "patch": f"Web Source ({clean_u})"
            })

    # Strip out the trailing "Sources: ..." section and raw URLs from the message body so no raw link is printed
    clean_text = re.sub(r"(?i)\n*\s*(?:Sources?|References?):\s*(?:\[\d+\].*|\d+\..*|https?://.*|\w+.*)*$", "", text).strip()
    clean_text = re.sub(r"(?i)\n*\[\d+\]\s*.*https?://[^\s\)]+$", "", clean_text).strip()
    clean_text = re.sub(r"https?://[^\s\)]+", "", clean_text).strip()

    # Remove any citations in the middle of sentences so citation only appears cleanly at the end
    clean_text = re.sub(r"\s*\[\d+\]", "", clean_text).strip()
    # Now append clean [1] at the end
    clean_text = f"{clean_text} [1]"

    if not cites:
        cites = [{
            "n": 1,
            "source": "Official Web Source",
            "url": "https://www.google.com",
            "patch": "Web Reference [1]"
        }]

    return clean_text, cites[:1]

def answer(question, source_ids, history=None):
    hist: list[BaseMessage] = []
    if history:
        for m in history:
            if isinstance(m, BaseMessage):
                c = re.sub(r"\[\d+\]", "", m.content)
                hist.append(HumanMessage(content=c) if isinstance(m, HumanMessage) else AIMessage(content=c))
            elif isinstance(m, dict):
                c = re.sub(r"\[\d+\]", "", m.get("content", ""))
                hist.append(HumanMessage(content=c) if m.get("role") == "user" else AIMessage(content=c))

    q_strip = question.strip()
    search_query = q_strip
    try:
        # Use LangChain to analyze query and rephrase follow-up query using recent chat history
        condense_res = (CONDENSE_PROMPT | fast_llm()).invoke({"history": hist[-4:], "question": q_strip})
        sq = condense_res.content.strip()
        if sq:
            search_query = sq
    except Exception:
        search_query = q_strip

    # Check if the user is confirming a web search offer
    prev_ai_offered_web = False
    if hist:
        for m in reversed(hist):
            if isinstance(m, AIMessage):
                if "web search results" in m.content.lower():
                    prev_ai_offered_web = True
                break

    is_affirmation = q_strip.lower().rstrip(".!") in (
        "yes", "y", "sure", "please", "yes please", "ok", "okay", "yeah", "yep",
        "do it", "search", "search the web", "provide web search", "yes provide web search results",
        "yes, please", "yes, provide web search results"
    )

    # If user agreed to web search, process via fast_llm (gpt-4o-mini) with web search citations!
    if (prev_ai_offered_web and is_affirmation) or "DO_WEB_SEARCH" in search_query:
        msg = (WEB_SEARCH_PROMPT | fast_llm()).invoke({"history": hist[-4:], "question": q_strip})
        text = msg.content
        u = msg.usage_metadata or {}
        cost = (u.get("input_tokens", 0) * PRICE["in"] + u.get("output_tokens", 0) * PRICE["out"])
        out_tok = u.get("output_tokens", 0)
        tot_tok = u.get("total_tokens", 0)
        in_tok = u.get("input_tokens", 0) or max(0, tot_tok - out_tok)
        clean_text, cites = process_web_answer_and_citations(text)
        return {"content": clean_text, "citations": cites, "cost": cost, "tokens": out_tok, "output_tokens": out_tok, "input_tokens": in_tok, "total_tokens": tot_tok}

    # Case 1: General conversation, math, logic, world knowledge (NO_SEARCH) or no sources checked
    if "NO_SEARCH" in search_query or not source_ids:
        # Answer general query or offer web search if outside notebook scope
        msg = (CHAT_PROMPT | llm()).invoke({"history": hist[-2:], "question": q_strip})
        text = msg.content
        u = msg.usage_metadata or {}
        cost = (u.get("input_tokens", 0) * PRICE["in"] + u.get("output_tokens", 0) * PRICE["out"])
        out_tok = u.get("output_tokens", 0)
        tot_tok = u.get("total_tokens", 0)
        in_tok = u.get("input_tokens", 0) or max(0, tot_tok - out_tok)
        return {"content": text, "citations": [], "cost": cost, "tokens": out_tok, "output_tokens": out_tok, "input_tokens": in_tok, "total_tokens": tot_tok}

    # Case 2: Document RAG with rephrased search query (retrieve top 2 focused chunks to prevent bloated context)
    docs = retrieve(search_query, source_ids, k=2)
    if not docs:
        msg = (CHAT_PROMPT | llm()).invoke({"history": hist[-2:], "question": q_strip})
        text = msg.content
        u = msg.usage_metadata or {}
        cost = (count_tokens(search_query) * PRICE["embed"] + u.get("input_tokens", 0) * PRICE["in"] + u.get("output_tokens", 0) * PRICE["out"])
        out_tok = u.get("output_tokens", 0)
        tot_tok = u.get("total_tokens", 0)
        in_tok = u.get("input_tokens", 0) or max(0, tot_tok - out_tok)
        return {"content": text, "citations": [], "cost": cost, "tokens": out_tok, "output_tokens": out_tok, "input_tokens": in_tok, "total_tokens": tot_tok}

    context = "\n\n".join(f"[{i+1}] ({d.metadata.get('source', 'Doc')}, p.{d.metadata.get('page', 1)})\n{d.page_content}"
                          for i, d in enumerate(docs))

    msg = (RAG_PROMPT | llm()).invoke({"history": hist[-2:], "context": context, "question": q_strip})
    text = msg.content

    # Citations: Only cite when grounded in the excerpts
    used = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    cites = []
    # If model replied that the documents do not have the answer, do not cite excerpts
    if "web search results" not in text.lower():
        for i, d in enumerate(docs):
            if i + 1 in used:
                patch = extract_concerned_patch(d.page_content, q_strip, text)
                cites.append({
                    "n": i + 1,
                    "chunk_id": d.metadata.get("chunk_id") or d.metadata.get("_id") or getattr(d, "id", None),
                    "source_id": d.metadata.get("source_id"),
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page", 1),
                    "text": d.page_content,
                    "patch": patch
                })

    u = msg.usage_metadata or {}
    cost = (count_tokens(search_query) * PRICE["embed"] + u.get("input_tokens", 0) * PRICE["in"]
            + u.get("output_tokens", 0) * PRICE["out"])
    out_tok = u.get("output_tokens", 0)
    tot_tok = u.get("total_tokens", 0)
    in_tok = u.get("input_tokens", 0) or max(0, tot_tok - out_tok)
    return {"content": text, "citations": cites, "cost": cost, "tokens": out_tok, "output_tokens": out_tok, "input_tokens": in_tok, "total_tokens": tot_tok}
