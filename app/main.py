import json, os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import db, guard, rag

app = FastAPI(title="Notebook")
STATIC = os.path.join(os.path.dirname(__file__), "..", "static")

class NameIn(BaseModel): name: str
class CheckIn(BaseModel): checked: bool
class ChatIn(BaseModel):
    question: str
    session_id: int | None = None

@app.get("/")
def index(): return FileResponse(os.path.join(STATIC, "index.html"))

@app.get("/api/notebooks")
def notebooks(): return db.q("SELECT * FROM notebooks ORDER BY id")

@app.post("/api/notebooks")
def new_notebook(b: NameIn):
    return {"id": db.ins("INSERT INTO notebooks(name) VALUES(?)", (b.name,)), "name": b.name}

@app.delete("/api/notebooks/{nid}")
def remove_notebook(nid: int):
    nb = db.q("SELECT * FROM notebooks WHERE id=?", (nid,), one=True)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    rag.delete_notebook(nid)
    return {"ok": True}

@app.get("/api/notebooks/{nid}")
def notebook(nid: int):
    nb = db.q("SELECT * FROM notebooks WHERE id=?", (nid,), one=True)
    if not nb: raise HTTPException(404)
    sc = db.q("SELECT COALESCE(SUM(cost),0) c FROM sources WHERE notebook_id=?", (nid,), one=True)["c"]
    mc = db.q("SELECT COALESCE(SUM(m.cost),0) c FROM messages m JOIN sessions s ON s.id=m.session_id "
              "WHERE s.notebook_id=?", (nid,), one=True)["c"]
    return {**nb, "total_cost": sc + mc,
            "sources": db.q("SELECT * FROM sources WHERE notebook_id=? ORDER BY id", (nid,)),
            "sessions": db.q("SELECT s.*, COUNT(m.id) as msg_count FROM sessions s LEFT JOIN messages m ON m.session_id=s.id WHERE s.notebook_id=? GROUP BY s.id ORDER BY msg_count DESC, s.id DESC", (nid,))}

# ---- Phase 1: sources
@app.post("/api/notebooks/{nid}/sources")
async def add_source(nid: int, file: UploadFile = File(...)):
    try:
        sid = rag.ingest(nid, file.filename, await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Ingestion failed: {e}")
    return db.q("SELECT * FROM sources WHERE id=?", (sid,), one=True)

@app.patch("/api/sources/{sid}")
def toggle(sid: int, b: CheckIn):
    db.q("UPDATE sources SET checked=? WHERE id=?", (int(b.checked), sid)); return {"ok": True}

@app.patch("/api/notebooks/{nid}/sources/toggle-all")
def toggle_all(nid: int, b: CheckIn):
    db.q("UPDATE sources SET checked=? WHERE notebook_id=?", (int(b.checked), nid)); return {"ok": True}

@app.delete("/api/sources/{sid}")
def remove(sid: int): rag.delete_source(sid); return {"ok": True}

@app.get("/api/sources/{sid}/chunks")   # Phase 3: the viewer
def chunks(sid: int):
    src = db.q("SELECT * FROM sources WHERE id=?", (sid,), one=True)
    if not src:
        raise HTTPException(404, "Source not found")
    chunks_list = db.q("SELECT id,idx,page,text FROM chunks WHERE source_id=? ORDER BY idx", (sid,))
    pages_data = None
    if src.get("full_text"):
        try:
            pages_data = json.loads(src["full_text"])
        except Exception:
            pages_data = [{"page": 1, "text": src["full_text"]}]
    return {
        "source": src,
        "chunks": chunks_list,
        "pages": pages_data
    }

# ---- Phase 2 + 4: notebook conversation & LangChain message history
@app.get("/api/notebooks/{nid}/messages")
def notebook_messages(nid: int, sid: int | None = None):
    history = rag.get_notebook_history(nid, sid)
    result = []
    for m in history.messages:
        role = "user" if isinstance(m, rag.HumanMessage) else "assistant"
        extra = m.additional_kwargs or {}
        result.append({
            "id": extra.get("id"),
            "role": role,
            "content": m.content,
            "citations": extra.get("citations", []),
            "cost": extra.get("cost", 0.0),
            "tokens": extra.get("tokens", 0),
            "input_tokens": extra.get("input_tokens", max(0, extra.get("total_tokens", 0) - extra.get("tokens", 0))),
            "total_tokens": extra.get("total_tokens", 0),
        })
    return result

@app.post("/api/notebooks/{nid}/chat")
def notebook_chat(nid: int, b: ChatIn):
    nb = db.q("SELECT * FROM notebooks WHERE id=?", (nid,), one=True)
    if not nb: raise HTTPException(404)
    history = rag.get_notebook_history(nid, b.session_id)

    blocked = guard.check(b.question)
    if blocked:
        history.add_message(rag.HumanMessage(content=b.question))
        ai = rag.AIMessage(content=blocked, additional_kwargs={"citations": [], "cost": 0.0, "tokens": 0, "total_tokens": 0})
        history.add_message(ai)
        return {"id": ai.additional_kwargs.get("id"), "role": "assistant", "content": blocked, "citations": [], "cost": 0.0, "tokens": 0, "total_tokens": 0}

    ids = [r["id"] for r in db.q("SELECT id FROM sources WHERE notebook_id=? AND checked=1", (nid,))]

    user_msg = rag.HumanMessage(content=b.question)
    history.add_message(user_msg)
    try:
        # Pass conversation history (excluding the current user question just added)
        res = rag.answer(b.question, ids, history.messages[:-1])
    except Exception as e:
        raise HTTPException(502, str(e))

    ai_msg = rag.AIMessage(content=res["content"], additional_kwargs={
        "citations": res["citations"],
        "cost": res["cost"],
        "tokens": res["tokens"],
        "input_tokens": res.get("input_tokens", max(0, res.get("total_tokens", 0) - res.get("tokens", 0))),
        "total_tokens": res.get("total_tokens", 0)
    })
    history.add_message(ai_msg)
    return {"id": ai_msg.additional_kwargs.get("id"), "role": "assistant", **res}

@app.post("/api/notebooks/{nid}/sessions")
def new_session(nid: int):
    return {"id": db.ins("INSERT INTO sessions(notebook_id,title) VALUES(?,?)", (nid, "New chat")), "title": "New chat"}

@app.get("/api/sessions/{sid}/messages")
def messages(sid: int):
    s = db.q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)
    if not s: return []
    return notebook_messages(s["notebook_id"], sid)

@app.post("/api/sessions/{sid}/chat")
def chat(sid: int, b: ChatIn):
    s = db.q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)
    if not s: raise HTTPException(404)
    b.session_id = sid
    return notebook_chat(s["notebook_id"], b)
