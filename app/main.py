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
class ChatIn(BaseModel): question: str

@app.get("/")
def index(): return FileResponse(os.path.join(STATIC, "index.html"))

@app.get("/api/notebooks")
def notebooks(): return db.q("SELECT * FROM notebooks ORDER BY id")

@app.post("/api/notebooks")
def new_notebook(b: NameIn):
    return {"id": db.ins("INSERT INTO notebooks(name) VALUES(?)", (b.name,)), "name": b.name}

@app.get("/api/notebooks/{nid}")
def notebook(nid: int):
    nb = db.q("SELECT * FROM notebooks WHERE id=?", (nid,), one=True)
    if not nb: raise HTTPException(404)
    sc = db.q("SELECT COALESCE(SUM(cost),0) c FROM sources WHERE notebook_id=?", (nid,), one=True)["c"]
    mc = db.q("SELECT COALESCE(SUM(m.cost),0) c FROM messages m JOIN sessions s ON s.id=m.session_id "
              "WHERE s.notebook_id=?", (nid,), one=True)["c"]
    return {**nb, "total_cost": sc + mc,
            "sources": db.q("SELECT * FROM sources WHERE notebook_id=? ORDER BY id", (nid,)),
            "sessions": db.q("SELECT * FROM sessions WHERE notebook_id=? ORDER BY id DESC", (nid,))}

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
    return {"source": db.q("SELECT * FROM sources WHERE id=?", (sid,), one=True),
            "chunks": db.q("SELECT id,idx,page,text FROM chunks WHERE source_id=? ORDER BY idx", (sid,))}

# ---- Phase 2 + 4: sessions and chat
@app.post("/api/notebooks/{nid}/sessions")
def new_session(nid: int):
    return {"id": db.ins("INSERT INTO sessions(notebook_id,title) VALUES(?,?)", (nid, "New chat")), "title": "New chat"}

@app.get("/api/sessions/{sid}/messages")
def messages(sid: int):
    rows = db.q("SELECT * FROM messages WHERE session_id=? ORDER BY id", (sid,))
    for r in rows: r["citations"] = json.loads(r["citations"])
    return rows

@app.post("/api/sessions/{sid}/chat")
def chat(sid: int, b: ChatIn):
    s = db.q("SELECT * FROM sessions WHERE id=?", (sid,), one=True)
    if not s: raise HTTPException(404)
    hist = [{"role": m["role"], "content": __import__("re").sub(r"\[\d+\]", "", m["content"])}
            for m in db.q("SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 6", (sid,))][::-1]
    db.ins("INSERT INTO messages(session_id,role,content) VALUES(?,?,?)", (sid, "user", b.question))
    if s["title"] == "New chat":
        db.q("UPDATE sessions SET title=? WHERE id=?", (b.question[:40], sid))
    res = {"content": "", "citations": [], "cost": 0.0, "tokens": 0}
    blocked = guard.check(b.question)
    if blocked:
        res["content"] = blocked
    else:
        ids = [r["id"] for r in db.q("SELECT id FROM sources WHERE notebook_id=? AND checked=1", (s["notebook_id"],))]
        if not ids:
            res["content"] = "Please check at least one source in the left panel first."
        else:
            try: res = rag.answer(b.question, ids, hist)
            except Exception as e: raise HTTPException(502, str(e))
    mid = db.ins("INSERT INTO messages(session_id,role,content,citations,cost,tokens) VALUES(?,?,?,?,?,?)",
                 (sid, "assistant", res["content"], json.dumps(res["citations"]), res["cost"], res["tokens"]))
    return {"id": mid, "role": "assistant", **res}
