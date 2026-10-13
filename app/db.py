"""SQLite: notebooks, sources, chunk text, chat sessions, messages. Vectors live in Qdrant."""
import os, sqlite3, threading

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
_conn = sqlite3.connect(os.path.join(DATA_DIR, "app.db"), check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()
_conn.executescript("""
CREATE TABLE IF NOT EXISTS notebooks(id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, notebook_id INT, name TEXT,
  n_chunks INT, cost REAL DEFAULT 0, checked INT DEFAULT 1);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, source_id INT, idx INT, page INT, text TEXT);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, notebook_id INT, title TEXT);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, session_id INT, role TEXT,
  content TEXT, citations TEXT DEFAULT '[]', cost REAL DEFAULT 0, tokens INT DEFAULT 0);
""")

def q(sql, args=(), one=False):
    with _lock:
        rows = [dict(r) for r in _conn.execute(sql, args).fetchall()]
        _conn.commit()
    return (rows[0] if rows else None) if one else rows

def ins(sql, args=()):
    with _lock:
        cur = _conn.execute(sql, args)
        _conn.commit()
        return cur.lastrowid
