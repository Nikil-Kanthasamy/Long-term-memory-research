"""SQLite persistence for conversation turns and long-term memory items."""
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat()

class MemoryStore:
    def __init__(self, path):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                text TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                content TEXT NOT NULL, category TEXT NOT NULL DEFAULT 'fact',
                importance REAL NOT NULL DEFAULT 0.5, confidence REAL NOT NULL DEFAULT 0.7,
                embedding TEXT, source_turn_id INTEGER, status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_mem_user_status ON memories(user_id,status);
            """)

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def add_turn(self, user_id, text):
        with self.connect() as db:
            cur = db.execute("INSERT INTO turns(user_id,text,created_at) VALUES(?,?,?)",
                             (user_id,text,now()))
            return int(cur.lastrowid)

    def add_memory(self, user_id, content, category="fact", importance=0.5,
                   confidence=0.7, embedding=None, source_turn_id=None):
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        with self.connect() as db:
            old = db.execute("""SELECT id FROM memories WHERE user_id=? AND status='active'
                AND lower(content)=lower(?)""",(user_id,content)).fetchone()
            if old:
                db.execute("UPDATE memories SET updated_at=? WHERE id=?",(now(),old["id"]))
                return int(old["id"])
            cur = db.execute("""INSERT INTO memories
                (user_id,content,category,importance,confidence,embedding,source_turn_id,status,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,'active',?,?)""",
                (user_id,content,category,float(importance),float(confidence),
                 json.dumps(embedding) if embedding else None,source_turn_id,now(),now()))
            return int(cur.lastrowid)

    def active_memories(self, user_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM memories WHERE user_id=? AND status='active' ORDER BY updated_at DESC",
                              (user_id,)).fetchall()
        return [dict(row) for row in rows]

    def delete_memory(self, user_id, memory_id):
        with self.connect() as db:
            cur = db.execute("DELETE FROM memories WHERE user_id=? AND id=?",(user_id,memory_id))
            return cur.rowcount > 0

    def search(self, user_id, query, query_embedding=None, limit=5):
        q = set(re.findall(r"[a-z0-9_]+",query.lower()))
        results = []
        for row in self.active_memories(user_id):
            terms = set(re.findall(r"[a-z0-9_]+",row["content"].lower()))
            lexical = len(q & terms) / max(1,len(q | terms))
            semantic = 0.0
            if query_embedding and row["embedding"]:
                try:
                    semantic = cosine(query_embedding,json.loads(row["embedding"]))
                except (ValueError,TypeError):
                    pass
            score = (0.75*semantic + 0.25*lexical) if query_embedding else lexical
            score += 0.05*float(row["importance"])
            if score > 0:
                row["score"] = score
                results.append(row)
        return sorted(results,key=lambda x:x["score"],reverse=True)[:limit]

def cosine(a,b):
    if len(a)!=len(b) or not a:
        return 0.0
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0
