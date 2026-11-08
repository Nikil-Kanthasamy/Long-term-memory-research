"""SQLite persistence with temporal validity and versioned memories."""
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
            # Additive migrations preserve databases created by Phase 1.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memories)")}
            for name, declaration in (
                ("valid_from", "TEXT"),
                ("valid_until", "TEXT"),
                ("supersedes_id", "INTEGER"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def add_turn(self, user_id, text):
        with self.connect() as db:
            cur = db.execute("INSERT INTO turns(user_id,text,created_at) VALUES(?,?,?)",
                             (user_id, text, now()))
            return int(cur.lastrowid)

    def add_memory(self, user_id, content, category="fact", importance=0.5,
                   confidence=0.7, embedding=None, source_turn_id=None,
                   valid_from=None, valid_until=None, supersedes_id=None):
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")
        timestamp = now()
        with self.connect() as db:
            if supersedes_id is not None:
                old = db.execute(
                    "SELECT id FROM memories WHERE id=? AND user_id=? AND status='active'",
                    (int(supersedes_id), user_id)).fetchone()
                if old is None:
                    raise ValueError("The memory to supersede does not exist or is not active for this user")
                db.execute("UPDATE memories SET status='superseded', valid_until=?, updated_at=? WHERE id=?",
                           (timestamp, timestamp, int(supersedes_id)))
            else:
                old = db.execute("""SELECT id FROM memories WHERE user_id=? AND status='active'
                    AND lower(content)=lower(?)""", (user_id, content)).fetchone()
                if old:
                    db.execute("UPDATE memories SET updated_at=? WHERE id=?", (timestamp, old["id"]))
                    return int(old["id"])
            cur = db.execute("""INSERT INTO memories
                (user_id,content,category,importance,confidence,embedding,source_turn_id,
                 status,created_at,updated_at,valid_from,valid_until,supersedes_id)
                VALUES(?,?,?,?,?,?,?,'active',?,?,?,?,?)""",
                (user_id, content, category, float(importance), float(confidence),
                 json.dumps(embedding) if embedding else None, source_turn_id,
                 timestamp, timestamp, valid_from or timestamp, valid_until, supersedes_id))
            return int(cur.lastrowid)

    def add_memories_batch(self, user_id, records):
        """Insert benchmark/session records in one transaction; return their IDs."""
        timestamp = now()
        ids = []
        with self.connect() as db:
            for record in records:
                content = str(record.get("content", "")).strip()
                if not content:
                    continue
                cur = db.execute("""INSERT INTO memories
                    (user_id,content,category,importance,confidence,embedding,source_turn_id,
                     status,created_at,updated_at,valid_from,valid_until,supersedes_id)
                    VALUES(?,?,?, ?,?,?,NULL,'active',?,?,?,NULL,NULL)""",
                    (user_id, content, str(record.get("category", "session")),
                     float(record.get("importance", 0.5)),
                     float(record.get("confidence", 1.0)), None,
                     timestamp, timestamp, record.get("valid_from")))
                ids.append(int(cur.lastrowid))
        return ids

    def active_memories(self, user_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM memories WHERE user_id=? AND status='active' ORDER BY updated_at DESC",
                              (user_id,)).fetchall()
        return [dict(row) for row in rows]

    def memory_history(self, user_id, memory_id):
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM memories WHERE user_id=?
                AND (id=? OR supersedes_id=?) ORDER BY created_at""",
                (user_id, memory_id, memory_id)).fetchall()
        return [dict(row) for row in rows]

    def delete_memory(self, user_id, memory_id):
        with self.connect() as db:
            cur = db.execute("DELETE FROM memories WHERE user_id=? AND id=?", (user_id, memory_id))
            return cur.rowcount > 0

    def search_turns(self, user_id, query, limit=5):
        query_terms = set(_terms(query))
        with self.connect() as db:
            rows = db.execute("SELECT * FROM turns WHERE user_id=? ORDER BY id DESC",
                              (user_id,)).fetchall()
        found = []
        for row in rows:
            terms = set(_terms(row["text"]))
            score = len(query_terms & terms) / max(1, len(query_terms | terms))
            if score:
                item = dict(row)
                item["score"] = score
                found.append(item)
        return sorted(found, key=lambda x: x["score"], reverse=True)[:limit]

    def search(self, user_id, query, query_embedding=None, limit=5, as_of=None, mode="hybrid"):
        if mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError("mode must be lexical, semantic, or hybrid")
        if mode == "semantic" and query_embedding is None:
            raise ValueError("semantic retrieval requires a query embedding")
        query_terms = set(_terms(query))
        as_of = as_of or now()
        results = []
        for row in self.active_memories(user_id):
            if row.get("valid_from") and row["valid_from"] > as_of:
                continue
            if row.get("valid_until") and row["valid_until"] <= as_of:
                continue
            terms = set(_terms(row["content"]))
            lexical = len(query_terms & terms) / max(1, len(query_terms | terms))
            semantic = 0.0
            if query_embedding and row["embedding"]:
                try:
                    semantic = cosine(query_embedding, json.loads(row["embedding"]))
                except (ValueError, TypeError):
                    pass
            if mode == "semantic":
                score = semantic
            elif mode == "hybrid":
                score = 0.75 * semantic + 0.25 * lexical if query_embedding else lexical
            else:
                score = lexical
            if score > 0:
                score += 0.05 * float(row["importance"])
                row["score"] = score
                results.append(row)
        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

def _terms(text):
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 1]

def cosine(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x*y for x, y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0
