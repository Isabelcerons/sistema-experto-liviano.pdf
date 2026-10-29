"""SQLite: documentos, páginas, imágenes (con su texto OCR y descripción), fragmentos con embedding,
mensajes y ajustes. Un solo archivo, sin instalar servidores de base de datos."""

import json
import sqlite3
import time
from contextlib import contextmanager

import numpy as np

from .config import DB_PATH, DEFAULT_SETTINGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL DEFAULT '',
    pages INTEGER NOT NULL DEFAULT 0,
    images INTEGER NOT NULL DEFAULT 0,
    ocr_pages INTEGER NOT NULL DEFAULT 0,
    embed_model TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'pdf',      -- pdf (texto real) | ocr (página escaneada leída con Tesseract)
    PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    path TEXT NOT NULL,
    width INTEGER, height INTEGER,
    ocr_text TEXT NOT NULL DEFAULT '',       -- palabras escritas dentro de la imagen (Tesseract)
    description TEXT NOT NULL DEFAULT ''     -- lo que "ve" el modelo de visión (se llena bajo demanda)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'text',       -- text | image
    image_id INTEGER,
    text TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_images_doc ON images(doc_id);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                      -- summary
    content TEXT NOT NULL,
    created REAL NOT NULL,
    PRIMARY KEY (doc_id, kind)
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def to_blob(v) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return np.frombuffer(b, dtype=np.float32) if b else None


# ---------- ajustes ----------

def get_settings() -> dict:
    with connect() as con:
        rows = con.execute("SELECT key, value FROM settings").fetchall()
    s = dict(DEFAULT_SETTINGS)
    s.update({r["key"]: r["value"] for r in rows})
    return s


def save_settings(values: dict) -> None:
    with connect() as con:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS:
                con.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)", (k, str(v)))


# ---------- documentos ----------

def create_document(name: str) -> int:
    with connect() as con:
        return con.execute("INSERT INTO documents(name, created) VALUES(?, ?)", (name, time.time())).lastrowid


def update_document(doc_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE documents SET {cols} WHERE id = ?", (*fields.values(), doc_id))


def list_documents() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM documents WHERE pages > 0 ORDER BY id")]


def get_document(doc_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(r) if r else None


def delete_document(doc_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        con.execute("DELETE FROM messages WHERE scope = ?", (f"doc:{doc_id}",))


# ---------- páginas e imágenes ----------

def add_pages(doc_id: int, pages: list[tuple[int, str, str]]) -> None:
    with connect() as con:
        con.executemany("INSERT OR REPLACE INTO pages(doc_id, page, text, source) VALUES(?,?,?,?)",
                        [(doc_id, p, t, src) for p, t, src in pages])


def get_pages(doc_ids: list[int]) -> list[dict]:
    if not doc_ids:
        return []
    q = ",".join("?" * len(doc_ids))
    with connect() as con:
        return [dict(r) for r in con.execute(
            f"SELECT doc_id, page, text, source FROM pages WHERE doc_id IN ({q}) ORDER BY doc_id, page", doc_ids)]


def add_image(doc_id: int, page: int, path: str, w: int, h: int, ocr_text: str) -> int:
    with connect() as con:
        return con.execute("INSERT INTO images(doc_id, page, path, width, height, ocr_text) VALUES(?,?,?,?,?,?)",
                           (doc_id, page, path, w, h, ocr_text)).lastrowid


def get_images(doc_ids: list[int]) -> list[dict]:
    if not doc_ids:
        return []
    q = ",".join("?" * len(doc_ids))
    with connect() as con:
        return [dict(r) for r in con.execute(
            f"SELECT * FROM images WHERE doc_id IN ({q}) ORDER BY doc_id, page, id", doc_ids)]


def get_image(image_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(r) if r else None


def set_image_description(image_id: int, description: str) -> None:
    with connect() as con:
        con.execute("UPDATE images SET description = ? WHERE id = ?", (description, image_id))


# ---------- fragmentos ----------

def add_chunks(doc_id: int, items: list[dict]) -> None:
    with connect() as con:
        con.executemany(
            "INSERT INTO chunks(doc_id, page, kind, image_id, text, embedding) VALUES(?,?,?,?,?,?)",
            [(doc_id, it["page"], it.get("kind", "text"), it.get("image_id"), it["text"],
              to_blob(it["embedding"]) if it.get("embedding") is not None else None) for it in items])


def replace_image_chunk(doc_id: int, image_id: int, page: int, text: str, embedding) -> None:
    with connect() as con:
        con.execute("DELETE FROM chunks WHERE image_id = ?", (image_id,))
        con.execute("INSERT INTO chunks(doc_id, page, kind, image_id, text, embedding) VALUES(?,?,?,?,?,?)",
                    (doc_id, page, "image", image_id, text, to_blob(embedding) if embedding is not None else None))


def get_chunks(doc_ids: list[int]) -> list[dict]:
    if not doc_ids:
        return []
    q = ",".join("?" * len(doc_ids))
    with connect() as con:
        rows = con.execute(f"SELECT * FROM chunks WHERE doc_id IN ({q}) ORDER BY doc_id, page, id", doc_ids)
        out = []
        for r in rows:
            d = dict(r)
            d["embedding"] = from_blob(d["embedding"])
            out.append(d)
        return out


# ---------- mensajes ----------

def add_message(scope: str, role: str, content: str, meta: dict | None = None) -> None:
    with connect() as con:
        con.execute("INSERT INTO messages(scope, role, content, meta, created) VALUES(?,?,?,?,?)",
                    (scope, role, content, json.dumps(meta, ensure_ascii=False) if meta else None, time.time()))


def get_messages(scope: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT role, content, meta FROM messages WHERE scope = ? ORDER BY id", (scope,))
        return [{"role": r["role"], "content": r["content"], "meta": json.loads(r["meta"]) if r["meta"] else {}}
                for r in rows]


def clear_messages(scope: str) -> None:
    with connect() as con:
        con.execute("DELETE FROM messages WHERE scope = ?", (scope,))


# ---------- análisis (resumen) ----------

def get_analysis(doc_id: int, kind: str) -> str | None:
    with connect() as con:
        r = con.execute("SELECT content FROM analysis WHERE doc_id = ? AND kind = ?", (doc_id, kind)).fetchone()
        return r["content"] if r else None


def save_analysis(doc_id: int, kind: str, content: str) -> None:
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO analysis(doc_id, kind, content, created) VALUES(?,?,?,?)",
                    (doc_id, kind, content, time.time()))
