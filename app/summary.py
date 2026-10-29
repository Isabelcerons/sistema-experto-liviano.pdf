"""Resumen RÁPIDO del PDF, pensado para un PC sin tarjeta gráfica.

En vez de que la IA lea todo el documento (lento), se hace en 2 pasos:
1. SIN IA (instantáneo): con los embeddings que ya se calcularon al subir el PDF, se agrupan los
   fragmentos por tema (k-means) y de cada grupo se toma el fragmento más representativo, con su página.
   Así salen las "ideas clave" de todo el documento, de principio a fin.
2. CON IA (una sola llamada, ~1 min): el modelo redacta el resumen a partir de esas ideas clave.
El resultado se guarda: la segunda vez se muestra al instante."""

import re
from typing import AsyncIterator

import numpy as np

from . import db, llm

N_KEY = 8            # ideas clave (grupos de temas)
KEY_CHARS = 380      # texto que se toma de cada idea clave

SUMMARY_PROMPT = """Estas son las IDEAS CLAVE del documento "{doc}" ({pages} páginas), en orden, con su página.
Escribe un resumen en español con esta estructura:

## 📌 De qué trata
2 o 3 oraciones sobre el tema central.

## 💡 Ideas principales
5 a 7 viñetas; cada una termina con su cita [{doc}, pág. N].

## ✅ Conclusión
2 oraciones.

Usa SOLO la información de las ideas clave. No inventes.

IDEAS CLAVE:
{ideas}"""


def _kmeans(x: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    """k-means simple (coseno). Devuelve la etiqueta de grupo de cada fila."""
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    idx = np.linspace(0, len(x) - 1, k).astype(int)          # semillas repartidas por todo el documento
    cent = x[idx].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        labels = np.argmax(x @ cent.T, axis=1)
        for j in range(k):
            m = x[labels == j]
            if len(m):
                c = m.mean(axis=0)
                cent[j] = c / (np.linalg.norm(c) + 1e-9)
    return labels, cent, x


def _clean(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    # los fragmentos se solapan: si empieza a mitad de frase, se arranca en la siguiente oración
    if t and not (t[0].isupper() or t[0].isdigit() or t[0] in "¿¡\"«(-•"):
        k = t.find(". ")
        if 0 <= k < 200:
            t = t[k + 2:]
    if len(t) <= KEY_CHARS:
        return t
    cut = t[:KEY_CHARS]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > KEY_CHARS * 0.5 else cut.rsplit(" ", 1)[0] + "…")


def key_ideas(doc_id: int) -> list[dict]:
    chunks = [c for c in db.get_chunks([doc_id]) if c["embedding"] is not None and len(c["text"]) > 80]
    if not chunks:
        return []
    if len(chunks) <= N_KEY:
        chosen = chunks
    else:
        x = np.vstack([c["embedding"] for c in chunks]).astype(np.float32)
        labels, cent, xn = _kmeans(x, N_KEY)
        chosen = []
        for j in range(N_KEY):
            members = np.where(labels == j)[0]
            if len(members):
                best = members[np.argmax(xn[members] @ cent[j])]   # el más representativo del grupo
                chosen.append(chunks[best])
    chosen.sort(key=lambda c: (c["page"], c["id"]))
    return [{"page": c["page"], "text": _clean(c["text"])} for c in chosen]


def ideas_markdown(doc: dict, ideas: list[dict]) -> str:
    lines = ["## 🔑 Ideas clave del documento"]
    lines += [f"- {i['text']} [{doc['name']}, pág. {i['page']}]" for i in ideas]
    return "\n".join(lines)


async def summary_stream(doc: dict, s: dict, force: bool = False) -> AsyncIterator[dict]:
    if not force and (cached := db.get_analysis(doc["id"], "summary")):
        yield {"type": "done", "content": cached, "cached": True}
        return
    ideas = key_ideas(doc["id"])
    if not ideas:
        raise ValueError("El documento no tiene texto para resumir.")
    ideas_md = ideas_markdown(doc, ideas)
    # 1) lo instantáneo: las ideas clave ya se pueden leer mientras la IA redacta
    yield {"type": "ideas", "content": ideas_md}
    yield {"type": "progress", "pct": 40, "message": "Redactando el resumen con la IA (≈1 minuto)…"}
    # 2) una sola llamada a la IA
    ideas_txt = "\n".join(f"- (pág. {i['page']}) {i['text']}" for i in ideas)
    prompt = SUMMARY_PROMPT.format(doc=doc["name"], pages=doc["pages"], ideas=ideas_txt)
    out = []
    async for piece in llm.chat_stream([{"role": "user", "content": prompt}], s, max_tokens=550):
        out.append(piece)
        yield {"type": "token", "text": piece}
    content = "".join(out).strip() + "\n\n" + ideas_md
    db.save_analysis(doc["id"], "summary", content)
    yield {"type": "done", "content": content}
