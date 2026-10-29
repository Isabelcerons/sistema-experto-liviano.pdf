"""Sistema Experto PDF (liviano): API (FastAPI) + interfaz web.

Pensado para PC sin tarjeta gráfica:
- "¿Dónde dice …?" se responde al instante, SIN IA (búsqueda literal en texto + texto de imágenes).
- Las palabras dentro de imágenes y páginas escaneadas se leen con Tesseract (OCR rápido).
- Las imágenes se "miran" con IA solo cuando preguntas por ellas (o con Gemini, en la nube, si quieres).
- El chat usa un modelo pequeño (qwen2.5:1.5b) que se queda cargado en memoria."""

import asyncio
import json
import shutil

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, llm, ocr, pdf, rag, search, summary
from .config import (BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_SETTINGS, IMG_DIR, MAX_FILE_MB, MAX_IMAGES,
                     MAX_PAGES, MIN_IMAGE_PX, PDF_DIR, TOP_K)

app = FastAPI(title="Sistema Experto PDF")
db.init()


@app.on_event("startup")
async def _startup():
    asyncio.create_task(llm.warm_up(db.get_settings()))
    describe_pending()  # imágenes que quedaron sin mirar la vez anterior


def ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def resolve_scope(scope: str) -> list[int]:
    if scope == "all":
        return [d["id"] for d in db.list_documents()]
    if scope.startswith("doc:") and scope[4:].isdigit():
        if not db.get_document(int(scope[4:])):
            raise HTTPException(404, "Documento no encontrado")
        return [int(scope[4:])]
    raise HTTPException(400, "scope inválido")


def names_map() -> dict[int, str]:
    return {d["id"]: d["name"] for d in db.list_documents()}


def search_entries(ids: list[int]) -> list[dict]:
    """Todo lo que se puede buscar literalmente: texto de cada página + texto de cada imagen."""
    entries = db.get_pages(ids)
    for im in db.get_images(ids):
        txt = "\n".join(t for t in (im["ocr_text"], im["description"]) if t)
        if txt:
            entries.append({"doc_id": im["doc_id"], "page": im["page"], "text": txt, "image_id": im["id"]})
    return entries


# ---------------- estado y ajustes ----------------

@app.get("/api/status")
async def status():
    s = db.get_settings()
    out = {"vision_provider": s["vision_provider"], "chat_model": s["chat_model"], "ocr": ocr.status(s)}
    try:
        have = await llm.ollama_models(s)
        need = [s["chat_model"], s["embed_model"]] + ([s["vision_model"]] if s["vision_provider"] == "ollama" else [])
        missing = [m for m in need if m not in have]
        out["ollama"] = {"ok": not missing, "models": have, "missing": missing,
                         "error": ("Falta descargar: " + ", ".join(f"ollama pull {m}" for m in missing)) if missing else None}
    except llm.LLMError as e:
        out["ollama"] = {"ok": False, "models": [], "missing": [], "error": str(e)}
    if s["vision_provider"] == "gemini" and not s.get("gemini_api_key"):
        out["vision_warning"] = "Visión con Gemini sin clave: ponla en ⚙ Ajustes."
    return out


def _public(s: dict) -> dict:
    out = dict(s)
    key = out.pop("gemini_api_key", "")
    out["gemini_key_set"] = bool(key)
    return out


@app.get("/api/settings")
def get_settings():
    return _public(db.get_settings())


@app.put("/api/settings")
def put_settings(values: dict):
    values = {k: str(v).strip() for k, v in values.items() if k in DEFAULT_SETTINGS and v is not None}
    if values.get("vision_provider") not in (None, "ollama", "gemini"):
        raise HTTPException(400, "vision_provider debe ser 'ollama' o 'gemini'")
    if values.get("gemini_api_key") == "":
        values.pop("gemini_api_key")
    db.save_settings(values)
    ocr.find_tesseract.cache_clear()
    return _public(db.get_settings())


# ---------------- documentos ----------------

@app.get("/api/documents")
def documents():
    return db.list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Documento no encontrado")
    db.delete_document(doc_id)
    if doc["path"]:
        (PDF_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    shutil.rmtree(IMG_DIR / str(doc_id), ignore_errors=True)
    return {"ok": True}


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "documento.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF")
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"El PDF supera {MAX_FILE_MB} MB")
    if any(d["name"].lower() == name.lower() for d in db.list_documents()):
        raise HTTPException(400, f'"{name}" ya está cargado. Si quieres volver a subirlo, primero elimínalo de la lista (✕).')

    async def gen():
        s = db.get_settings()
        doc_id = None
        try:
            yield ndjson({"pct": 2, "msg": "Leyendo el PDF…"})
            try:
                parsed = await asyncio.to_thread(pdf.parse, data, MAX_PAGES, MAX_IMAGES, MIN_IMAGE_PX)
            except Exception as e:
                raise ValueError("No se pudo leer el PDF: puede estar dañado o protegido con contraseña. "
                                 f"({type(e).__name__})") from e
            if parsed.total_pages > MAX_PAGES:
                yield ndjson({"warning": f"El PDF tiene {parsed.total_pages} páginas: solo se leen las primeras {MAX_PAGES}."})
            if parsed.skipped_images:
                yield ndjson({"warning": f"Se omitieron {parsed.skipped_images} imágenes (máximo {MAX_IMAGES} por PDF)."})
            doc_id = db.create_document(name)
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            pdf_path = PDF_DIR / f"{doc_id}.pdf"
            pdf_path.write_bytes(data)
            img_dir = IMG_DIR / str(doc_id)
            img_dir.mkdir(parents=True, exist_ok=True)
            st = ocr.status(s)
            if (parsed.scanned or parsed.images) and not st["ok"]:
                yield ndjson({"warning": st["error"]})

            # 1) Páginas escaneadas → OCR (Tesseract)
            page_text = {p: (t, "pdf") for p, t in parsed.pages}
            for k, sp in enumerate(parsed.scanned, 1):
                yield ndjson({"pct": 5 + int(35 * (k - 1) / len(parsed.scanned)),
                              "msg": f"Leyendo página escaneada {k}/{len(parsed.scanned)} (pág. {sp.page}) con OCR…"})
                text = await asyncio.to_thread(ocr.read, sp.png, s)
                if not text and not st["ok"] and s["vision_provider"] == "gemini" and s.get("gemini_api_key"):
                    text = await llm.describe_image(sp.png, "Transcribe TODO el texto de esta página, tal cual.", s)
                page_text[sp.page] = (text, "ocr")
            db.add_pages(doc_id, [(p, t, src) for p, (t, src) in sorted(page_text.items())])

            # 2) Imágenes → guardar + leer las palabras que tengan (OCR) + (opcional) describirlas
            describe_now = s["describe_on_upload"] == "yes"
            images = []
            for k, im in enumerate(parsed.images, 1):
                yield ndjson({"pct": 40 + int(30 * (k - 1) / len(parsed.images)),
                              "msg": f"Revisando imagen {k}/{len(parsed.images)} (pág. {im.page})…"})
                path = img_dir / f"p{im.page}_{k}.png"
                path.write_bytes(im.png)
                text = await asyncio.to_thread(ocr.read, im.png, s, True)
                img = {"id": db.add_image(doc_id, im.page, str(path), im.width, im.height, text),
                       "page": im.page, "ocr_text": text, "description": ""}
                if describe_now:
                    try:
                        img["doc_id"], img["path"] = doc_id, str(path)
                        await describe(img, s, {doc_id: name}, reindex=False)
                    except llm.LLMError as e:
                        describe_now = False
                        yield ndjson({"warning": f"No se pudieron describir las imágenes ({e}). "
                                                 "Se describirán cuando preguntes por ellas."})
                images.append(img)

            # 3) Fragmentos + embeddings
            items = [{"page": p, "kind": "text", "text": t}
                     for p, t in pdf.chunk_pages([(p, t) for p, (t, _) in sorted(page_text.items())],
                                                 CHUNK_SIZE, CHUNK_OVERLAP)]
            items += [{"page": im["page"], "kind": "image", "image_id": im["id"], "text": rag.image_chunk_text(im)}
                      for im in images]
            if not items:
                raise ValueError("El PDF no tiene texto legible. Si es escaneado, instala Tesseract (ver README).")
            for i in range(0, len(items), 16):
                batch = items[i:i + 16]
                for b, v in zip(batch, await llm.embed([b["text"] for b in batch], s)):
                    b["embedding"] = v
                yield ndjson({"pct": 70 + int(30 * min(i + 16, len(items)) / len(items)),
                              "msg": f"Indexando {min(i + 16, len(items))}/{len(items)} fragmentos…"})
            db.add_chunks(doc_id, items)
            db.update_document(doc_id, path=str(pdf_path), pages=len(parsed.pages), images=len(images),
                               ocr_pages=len(parsed.scanned), embed_model=s["embed_model"])
            yield ndjson({"pct": 100, "msg": "Listo", "done": True, "document": db.get_document(doc_id)})
            if images and not describe_now:
                describe_pending(doc_id)
        except Exception as e:
            if not isinstance(e, (llm.LLMError, ValueError)):
                e = ValueError(f"Error inesperado al procesar el PDF: {e}")
            if doc_id:
                db.delete_document(doc_id)
                shutil.rmtree(IMG_DIR / str(doc_id), ignore_errors=True)
            yield ndjson({"error": str(e)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/documents/{doc_id}/page/{page}")
def page_png(doc_id: int, page: int, hl: str | None = None):
    doc = db.get_document(doc_id)
    if not doc or not 1 <= page <= doc["pages"]:
        raise HTTPException(404, "Página no encontrada")
    return Response(pdf.render_page(doc["path"], page, hl), media_type="image/png",
                    headers={"Cache-Control": "no-store" if hl else "max-age=3600"})


# ---------------- imágenes ----------------

@app.get("/api/documents/{doc_id}/images")
def doc_images(doc_id: int):
    return [{k: v for k, v in im.items() if k != "path"} for im in db.get_images([doc_id])]


@app.get("/api/images/{image_id}/file")
def image_file(image_id: int):
    im = db.get_image(image_id)
    if not im:
        raise HTTPException(404, "Imagen no encontrada")
    return FileResponse(im["path"], media_type="image/png")


_locks: dict[int, asyncio.Lock] = {}


async def describe(im: dict, s: dict, names: dict, reindex: bool = True) -> str:
    """Mira la imagen con IA (una sola vez: queda guardado) y la vuelve a indexar con su descripción."""
    lock = _locks.setdefault(im["id"], asyncio.Lock())
    async with lock:  # si ya la está mirando el proceso de fondo, espera y reutiliza el resultado
        saved = db.get_image(im["id"])
        if saved and saved["description"]:
            im["description"] = saved["description"]
            return im["description"]
        return await _describe(im, s, names, reindex)


_bg = asyncio.Lock()
_busy = {"n": 0}  # preguntas o resúmenes en curso: el trabajo de fondo espera para no competir por el procesador


_focus = {"doc": None}            # documento que el usuario está consultando: sus imágenes van primero
_wake = asyncio.Event()
_worker = {"task": None}


def describe_pending(doc_id: int | None = None) -> None:
    """Pide al trabajador de fondo que revise las imágenes que falten (las del doc_id primero)."""
    if doc_id:
        _focus["doc"] = doc_id
    if not _worker["task"] or _worker["task"].done():
        _worker["task"] = asyncio.create_task(_bg_worker())
    _wake.set()


def _next_pending() -> dict | None:
    """Siguiente imagen sin describir: primero el documento en foco, luego el más reciente."""
    imgs = [im for im in db.get_images([d["id"] for d in db.list_documents()]) if not im["description"]]
    if not imgs:
        return None
    imgs.sort(key=lambda im: (im["doc_id"] != _focus["doc"], -im["doc_id"], im["page"], im["id"]))
    return imgs[0]


async def _bg_worker() -> None:
    """Un solo trabajador en segundo plano: describe las imágenes de una en una, sin saturar el PC."""
    await asyncio.sleep(2)
    fails = 0
    while True:
        im = _next_pending()
        if im is None:
            _wake.clear()
            await _wake.wait()
            continue
        while _busy["n"] > 0:  # primero lo que pidió el usuario
            await asyncio.sleep(2)
        try:
            await describe(im, db.get_settings(), names_map())
            fails = 0
        except Exception:      # Ollama ocupado o apagado: reintenta más tarde
            fails += 1
            await asyncio.sleep(min(60, 10 * fails))


async def _describe(im: dict, s: dict, names: dict, reindex: bool) -> str:
    png = open(im["path"], "rb").read()
    desc = await llm.describe_image(png, rag.image_prompt(s, im["page"], names.get(im["doc_id"], "")), s)
    if rag.needs_translation(s) and desc:
        # moondream describe en inglés: se traduce para poder buscar en español ("carro", "persona"…)
        try:
            es = await llm.chat_once(rag.translate_messages(desc), s)
            desc = f"{es}\n(original en inglés: {desc})" if es else desc
        except llm.LLMError:
            pass
    db.set_image_description(im["id"], desc)
    im["description"] = desc
    if not reindex:  # al subir: el fragmento se indexa después junto con los demás
        return desc
    text = rag.image_chunk_text(im)
    vec = (await llm.embed([text], s))[0]
    db.replace_image_chunk(im["doc_id"], im["id"], im["page"], text, vec)
    return desc


@app.post("/api/images/{image_id}/describe")
async def describe_one(image_id: int):
    im = db.get_image(image_id)
    if not im:
        raise HTTPException(404, "Imagen no encontrada")
    try:
        return {"description": await describe(im, db.get_settings(), names_map())}
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


# ---------------- resumen ----------------

@app.get("/api/documents/{doc_id}/summary")
def get_summary(doc_id: int):
    if not db.get_document(doc_id):
        raise HTTPException(404, "Documento no encontrado")
    return {"content": db.get_analysis(doc_id, "summary")}


@app.post("/api/documents/{doc_id}/summary")
async def make_summary(doc_id: int, force: bool = False):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Documento no encontrado")

    async def gen():
        _busy["n"] += 1
        try:
            async for ev in summary.summary_stream(doc, db.get_settings(), force):
                yield ndjson(ev)
        except (llm.LLMError, ValueError) as e:
            yield ndjson({"type": "error", "text": str(e)})
        finally:
            _busy["n"] -= 1

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------------- búsqueda literal ----------------

@app.get("/api/search")
def phrase_search(q: str = Query(..., min_length=2), scope: str = "all"):
    ids = resolve_scope(scope)
    res = search.locate(q, search_entries(ids))
    names = names_map()
    for h in res["hits"]:
        h["doc_name"] = names.get(h["doc_id"], "?")
    return res


# ---------------- chat ----------------

class Ask(BaseModel):
    scope: str
    question: str


@app.post("/api/chat")
async def chat(body: Ask):
    ids = resolve_scope(body.scope)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "La pregunta está vacía")
    if not ids:
        raise HTTPException(400, "Primero sube un PDF")

    async def gen():
        _busy["n"] += 1
        try:
            async for line in _chat(body, ids, question):
                yield line
        finally:
            _busy["n"] -= 1

    async def _chat(body, ids, question):
        s = db.get_settings()
        names = names_map()
        try:
            def instant(text: str, meta: dict | None = None):
                meta = {"locations": [], "sources": [], **(meta or {}), "instant": True}
                db.add_message(body.scope, "user", question)
                db.add_message(body.scope, "assistant", text, meta)
                return [ndjson({"type": "meta", **meta}), ndjson({"type": "token", "text": text}), ndjson({"type": "done"})]

            # 0) Saludos y preguntas sobre el propio documento (páginas, imágenes) → sin IA
            small = rag.small_talk(question)
            if small:
                for line in instant(small):
                    yield line
                return
            docs = [db.get_document(i) for i in ids]
            count = rag.count_answer(question, docs, db.get_images(ids))
            if count:
                for line in instant(count):
                    yield line
                return

            # 1) ¿Pregunta dónde está una palabra/frase? → respuesta instantánea, sin IA
            locations = [search.locate(p, search_entries(ids)) for p in search.candidate_phrases(question)]
            for loc in locations:
                for h in loc["hits"]:
                    h["doc_name"] = names.get(h["doc_id"], "?")
            if locations and search.is_locate_question(question) and all(l["hits"] for l in locations) \
                    and not rag.is_image_question(question):
                for line in instant(search.locate_answer(locations), {"locations": locations}):
                    yield line
                return
            if locations and search.is_locate_question(question) and not all(l["hits"] for l in locations):
                missing = [l["phrase"] for l in locations if not l["hits"]]
                # ¿no está escrito pero sí se VE en una imagen? ("¿en qué página hay un carro?")
                words = [w for p in missing for w in search.content_words(p)]
                described = [im for im in db.get_images(ids) if im["description"]]
                # todas las palabras deben verse en la MISMA imagen ("mariposa verde" ≠ "árbol verde")
                seen = [im for im in described if words and all(rag.find_object([im], [w]) for w in words)]
                if seen:
                    all_imgs = db.get_images(ids)
                    pos = {im["id"]: i + 1 for i, im in enumerate(all_imgs)}
                    text = (f'**"{", ".join(missing)}"** no aparece escrito, pero sí se ve en '
                            f'{"una imagen" if len(seen) == 1 else f"{len(seen)} imágenes"}:\n\n'
                            + rag.describe_answer([(pos[im["id"]], im) for im in seen[:3]], names, len(all_imgs)))
                    for line in instant(text):
                        yield line
                    return
                if search.is_literal_question(question):
                    text = (f'**"{", ".join(missing)}"** no aparece en '
                            f'{"ninguno de los documentos" if len(ids) > 1 else "el documento"}, ni en el texto ni '
                            f'dentro de las imágenes revisadas.\n\nRevisa cómo está escrito o pregúntame con otras palabras '
                            f'(por ejemplo: *¿qué dice el documento sobre …?*).')
                    for line in instant(text, {"locations": locations}):
                        yield line
                    return

            # 1b') "¿Hay fotos de gatos?" → se revisan las descripciones guardadas de TODAS las imágenes
            if len(ids) == 1:
                _focus["doc"] = ids[0]
            obj = rag.object_query(question)
            if not obj:  # "¿y de un carro?" justo después de preguntar por otra imagen
                prev = [m["content"] for m in db.get_messages(body.scope) if m["role"] == "user"]
                obj = rag.followup_object(question, prev[-1] if prev else None)
            if obj:
                all_imgs = db.get_images(ids)
                if not all_imgs:
                    text = "Este documento no tiene imágenes."
                else:
                    # NO se espera a la IA: se responde con lo que ya está revisado (descripciones + OCR)
                    pending = [im for im in all_imgs if not im["description"]]
                    found = rag.find_object(all_imgs, obj)
                    # si no aparece y faltan pocas imágenes por revisar, se revisan ya (respuesta correcta)
                    if not found and 0 < len(pending) <= 4:
                        for k, im in enumerate(pending, 1):
                            yield ndjson({"type": "status", "text": f"👁 Revisando imagen {k} de {len(pending)} (pág. {im['page']})…"})
                            try:
                                await describe(im, s, names)
                            except llm.LLMError:
                                break
                        found = rag.find_object(all_imgs, obj)
                        pending = [im for im in all_imgs if not im["description"]]
                    if pending:  # las que falten se siguen revisando solas en segundo plano
                        describe_pending(pending[0]["doc_id"])
                    what = " ".join(obj)
                    falta = (f"\n\n⏳ Aún faltan **{len(pending)}** de {len(all_imgs)} imágenes por revisar con la IA "
                             f"(se están revisando solas). Pregúntame de nuevo en un momento para una respuesta completa."
                             if pending else "")
                    if found:
                        text = f"Sí. Encontré **{len(found)}** imagen(es) relacionadas con **{what}**:\n\n" + rag.describe_answer(
                            [(all_imgs.index(im) + 1, im) for im in found[:3]], names, len(all_imgs))
                    else:
                        done_imgs = [im for im in all_imgs if im["description"]]
                        lst = "\n".join(f"- Imagen {all_imgs.index(im) + 1} · [{names.get(im['doc_id'], '?')}, pág. {im['page']}]: "
                                        f"{im['description'].split(chr(10))[0][:140]}" for im in done_imgs[:8])
                        if done_imgs:
                            text = (f"No. Revisé **{len(done_imgs)}** imágenes y ninguna muestra **{what}**."
                                    f"\n\nEsto es lo que hay en ellas:\n{lst}") + falta
                        else:
                            text = (f"Todavía estoy revisando las **{len(all_imgs)}** imágenes con la IA (cada una tarda "
                                    f"unos segundos y se hace sola en segundo plano). Pregúntame de nuevo en un momento.")
                srcs = []
                meta = {"locations": [], "sources": srcs, "instant": True}
                yield ndjson({"type": "meta", **meta})
                yield ndjson({"type": "token", "text": text})
                db.add_message(body.scope, "user", question)
                db.add_message(body.scope, "assistant", text, meta)
                yield ndjson({"type": "done"})
                return

            # 1c) "¿Qué hay en la segunda foto?" / "imagen de la página 8" → respuesta directa, sin IA de chat
            if rag.is_image_question(question) and not search.is_locate_question(question):
                all_imgs = db.get_images(ids)
                pos = rag.image_refs(question)
                pages = rag.page_refs(question)
                targets = []
                for p in pos:
                    i = p - 1 if p > 0 else len(all_imgs) + p
                    if 0 <= i < len(all_imgs):
                        targets.append((i + 1, all_imgs[i]))
                if not targets and pages:
                    targets = [(i + 1, im) for i, im in enumerate(all_imgs) if im["page"] in pages]
                if pos and not targets:
                    text = f"El documento solo tiene {len(all_imgs)} imagen(es)." if all_imgs else \
                        "Este documento no tiene imágenes."
                    yield ndjson({"type": "meta", "locations": [], "sources": [], "instant": True})
                    yield ndjson({"type": "token", "text": text})
                    db.add_message(body.scope, "user", question)
                    db.add_message(body.scope, "assistant", text, {"instant": True})
                    yield ndjson({"type": "done"})
                    return
                if targets:
                    for _, im in targets[:3]:
                        if not im["description"]:
                            who = "Gemini" if s["vision_provider"] == "gemini" else "el modelo de visión"
                            yield ndjson({"type": "status", "text": f"👁 Mirando la imagen de la pág. {im['page']} con {who}…"})
                        try:
                            await describe(im, s, names)
                        except llm.LLMError:
                            pass  # se muestra al menos el texto leído por OCR
                    text = rag.describe_answer(targets[:3], names, len(all_imgs))
                    srcs = [{"doc_id": im["doc_id"], "doc_name": names.get(im["doc_id"], "?"), "page": im["page"],
                             "kind": "image", "image_id": im["id"], "score": 1.0, "text": ""} for _, im in targets[:3]]
                    meta = {"locations": [], "sources": srcs, "instant": True}
                    yield ndjson({"type": "meta", **meta})
                    yield ndjson({"type": "token", "text": text})
                    db.add_message(body.scope, "user", question)
                    db.add_message(body.scope, "assistant", text, meta)
                    yield ndjson({"type": "done"})
                    return

            # 2) Fragmentos relevantes
            qvec = (await llm.embed([question], s))[0]
            chunks = db.get_chunks(ids)
            hits = rag.retrieve(qvec, question, chunks, TOP_K)

            # 3) ¿Pregunta por una imagen? → mirarla con IA ahora (solo la primera vez)
            if rag.is_image_question(question):
                all_imgs = db.get_images(ids)
                pages = rag.page_refs(question)
                if pages:
                    targets = [im for im in all_imgs if im["page"] in pages]
                else:
                    img_chunks = [c for c in chunks if c["kind"] == "image"]
                    best = rag.retrieve(qvec, question, img_chunks, 2) if img_chunks else []
                    targets = [im for im in all_imgs if im["id"] in {c["image_id"] for c in best}]
                    if len(all_imgs) <= 2:
                        targets = all_imgs
                for im in targets[:3]:
                    if not im["description"]:
                        who = "Gemini" if s["vision_provider"] == "gemini" else "el modelo de visión"
                        yield ndjson({"type": "status", "text": f"👁 Mirando la imagen de la pág. {im['page']} con {who}…"})
                    await describe(im, s, names)
                if targets:
                    tp = {(im["doc_id"], im["page"]) for im in targets[:3]}
                    texts = [h for h in hits if h["kind"] != "image"]
                    # texto de la misma página (p. ej. el pie de figura) + 1 fragmento más: prompt corto = rápido
                    keep = [h for h in texts if (h["doc_id"], h["page"]) in tp][:2] + \
                           [h for h in texts if (h["doc_id"], h["page"]) not in tp][:1]
                    hits = keep + [{"doc_id": im["doc_id"], "page": im["page"], "kind": "image", "image_id": im["id"],
                                    "text": rag.image_chunk_text(im), "score": 1.0} for im in targets[:3]]

            # páginas donde se encontró la frase literal: asegurar que estén en el contexto
            have = {(h["doc_id"], h["page"]) for h in hits}
            for loc in locations:
                for lh in loc["hits"][:2]:
                    if (lh["doc_id"], lh["page"]) not in have:
                        for c in chunks:
                            if (c["doc_id"], c["page"]) == (lh["doc_id"], lh["page"]) and \
                                    search.norm(lh["match"]) in search.norm(c["text"]):
                                hits.append(c | {"score": 1.0})
                                have.add((c["doc_id"], c["page"]))
                                break

            sources = [{"doc_id": h["doc_id"], "doc_name": names.get(h["doc_id"], "?"), "page": h["page"],
                        "kind": h["kind"], "image_id": h.get("image_id"), "score": round(float(h["score"]), 2),
                        "text": h["text"][:400]} for h in hits]
            meta = {"locations": locations, "sources": sources}
            yield ndjson({"type": "meta", **meta})
            history = db.get_messages(body.scope)
            db.add_message(body.scope, "user", question)
            answer = []
            async for piece in llm.chat_stream(rag.build_messages(question, hits, names, history), s):
                answer.append(piece)
                yield ndjson({"type": "token", "text": piece})
            # los modelos pequeños a veces olvidan citar: se agregan las páginas usadas
            full = "".join(answer)
            if "pág" not in full.lower() and rag.NOT_FOUND.lower()[:30] not in full.lower() and hits:
                pages = []
                for h in hits:
                    ref = f"[{names.get(h['doc_id'], '?')}, pág. {h['page']}]"
                    if ref not in pages:
                        pages.append(ref)
                extra = "\n\n📄 **Fuentes:** " + " ".join(pages[:3])
                answer.append(extra)
                yield ndjson({"type": "token", "text": extra})
            db.add_message(body.scope, "assistant", "".join(answer), meta)
            yield ndjson({"type": "done"})
        except (llm.LLMError, ValueError) as e:
            yield ndjson({"type": "error", "text": str(e)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/messages")
def messages(scope: str):
    resolve_scope(scope)
    return db.get_messages(scope)


@app.delete("/api/messages")
def clear_messages(scope: str):
    resolve_scope(scope)
    db.clear_messages(scope)
    return {"ok": True}


@app.middleware("http")
async def _no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"   # la página siempre se carga actualizada
    return resp


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
