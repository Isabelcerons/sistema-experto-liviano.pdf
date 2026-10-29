"""IA: Ollama (local) para chat, embeddings y visión; Gemini (nube, opcional) solo para mirar imágenes."""

import base64
import io
import json
from typing import AsyncIterator

import httpx

TIMEOUT = httpx.Timeout(600.0, connect=10.0)
KEEP_ALIVE = "60m"   # el modelo de chat se queda en RAM: así no tarda en recargarse en cada pregunta
NUM_CTX = 4096       # contexto pequeño y FIJO: menos RAM y nada de recargas
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class LLMError(Exception):
    pass


def _ollama(s: dict) -> str:
    return s["ollama_url"].rstrip("/")


def _check(r: httpx.Response, who: str, model: str) -> None:
    if r.status_code < 400:
        return
    try:
        detail = r.json()
        if isinstance(detail, list):
            detail = detail[0]
        detail = detail.get("error", detail)
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
    except ValueError:
        detail = r.text[:300]
    if who == "ollama" and r.status_code == 404:
        raise LLMError(f"Ollama no tiene el modelo '{model}'. Descárgalo con:  ollama pull {model}")
    raise LLMError(f"{who} respondió {r.status_code} ({model}): {detail}")


def _conn(s: dict) -> str:
    return f"No me pude conectar a Ollama en {s['ollama_url']}. Ábrelo desde el menú Inicio."


async def embed(texts: list[str], s: dict) -> list[list[float]]:
    model = s["embed_model"]
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        try:
            r = await c.post(f"{_ollama(s)}/api/embed",
                             json={"model": model, "input": texts, "keep_alive": KEEP_ALIVE})
        except httpx.ConnectError as e:
            raise LLMError(_conn(s)) from e
        _check(r, "ollama", model)
        return r.json()["embeddings"]


async def chat_stream(messages: list[dict], s: dict, max_tokens: int = 500) -> AsyncIterator[str]:
    model = s["chat_model"]
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.1, "num_ctx": NUM_CTX, "num_predict": max_tokens}}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        try:
            async with c.stream("POST", f"{_ollama(s)}/api/chat", json=body) as r:
                if r.status_code >= 400:
                    await r.aread()
                    _check(r, "ollama", model)
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise LLMError(obj["error"])
                    piece = obj.get("message", {}).get("content", "")
                    if piece:
                        yield piece
        except httpx.ConnectError as e:
            raise LLMError(_conn(s)) from e


async def chat_once(messages: list[dict], s: dict, max_tokens: int = 400) -> str:
    return "".join([p async for p in chat_stream(messages, s, max_tokens)]).strip()


async def describe_image(png: bytes, prompt: str, s: dict) -> str:
    """Le pide al modelo de visión que mire una imagen. Local (Ollama) o nube (Gemini)."""
    png = _shrink(png)
    b64 = base64.b64encode(png).decode()
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        if s["vision_provider"] == "gemini":
            key = s.get("gemini_api_key", "").strip()
            if not key:
                raise LLMError("Falta la clave de Gemini: ponla en ⚙ Ajustes o cambia la visión a Local.")
            model = s["gemini_model"]
            body = {"model": model, "temperature": 0.1, "max_tokens": 600, "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
            try:
                r = await c.post(GEMINI_URL, json=body, headers={"Authorization": f"Bearer {key}"})
            except httpx.HTTPError as e:
                raise LLMError(f"No me pude conectar a Gemini (¿hay internet?): {e}") from e
            _check(r, "Gemini", model)
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        model = s["vision_model"]
        body = {"model": model, "stream": False, "keep_alive": "5m",   # libera la RAM al rato
                "options": {"temperature": 0.1, "num_ctx": 2048, "num_predict": 160},
                "messages": [{"role": "user", "content": prompt, "images": [b64]}]}
        try:
            r = await c.post(f"{_ollama(s)}/api/chat", json=body)
        except httpx.ConnectError as e:
            raise LLMError(_conn(s)) from e
        _check(r, "ollama", model)
        return r.json()["message"]["content"].strip()


def _shrink(png: bytes, side: int = 640) -> bytes:
    """Reduce la imagen antes de enviarla: menos píxeles = el modelo de visión responde antes."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png))
        if max(img.size) <= side:
            return png
        img.thumbnail((side, side))
        out = io.BytesIO()
        img.convert("RGB").save(out, "PNG")
        return out.getvalue()
    except Exception:
        return png


async def ollama_models(s: dict) -> list[str]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as c:
        try:
            r = await c.get(f"{_ollama(s)}/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(_conn(s)) from e
        names = [m["name"] for m in r.json().get("models", [])]
        return sorted(set(names) | {n.removesuffix(":latest") for n in names})


async def warm_up(s: dict) -> None:
    """Carga el modelo de chat y el de embeddings al arrancar: la primera pregunta no espera."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        try:
            await c.post(f"{_ollama(s)}/api/embed", json={"model": s["embed_model"], "input": ["hola"],
                                                          "keep_alive": KEEP_ALIVE})
            await c.post(f"{_ollama(s)}/api/chat", json={"model": s["chat_model"], "messages": [],
                                                         "keep_alive": KEEP_ALIVE,
                                                         "options": {"num_ctx": NUM_CTX}})
        except httpx.HTTPError:
            pass
