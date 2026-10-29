"""Búsqueda de fragmentos relevantes (RAG) y armado del prompt. Prompts cortos: el modelo es pequeño."""

import re

import numpy as np

from .search import content_words, norm

NOT_FOUND = "No encontré esa información en los documentos cargados."

SYSTEM_PROMPT = f"""Eres un asistente que responde preguntas sobre documentos PDF.
Reglas:
1. Responde SOLO con la información del CONTEXTO. Si no está, responde exactamente: "{NOT_FOUND}"
2. Después de cada dato pon su fuente así: [archivo.pdf, pág. N] (copia el nombre y la página del contexto).
3. Los bloques [IMAGEN ...] describen imágenes del PDF: úsalos para preguntas sobre imágenes
   (si la descripción está en inglés, respóndela en español).
4. Responde en español, claro y breve: primero la respuesta en 1-2 oraciones y, si hace falta, viñetas."""

# moondream es pequeño y entiende mejor instrucciones cortas en inglés
IMAGE_PROMPT_SMALL = "Describe this image in two or three sentences: the main objects, people or animals, colors, and what it shows."

IMAGE_PROMPT = """Mira esta imagen de la página {page} del documento "{doc}".
Describe en español, sin inventar:
- Los objetos, personas, animales y elementos que se ven, con colores y posición.
- Todo el texto que aparezca, tal cual.
- Si es un gráfico, diagrama o tabla: qué representa y sus partes.
Máximo 120 palabras."""

def needs_translation(s: dict) -> bool:
    return s.get("vision_provider") == "ollama" and "moondream" in s.get("vision_model", "")


def translate_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": "Traduce al español el texto que te den. Responde solo con la traducción."},
            {"role": "user", "content": text}]


def image_prompt(s: dict, page: int, doc: str) -> str:
    if s.get("vision_provider") == "ollama" and "moondream" in s.get("vision_model", ""):
        return IMAGE_PROMPT_SMALL
    return IMAGE_PROMPT.format(page=page, doc=doc)


IMAGE_Q = re.compile(
    r"\b(imagen|im[aá]genes|foto|fotos|fotograf|figura|gr[aá]fic|dibujo|diagrama|ilustraci|logo|esquema|"
    r"plano|mapa|se ve|se observa|aparece|muestra|objeto|color|personas?|animal)", re.IGNORECASE)
ORDINALS = {"primer": 1, "primera": 1, "segunda": 2, "segundo": 2, "tercera": 3, "tercer": 3, "tercero": 3,
            "cuarta": 4, "cuarto": 4, "quinta": 5, "quinto": 5, "sexta": 6, "sexto": 6, "septima": 7,
            "octava": 8, "novena": 9, "decima": 10, "ultima": -1, "ultimo": -1}
IMG_WORD = r"(?:imagen|foto|fotograf[ií]a|figura|gr[aá]fico|diagrama|ilustraci[oó]n)"
ORD_RE = re.compile(r"\b(" + "|".join(ORDINALS) + r")\s+" + IMG_WORD, re.IGNORECASE)
NUM_RE = re.compile(IMG_WORD + r"\s*(?:n[uú]mero\s*|#\s*|no\.?\s*)?(\d{1,3})\b", re.IGNORECASE)
PAGE_REF = re.compile(r"p[aá]g(?:ina|\.)?\s*(\d{1,4})", re.IGNORECASE)


def is_image_question(q: str) -> bool:
    return bool(IMAGE_Q.search(q))


def page_refs(q: str) -> list[int]:
    return [int(n) for n in PAGE_REF.findall(q)]


def image_refs(q: str) -> list[int]:
    """'la segunda foto', 'última imagen', 'imagen 3' → posiciones (1 = primera, -1 = última)."""
    from .search import norm
    nq = norm(q)
    out = [ORDINALS[m.group(1)] for m in ORD_RE.finditer(nq)]
    out += [int(n) for n in NUM_RE.findall(nq)]
    return out


EXIST_RE = re.compile(r"\b(hay|habia|existe[n]?|aparece[n]?|tiene|salen?|muestra[n]?)\b", re.IGNORECASE)
IMG_WORDS = {"imagen", "imagenes", "foto", "fotos", "fotografia", "fotografias", "figura", "figuras", "grafico",
             "graficos", "dibujo", "dibujos", "ilustracion", "ilustraciones", "diagrama", "diagramas"}
FILLER_WORDS = {"hay", "habia", "existe", "existen", "aparece", "aparecen", "tiene", "sale", "salen", "muestra",
                "muestran", "algun", "alguna", "algunos", "algunas", "este", "esta", "documento", "pdf", "archivo",
                "en", "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas", "con", "que", "donde",
                "pagina", "si", "no", "o", "y", "d", "e"}


def object_query(q: str) -> list[str]:
    """'¿hay fotos de gatos?' → ['gatos'] (lo que se busca dentro de las imágenes)."""
    from .search import norm
    nq = norm(q)
    words = re.findall(r"\w+", nq)
    if not EXIST_RE.search(nq) or not any(w in IMG_WORDS for w in words):
        return []
    return [w for w in words if w not in IMG_WORDS and w not in FILLER_WORDS and len(w) >= 3]


def _stem(w: str) -> str:
    for suf in ("es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


# Sinónimos (español de varios países + inglés, porque moondream describe en inglés)
SYNONYMS = [
    {"carro", "coche", "auto", "automovil", "vehiculo", "camioneta", "taxi", "car", "vehicle", "automobile", "suv"},
    {"perro", "perrito", "cachorro", "can", "dog", "puppy"},
    {"gato", "gatito", "felino", "cat", "kitten"},
    {"persona", "gente", "hombre", "mujer", "nino", "nina", "senor", "senora", "person", "people", "man", "woman",
     "child", "boy", "girl", "human"},
    {"casa", "vivienda", "hogar", "edificio", "house", "home", "building"},
    {"sol", "sun", "sunny"},
    {"arbol", "planta", "tree", "plant"},
    {"robot", "robotico", "android"},
    {"computador", "computadora", "ordenador", "portatil", "pantalla", "monitor", "computer", "laptop", "screen"},
    {"panel", "paneles", "placa", "solar", "panels"},
    {"avion", "aeronave", "airplane", "plane", "aircraft"},
    {"dron", "drone", "quadcopter"},
    {"moto", "motocicleta", "motorcycle", "motorbike"},
    {"bicicleta", "bici", "bicycle", "bike"},
    {"camion", "truck"},
    {"flor", "flores", "flower"},
    {"telefono", "celular", "movil", "phone", "smartphone"},
    {"grafico", "grafica", "chart", "graph", "barras", "bar"},
    {"diagrama", "esquema", "diagram", "flowchart"},
    {"mapa", "map"},
    {"libro", "book"},
    {"pajaro", "ave", "bird"},
    {"caballo", "horse"},
    {"agua", "mar", "rio", "water", "sea", "river", "ocean"},
    {"montana", "mountain"},
]


def _expand(word: str) -> set[str]:
    st = _stem(word)
    out = {st, word}
    for group in SYNONYMS:
        if word in group or st in group or any(_stem(g) == st for g in group):
            out |= group | {_stem(g) for g in group}
    return {w for w in out if len(w) >= 3}


def find_object(images: list[dict], words: list[str]) -> list[dict]:
    """Imágenes cuya descripción (o texto) menciona lo buscado. Usa sinónimos (carro = coche = car),
    plurales y tolera errores de tipeo leves."""
    from difflib import SequenceMatcher
    from .search import norm
    groups = [_expand(w) for w in words]
    hits = []
    for im in images:
        toks = {t for t in re.findall(r"\w+", norm((im.get("description") or "") + " " + (im.get("ocr_text") or "")))
                if len(t) >= 3}
        stoks = {_stem(t) for t in toks}
        for g in groups:
            exact = bool(g & toks or g & stoks)
            # errores de tipeo ("egatos" → "gatos"): solo palabras largas y muy parecidas
            typo = not exact and any(len(w) >= 5 and SequenceMatcher(None, w, t).ratio() >= 0.86
                                     for w in g for t in stoks)
            if exact or typo:
                hits.append(im)
                break
    return hits


FOLLOW_RE = re.compile(r"^\s*(y|tambien|ademas|ahora)\b", re.IGNORECASE)


def followup_object(q: str, last_user: str | None) -> list[str]:
    """'y de un carro?' justo después de '¿hay imágenes de casas?' → ['carro']."""
    from .search import norm
    if not last_user or not object_query(last_user) or not FOLLOW_RE.search(norm(q)):
        return []
    words = re.findall(r"\w+", norm(q))
    return [w for w in words if w not in IMG_WORDS and w not in FILLER_WORDS and len(w) >= 3
            and w not in ("tambien", "ademas", "ahora")]


def describe_answer(targets: list[tuple[int, dict]], names: dict, total: int) -> str:
    """Respuesta directa, sin IA de chat: lo que vio el modelo de visión + las letras leídas por OCR."""
    parts = []
    for pos, im in targets:
        desc = (im.get("description") or "").split("\n(original en inglés:")[0].strip()
        lines = [f"### 🖼 Imagen {pos} de {total} · [{names.get(im['doc_id'], '?')}, pág. {im['page']}]"]
        if desc:
            lines.append(desc)
        if im.get("ocr_text"):
            lines.append("**Texto escrito en la imagen:** " + " · ".join(l for l in im["ocr_text"].splitlines() if l.strip()))
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts)


def image_chunk_text(img: dict) -> str:
    parts = [f"[IMAGEN en la página {img['page']}]"]
    if img.get("description"):
        parts.append(img["description"])
    if img.get("ocr_text"):
        parts.append("Texto escrito en la imagen: " + img["ocr_text"])
    if len(parts) == 1:
        parts.append("(imagen sin texto; aún no se ha mirado con el modelo de visión)")
    return " ".join(parts)


def retrieve(qvec, question: str, chunks: list[dict], k: int) -> list[dict]:
    """Híbrido: 75 % similitud de significado (embeddings) + 25 % palabras clave en común."""
    usable = [c for c in chunks if c["embedding"] is not None]
    if not usable:
        return []
    m = np.vstack([c["embedding"] for c in usable]).astype(np.float32)
    q = np.asarray(qvec, dtype=np.float32)
    if m.shape[1] != q.shape[0]:
        raise ValueError("Los documentos se indexaron con otro modelo de embeddings: vuelve a subirlos.")
    cos = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
    qwords = set(content_words(question))
    scored = []
    for c, sc in zip(usable, cos):
        kw = 0.0
        if qwords:
            words = set(norm(c["text"]).split())
            kw = sum(1 for w in qwords if w in words or any(x.startswith(w) for x in words)) / len(qwords)
        scored.append((0.75 * float(sc) + 0.25 * kw, c))
    scored.sort(key=lambda x: -x[0])
    return [c | {"score": s} for s, c in scored[:k]]


def build_messages(question: str, hits: list[dict], names: dict, history: list[dict]) -> list[dict]:
    blocks = []
    for h in hits:
        blocks.append(f"[{names.get(h['doc_id'], '?')}, pág. {h['page']}]\n{h['text']}")
    context = "\n\n".join(blocks) or "(sin fragmentos)"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-2:]:  # solo el último intercambio: menos texto = más rápido
        msgs.append({"role": m["role"], "content": m["content"][:600]})
    msgs.append({"role": "user", "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"})
    return msgs


GREET_RE = re.compile(r"^\s*(hola|buenas|buenos dias|buenas tardes|buenas noches|hey|que tal|saludos)\b[\s!.,?¿¡]*$", re.IGNORECASE)
THANKS_RE = re.compile(r"^\s*(gracias|muchas gracias|mil gracias|ok|okey|listo|perfecto|genial|excelente)\b[\s!.,]*$", re.IGNORECASE)
COUNT_RE = re.compile(r"cu[aá]nt[oa]s\s+(p[aá]ginas|im[aá]genes|fotos|figuras|hojas)", re.IGNORECASE)


def small_talk(q: str) -> str | None:
    if GREET_RE.match(q):
        return ("¡Hola! 👋 Pregúntame lo que quieras sobre el PDF. Por ejemplo:\n"
                "- *¿Dónde dice …?* (te digo la página al instante)\n- *¿Hay imágenes de …?*\n"
                "- *¿Qué hay en la segunda imagen?*\n- *¿De qué trata el documento?*")
    if THANKS_RE.match(q):
        return "¡Con gusto! Si tienes otra pregunta sobre el documento, aquí estoy."
    return None


def count_answer(q: str, docs: list[dict], images: list[dict]) -> str | None:
    m = COUNT_RE.search(q)
    if not m:
        return None
    what = m.group(1).lower()
    parts = []
    for d in docs:
        n_img = sum(1 for im in images if im["doc_id"] == d["id"])
        if what.startswith(("p", "h")):
            parts.append(f"**{d['name']}** tiene **{d['pages']} páginas**"
                         + (f" ({d['ocr_pages']} escaneadas)." if d.get("ocr_pages") else "."))
        else:
            pages = sorted({im["page"] for im in images if im["doc_id"] == d["id"]})
            parts.append(f"**{d['name']}** tiene **{n_img} imágenes**"
                         + (f", en las páginas {', '.join(map(str, pages))}." if pages else "."))
    return "\n".join(f"- {p}" if len(parts) > 1 else p for p in parts)
