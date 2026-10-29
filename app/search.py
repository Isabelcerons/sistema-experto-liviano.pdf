"""Localizador de frases: encuentra en qué página(s) aparece un texto.

Es determinista (no depende de la IA): normaliza mayúsculas, tildes, puntuación y saltos de
línea, busca la frase exacta y, si no está tal cual, busca la más parecida (búsqueda difusa)."""

import re
import unicodedata
from difflib import SequenceMatcher

WORD = re.compile(r"\w+", re.UNICODE)
QUOTED = re.compile(r'["“”«»]([^"“”«»]{3,300})["“”«»]|\'([^\']{3,300})\'')
LOCATE_TRIGGER = re.compile(
    r"(en\s+qu[eé]\s+p[aá]gina|qu[eé]\s+p[aá]gina|d[oó]nde\s+(est[aá]|aparece|dice|se\s+(encuentra|menciona|habla))|"
    r"d[oó]nde\s+(sale|figura|se\s+lee|se\s+ve)|ubica(r|ci[oó]n)?\s+(de\s+)?la\s+(frase|palabra)|"
    r"busca(r|me)?\s+(la\s+)?(frase|palabra)|en\s+qu[eé]\s+parte)",
    re.IGNORECASE,
)
FILLER = re.compile(
    r"^(\s*(la|el|esta|esa|siguiente)?\s*(frase|oraci[oó]n|texto|palabra|parte|p[aá]rrafo)\s*[:,]?\s*)|"
    r"^\s*(dice|est[aá]|aparece|se\s+encuentra|se\s+menciona|se\s+habla\s+de|habla\s+de|menciona|hay|sale)\s+(que\s+)?((un|una|unos|unas|el|la|los|las)\s+)?",
    re.IGNORECASE,
)
STOP = set("""a al algo como con de del el en es esta este esto la las lo los mas me mi o para pero por que se si sin
sobre su sus tambien un una uno unos unas y ya hay cual cuales donde cuando quien pagina documento pdf dice esta
aparece frase""".split())


SUFFIX = re.compile(r"\s+(en\s+(el|este|la|esta)\s+(documento|pdf|archivo|texto|libro|trabajo))$", re.IGNORECASE)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(text: str) -> list[tuple[str, int, int]]:
    """[(token_normalizado, inicio, fin)] sobre el texto original."""
    return [(norm(m.group()), m.start(), m.end()) for m in WORD.finditer(text)]


def content_words(text: str) -> list[str]:
    return [t for t, _, _ in tokens(text) if len(t) > 2 and t not in STOP]


def candidate_phrases(question: str) -> list[str]:
    """Frases que el usuario quiere ubicar: lo que está entre comillas o lo que sigue a
    expresiones como '¿en qué página dice…?' / '¿dónde aparece…?'."""
    quoted = [a or b for a, b in QUOTED.findall(question)]
    if quoted:
        return [q.strip() for q in quoted if len(WORD.findall(q)) >= 1]
    m = LOCATE_TRIGGER.search(question)
    if not m:
        return []
    rest = question[m.end():].strip(" ¿?¡!.:")
    for _ in range(3):
        rest = SUFFIX.sub("", FILLER.sub("", rest)).strip(" ¿?¡!.:,")
    words = [norm(w) for w in WORD.findall(rest)]
    # vale también UNA sola palabra (p. ej. "¿dónde dice isa?"), siempre que no sea una palabra vacía
    if not words or all(w in STOP for w in words) or len(rest) < 2:
        return []
    return [rest]


def _snippet(text: str, start: int, end: int, pad: int = 70) -> str:
    a, b = max(0, start - pad), min(len(text), end + pad)
    s = text[a:b].replace("\n", " ")
    return ("…" if a > 0 else "") + s + ("…" if b < len(text) else "")


def locate(phrase: str, pages: list[dict], max_hits: int = 12) -> dict:
    """pages: [{doc_id, page, text, image_id?}] (las imágenes entran como "páginas" con image_id) → {'exact': bool, 'hits': [{doc_id, page, score, snippet, match}]}"""
    ptoks = [t for t, _, _ in tokens(phrase)]
    if not ptoks:
        return {"phrase": phrase, "exact": False, "hits": []}
    n = len(ptoks)
    target = " ".join(ptoks)
    exact_hits, fuzzy_best, partial = [], [], []
    for pg in pages:
        toks = tokens(pg["text"])
        words = [t for t, _, _ in toks]
        found = False
        for i in range(len(words) - n + 1):
            if words[i] == ptoks[0] and words[i:i + n] == ptoks:
                if found:  # una entrada por página, contando las repeticiones
                    exact_hits[-1]["count"] += 1
                    continue
                s, e = toks[i][1], toks[i + n - 1][2]
                exact_hits.append({"doc_id": pg["doc_id"], "page": pg["page"], "image_id": pg.get("image_id"), "score": 1.0, "count": 1,
                                   "snippet": _snippet(pg["text"], s, e), "match": pg["text"][s:e]})
                found = True
        if found or exact_hits:
            continue
        if n == 1:
            # una sola palabra: si no está suelta, se busca DENTRO de otras palabras ("isa" → "Isabel")
            # pero no se usa la búsqueda difusa ("isa" no debe confundirse con "usa")
            for t, s0, e0 in toks:
                if len(ptoks[0]) >= 3 and ptoks[0] in t:
                    if partial and partial[-1]["page"] == pg["page"] and partial[-1]["doc_id"] == pg["doc_id"]:
                        partial[-1]["count"] += 1
                        continue
                    partial.append({"doc_id": pg["doc_id"], "page": pg["page"], "image_id": pg.get("image_id"), "score": 0.9, "count": 1,
                                    "partial": True, "snippet": _snippet(pg["text"], s0, e0),
                                    "match": pg["text"][s0:e0]})
            continue
        # búsqueda difusa: ventana de tamaño parecido a la frase
        best, best_span = 0.0, None
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(0, max(1, len(words) - size + 1)):
                cand = " ".join(words[i:i + size])
                sm = SequenceMatcher(None, target, cand)
                if sm.real_quick_ratio() < best or sm.quick_ratio() < max(best, 0.8):
                    continue
                r = sm.ratio()
                if r > best:
                    best, best_span = r, (i, min(i + size, len(toks)) - 1)
        if best_span and best >= 0.82:  # solo errores leves de tipeo
            s, e = toks[best_span[0]][1], toks[best_span[1]][2]
            fuzzy_best.append({"doc_id": pg["doc_id"], "page": pg["page"], "image_id": pg.get("image_id"), "score": round(best, 3), "count": 1,
                               "snippet": _snippet(pg["text"], s, e), "match": pg["text"][s:e]})
    if exact_hits:
        return {"phrase": phrase, "exact": True, "hits": exact_hits[:max_hits]}
    if partial:
        return {"phrase": phrase, "exact": False, "partial": True, "hits": partial[:max_hits]}
    fuzzy_best.sort(key=lambda h: -h["score"])
    return {"phrase": phrase, "exact": False, "hits": fuzzy_best[:5]}


def is_locate_question(question: str) -> bool:
    """¿La pregunta es SOLO de ubicar una frase? (p. ej. '¿dónde dice isabel?', '¿en qué página dice "..."?')"""
    return bool(LOCATE_TRIGGER.search(question))


def locate_answer(locations: list[dict]) -> str:
    """Respuesta en Markdown armada directamente con las ubicaciones verificadas (sin IA)."""
    parts = []
    for loc in locations:
        hits = loc["hits"]
        pages = sorted({(h["doc_name"], h["page"]) for h in hits}, key=lambda x: (x[0], x[1]))
        if loc["exact"]:
            head = f'**"{loc["phrase"]}"** aparece en'
        elif loc.get("partial"):
            words = sorted({h["match"] for h in hits})
            head = f'**"{loc["phrase"]}"** aparece dentro de la palabra **{", ".join(words)}** en'
        else:
            head = f'**"{loc["phrase"]}"** no está escrito tal cual; lo más parecido está en'
        if len(pages) == 1:
            head += f" la **página {pages[0][1]}**:"
        else:
            head += f" **{len(pages)} páginas** ({', '.join(str(p) for _, p in pages)}):"
        lines = [head, ""]
        for h in hits:
            times = f" — aparece {h['count']} veces" if h.get("count", 1) > 1 else ""
            where = " 🖼 **dentro de una imagen**" if h.get("image_id") else ""
            lines.append(f'- [{h["doc_name"]}, pág. {h["page"]}]{where}{times}: «{h["snippet"].replace("*", "").strip()}»')
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n\nHaz clic en la cita para ver la página con la frase resaltada en amarillo."


LITERAL_RE = re.compile(r"(\bdice\b|palabra|frase|escrit|textual|literal|aparece|\bsale\b|[\"\u201c\u201d\u00ab\u00bb])", re.IGNORECASE)


def is_literal_question(question: str) -> bool:
    """'¿dónde dice X?' / 'la palabra X' → se busca el texto tal cual; '¿dónde habla de X?' → por significado."""
    return bool(LITERAL_RE.search(question))
