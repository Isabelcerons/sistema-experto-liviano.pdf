"""Configuración. Se cambia desde la interfaz (⚙ Ajustes) o con un archivo .env."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

_env = BASE_DIR / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = DATA_DIR / "chatpdf.db"
PDF_DIR = DATA_DIR / "pdfs"
IMG_DIR = DATA_DIR / "imagenes"

MAX_PAGES = int(os.getenv("MAX_PAGES", "150"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "80"))
MIN_IMAGE_PX = int(os.getenv("MIN_IMAGE_PX", "80"))   # ignora íconos y adornos
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4            # pocos fragmentos = menos texto para la IA = respuesta más rápida en CPU

DEFAULT_SETTINGS = {
    "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "chat_model": os.getenv("CHAT_MODEL", "qwen2.5:1.5b"),          # 1 GB, rápido en CPU, buen español
    "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),    # 0,3 GB
    # Visión (mirar imágenes): "ollama" (local, sin internet) o "gemini" (nube, rápido)
    "vision_provider": os.getenv("VISION_PROVIDER", "ollama"),
    "vision_model": os.getenv("VISION_MODEL", "moondream"),           # 1,7 GB, el más liviano para mirar imágenes
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    # ¿Describir TODAS las imágenes al subir? "no" = solo cuando preguntes por una (más rápido)
    "describe_on_upload": os.getenv("DESCRIBE_ON_UPLOAD", "no"),
    "ocr_lang": os.getenv("OCR_LANG", "spa+eng"),
    "tesseract_cmd": os.getenv("TESSERACT_CMD", ""),                # vacío = buscarlo solo
}
