# 📚 Sistema Experto PDF (liviano)

Sistema experto que lee tus PDF y responde preguntas citando la página. Pensado para un PC **sin tarjeta gráfica**
(Intel i5-10210U, 12 GB de RAM): usa solo modelos pequeños de **Ollama** y responde "¿dónde dice…?" **sin IA, al instante**.

## Cómo arrancarlo

**Opción 1 – Doble clic:** abre la carpeta en el Explorador y haz doble clic en **`iniciar.bat`**.

**Opción 2 – Visual Studio Code:**
1. *Archivo → Abrir carpeta…* → elige esta carpeta.
2. *Terminal → Nueva terminal* y escribe `.\iniciar.bat`
3. Después de la primera vez también puedes usar **Ctrl+Shift+B** (iniciar) o **F5** (depurar).

La primera vez instala las librerías y revisa los modelos. Luego abre **http://localhost:8000** solo.
Para cerrarlo: **Ctrl+C** en la ventana negra o en la terminal.

## Qué responde y cuánto tarda

| Pregunta | Cómo lo resuelve | Tiempo |
|---|---|---|
| **"¿Dónde dice Isabel?"** · "¿En qué página dice …?" · "Busca la palabra …" | Búsqueda literal, **sin IA**. Ignora mayúsculas y tildes; busca también dentro de otras palabras ("isa" → "Isabel") | **Instantáneo** |
| Palabra escrita **dentro de una imagen** o en una **página escaneada** | **Tesseract** lee esas letras al subir el PDF | **Instantáneo** |
| **"¿Qué aparece en la imagen de la página 8?"** | **moondream** mira esa imagen (solo la primera vez; queda guardado) y la descripción se traduce al español | ~20-60 s la primera vez |
| **📋 Resumen** (pestaña) | Sin IA agrupa los fragmentos por tema (k-means sobre los embeddings) y toma la idea más representativa de cada grupo con su página; luego la IA redacta de qué trata, ideas principales y conclusión en una sola llamada. Queda guardado | Ideas clave: instantáneo · resumen: ~1-2 min, solo la primera vez |
| Preguntas generales ("¿de qué trata?") | Busca los 4 fragmentos más parecidos y responde **qwen2.5:1.5b** citando `[archivo.pdf, pág. N]` | ~15-40 s |

## Modelos (todos locales, ≈3 GB en total)

| Modelo | Tamaño | Uso |
|---|---|---|
| `qwen2.5:1.5b` | 1,0 GB | Redactar las respuestas |
| `nomic-embed-text` | 0,3 GB | Buscar por significado (embeddings) |
| `moondream` | 1,7 GB | Mirar imágenes |
| Tesseract OCR | 0,1 GB | Leer letras en imágenes y páginas escaneadas |

## Arquitectura

```
PDF ──► PyMuPDF ──► texto por página ─────────────┐
           │                                       ├──► fragmentos ──► nomic-embed-text ──► SQLite
           ├──► páginas escaneadas ──► Tesseract ──┤
           └──► imágenes ──► Tesseract (letras) ───┘
                        └──► moondream (lo que se ve, bajo demanda)

Pregunta ──► ¿"dónde dice…"? ──sí──► búsqueda literal ──► respuesta instantánea + página resaltada
                 │no
                 └──► 4 fragmentos más parecidos ──► qwen2.5:1.5b ──► respuesta con [archivo.pdf, pág. N]
```

| Carpeta / archivo | Qué hace |
|---|---|
| `app/main.py` | API (FastAPI): subir PDF, chat, búsqueda, imágenes |
| `app/pdf.py` | PyMuPDF: texto, imágenes, páginas escaneadas, página con resaltado |
| `app/ocr.py` | Tesseract |
| `app/search.py` | Búsqueda literal de palabras y frases |
| `app/summary.py` | Resumen rápido: ideas clave por k-means + una sola llamada a la IA |
| `app/rag.py` | Selección de fragmentos y prompt |
| `app/llm.py` | Conexión con Ollama (y Gemini opcional) |
| `app/db.py` | Base de datos SQLite (un solo archivo, sin Docker) |
| `static/index.html` | Página web |
| `config-vscode/` | Configuración de VS Code (iniciar.bat la copia a `.vscode`) |
| `data/` | PDF, imágenes y base de datos (se crea sola) |

## Requisitos
- Python 3.10+ (con *Add Python to PATH*), Ollama, y Tesseract (https://github.com/UB-Mannheim/tesseract/wiki).
- Cierra programas pesados (por ejemplo, el navegador con muchas pestañas) para dejar RAM libre a los modelos.
