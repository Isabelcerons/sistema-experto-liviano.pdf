@echo off
title Sistema Experto PDF
REM ============================================================
REM  Doble clic en este archivo (desde la carpeta) o, en VS Code,
REM  escribe en la terminal:  .\iniciar.bat
REM  La primera vez instala todo; las siguientes arranca en segundos.
REM ============================================================
cd /d "%~dp0"

set PY=
where python >nul 2>nul && set PY=python
if not defined PY ( where py >nul 2>nul && set PY=py )
if not defined PY (
  echo [ERROR] No encontre Python. Instalalo desde https://www.python.org/downloads/
  echo         y marca la casilla "Add Python to PATH".
  goto :fin
)

if not exist .venv\Scripts\python.exe (
  echo === Primera vez: creando entorno e instalando librerias ^(unos minutos^) ===
  %PY% -m venv .venv || goto :error
)
if not exist .venv\instalado.txt (
  echo Instalando librerias...
  .venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements.txt || goto :error
  echo ok> .venv\instalado.txt
)

REM ---------- VS Code: crea la configuracion (Ctrl+Shift+B inicia, F5 depura) ----------
if not exist .vscode\tasks.json if exist config-vscode xcopy /E /I /Y /Q config-vscode .vscode >nul

REM ---------- Ollama ----------
where ollama >nul 2>nul || (
  echo [ERROR] No encontre Ollama. Instalalo desde https://ollama.com/download
  goto :fin
)
powershell -NoProfile -Command "try{Invoke-WebRequest http://localhost:11434 -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0}catch{exit 1}" || (
  echo Encendiendo Ollama...
  start "" /min ollama serve
  powershell -NoProfile -Command "for($i=0;$i -lt 30;$i++){try{Invoke-WebRequest http://localhost:11434 -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0}catch{Start-Sleep 1}}; exit 1" || echo [AVISO] Ollama no responde: abrelo desde el menu Inicio.
)
if not exist .venv\modelos.txt (
  echo Revisando modelos livianos de Ollama...
  ollama list > .venv\lista.txt 2>nul
  findstr /i "qwen2.5:1.5b" .venv\lista.txt >nul || ollama pull qwen2.5:1.5b
  findstr /i "nomic-embed-text" .venv\lista.txt >nul || ollama pull nomic-embed-text
  findstr /i "moondream" .venv\lista.txt >nul || ollama pull moondream
  echo ok> .venv\modelos.txt
)

REM ---------- Tesseract (OCR) ----------
set TESS=
where tesseract >nul 2>nul && set TESS=1
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" set TESS=1
if not defined TESS (
  echo.
  echo [AVISO] Tesseract no esta instalado: no se podran buscar palabras dentro de imagenes
  echo         ni en paginas escaneadas. Instalador: https://github.com/UB-Mannheim/tesseract/wiki
  echo.
)
if not exist tessdata\spa.traineddata (
  echo Descargando el idioma espanol para el OCR ^(unos 4 MB^)...
  mkdir tessdata 2>nul
  powershell -NoProfile -Command "$u='https://github.com/tesseract-ocr/tessdata_fast/raw/main/'; foreach($l in 'spa','eng','osd'){ try{ Invoke-WebRequest ($u+$l+'.traineddata') -OutFile ('tessdata\'+$l+'.traineddata') -UseBasicParsing }catch{ Write-Host ('No se pudo descargar '+$l) } }"
)

REM ---------- Si ya hay un servidor en el puerto 8000, solo abre la pagina ----------
powershell -NoProfile -Command "try{Invoke-WebRequest http://localhost:8000/api/status -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0}catch{exit 1}" && (
  echo El sistema ya estaba abierto. Abriendo la pagina...
  start "" http://localhost:8000
  goto :fin
)

echo.
echo === Sistema Experto PDF en http://localhost:8000 ===
echo     Deja esta ventana abierta mientras lo usas. Para cerrar: Ctrl+C
echo.
start "" "%~dp0cargando.html"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
goto :fin

:error
echo.
echo [ERROR] Algo fallo durante la instalacion. Toma una captura de esta ventana.

:fin
echo.
pause
