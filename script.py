# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Generador de Dataset desde PDF de Acreditación                             ║
# ║  Ejecutar en tu máquina local con acceso a OpenRouter API                   ║
# ║  Uso:  python generar_dataset_openrouter.py                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import re
import json
import time
import random
from pathlib import Path
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Tu API key de OpenRouter
PDF_DIR            = "./test"                          # carpeta con varios PDFs
OUTPUT_JSON        = "dataset_acreditacion.json"
PARES_POR_CHUNK    = 3                               # preguntas por chunk
WINDOW             = 700                             # tamaño del chunk en chars
STEP               = 450                             # paso de la ventana
SEED               = 42                              # para reproducibilidad
PROMPT_VERSION     = "v1.0"                          # versión del prompt

# Modelo de OpenRouter (opciones populares)
# - "meta-llama/llama-3.3-70b-instruct"
# - "anthropic/claude-3.5-sonnet"
# - "openai/gpt-4o-mini"
# - "google/gemini-2.0-flash-exp:free"
MODEL              = "meta-llama/llama-3.3-70b-instruct"

# Fijar seed para reproducibilidad
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXTRAER Y LIMPIAR TEXTO DE TODOS LOS PDFs EN PDF_DIR
# ─────────────────────────────────────────────────────────────────────────────
def limpiar_texto(full_text: str) -> str:
    """Elimina encabezados, pies de página y normaliza espacios."""
    full_text = re.sub(r'Página \| \d+', '', full_text)
    full_text = re.sub(r'Documento para Solicitud de Renovación[^\n]*\n', '', full_text)
    full_text = re.sub(r'Programa Académico Ingeniería[^\n]*\n', '', full_text)
    full_text = re.sub(r'Versión\. Noviembre de 2024\n?', '', full_text)
    full_text = re.sub(r'Vicerrectoría\s*\nAcadémica\s*\n', '', full_text)
    full_text = re.sub(r'Dirección de Autoevaluación[^\n]*\n', '', full_text)

    # Solo contenido narrativo desde la Introducción (si existe esa sección)
    intro_idx = full_text.find('1. Introducción')
    if intro_idx >= 0:
        full_text = full_text[intro_idx:]

    text_clean = re.sub(r'\n+', ' ', full_text)
    text_clean = re.sub(r'\s{2,}', ' ', text_clean).strip()
    return text_clean


pdf_dir = Path(PDF_DIR)
if not pdf_dir.is_dir():
    raise SystemExit(f"❌ La carpeta '{PDF_DIR}' no existe. Crea la carpeta y coloca tus PDFs ahí.")

pdf_files = sorted(pdf_dir.glob("*.pdf"))
if not pdf_files:
    raise SystemExit(f"❌ No se encontraron archivos .pdf en '{PDF_DIR}'.")

print(f"📂 PDFs encontrados: {len(pdf_files)}")
for p in pdf_files:
    print(f"   - {p.name}")

all_chunks_with_source = []   # [(chunk_text, nombre_pdf), ...]

for pdf_path in pdf_files:
    print(f"\n📄 Procesando: {pdf_path.name}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        print(f"   ⚠️  No se pudo leer {pdf_path.name}: {e}")
        continue

    text_clean = limpiar_texto(full_text)

    # ── Chunking semántico (ventana deslizante + filtro de listas) ───────────
    raw_chunks = []
    for i in range(0, max(len(text_clean) - WINDOW, 0), STEP):
        chunk = text_clean[i:i+WINDOW].strip()
        if (len(chunk) > 200
                and chunk.lower().count('tabla ') < 5
                and chunk.lower().count('figura ') < 5):
            raw_chunks.append(chunk)

    # Si el documento es más corto que WINDOW, tomarlo completo
    if not raw_chunks and len(text_clean) > 200:
        raw_chunks = [text_clean]

    n_antes = len(all_chunks_with_source)
    for c in raw_chunks:
        all_chunks_with_source.append((c, pdf_path.name))
    print(f"   ✅ {len(all_chunks_with_source) - n_antes} chunks extraídos")

# ── Deduplicar por prefijo (independiente del PDF de origen) ────────────────
chunks_with_source, seen = [], set()
for c, src in all_chunks_with_source:
    key = c[:80]
    if key not in seen:
        seen.add(key)
        chunks_with_source.append((c, src))

chunks = [c for c, _ in chunks_with_source]
print(f"\n✅ Total chunks únicos (todos los PDFs): {len(chunks)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERACIÓN DE PARES Q/A CON OPENROUTER
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM_PROMPT = f"""Eres un generador experto de datasets para modelos de embeddings.
Dado un contexto de un documento universitario, genera exactamente {PARES_POR_CHUNK} pares \
pregunta-respuesta en formato JSON.

Reglas estrictas:
- La respuesta DEBE estar contenida literalmente en el contexto proporcionado.
- Las preguntas deben ser variadas: factual, inferencial y de detalle específico.
- Responde ÚNICAMENTE con un array JSON válido, sin texto adicional, sin markdown, sin backticks.
- Formato exacto: [{{"pregunta": "...", "respuesta": "..."}}, ...]"""

# Guardar el prompt para versionado
with open(f'prompt_system_{PROMPT_VERSION}.txt', 'w', encoding='utf-8') as f:
    f.write(SYSTEM_PROMPT)
print(f"\n💾 Prompt guardado: prompt_system_{PROMPT_VERSION}.txt")

dataset = []
errores = 0

print(f"\n🤖 Generando pares Q/A con {MODEL}...\n")
for i, (chunk, fuente) in enumerate(chunks_with_source):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Contexto:\n{chunk}"}
            ],
            temperature=0.4,
            max_tokens=700,
            seed=SEED,
        )
        raw = resp.choices[0].message.content.strip()
        # Limpiar posibles backticks residuales
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        pares = json.loads(raw)
        count = 0
        for par in pares:
            if "pregunta" in par and "respuesta" in par:
                dataset.append({
                    "contexto":  chunk,
                    "pregunta":  par["pregunta"].strip(),
                    "respuesta": par["respuesta"].strip(),
                    "fuente":    fuente
                })
                count += 1

        print(f"  [{i+1:02d}/{len(chunks_with_source)}] ✅  {count} pares · {fuente}")
        time.sleep(0.5)   # respetar rate limit de OpenRouter

    except json.JSONDecodeError as e:
        errores += 1
        print(f"  [{i+1:02d}/{len(chunks_with_source)}] ⚠️  JSON inválido: {str(e)[:60]}")
    except Exception as e:
        errores += 1
        print(f"  [{i+1:02d}/{len(chunks_with_source)}] ⚠️  Error: {str(e)[:80]}")
        time.sleep(1.5)   # pausa extra en error de red

# ─────────────────────────────────────────────────────────────────────────────
# 3. GUARDAR DATASET
# ─────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"""
╔══════════════════════════════════════════╗
║  Dataset generado                        ║
║  PDFs procesados : {len(pdf_files):<22}║
║  Total pares     : {len(dataset):<22}║
║  Chunks proc.     : {len(chunks_with_source):<22}║
║  Errores          : {errores:<22}║
║  Archivo          : {OUTPUT_JSON:<22}║
║  Prompt versión   : {PROMPT_VERSION:<22}║
║  Seed             : {SEED:<22}║
╚══════════════════════════════════════════╝
""")

# Desglose por fuente
from collections import Counter
por_fuente = Counter(item["fuente"] for item in dataset)
print("📊 Pares generados por PDF:")
for fuente, n in por_fuente.items():
    print(f"   - {fuente}: {n} pares")

# Muestra 2 ejemplos
print("\n📋 Ejemplos del dataset:")
for item in dataset[:2]:
    print(f"\n  Fuente   : {item['fuente']}")
    print(f"  Contexto : {item['contexto'][:120]}...")
    print(f"  Pregunta : {item['pregunta']}")
    print(f"  Respuesta: {item['respuesta']}")