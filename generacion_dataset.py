# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Generador de Dataset Alineado con Pipeline RAG                             ║
# ║  - Ingesta con pymupdf4llm + PyMuPDF                                        ║
# ║  - Chunking recursivo (800 tokens, 150 overlap)                             ║
# ║  - Separación de tablas                                                     ║
# ║                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pymupdf4llm

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN (ALINEADA CON PIPELINE RAG)
# ─────────────────────────────────────────────────────────────────────────────
fragmento_pdf = 11
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PDF_DIR            = "./test"
OUTPUT_JSON        = f"dataset_acreditacion_{fragmento_pdf}.json"
PARES_POR_CHUNK    = 3
SEED               = 42
PROMPT_VERSION     = "v2.0_alineado"
MODEL              = "meta-llama/llama-3.3-70b-instruct"

# Parámetros de chunking (MISMOS que pipeline RAG)
CHUNK_SIZE = 800        # tokens
CHUNK_OVERLAP = 150     # tokens
TABLE_MAX_TOKENS = 800  # tokens

random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE EXTRACCIÓN (DEL PIPELINE RAG)
# ─────────────────────────────────────────────────────────────────────────────
def extract_real_page_number(texto: str) -> int | None:
    """Extrae número de página real desde patrones como 'Página | 12'."""
    patron = r"(?:P[áa]gin[aa]|Pag\.?|P\.?)\s*\|?\s*(\d+)"
    match = re.search(patron, texto, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extraer_tablas_markdown(texto: str) -> Tuple[List[str], str]:
    """Extrae tablas Markdown y el string de página."""
    patron_tabla = re.compile(
        r"(^\|.*\|\s*\n"
        r"\|(?:\s*:?-{3,}:?\s*\|)+\s*\n"
        r"(?:\|.*\|\s*\n?)+)",
        re.MULTILINE,
    )
    tablas = patron_tabla.findall(texto or "")

    patron_pagina = re.compile(
        r"Página\s*\|\s*(\d+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    match = patron_pagina.search((texto or "").strip())
    pagina = match.group(0) if match else "página no identificada"

    return tablas, pagina


def limpiar_markdown_para_texto(md: str) -> str:
    """
    Remueve tablas Markdown del texto para evitar duplicarlas.
    """
    if not md:
        return ""

    patron_tabla = re.compile(
        r"(^\|.*\|\s*\n"
        r"\|(?:\s*:?-{3,}:?\s*\|)+\s*\n"
        r"(?:\|.*\|\s*\n?)+)",
        re.MULTILINE,
    )
    texto = patron_tabla.sub("\n", md)

    # Limpieza básica de Markdown
    texto = re.sub(r"^#{1,6}\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\*\*(.*?)\*\*", r"\1", texto)
    texto = re.sub(r"\*(.*?)\*", r"\1", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)

    return texto.strip()


def extraer_texto_plano_pymupdf(doc: fitz.Document, page_num: int) -> str:
    """
    Extrae texto plano con PyMuPDF nativo como respaldo.
    """
    page = doc[page_num]
    texto_plano = ""
    blocks = page.get_text("dict").get("blocks", [])

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            linea = "".join(span["text"] for span in line["spans"]).strip()
            linea = re.sub(r"\. (?!\d|\|)", ".\n", linea)
            linea = re.sub(r"(Página \| \d+)", r"\1\n", linea)
            texto_plano += linea + " "

    return texto_plano.strip()


def load_pdf_separado(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Carga PDF extrayendo texto plano, tablas Markdown y número de página.
    Usa pymupdf4llm directo.
    """
    print(f"📄 Cargando PDF: {pdf_path}...", flush=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  📚 Total de páginas: {total_pages}", flush=True)

    try:
        page_chunks = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            table_strategy="lines_strict",
            show_progress=False,
        )
    except TypeError:
        page_chunks = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
        )

    paginas = []
    for page_num in range(total_pages):
        chunk = page_chunks[page_num] if page_num < len(page_chunks) else {}
        md_content = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        tablas, pagina_str = extraer_tablas_markdown(md_content)
        texto_desde_md = limpiar_markdown_para_texto(md_content)
        texto_desde_fitz = extraer_texto_plano_pymupdf(doc, page_num)
        texto_plano = texto_desde_fitz.strip() or texto_desde_md.strip()
        pagina_real_numero = extract_real_page_number(pagina_str)
        pagina_real_numero = pagina_real_numero or extract_real_page_number(texto_plano)

        paginas.append({
            "index_pagina": page_num + 1,
            "pagina_str": pagina_str,
            "pagina_numero": pagina_real_numero,
            "tablas": tablas,
            "texto_plano": texto_plano.strip(),
        })
        print(f"  🔹 Página {page_num + 1}/{total_pages} ({len(tablas)} tablas)", flush=True)

    doc.close()
    print(f"  ✅ Cargadas {len(paginas)} páginas", flush=True)
    return paginas


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTER RECURSIVO (DEL PIPELINE RAG)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleRecursiveTextSplitter:
    """Splitter recursivo simple (idéntico al pipeline RAG)."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        pieces = self._split_recursive(text, self.separators)
        chunks = self._merge_pieces(pieces)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return self._split_by_chars(text)
        separator = separators[0]
        remaining_separators = separators[1:]
        if separator == "":
            return self._split_by_chars(text)
        if separator not in text:
            return self._split_recursive(text, remaining_separators)
        raw_parts = text.split(separator)
        parts = []
        for i, part in enumerate(raw_parts):
            if not part:
                continue
            if i < len(raw_parts) - 1:
                part = part + separator
            if len(part) > self.chunk_size:
                parts.extend(self._split_recursive(part, remaining_separators))
            else:
                parts.append(part)
        return parts

    def _split_by_chars(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end
        return chunks

    def _merge_pieces(self, pieces: List[str]) -> List[str]:
        chunks = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                chunks.append(current)
                if self.chunk_overlap > 0:
                    overlap = current[-self.chunk_overlap:]
                    current = overlap + piece
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks


def count_tokens_approx(text: str) -> int:
    """Conteo aproximado de tokens."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKER (DEL PIPELINE RAG)
# ─────────────────────────────────────────────────────────────────────────────
class RecursiveChunker:
    """Chunker recursivo (idéntico al pipeline RAG)."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = SimpleRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )
        self.table_splitter = SimpleRecursiveTextSplitter(
            chunk_size=TABLE_MAX_TOKENS,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, paginas: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """
        Genera chunks de texto y tablas.
        Retorna lista de dicts con: texto, tipo, metadata
        """
        chunks = []

        for p in paginas:
            meta_base = {
                "source": source,
                "page_number": p["pagina_numero"] or p["index_pagina"],
            }

            # Chunks de texto
            if p["texto_plano"] and p["texto_plano"].strip():
                texto_chunks = self.text_splitter.split_text(p["texto_plano"])
                for i, chunk_text in enumerate(texto_chunks):
                    chunks.append({
                        "texto": chunk_text,
                        "tipo": "texto",
                        "metadata": {
                            **meta_base,
                            "token_count": count_tokens_approx(chunk_text),
                            "chunk_idx": i,
                        }
                    })

            # Chunks de tablas
            for i, tabla in enumerate(p["tablas"] or []):
                if not tabla.strip():
                    continue
                token_count = count_tokens_approx(tabla)
                if token_count > TABLE_MAX_TOKENS:
                    sub_chunks = self.table_splitter.split_text(tabla)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            "texto": sub_chunk,
                            "tipo": "tabla",
                            "metadata": {
                                **meta_base,
                                "token_count": count_tokens_approx(sub_chunk),
                                "chunk_idx": j,
                                "tabla_original_idx": i,
                            }
                        })
                else:
                    chunks.append({
                        "texto": tabla,
                        "tipo": "tabla",
                        "metadata": {
                            **meta_base,
                            "token_count": token_count,
                            "chunk_idx": 0,
                            "tabla_original_idx": i,
                        }
                    })

        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE PDFs
# ─────────────────────────────────────────────────────────────────────────────
pdf_dir = Path(PDF_DIR)
if not pdf_dir.is_dir():
    raise SystemExit(f"❌ La carpeta '{PDF_DIR}' no existe.")

pdf_files = sorted(pdf_dir.glob("*.pdf"))
if not pdf_files:
    raise SystemExit(f"❌ No se encontraron archivos .pdf en '{PDF_DIR}'.")

print(f"📂 PDFs encontrados: {len(pdf_files)}")
for p in pdf_files:
    print(f"   - {p.name}")

all_chunks = []

for pdf_path in pdf_files:
    paginas = load_pdf_separado(str(pdf_path))
    chunker = RecursiveChunker()
    chunks = chunker.chunk(paginas, pdf_path.name)
    all_chunks.extend(chunks)
    print(f"   ✅ {len(chunks)} chunks extraídos")

print(f"\n✅ Total chunks: {len(all_chunks)}")
print(f"   - Texto: {sum(1 for c in all_chunks if c['tipo'] == 'texto')}")
print(f"   - Tablas: {sum(1 for c in all_chunks if c['tipo'] == 'tabla')}")

# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE PARES Q/A
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

with open(f'prompt_system_{PROMPT_VERSION}.txt', 'w', encoding='utf-8') as f:
    f.write(SYSTEM_PROMPT)
print(f"\n💾 Prompt guardado: prompt_system_{PROMPT_VERSION}.txt")

dataset = []
errores = 0

print(f"\n🤖 Generando pares Q/A con {MODEL}...\n")
for i, chunk_data in enumerate(all_chunks):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Contexto:\n{chunk_data['texto']}"}
            ],
            temperature=0.4,
            max_tokens=1500,
            seed=SEED,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        pares = json.loads(raw)
        count = 0
        for par in pares:
            if "pregunta" in par and "respuesta" in par:
                dataset.append({
                    "contexto": chunk_data['texto'],
                    "pregunta": par["pregunta"].strip(),
                    "respuesta": par["respuesta"].strip(),
                    "fuente": chunk_data['metadata']['source'],
                    "tipo": chunk_data['tipo'],
                    "page_number": chunk_data['metadata']['page_number'],
                    "chunk_idx": chunk_data['metadata']['chunk_idx'],
                    "token_count": chunk_data['metadata']['token_count'],
                })
                count += 1

        print(f"  [{i+1:03d}/{len(all_chunks)}] ✅  {count} pares · {chunk_data['tipo']} · pág {chunk_data['metadata']['page_number']}")
        time.sleep(0.5)

    except json.JSONDecodeError as e:
        errores += 1
        print(f"  [{i+1:03d}/{len(all_chunks)}] ⚠️  JSON inválido: {str(e)[:60]}")
    except Exception as e:
        errores += 1
        print(f"  [{i+1:03d}/{len(all_chunks)}] ⚠️  Error: {str(e)[:80]}")
        time.sleep(1.5)

# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR DATASET
# ─────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"""
╔══════════════════════════════════════════════╗
║  Dataset Alineado generado                   ║
║  PDFs procesados      : {len(pdf_files):<22}║
║  Total pares          : {len(dataset):<22}║
║  Chunks texto         : {sum(1 for d in dataset if d['tipo'] == 'texto'):<22}║
║  Chunks tablas        : {sum(1 for d in dataset if d['tipo'] == 'tabla'):<22}║
║  Errores              : {errores:<22}║
║  Archivo              : {OUTPUT_JSON:<22}║
║  Prompt versión       : {PROMPT_VERSION:<22}║
║  Chunk size           : {CHUNK_SIZE} tokens{' ' * 14}║
║  Chunk overlap        : {CHUNK_OVERLAP} tokens{' ' * 14}║
╚══════════════════════════════════════════════╝
""")

# Desglose por tipo
from collections import Counter
por_tipo = Counter(item["tipo"] for item in dataset)
print("📊 Pares generados por tipo:")
for tipo, n in por_tipo.items():
    print(f"   - {tipo}: {n} pares")

# Muestra 2 ejemplos
print("\n📋 Ejemplos del dataset:")
for item in dataset[:2]:
    print(f"\n  Tipo       : {item['tipo']}")
    print(f"  Fuente     : {item['fuente']}")
    print(f"  Página     : {item['page_number']}")
    print(f"  Chunk idx  : {item['chunk_idx']}")
    print(f"  Tokens     : {item['token_count']}")
    print(f"  Contexto   : {item['contexto'][:120]}...")
    print(f"  Pregunta   : {item['pregunta']}")
    print(f"  Respuesta  : {item['respuesta']}")