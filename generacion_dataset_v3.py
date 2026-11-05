import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv
import fitz
import pymupdf4llm
import tiktoken


load_dotenv()


# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────

fragmento_pdf = "1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

PDF_DIR = "./test"
OUTPUT_JSON = f"dataset_acreditacion_{fragmento_pdf}.json"

PARES_POR_CHUNK = 4
SEED = 42
PROMPT_VERSION = "v2.0_alineado"
MODEL = "meta-llama/llama-3.3-70b-instruct"

# Chunks de texto
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Tablas
TABLE_MAX_TOKENS = 2000
TABLE_CHUNK_OVERLAP = 50


random.seed(SEED)
encoding = tiktoken.get_encoding("cl100k_base")


# ─── FUNCIONES DE EXTRACCIÓN ─────────────────────────────────────────────────

def extract_real_page_number(texto: str) -> int | None:
    """
    Intenta identificar el número de página impreso dentro del documento.

    Si no encuentra un número de página explícito, se utilizará como
    respaldo el número físico de página del PDF (index_pagina).
    """
    if not texto:
        return None

    match = re.search(
        r"(?:P[áa]gin[aa]|Pag\.?|P\.?)\s*\|?\s*(\d+)",
        texto,
        re.IGNORECASE,
    )

    return int(match.group(1)) if match else None


def extraer_tablas_markdown(texto: str) -> Tuple[List[str], str]:
    """
    Extrae tablas Markdown y conserva el texto utilizado para identificar
    la página cuando aparece en el Markdown generado por PyMuPDF4LLM.
    """
    patron_tabla = re.compile(
        r"(^\|.*\|\s*\n"
        r"\|(?:\s*:?-{3,}:?\s*\|)+\s*\n"
        r"(?:\|.*\|\s*\n?)+)",
        re.MULTILINE,
    )

    tablas = patron_tabla.findall(texto or "")

    match = re.search(
        r"Página\s*\|\s*(\d+)$",
        (texto or "").strip(),
        re.IGNORECASE | re.MULTILINE,
    )

    pagina = match.group(0) if match else "página no identificada"

    return tablas, pagina


def limpiar_markdown_para_texto(md: str) -> str:
    if not md:
        return ""

    texto = re.sub(
        r"(^\|.*\|\s*\n"
        r"\|(?:\s*:?-{3,}:?\s*\|)+\s*\n"
        r"(?:\|.*\|\s*\n?)+)",
        "\n",
        md,
        flags=re.MULTILINE,
    )

    texto = re.sub(r"^#{1,6}\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\*\*(.*?)\*\*", r"\1", texto)
    texto = re.sub(r"\*(.*?)\*", r"\1", texto)

    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def extraer_texto_plano_pymupdf(doc: fitz.Document, page_num: int) -> str:
    """
    Extrae texto plano respetando el orden de bloques/líneas de PyMuPDF.
    """
    texto_plano = ""

    for block in doc[page_num].get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue

        for line in block["lines"]:
            linea = "".join(span["text"] for span in line["spans"]).strip()

            # Ayuda a separar oraciones para el chunking.
            linea = re.sub(r"\. (?!\d|\|)", ".\n", linea)

            texto_plano += linea + " "

    return texto_plano.strip()


def load_pdf_separado(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Carga cada página individualmente.

    Se conservan DOS referencias de página:
      - index_pagina: página física dentro del PDF (siempre confiable).
      - pagina_numero: número de página identificado dentro del documento.

    Esto permite ordenar siempre por la página física del PDF sin perder
    el número de página real/impreso cuando está disponible.
    """
    print(f"📄 Cargando PDF: {pdf_path}...", flush=True)

    doc = fitz.open(pdf_path)

    try:
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

        for page_num in range(len(doc)):
            chunk = page_chunks[page_num] if page_num < len(page_chunks) else {}

            md_content = (
                chunk.get("text", "")
                if isinstance(chunk, dict)
                else str(chunk)
            )

            tablas, pagina_str = extraer_tablas_markdown(md_content)

            texto_plano = (
                extraer_texto_plano_pymupdf(doc, page_num).strip()
                or limpiar_markdown_para_texto(md_content).strip()
            )

            pagina_real = (
                extract_real_page_number(pagina_str)
                or extract_real_page_number(texto_plano)
            )

            paginas.append(
                {
                    # Página física dentro del PDF. Nunca debe utilizarse
                    # find() para volver a descubrir esta posición.
                    "index_pagina": page_num + 1,

                    # Número de página impreso/documental, si fue detectado.
                    "pagina_numero": pagina_real,

                    "tablas": tablas,
                    "texto_plano": texto_plano,
                }
            )

        return paginas

    finally:
        doc.close()


# ─── PIEZA DE TEXTO CON POSICIÓN ─────────────────────────────────────────────

@dataclass
class TextPiece:
    """
    Fragmento de texto que conserva su posición exacta dentro de
    texto_global.
    """
    text: str
    start: int
    end: int


# ─── SPLITTER POR TOKENS + OFFSETS ───────────────────────────────────────────

class SimpleRecursiveTextSplitter:
    """
    Splitter recursivo basado en tokens que, además de generar el texto,
    conserva los offsets de caracteres dentro del documento original.

    Esto evita tener que hacer posteriormente:

        texto_global.find(chunk_text[:100], ...)

    Los chunks pueden cruzar páginas porque el splitter trabaja sobre
    texto_global completo.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
        enc,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.enc = enc

    def token_count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def split_text(self, text: str) -> List[str]:
        """
        Mantiene la API original: devuelve únicamente los textos.
        """
        return [
            piece.text
            for piece in self.split_text_with_offsets(text)
        ]

    def split_text_with_offsets(self, text: str) -> List[TextPiece]:
        """
        Genera chunks y conserva start/end en caracteres dentro de `text`.
        """
        text = text.strip()

        if not text:
            return []

        # El strip cambia potencialmente el offset respecto al string
        # original. Para este script se usa siempre el texto global ya
        # construido y sin espacios externos relevantes.
        pieces = self._split_recursive(
            text=text,
            start_offset=0,
            separators=self.separators,
        )

        return self._merge_pieces(pieces)

    def _split_recursive(
        self,
        text: str,
        start_offset: int,
        separators: List[str],
    ) -> List[TextPiece]:

        if self.token_count(text) <= self.chunk_size:
            return [
                TextPiece(
                    text=text,
                    start=start_offset,
                    end=start_offset + len(text),
                )
            ]

        if not separators:
            return self._split_by_tokens(text, start_offset)

        sep = separators[0]

        if sep == "" or sep not in text:
            return self._split_recursive(
                text,
                start_offset,
                separators[1:],
            )

        pieces: List[TextPiece] = []

        # Mantener las posiciones exactas dentro de `text`.
        cursor = 0

        for i, part in enumerate(text.split(sep)):
            if not part:
                cursor += len(sep)
                continue

            part_with_sep = (
                part + sep
                if i < len(text.split(sep)) - 1
                else part
            )

            local_start = cursor
            local_end = cursor + len(part_with_sep)

            if self.token_count(part_with_sep) > self.chunk_size:
                pieces.extend(
                    self._split_recursive(
                        text=part_with_sep,
                        start_offset=start_offset + local_start,
                        separators=separators[1:],
                    )
                )
            else:
                pieces.append(
                    TextPiece(
                        text=part_with_sep,
                        start=start_offset + local_start,
                        end=start_offset + local_end,
                    )
                )

            cursor = local_end

        return pieces

    def _split_by_tokens(
        self,
        text: str,
        start_offset: int,
    ) -> List[TextPiece]:

        tokens = self.enc.encode(text)
        pieces = []

        for i in range(0, len(tokens), self.chunk_size):
            token_slice = tokens[i:i + self.chunk_size]
            piece_text = self.enc.decode(token_slice)

            # El texto decodificado corresponde a una porción del texto
            # original. Se calcula el offset mediante acumulación de
            # caracteres para no volver a buscar el chunk globalmente.
            if i == 0:
                local_start = 0
            else:
                local_start = len(
                    self.enc.decode(tokens[:i])
                )

            local_end = local_start + len(piece_text)

            pieces.append(
                TextPiece(
                    text=piece_text,
                    start=start_offset + local_start,
                    end=start_offset + local_end,
                )
            )

        return pieces

    def _merge_pieces(
        self,
        pieces: List[TextPiece],
    ) -> List[TextPiece]:
        """
        Une piezas hasta CHUNK_SIZE tokens.

        Cuando empieza un nuevo chunk, conserva CHUNK_OVERLAP tokens
        del chunk anterior siempre que sea posible.

        El nuevo chunk conserva como `start` la posición real del texto
        correspondiente al comienzo del overlap.
        """
        if not pieces:
            return []

        chunks: List[TextPiece] = []

        current_text = ""
        current_start = None
        current_end = None

        for piece in pieces:
            piece_tokens = self.token_count(piece.text)

            if not current_text:
                current_text = piece.text
                current_start = piece.start
                current_end = piece.end
                continue

            current_tokens = self.token_count(current_text)

            if current_tokens + piece_tokens <= self.chunk_size:
                current_text += piece.text
                current_end = piece.end
                continue

            # Guardar chunk actual.
            chunks.append(
                TextPiece(
                    text=current_text.strip(),
                    start=current_start,
                    end=current_end,
                )
            )

            # Intentar conservar overlap.
            overlap_tokens = min(
                self.chunk_overlap,
                current_tokens,
            )

            # Si la pieza siguiente es muy grande, reducimos el overlap
            # para no superar CHUNK_SIZE.
            max_allowed_overlap = max(
                0,
                self.chunk_size - piece_tokens,
            )

            overlap_tokens = min(
                overlap_tokens,
                max_allowed_overlap,
            )

            if overlap_tokens > 0:
                current_tokens_encoded = self.enc.encode(current_text)
                overlap_encoded = current_tokens_encoded[-overlap_tokens:]
                overlap_text = self.enc.decode(overlap_encoded)

                # El overlap corresponde al final exacto de current_text.
                overlap_start = current_end - len(overlap_text)

                current_text = overlap_text + piece.text
                current_start = overlap_start
                current_end = piece.end
            else:
                current_text = piece.text
                current_start = piece.start
                current_end = piece.end

        if current_text:
            chunks.append(
                TextPiece(
                    text=current_text.strip(),
                    start=current_start,
                    end=current_end,
                )
            )

        return chunks


# ─── CHUNKER GLOBAL ──────────────────────────────────────────────────────────

class GlobalRecursiveChunker:

    def __init__(self):
        self.enc = encoding

        self.text_splitter = SimpleRecursiveTextSplitter(
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            ["\n\n", "\n", ". ", "; ", " ", ""],
            self.enc,
        )

        self.table_splitter = SimpleRecursiveTextSplitter(
            TABLE_MAX_TOKENS,
            TABLE_CHUNK_OVERLAP,
            ["\n\n", "\n", " ", ""],
            self.enc,
        )

    @staticmethod
    def _page_number(page: Dict[str, Any]) -> int:
        """
        Número mostrado al usuario.

        Prioridad:
          1. número real/impreso detectado
          2. número físico del PDF
        """
        return page["pagina_numero"] or page["index_pagina"]

    def _build_global_text(
        self,
        paginas: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:

        partes: List[str] = []
        page_map: List[Dict[str, Any]] = []
        current_offset = 0

        for p in paginas:
            texto = (p["texto_plano"] or "").strip()

            if not texto:
                continue

            if partes:
                separador = "\n\n"
                partes.append(separador)
                current_offset += len(separador)

            start = current_offset
            partes.append(texto)
            end = current_offset + len(texto)

            page_map.append(
                {
                    "start": start,
                    "end": end,

                    # Página física para ordenar.
                    "pdf_page": p["index_pagina"],

                    # Página documental para mostrar.
                    "page_number": self._page_number(p),
                }
            )

            current_offset = end

        return "".join(partes), page_map

    @staticmethod
    def _pages_for_span(
        start: int,
        end: int,
        page_map: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Devuelve las páginas cuyo rango de caracteres intersecta el
        rango del chunk.

        No busca el texto. Utiliza exclusivamente offsets conocidos.
        """
        pages = []

        for page in page_map:
            if page["start"] < end and page["end"] > start:
                pages.append(page)

        # El orden SIEMPRE es el del PDF.
        pages.sort(key=lambda p: p["pdf_page"])

        return pages

    def _make_text_chunks(
        self,
        texto_global: str,
        page_map: List[Dict[str, Any]],
        source: str,
    ) -> List[Dict[str, Any]]:

        pieces = self.text_splitter.split_text_with_offsets(texto_global)

        chunks = []

        for idx, piece in enumerate(pieces):

            pages = self._pages_for_span(
                piece.start,
                piece.end,
                page_map,
            )

            if not pages:
                continue

            first_page = pages[0]
            last_page = pages[-1]

            chunks.append(
                {
                    "texto": piece.text,
                    "tipo": "texto",
                    "_sort_pdf_page": first_page["pdf_page"],
                    "_sort_start": piece.start,
                    "metadata": {
                        "source": source,

                        # Página donde comienza el chunk.
                        "page_number": first_page["page_number"],

                        # Todas las páginas que toca el chunk.
                        "pages_span": [
                            p["page_number"]
                            for p in pages
                        ],

                        # También conservamos las páginas físicas para
                        # poder verificar la trazabilidad del PDF.
                        "pdf_page_start": first_page["pdf_page"],
                        "pdf_page_end": last_page["pdf_page"],
                        "pdf_pages_span": [
                            p["pdf_page"]
                            for p in pages
                        ],

                        "token_count": self.enc.encode(piece.text).__len__(),

                        # Se reasigna posteriormente globalmente.
                        "chunk_idx": None,

                        "is_cross_page": len(pages) > 1,
                    },
                }
            )

        return chunks

    def _make_table_chunks(
        self,
        paginas: List[Dict[str, Any]],
        source: str,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for p in paginas:
            page_number = self._page_number(p)
            pdf_page = p["index_pagina"]

            for table_idx, tabla in enumerate(p["tablas"] or []):

                if not tabla.strip():
                    continue

                token_count = len(self.enc.encode(tabla))

                if token_count > TABLE_MAX_TOKENS:

                    sub_chunks = self.table_splitter.split_text_with_offsets(
                        tabla
                    )

                    for sub_idx, sub in enumerate(sub_chunks):

                        chunks.append(
                            {
                                "texto": sub.text,
                                "tipo": "tabla",

                                # Las tablas se ordenan por página física.
                                # Dentro de una página conservan su índice.
                                "_sort_pdf_page": pdf_page,
                                "_sort_start": 10_000_000 + table_idx * 10_000 + sub_idx,

                                "metadata": {
                                    "source": source,
                                    "page_number": page_number,
                                    "pages_span": [page_number],

                                    "pdf_page_start": pdf_page,
                                    "pdf_page_end": pdf_page,
                                    "pdf_pages_span": [pdf_page],

                                    "token_count": len(
                                        self.enc.encode(sub.text)
                                    ),

                                    "chunk_idx": None,
                                    "tabla_original_idx": table_idx,
                                    "tabla_subchunk_idx": sub_idx,
                                    "is_cross_page": False,
                                },
                            }
                        )

                else:
                    chunks.append(
                        {
                            "texto": tabla,
                            "tipo": "tabla",

                            "_sort_pdf_page": pdf_page,
                            "_sort_start": 10_000_000 + table_idx * 10_000,

                            "metadata": {
                                "source": source,
                                "page_number": page_number,
                                "pages_span": [page_number],

                                "pdf_page_start": pdf_page,
                                "pdf_page_end": pdf_page,
                                "pdf_pages_span": [pdf_page],

                                "token_count": token_count,

                                "chunk_idx": None,
                                "tabla_original_idx": table_idx,
                                "tabla_subchunk_idx": 0,
                                "is_cross_page": False,
                            },
                        }
                    )

        return chunks

    def chunk(
        self,
        paginas: List[Dict[str, Any]],
        source: str,
    ) -> List[Dict[str, Any]]:

        # 1. Construir documento global respetando el orden físico del PDF.
        texto_global, page_map = self._build_global_text(paginas)

        if not texto_global.strip():
            return []

        # 2. Chunking por TOKENS sobre TODO el documento.
        #
        # Esto permite que un chunk continúe desde una página hacia la
        # siguiente y que el overlap de 150 tokens también pueda cruzar
        # páginas.
        text_chunks = self._make_text_chunks(
            texto_global,
            page_map,
            source,
        )

        # 3. Tablas.
        table_chunks = self._make_table_chunks(
            paginas,
            source,
        )

        # 4. Unir y ordenar.
        #
        # El criterio principal es la página física del PDF.
        # El criterio secundario es la posición global del texto.
        #
        # Las tablas se colocan después de los chunks de texto de su página
        # porque actualmente no se conserva una coordenada física de tabla
        # dentro de la página.
        chunks = text_chunks + table_chunks

        chunks.sort(
            key=lambda c: (
                c["_sort_pdf_page"],
                c["_sort_start"],
            )
        )

        # 5. Índice GLOBAL y SECUENCIAL.
        #
        # Nunca se reinicia al cambiar de página o de tipo.
        for global_idx, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_idx"] = global_idx

            # Campos internos usados solo para ordenar.
            chunk.pop("_sort_pdf_page", None)
            chunk.pop("_sort_start", None)

        return chunks


# ─── PROCESAMIENTO DE PDFs ───────────────────────────────────────────────────

pdf_dir = Path(PDF_DIR)

if not pdf_dir.is_dir():
    raise SystemExit(
        f"❌ La carpeta '{PDF_DIR}' no existe."
    )

pdf_files = sorted(pdf_dir.glob("*.pdf"))

if not pdf_files:
    raise SystemExit(
        f"❌ No se encontraron archivos .pdf en '{PDF_DIR}'."
    )

print(f"📂 PDFs encontrados: {len(pdf_files)}")

for p in pdf_files:
    print(f"   - {p.name}")


all_chunks = []

for pdf_path in pdf_files:

    paginas = load_pdf_separado(str(pdf_path))

    chunker = GlobalRecursiveChunker()

    chunks = chunker.chunk(
        paginas,
        pdf_path.name,
    )

    all_chunks.extend(chunks)

    print(
        f"✅ {pdf_path.name}: "
        f"{len(chunks)} chunks extraídos"
    )


print(f"\n✅ Total chunks: {len(all_chunks)}")

print(
    f"   - Texto: "
    f"{sum(1 for c in all_chunks if c['tipo'] == 'texto')}"
)

print(
    f"   - Tablas: "
    f"{sum(1 for c in all_chunks if c['tipo'] == 'tabla')}"
)


# ─── GENERACIÓN DE PARES Q/A ─────────────────────────────────────────────────

if not OPENROUTER_API_KEY:
    raise SystemExit(
        "❌ No se encontró OPENROUTER_API_KEY en las variables de entorno."
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


SYSTEM_PROMPT = f"""Eres un generador experto de datasets para modelos de embeddings.
Dado un contexto de un documento universitario, genera exactamente {PARES_POR_CHUNK} pares pregunta-respuesta en formato JSON.

Reglas estrictas:
1. La respuesta DEBE estar contenida literalmente en el contexto.
2. Las preguntas deben ser variadas: factual, inferencial y de detalle específico.
3. Responde ÚNICAMENTE con un array JSON válido, sin texto adicional, sin markdown, sin backticks.
4. Formato exacto: [{{"pregunta": "...", "respuesta": "..."}}, ...]"""


with open(
    f"prompt_system_{PROMPT_VERSION}.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(SYSTEM_PROMPT)


dataset = []
errores = 0

print(
    f"\n🤖 Generando pares Q/A con {MODEL}...\n"
)


for i, chunk_data in enumerate(all_chunks):

    try:

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Contexto:\n"
                        f"{chunk_data['texto']}"
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=1500,
            seed=SEED,
        )

        raw = resp.choices[0].message.content.strip()

        raw = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            raw,
            flags=re.DOTALL,
        )

        pares = json.loads(raw)

        count = 0

        for par in pares:

            if (
                "pregunta" in par
                and "respuesta" in par
            ):

                dataset.append(
                    {
                        "contexto": chunk_data["texto"],
                        "pregunta": par["pregunta"].strip(),
                        "respuesta": par["respuesta"].strip(),

                        "fuente": chunk_data["metadata"]["source"],
                        "tipo": chunk_data["tipo"],

                        "page_number": chunk_data["metadata"]["page_number"],
                        "pages_span": chunk_data["metadata"].get(
                            "pages_span",
                            [chunk_data["metadata"]["page_number"]],
                        ),

                        "pdf_page_start": chunk_data["metadata"].get(
                            "pdf_page_start"
                        ),
                        "pdf_page_end": chunk_data["metadata"].get(
                            "pdf_page_end"
                        ),
                        "pdf_pages_span": chunk_data["metadata"].get(
                            "pdf_pages_span",
                            [],
                        ),

                        "is_cross_page": chunk_data["metadata"].get(
                            "is_cross_page",
                            False,
                        ),

                        "chunk_idx": chunk_data["metadata"]["chunk_idx"],
                        "token_count": chunk_data["metadata"]["token_count"],
                    }
                )

                count += 1

        print(
            f"[{i + 1:03d}/{len(all_chunks)}] "
            f"✅ {count} pares · "
            f"{chunk_data['tipo']} · "
            f"pág {chunk_data['metadata']['page_number']} · "
            f"PDF pág {chunk_data['metadata'].get('pdf_page_start')}"
        )

        time.sleep(0.5)

    except Exception as e:

        errores += 1

        print(
            f"[{i + 1:03d}/{len(all_chunks)}] "
            f"⚠️ Error: {str(e)[:120]}"
        )

        time.sleep(1.5)


# ─── GUARDAR DATASET ─────────────────────────────────────────────────────────

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        dataset,
        f,
        ensure_ascii=False,
        indent=2,
    )


print(
    f"\n✅ Dataset guardado en: {OUTPUT_JSON}"
)

print(
    f"📊 Total pares: {len(dataset)} | "
    f"Errores: {errores}"
)


print("\n--- Ejemplos del dataset ---")

for item in dataset[:2]:

    print(
        f"Páginas: {item['pages_span']} "
        f"(Cruza páginas: "
        f"{'Sí' if item['is_cross_page'] else 'No'})"
    )

    print(
        f"PDF: {item['pdf_pages_span']}"
    )

    print(
        f"Chunk: {item['chunk_idx']} | "
        f"Tokens: {item['token_count']}"
    )

    print(
        f"Q: {item['pregunta']}"
    )

    print(
        f"A: {item['respuesta']}\n"
    )
