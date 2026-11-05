"""
preparar_corpus_raft.py
=======================
Convierte los pares pregunta-respuesta por chunk en un dataset de entrenamiento
RAFT (Zhang et al., 2024, "RAFT: Adapting Language Model to Domain Specific RAG").

Etapa 1  cot        (usa un modelo barato de OpenRouter)
    Para cada par (D*, Q, A) el LLM:
      a) hace control de calidad: ACEPTAR / AJUSTAR (reescribe la pregunta para que
         sea autocontenida o corrige la respuesta) / DESCARTAR;
      b) genera la respuesta con razonamiento (Chain-of-Thought) citando D* entre
         ##begin_quote## y ##end_quote## (Sección 3 y Fig. 3 del paper).
    El script verifica que cada cita exista literalmente en D*; si no, reintenta.
    Los resultados se guardan en salida/cot_cache.jsonl (se puede interrumpir y
    reanudar sin pagar dos veces).

Etapa 2  ensamblar  (local, sin LLM, gratis y reproducible)
    Para cada pregunta aceptada arma el contexto de entrenamiento:
      P %     : Q + D* + D1 + ... + Dk  -> A*
      (1-P) % : Q + D1 + ... + Dk+1     -> A*      (sin documento dorado)
    con k distractores (por defecto 4, como en el paper) y P = 0.8 (Sección 4.4).
    Exporta JSONL en formato chat (system/user/assistant), que aceptan Unsloth,
    TRL, Together, Fireworks y OpenAI.

Ejemplos:
    python preparar_corpus_raft.py --etapa cot --limite 30          # piloto barato
    python preparar_corpus_raft.py --etapa cot                      # todo el corpus
    python preparar_corpus_raft.py --etapa ensamblar --k 4 --p 0.8
    python preparar_corpus_raft.py --etapa todo --excluir-preguntas ruta/dataset_validacion.json
    python preparar_corpus_raft.py --etapa todo --simular --limite 50   # prueba sin API
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raft_formato import (  # noqa: E402
    CITA_FIN, CITA_INICIO, MARCA_RAZON, MARCA_RESPUESTA,
    construir_mensajes, construir_objetivo, extraer_citas,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola de Windows
except Exception:
    pass

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────

DIR_SCRIPT = Path(__file__).resolve().parent
DIR_PROYECTO = DIR_SCRIPT.parent
ARCHIVOS_ENTRADA = [
    DIR_PROYECTO / "JsonsDatasetByChars" / "dataset_acreditacion_primera_mitad.json",
    DIR_PROYECTO / "JsonsDatasetByChars" / "dataset_acreditacion_segunda_mitad.json",
]
VALIDACION_DEFECTO = DIR_SCRIPT / "datos" / "dataset_validacion_final.json"
TOP_K_APP = 5                    # chunks que recupera el RAG en la app
DIR_SALIDA = DIR_SCRIPT / "salida"
ARCHIVO_CACHE = "cot_cache.jsonl"

MODELO_DEFECTO = os.getenv("RAFT_MODELO_GENERADOR", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_REINTENTOS_FORMATO = 2      # reintentos si las citas no son literales o falta el formato
MAX_REINTENTOS_RED = 5

DESCRIPCION_DOMINIO = (
    "el informe de autoevaluación con fines de acreditación de alta calidad del programa "
    "de Ingeniería de Sistemas de la Universidad del Valle"
)

# ─── PROMPT DEL GENERADOR (adaptación de la Fig. 3 del paper) ─────────────────

PROMPT_SISTEMA_GENERADOR = f"""Eres un anotador experto que prepara datos de entrenamiento para un asistente que responde preguntas sobre {DESCRIPCION_DOMINIO}.
Recibes un CONTEXTO (un fragmento del documento), una PREGUNTA, una RESPUESTA DE REFERENCIA y una lista de PREGUNTAS DE EVALUACIÓN que están reservadas para medir el sistema y no deben filtrarse al entrenamiento.

Tareas:
1. Revisa la calidad del par:
   - ¿La pregunta se puede responder con el CONTEXTO?
   - ¿La respuesta de referencia es correcta y suficiente según el CONTEXTO?
   - ¿La pregunta se entiende por sí sola, sin ver el CONTEXTO, como la formularía un usuario real? (no debe decir "según el texto", ni "el curso"/"la tabla" sin decir cuál)
   - ¿Es útil? (no debe preguntar por números de página, encabezados, pies de página u otros metadatos del PDF)
2. Elige un VEREDICTO:
   - ACEPTAR: el par está bien tal como está.
   - AJUSTAR: el par es aprovechable, pero la pregunta es ambigua o la respuesta es imprecisa o incompleta. Reescribe la pregunta para que sea autocontenida usando solo información del CONTEXTO y usa la respuesta corregida.
   - DESCARTAR: no se responde con el CONTEXTO, es trivial, pregunta por metadatos o no tiene arreglo.
3. Indica en SOLAPA_EVALUACION si la PREGUNTA (o su respuesta) pide el mismo dato o hecho que alguna PREGUNTA DE EVALUACIÓN, aunque esté redactada distinto. Escribe el número de esa pregunta, o NO si no coincide con ninguna. Tema parecido no basta: solo cuenta si la respuesta de una serviría para responder la otra.
4. Si no descartas, escribe un razonamiento breve y lógico que lleve a la respuesta. Cita los fragmentos del CONTEXTO que lo sustentan entre {CITA_INICIO} y {CITA_FIN}. Cada cita debe copiarse EXACTAMENTE como aparece en el CONTEXTO (carácter por carácter) y ser corta: una oración, parte de ella o una fila de tabla.
   Luego escribe la respuesta final: clara, en español, de una a tres oraciones, fiel al CONTEXTO.

Responde EXACTAMENTE con este formato y nada más:
VEREDICTO: <ACEPTAR|AJUSTAR|DESCARTAR>
MOTIVO: <una línea>
SOLAPA_EVALUACION: <NO|número>
PREGUNTA: <pregunta final; idéntica a la original si ACEPTAR>
{MARCA_RAZON} <razonamiento con citas>
{MARCA_RESPUESTA} <respuesta final>

Si DESCARTAR, escribe "-" en PREGUNTA, {MARCA_RAZON} y {MARCA_RESPUESTA}.

Ejemplo:
CONTEXTO: \"\"\"El programa cuenta con 18 profesores de tiempo completo, de los cuales 12 tienen título de doctorado. La relación estudiante/profesor es de 25:1.\"\"\"
PREGUNTA: ¿Cuántos profesores tienen doctorado?
RESPUESTA DE REFERENCIA: 12
PREGUNTAS DE EVALUACIÓN:
1. ¿Cuál es la relación estudiante/profesor del programa?
2. ¿Cuántos grupos de investigación tiene la Escuela?
VEREDICTO: AJUSTAR
MOTIVO: La pregunta no indica de qué programa o grupo de profesores se habla.
SOLAPA_EVALUACION: NO
PREGUNTA: ¿Cuántos profesores de tiempo completo del programa de Ingeniería de Sistemas tienen título de doctorado?
{MARCA_RAZON} El documento indica {CITA_INICIO}El programa cuenta con 18 profesores de tiempo completo, de los cuales 12 tienen título de doctorado.{CITA_FIN} Por lo tanto, de los 18 profesores de tiempo completo, 12 son doctores.
{MARCA_RESPUESTA} 12 de los 18 profesores de tiempo completo del programa tienen título de doctorado."""


def prompt_usuario_generador(par: Dict[str, Any]) -> str:
    nota = f"\nNOTA DEL REVISOR HUMANO: {par['nota']}" if par.get("nota") else ""
    cercanas = par.get("val_cercanas") or []
    lista = "\n".join(f"{i}. {q}" for i, q in enumerate(cercanas, 1)) or "(ninguna)"
    return (
        f'CONTEXTO: """{par["contexto"]}"""\n'
        f"PREGUNTA: {par['pregunta']}\n"
        f"RESPUESTA DE REFERENCIA: {par['respuesta']}{nota}\n"
        f"PREGUNTAS DE EVALUACIÓN:\n{lista}"
    )


# ─── UTILIDADES ──────────────────────────────────────────────────────────────

def sha(texto: str, n: int = 16) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()[:n]


def norm_literal(s: str) -> str:
    """Normaliza para verificar citas: ignora mayúsculas, espacios, puntuación,
    pipes de tablas Markdown, ** y <br>. Conserva letras con tilde y números."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    return re.sub(r"[\W_]+", "", s)


def norm_pregunta(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def tokens_pregunta(s: str) -> set:
    return {t for t in norm_pregunta(s).split() if len(t) > 2}


try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def contar_tokens(t: str) -> int:
        return len(_ENC.encode(t))
except Exception:  # aproximación si no está tiktoken
    def contar_tokens(t: str) -> int:
        return math.ceil(len(t) / 3.5)


# ─── CARGA Y LIMPIEZA ────────────────────────────────────────────────────────

def cargar_pares(archivos: List[Path]):
    pares, chunks = [], {}
    for ruta in archivos:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        for x in datos:
            ctx = x["contexto"]
            id_chunk = sha(ctx, 12)
            chunks.setdefault(id_chunk, {
                "id_chunk": id_chunk, "texto": ctx, "fuente": x.get("fuente"),
                "page_number": x.get("page_number"), "chunk_idx": x.get("chunk_idx"),
                "tipo": x.get("tipo"),
            })
            pares.append({
                "id": sha(f"{ctx}\x1f{x['pregunta']}\x1f{x['respuesta']}"),
                "id_chunk": id_chunk, "contexto": ctx,
                "pregunta": (x.get("pregunta") or "").strip(),
                "respuesta": (x.get("respuesta") or "").strip(),
                "nota": x.get("nota"), "tipo": x.get("tipo"),
                "fuente": x.get("fuente"), "page_number": x.get("page_number"),
                "archivo_origen": ruta.name,
            })
    return pares, chunks


def leer_items_validacion(rutas: List[str]) -> List[Dict[str, Any]]:
    """Lee el/los dataset(s) de validación (JSON, JSON con lista anidada como
    {"pares": [...]}, o JSONL) y devuelve [{"pregunta", "evidencias"}]."""
    claves_q = ("pregunta", "question", "query", "user_input", "input")
    claves_r = ("respuesta_esperada", "respuesta", "answer", "reference", "ground_truth")
    claves_ev = ("evidencia_gold", "evidencia", "reference_contexts", "contexts", "contexto")
    items = []
    for ruta in rutas:
        texto = Path(ruta).read_text(encoding="utf-8").strip()
        try:
            datos = json.loads(texto)
            if isinstance(datos, dict):
                datos = next((v for v in datos.values() if isinstance(v, list)), [datos])
        except json.JSONDecodeError:
            datos = [json.loads(l) for l in texto.splitlines() if l.strip()]
        for d in datos:
            if not isinstance(d, dict):
                continue
            q = next((d[k] for k in claves_q if isinstance(d.get(k), str)), None)
            ev = next((d[k] for k in claves_ev if d.get(k)), [])
            ev = [ev] if isinstance(ev, str) else [e for e in ev if isinstance(e, str)]
            r = next((d[k] for k in claves_r if isinstance(d.get(k), str)), "")
            if q:
                items.append({"pregunta": q, "respuesta": r, "evidencias": ev})
    return items


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def limpiar(pares, chunks, excluir_rutas: List[str], modo: str = "hecho",
            umbral_global: float = 0.6, umbral_mismo_chunk: float = 0.4):
    """Limpieza básica + exclusión de pares que 'filtran' el dataset de validación.

    modo='pregunta': quita preguntas iguales o muy parecidas (Jaccard >= umbral_global)
                     a alguna de validación.
    modo='hecho'   : lo anterior + en los chunks que contienen una evidencia_gold de
                     validación, quita los pares que preguntan por ese mismo hecho
                     (su respuesta está dentro de la evidencia, o su pregunta se parece
                     a la de validación). Recomendado.
    modo='chunk'   : lo anterior + quita TODOS los pares cuyo chunk contiene evidencia
                     de validación (el más estricto; esos chunks siguen como distractores).
    """
    stats = Counter()
    vistos, limpios = set(), []
    for p in pares:
        if not p["pregunta"] or not p["respuesta"]:
            stats["vacios"] += 1
            continue
        clave = (p["id_chunk"], norm_pregunta(p["pregunta"]))
        if clave in vistos:
            stats["duplicados_mismo_chunk"] += 1
            continue
        vistos.add(clave)
        limpios.append(p)

    if not excluir_rutas:
        return limpios, stats

    val = leer_items_validacion(excluir_rutas)
    stats["validacion_preguntas"] = len(val)
    exactas = {norm_pregunta(v["pregunta"]) for v in val}
    conj_val = [tokens_pregunta(v["pregunta"]) for v in val]
    indice = defaultdict(list)                      # token -> índices de validación
    for i, c in enumerate(conj_val):
        for t in c:
            indice[t].append(i)

    # Ubicar cada evidencia de validación en los chunks del corpus.
    norm_chunks = {k: norm_literal(c["texto"]) for k, c in chunks.items()}
    val_en_chunk = defaultdict(list)                # id_chunk -> [(idx_val, evidencia_norm)]
    encontradas = 0
    for i, v in enumerate(val):
        hallada = False
        for e in v["evidencias"]:
            en = norm_literal(e)
            if len(en) < 10:
                continue
            claves = [en] if len(en) <= 80 else [en, en[:80], en[-80:]]   # evidencia partida entre chunks
            for k, t in norm_chunks.items():
                if any(c in t for c in claves):
                    val_en_chunk[k].append((i, en))
                    hallada = True
        encontradas += hallada
    stats["validacion_con_evidencia_ubicada"] = encontradas
    stats["chunks_con_evidencia_de_validacion"] = len(val_en_chunk)

    filtrados = []
    for p in limpios:
        if norm_pregunta(p["pregunta"]) in exactas:
            stats["excluidas_pregunta_identica"] += 1
            continue
        tp = tokens_pregunta(p["pregunta"])
        candidatos = {i for t in tp for i in indice.get(t, ())}
        if any(jaccard(tp, conj_val[i]) >= umbral_global for i in candidatos):
            stats["excluidas_pregunta_similar"] += 1
            continue
        enlaces = val_en_chunk.get(p["id_chunk"], [])
        if enlaces and modo == "chunk":
            stats["excluidas_chunk_de_validacion"] += 1
            continue
        if enlaces and modo == "hecho":
            rn = norm_literal(p["respuesta"])
            mismo_hecho = any(len(rn) >= 3 and rn in en for _, en in enlaces)
            parecida = any(jaccard(tp, conj_val[i]) >= umbral_mismo_chunk for i, _ in enlaces)
            if mismo_hecho or parecida:
                stats["excluidas_mismo_hecho_que_validacion"] += 1
                continue
        filtrados.append(p)

    # Para el juicio semántico del LLM (etapa cot): las 3 preguntas de validación más
    # cercanas a cada par según BM25 sobre pregunta + respuesta esperada + evidencia.
    bm = BM25([f"{v['pregunta']} {v['respuesta']} {' '.join(v['evidencias'])}" for v in val])
    for p in filtrados:
        p["val_cercanas"] = [val[i]["pregunta"] for i in bm.ranking(f"{p['pregunta']} {p['respuesta']}")[:3]]
    return filtrados, stats


# ─── ETAPA 1: CONTROL DE CALIDAD + CoT ───────────────────────────────────────

def parsear_salida(texto: str) -> Dict[str, str]:
    def campo(nombre, siguiente):
        m = re.search(rf"{nombre}\s*(.*?)\s*(?={siguiente}|\Z)", texto, flags=re.S | re.I)
        return m.group(1).strip() if m else ""

    r = re.escape(MARCA_RAZON)
    a = re.escape(MARCA_RESPUESTA)
    veredicto = campo("VEREDICTO:", "MOTIVO:").split()[0].upper() if campo("VEREDICTO:", "MOTIVO:") else ""
    return {
        "veredicto": re.sub(r"[^A-Z]", "", veredicto),
        "motivo": campo("MOTIVO:", "SOLAPA_EVALUACION:|PREGUNTA:"),
        "solapa_evaluacion": (campo("SOLAPA_EVALUACION:", "PREGUNTA:").split() or [""])[0].upper().strip(".:"),
        "pregunta": campo("PREGUNTA:", r),
        "razonamiento": campo(r, a),
        "respuesta": campo(a, r"\Z"),
    }


def validar_salida(sal: Dict[str, str], contexto: str) -> List[str]:
    errores = []
    if sal["veredicto"] not in {"ACEPTAR", "AJUSTAR", "DESCARTAR"}:
        return ["Falta VEREDICTO válido (ACEPTAR, AJUSTAR o DESCARTAR)."]
    if sal["veredicto"] == "DESCARTAR":
        return []
    if not (sal["solapa_evaluacion"] == "NO" or sal["solapa_evaluacion"].isdigit()):
        errores.append("Falta SOLAPA_EVALUACION (NO o el número de la pregunta de evaluación).")
    if len(sal["pregunta"]) < 8 or sal["pregunta"] == "-":
        errores.append("Falta la PREGUNTA final.")
    if len(sal["respuesta"]) < 1 or sal["respuesta"] == "-":
        errores.append(f"Falta la respuesta final después de {MARCA_RESPUESTA}.")
    citas = extraer_citas(sal["razonamiento"])
    if not citas:
        errores.append(f"El razonamiento no tiene citas entre {CITA_INICIO} y {CITA_FIN}.")
    ctx_norm = norm_literal(contexto)
    for c in citas:
        cn = norm_literal(c)
        if len(cn) < 3 or cn not in ctx_norm:
            errores.append(f'La cita "{c[:120]}" no aparece literalmente en el CONTEXTO.')
    return errores


def respuesta_simulada(par: Dict[str, Any]) -> str:
    """Salida falsa para probar el flujo sin gastar en la API (--simular)."""
    ctx = par["contexto"]
    resp = par["respuesta"]
    if norm_literal(resp) and norm_literal(resp) in norm_literal(ctx):
        frase = next((s for s in re.split(r"(?<=[.\n])", ctx) if norm_literal(resp) in norm_literal(s)), resp)
        return (f"VEREDICTO: ACEPTAR\nMOTIVO: simulado\nSOLAPA_EVALUACION: NO\nPREGUNTA: {par['pregunta']}\n"
                f"{MARCA_RAZON} El documento dice {CITA_INICIO}{frase.strip()}{CITA_FIN} por lo tanto la respuesta es {resp}.\n"
                f"{MARCA_RESPUESTA} {resp}")
    return "VEREDICTO: DESCARTAR\nMOTIVO: simulado, respuesta no literal\nPREGUNTA: -\n##Reason: -\n##Answer: -"


class Generador:
    def __init__(self, modelo: str, simular: bool, temperatura: float):
        self.modelo, self.simular, self.temperatura = modelo, simular, temperatura
        self.uso = Counter()
        self.lock = threading.Lock()
        if not simular:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(DIR_PROYECTO / ".env")o
            load_dotenv()
            clave = os.getenv("OPENROUTER_API_KEY")
            if not clave:
                sys.exit("Falta OPENROUTER_API_KEY en el .env")
            self.cliente = OpenAI(base_url=OPENROUTER_URL, api_key=clave)

    def _llamar(self, mensajes, par) -> str:
        if self.simular:
            return respuesta_simulada(par)
        espera = 2.0
        for intento in range(MAX_REINTENTOS_RED):
            try:
                r = self.cliente.chat.completions.create(
                    model=self.modelo, messages=mensajes, temperature=self.temperatura,
                    max_tokens=900, extra_body={"usage": {"include": True}},
                )
                u = r.usage
                with self.lock:
                    if u:
                        self.uso["tokens_entrada"] += u.prompt_tokens or 0
                        self.uso["tokens_salida"] += u.completion_tokens or 0
                        costo = getattr(u, "cost", None) or (u.model_extra or {}).get("cost")
                        if costo:
                            self.uso["costo_usd_x1e6"] += int(float(costo) * 1e6)
                    self.uso["llamadas"] += 1
                contenido = r.choices[0].message.content if r.choices else None
                if contenido:
                    return contenido
                raise RuntimeError("respuesta vacía")
            except Exception:  # red, rate limit, 5xx
                if intento == MAX_REINTENTOS_RED - 1:
                    raise
                time.sleep(espera + random.random())
                espera *= 2
        return ""

    def procesar(self, par: Dict[str, Any]) -> Dict[str, Any]:
        mensajes = [
            {"role": "system", "content": PROMPT_SISTEMA_GENERADOR},
            {"role": "user", "content": prompt_usuario_generador(par)},
        ]
        registro = {
            "id": par["id"], "id_chunk": par["id_chunk"], "modelo": self.modelo,
            "pregunta_original": par["pregunta"], "respuesta_original": par["respuesta"],
            "nota": par.get("nota"), "val_cercanas": par.get("val_cercanas") or [],
        }
        errores: List[str] = []
        try:
            for intento in range(MAX_REINTENTOS_FORMATO + 1):
                crudo = self._llamar(mensajes, par)
                sal = parsear_salida(crudo)
                errores = validar_salida(sal, par["contexto"])
                if not errores:
                    estado = "descartado" if sal["veredicto"] == "DESCARTAR" else "ok"
                    registro.update(sal, estado=estado, intentos=intento + 1, salida_cruda=crudo)
                    return registro
                mensajes += [
                    {"role": "assistant", "content": crudo},
                    {"role": "user", "content": "Tu respuesta tiene estos problemas:\n- " + "\n- ".join(errores)
                     + "\nCorrígela respetando exactamente el formato pedido y copiando las citas tal cual."},
                ]
            registro.update(sal, estado="fallido", intentos=MAX_REINTENTOS_FORMATO + 1,
                            errores=errores, salida_cruda=crudo)
        except Exception as e:
            registro.update(estado="error_api", errores=[repr(e)])
        return registro


def leer_cache(ruta: Path) -> Dict[str, Dict[str, Any]]:
    cache = {}
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                if linea.strip():
                    r = json.loads(linea)
                    cache[r["id"]] = r   # la última versión gana
    return cache


def etapa_cot(pares, args):
    ruta_cache = DIR_SALIDA / ARCHIVO_CACHE
    cache = leer_cache(ruta_cache)
    reintentar = {"error_api", "fallido"} if args.reintentar_fallidos else {"error_api"}
    pendientes = [p for p in pares if p["id"] not in cache or cache[p["id"]]["estado"] in reintentar]
    if args.limite:
        pendientes = pendientes[: args.limite]
    print(f"[cot] pares válidos: {len(pares)} | en caché: {len(cache)} | por procesar: {len(pendientes)}")
    print(f"[cot] modelo: {args.modelo}{'  (SIMULADO)' if args.simular else ''}")
    if not pendientes:
        return

    gen = Generador(args.modelo, args.simular, args.temperatura)
    lock = threading.Lock()
    try:
        from tqdm import tqdm
        barra = tqdm(total=len(pendientes), unit="par")
    except Exception:
        barra = None

    estados = Counter()
    with open(ruta_cache, "a", encoding="utf-8") as fcache, \
            ThreadPoolExecutor(max_workers=1 if args.simular else args.workers) as ex:
        futuros = [ex.submit(gen.procesar, p) for p in pendientes]
        for i, fut in enumerate(as_completed(futuros), 1):
            r = fut.result()
            estados[r["estado"]] += 1
            with lock:
                fcache.write(json.dumps(r, ensure_ascii=False) + "\n")
                fcache.flush()
            if barra:
                barra.update(1)
            elif i % 50 == 0:
                print(f"  {i}/{len(pendientes)} {dict(estados)}")
    if barra:
        barra.close()

    costo = gen.uso["costo_usd_x1e6"] / 1e6
    print(f"[cot] resultado de esta corrida: {dict(estados)}")
    print(f"[cot] llamadas: {gen.uso['llamadas']} | tokens entrada: {gen.uso['tokens_entrada']:,} | "
          f"salida: {gen.uso['tokens_salida']:,} | costo reportado por OpenRouter: ${costo:.4f}")


# ─── ETAPA 2: ENSAMBLAJE RAFT ────────────────────────────────────────────────

class BM25:
    """BM25 mínimo (sin dependencias) para distractores 'difíciles' opcionales."""

    def __init__(self, textos: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [norm_pregunta(t).split() for t in textos]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def ranking(self, consulta: str) -> List[int]:
        q = [t for t in norm_pregunta(consulta).split() if t in self.idf]
        puntajes = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            puntajes.append((s, i))
        return [i for s, i in sorted(puntajes, reverse=True) if s > 0]


def etapa_ensamblar(pares, chunks, args, stats_limpieza=None):
    cache = leer_cache(DIR_SALIDA / ARCHIVO_CACHE)
    ids_validos = {p["id"] for p in pares}
    aceptados = [r for r in cache.values() if r["estado"] == "ok" and r["id"] in ids_validos]
    solapados = []
    if args.excluir_preguntas and args.modo_exclusion != "pregunta":
        solapados = [r for r in aceptados if r.get("solapa_evaluacion", "NO") != "NO"]
        aceptados = [r for r in aceptados if r.get("solapa_evaluacion", "NO") == "NO"]
    if solapados:
        with open(DIR_SALIDA / "excluidos_por_solapamiento_llm.jsonl", "w", encoding="utf-8") as f:
            for r in solapados:
                i = int(r["solapa_evaluacion"]) - 1
                cerc = r.get("val_cercanas") or []
                f.write(json.dumps({"pregunta_entrenamiento": r["pregunta"],
                                    "pregunta_evaluacion": cerc[i] if 0 <= i < len(cerc) else None},
                                   ensure_ascii=False) + "\n")
    if not aceptados:
        sys.exit("[ensamblar] No hay pares aceptados en la caché. Corre primero --etapa cot.")

    rng = random.Random(args.semilla)
    lista_chunks = list(chunks.values())
    ids_chunks = [c["id_chunk"] for c in lista_chunks]
    norm_chunks = {c["id_chunk"]: norm_literal(c["texto"]) for c in lista_chunks}
    bm25 = BM25([c["texto"] for c in lista_chunks]) if args.distractores == "mixtos" else None

    # División train/eval por chunk dorado: ninguna pregunta de eval comparte D* con train.
    chunks_con_preguntas = sorted({r["id_chunk"] for r in aceptados})
    rng_split = random.Random(args.semilla + 1)
    rng_split.shuffle(chunks_con_preguntas)
    n_eval = int(round(len(chunks_con_preguntas) * args.frac_eval))
    chunks_eval = set(chunks_con_preguntas[:n_eval])

    def es_distractor_valido(id_c: str, r: Dict[str, Any]) -> bool:
        """Un distractor no debe contener la respuesta (Sección 3: 'no contienen
        información relevante para la respuesta'). Se excluyen chunks que contienen
        alguna cita o la respuesta de referencia (p. ej. chunks vecinos con solapamiento)."""
        if id_c == r["id_chunk"]:
            return False
        texto = norm_chunks[id_c]
        for c in extraer_citas(r["razonamiento"]):
            cn = norm_literal(c)
            if len(cn) >= 15 and cn in texto:
                return False
        ref = norm_literal(r["respuesta_original"])
        return not (len(ref) >= 12 and ref in texto)

    def elegir_distractores(r, n: int) -> List[str]:
        elegidos: List[str] = []
        if bm25 is not None:
            n_dificiles = math.ceil(n / 2)
            for i in bm25.ranking(r["pregunta"])[:40]:
                if len(elegidos) >= n_dificiles:
                    break
                cid = ids_chunks[i]
                if cid not in elegidos and es_distractor_valido(cid, r):
                    elegidos.append(cid)
        intentos = 0
        while len(elegidos) < n and intentos < 10_000:
            cid = rng.choice(ids_chunks)
            intentos += 1
            if cid not in elegidos and es_distractor_valido(cid, r):
                elegidos.append(cid)
        return elegidos

    ejemplos_train, ejemplos_eval, completos = [], [], []
    stats = Counter()
    tokens = []
    for r in sorted(aceptados, key=lambda x: x["id"]):
        con_oraculo = rng.random() < args.p
        n_dist = args.k if con_oraculo else args.k + 1
        docs = elegir_distractores(r, n_dist)
        if con_oraculo:
            docs.append(r["id_chunk"])
        rng.shuffle(docs)

        mensajes = construir_mensajes(r["pregunta"], [chunks[d]["texto"] for d in docs])
        objetivo = r["respuesta"] if args.sin_cot else construir_objetivo(r["razonamiento"], r["respuesta"])
        mensajes.append({"role": "assistant", "content": objetivo})

        particion = "eval" if r["id_chunk"] in chunks_eval else "train"
        (ejemplos_eval if particion == "eval" else ejemplos_train).append({"messages": mensajes})
        n_tok = sum(contar_tokens(m["content"]) for m in mensajes)
        tokens.append(n_tok)
        completos.append({
            "id": r["id"], "particion": particion, "con_documento_dorado": con_oraculo,
            "id_chunk_dorado": r["id_chunk"], "ids_documentos": docs,
            "pregunta": r["pregunta"], "pregunta_original": r["pregunta_original"],
            "respuesta": r["respuesta"], "respuesta_original": r["respuesta_original"],
            "veredicto": r["veredicto"], "motivo": r["motivo"], "tokens_aprox": n_tok,
            "messages": mensajes,
        })
        stats["con_dorado" if con_oraculo else "sin_dorado"] += 1
        stats[f"veredicto_{r['veredicto']}"] += 1
        stats[particion] += 1
        if len(docs) < (args.k + 1):
            stats["pocos_distractores"] += 1

    sufijo = "_sincot" if args.sin_cot else ""
    def escribir(nombre, filas):
        with open(DIR_SALIDA / nombre, "w", encoding="utf-8") as f:
            for fila in filas:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    escribir(f"raft_train{sufijo}.jsonl", ejemplos_train)
    escribir(f"raft_eval{sufijo}.jsonl", ejemplos_eval)
    escribir(f"raft_completo{sufijo}.jsonl", completos)

    tokens.sort()
    estados_cache = Counter(r["estado"] for r in cache.values() if r["id"] in ids_validos)
    reporte = {
        "paper": "Zhang et al. (2024) RAFT: Adapting Language Model to Domain Specific RAG",
        "parametros": {"k_distractores": args.k, "P_documento_dorado": args.p,
                       "distractores": args.distractores, "sin_cot": args.sin_cot,
                       "frac_eval": args.frac_eval, "semilla": args.semilla},
        "pares_de_entrada_tras_limpieza": len(pares),
        "limpieza_y_exclusion_validacion": dict(stats_limpieza or {}),
        "excluidos_por_solapamiento_llm": len(solapados),
        "modo_exclusion": args.modo_exclusion,
        "estados_etapa_cot": dict(estados_cache),
        "ejemplos": dict(stats),
        "proporcion_real_con_dorado": round(stats["con_dorado"] / max(1, len(completos)), 3),
        "tokens_por_ejemplo": {"min": tokens[0], "mediana": tokens[len(tokens) // 2],
                               "p95": tokens[int(len(tokens) * 0.95)], "max": tokens[-1],
                               "total": sum(tokens)},
    }
    (DIR_SALIDA / f"reporte_raft{sufijo}.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    print(f"[ensamblar] archivos en {DIR_SALIDA}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Construye un dataset RAFT a partir de pares por chunk.")
    ap.add_argument("--etapa", choices=["cot", "ensamblar", "todo"], default="todo")
    ap.add_argument("--entrada", nargs="+", default=[str(p) for p in ARCHIVOS_ENTRADA])
    ap.add_argument("--excluir-preguntas", nargs="*", default=[str(VALIDACION_DEFECTO)] if VALIDACION_DEFECTO.exists() else [],
                    help="Dataset(s) de validación cuyos hechos no deben entrar al entrenamiento.")
    ap.add_argument("--modo-exclusion", choices=["pregunta", "hecho", "chunk"], default="hecho")
    # etapa cot
    ap.add_argument("--modelo", default=MODELO_DEFECTO, help="Modelo de OpenRouter para generar CoT.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperatura", type=float, default=0.2)
    ap.add_argument("--limite", type=int, default=0, help="Procesar solo N pares (piloto).")
    ap.add_argument("--reintentar-fallidos", action="store_true")
    ap.add_argument("--simular", action="store_true", help="No llama a la API (prueba del flujo).")
    # etapa ensamblar
    ap.add_argument("--k", type=int, default=TOP_K_APP - 1,
                    help="Distractores por ejemplo = top-k de la app - 1 (app: top-5 -> 4, igual que el paper).")
    ap.add_argument("--p", type=float, default=0.8, help="Fracción con documento dorado (paper: 0.8).")
    ap.add_argument("--distractores", choices=["aleatorios", "mixtos"], default="aleatorios",
                    help="aleatorios = como el paper; mixtos = mitad similares por BM25 + mitad aleatorios.")
    ap.add_argument("--sin-cot", action="store_true", help="Ablación 'RAFT sin CoT' (Tabla 2).")
    ap.add_argument("--frac-eval", type=float, default=0.05)
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    global DIR_SALIDA
    if args.simular:  # la simulación nunca toca la caché real
        DIR_SALIDA = DIR_SCRIPT / "salida_simulacion"
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    pares, chunks = cargar_pares([Path(p) for p in args.entrada])
    pares, stats = limpiar(pares, chunks, args.excluir_preguntas, args.modo_exclusion)
    print(f"[datos] chunks únicos: {len(chunks)} | pares tras limpieza: {len(pares)} | {dict(stats)}")

    if args.etapa in ("cot", "todo"):
        etapa_cot(pares, args)
    if args.etapa in ("ensamblar", "todo"):
        etapa_ensamblar(pares, chunks, args, stats)


if __name__ == "__main__":
    main()
