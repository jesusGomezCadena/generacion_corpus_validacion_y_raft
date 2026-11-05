"""
raft_formato.py
===============
Formato ÚNICO de prompt para el modelo RAFT.

Este módulo lo importan tanto el script que construye el dataset
(preparar_corpus_raft.py) como la app de Streamlit en inferencia.
RAFT asume que el modelo ve en la prueba exactamente el mismo formato
que vio en el entrenamiento: pregunta + top-k documentos -> respuesta
con razonamiento (Zhang et al., 2024, Sección 3). Si cambias algo aquí,
hay que regenerar el dataset y reentrenar.

Uso en la app:

    from raft_formato import construir_mensajes, extraer_respuesta
    mensajes = construir_mensajes(pregunta, [c.page_content for c in chunks])
    salida = cliente.chat.completions.create(model=MODELO, messages=mensajes,
                                             temperature=0).choices[0].message.content
    respuesta = extraer_respuesta(salida)        # lo que se le muestra al usuario
"""

from __future__ import annotations

import re
from typing import Iterable, List, Dict

# Marcadores tomados literalmente del paper (Fig. 3).
MARCA_RAZON = "##Reason:"
MARCA_RESPUESTA = "##Answer:"
CITA_INICIO = "##begin_quote##"
CITA_FIN = "##end_quote##"

SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre el informe de autoevaluación "
    "con fines de acreditación del programa de Ingeniería de Sistemas. "
    "Recibirás varios documentos recuperados; algunos pueden no ser relevantes para la pregunta. "
    "Usa solo la información pertinente: razona paso a paso, copia textualmente los fragmentos "
    f"que sustentan tu respuesta entre {CITA_INICIO} y {CITA_FIN}, y termina con la respuesta "
    f"final después de '{MARCA_RESPUESTA}'. Responde siempre en español."
)


def formatear_documentos(documentos: Iterable[str]) -> str:
    return "\n\n".join(f"<DOCUMENTO>\n{d.strip()}\n</DOCUMENTO>" for d in documentos)


def construir_mensajes(pregunta: str, documentos: Iterable[str]) -> List[Dict[str, str]]:
    """Mensajes (formato chat) que recibe el modelo, idénticos en entrenamiento e inferencia."""
    contenido = f"{formatear_documentos(documentos)}\n\nPregunta: {pregunta.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": contenido},
    ]


def construir_objetivo(razonamiento: str, respuesta: str) -> str:
    """Texto que el modelo aprende a generar (respuesta CoT del paper)."""
    return f"{MARCA_RAZON} {razonamiento.strip()}\n{MARCA_RESPUESTA} {respuesta.strip()}"


def extraer_respuesta(texto: str) -> str:
    """Devuelve solo la respuesta final (lo que va después de ##Answer:)."""
    if MARCA_RESPUESTA in texto:
        return texto.split(MARCA_RESPUESTA, 1)[1].strip()
    return texto.strip()


def extraer_citas(texto: str) -> List[str]:
    patron = re.escape(CITA_INICIO) + r"(.*?)" + re.escape(CITA_FIN)
    return [c.strip() for c in re.findall(patron, texto, flags=re.S)]
