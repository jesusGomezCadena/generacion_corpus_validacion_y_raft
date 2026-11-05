# RAFT para el RAG de acreditación — guía de trabajo

Referencia: Zhang, T., Patil, S. G., Jain, N., Shen, S., Zaharia, M., Stoica, I., & Gonzalez, J. E. (2024). *RAFT: Adapting Language Model to Domain Specific RAG*.

Ruta completa:

```
pares por chunk ──► [1] preparar_corpus_raft.py ──► raft_train.jsonl
                                                        │
                    [2] fine-tuning LoRA (Llama-3.1-8B) ◄┘
                                │
              ┌─────────────────┴──────────────────┐
     [3a] GGUF + Ollama (local)          [3b] HF Hub + vLLM serverless (API)
              └─────────────────┬──────────────────┘
                     [4] app Streamlit (mismo raft_formato.py)
                                │
                     [5] evaluación con el dataset de validación
```

---

## 1. Corpus RAFT (este script)

### Qué hace y qué parte del paper respalda cada paso

| Paso | Qué hace | Sustento en el paper |
|---|---|---|
| Limpieza | Quita pares vacíos o duplicados dentro del mismo chunk. | — |
| Exclusión de validación | Quita del entrenamiento los pares que preguntan por **el mismo hecho** que alguna pregunta de `datos/dataset_validacion_final.json` (detalle abajo). | La evaluación del paper usa preguntas distintas a las de entrenamiento sobre los mismos documentos (escenario *in-domain*, Sec. 2). |
| Control de calidad (LLM) | Clasifica cada par como ACEPTAR, AJUSTAR (reescribe la pregunta para que se entienda sola) o DESCARTAR. Las notas de revisión humana (`nota`) se le pasan al LLM. | Es un paso de calidad propio. El paper parte de datasets ya curados. |
| Respuesta CoT | Genera `##Reason:` con citas `##begin_quote##…##end_quote##` del documento dorado, seguido de `##Answer:`. | Sec. 3 y Fig. 3. En el paper, la ablación sin CoT pierde 9.7 puntos en HotpotQA y 14.9 en HuggingFace (Tabla 2). |
| Verificación de citas | Comprueba que cada cita exista literalmente en D\*. Si no existe, reintenta con retroalimentación; si vuelve a fallar, el par queda `fallido`. | Garantiza que el razonamiento "referencia el contexto original" (Sec. 3). |
| Distractores | Agrega k = 4 documentos (top-5 de la app = D\* + 4) que no contienen la respuesta. Se descartan los chunks vecinos que comparten la cita por el solapamiento del chunking. | Sec. 4.4 y 5: el paper usa 1 dorado + 4 distractores muestreados del corpus. |
| P | En el 80 % de los ejemplos va D\* + 4 distractores; en el 20 % restante van solo 5 distractores, con la misma respuesta objetivo. | Sec. 3 y 4.4 (Fig. 5). |
| Partición | Separa train y eval *por chunk dorado*, de modo que ningún D\* de eval aparece como dorado en train. | Sirve para monitorear la pérdida sin fuga. |

### Cómo correrlo (desde esta carpeta, con el `.venv` del proyecto)

```bash
# 0) prueba del flujo sin gastar (escribe en salida_simulacion/)
python preparar_corpus_raft.py --simular --limite 50

# 1) piloto real: 30 pares. Revisa salida/cot_cache.jsonl y el costo impreso
python preparar_corpus_raft.py --etapa cot --limite 30

# 2) corpus completo (~4.400 pares tras excluir validación; reanudable si se corta)
python preparar_corpus_raft.py --etapa cot

# 3) ensamblar (gratis; se puede repetir con otros k/P sin volver a llamar al LLM)
python preparar_corpus_raft.py --etapa ensamblar
```

Opciones útiles:

- `--modelo` cambia el generador. Por defecto es `meta-llama/llama-3.3-70b-instruct`, el mismo que ya usaron para generar el corpus. Se puede fijar también con la variable `RAFT_MODELO_GENERADOR`.
- `--reintentar-fallidos` vuelve a procesar los pares que fallaron.
- `--distractores mixtos` toma la mitad de los distractores entre los más parecidos por BM25 y la otra mitad al azar. Es una extensión que imita mejor al recuperador; por defecto se usa `aleatorios`, como en el paper.
- `--sin-cot` genera `raft_train_sincot.jsonl` para la ablación de la Tabla 2.

Salidas en `salida/`:

- `raft_train.jsonl` y `raft_eval.jsonl`: formato chat `{"messages":[system, user, assistant]}`, listo para Unsloth, TRL, Together, Fireworks u OpenAI.
- `raft_completo.jsonl`: trazabilidad de cada ejemplo (pregunta original o ajustada, veredicto, documentos usados, si incluyó D\*). Sirve para el capítulo de metodología.
- `reporte_raft.json`: cifras para la tesis (aceptados, ajustados, descartados, proporción real de P, tokens por ejemplo).
- `cot_cache.jsonl`: todo lo que devolvió el LLM. Es la caché; no la borres.

**Antes de la corrida completa**, revisa a mano unos 30–50 registros del piloto. Si el modelo descarta o ajusta demasiado, o si las respuestas no te gustan, cambia `PROMPT_SISTEMA_GENERADOR` o el modelo **antes** de gastar en las 5.000 llamadas.

Ya quedan calibrados por defecto: `TOP_K_APP = 5`, que da `--k 4`, y el dataset de validación en `datos/dataset_validacion_final.json`. Basta con correr el script sin parámetros extra.

### Exclusión del dataset de validación (`--modo-exclusion`)

El dataset de validación sale del mismo documento. Si el modelo se entrena con preguntas sobre los mismos hechos, el puntaje en validación se infla y el jurado lo puede objetar. Medí el corpus contra los 296 pares de validación: sus evidencias aparecen en 469 de los 1.750 chunks. Además había casos casi idénticos, por ejemplo "¿Dónde se desarrolla el programa…?" frente a "¿En qué sedes se desarrolla el programa…?".

Por eso la exclusión trabaja en tres capas (modo `hecho`, que es el predeterminado):

1. **Léxica global:** quita preguntas idénticas o con Jaccard ≥ 0,6 frente a cualquier pregunta de validación (69 pares).
2. **Por evidencia:** dentro de los chunks que contienen una `evidencia_gold`, quita los pares cuya respuesta está dentro de esa evidencia o cuya pregunta se parece a la de validación (619 pares).
3. **Semántica, con el LLM:** en la etapa `cot`, a cada par se le muestran las 3 preguntas de validación más cercanas según BM25, y el LLM marca `SOLAPA_EVALUACION` si piden el mismo dato. Esos pares se quitan al ensamblar y quedan listados en `salida/excluidos_por_solapamiento_llm.jsonl` para que los revises.

Antes del LLM quedan 4.397 pares. Otros modos:
- `pregunta`: solo la capa 1. Es el más permisivo.
- `chunk`: quita todos los pares de los chunks con evidencia de validación (quedan 3.725). Es el más estricto y sirve como análisis de sensibilidad.

El reporte guarda los conteos de cada capa para el capítulo de metodología.

### Dos decisiones a tener presentes

1. **El chunking debe ser el mismo de la app.** Los chunks de este corpus salen de `generacion_dataset_v3.py` (800 tokens, solapamiento de 150). Si el índice del RAG usa otra partición, el modelo verá en producción documentos de otro tamaño. Esto no lo invalida, pero conviene mencionarlo o alinearlo.
2. **El prompt vive en `raft_formato.py`.** La app debe importar ese mismo archivo para armar los mensajes. Si cambias el *system prompt* o el formato de `<DOCUMENTO>`, hay que regenerar el dataset y reentrenar.

---

## 2. Fine-tuning (lo más rápido y defendible)

- **Modelo base:** `meta-llama/Llama-3.1-8B-Instruct`. Es el sucesor directo del LLaMA-2-7B del paper (misma familia y tamaño), y eso facilita sustentarlo. Una alternativa razonable es `Qwen2.5-7B-Instruct` o `Qwen3-8B`, que suelen ir bien en español. Llama es *gated*: hay que aceptar la licencia en Hugging Face y usar el `HF_TOKEN` que ya tienen en el `.env`.
- **Método:** QLoRA con **Unsloth** en Google Colab o Kaggle. Una GPU T4 gratuita alcanza; una L4 o A100 lo hace varias veces más rápido. RAFT es SFT estándar sobre estos datos (Sec. 3), así que LoRA no se aparta del método. El paper hace fine-tuning pero no usa LoRA; esto se declara como decisión de eficiencia.
- **Hiperparámetros de partida** (el paper no los reporta todos; estos son valores estándar para QLoRA):
  - `r = 16`, `lora_alpha = 32`, `lr = 2e-4`, 2 épocas
  - `max_seq_length = 4096`
  - batch efectivo de 16
  - **calcular la pérdida solo sobre la respuesta del asistente** (`train_on_responses_only` en Unsloth)
- **Volumen esperado:** unos 3.500–4.000 ejemplos de aproximadamente 1.200 tokens (mediana medida en la simulación), es decir, unos 6M tokens por época. En una T4 son algunas horas; en una A100, bastante menos de una hora.
- **Exportar dos cosas al terminar:**
  - `push_to_hub_merged(..., save_method="merged_16bit")` a un repositorio **privado** de HF, para la API.
  - `save_pretrained_gguf(..., quantization_method="q4_k_m")`, para usarlo localmente.

---

## 3. Servir el modelo

La app ya habla con OpenRouter mediante el cliente de OpenAI. Las dos opciones siguientes también son compatibles con OpenAI, así que **solo cambian `base_url`, `api_key` y `model`**.

### 3a. Local, para pruebas: Ollama
```bash
ollama create raft-acreditacion -f Modelfile     # Modelfile: FROM ./modelo-q4_k_m.gguf (Unsloth lo genera)
# base_url = http://localhost:11434/v1   api_key = "ollama"   model = "raft-acreditacion"
```

### 3b. App desplegada: vLLM *serverless* en RunPod
Streamlit Cloud no puede alojar un modelo de 8B, así que el modelo debe estar detrás de una API.

1. Crea un endpoint **Serverless → vLLM** con `MODEL_NAME = tu-usuario/raft-acreditacion-8b` y `HF_TOKEN`, porque el repositorio es privado. Una GPU de 24 GB alcanza.
2. Usa `base_url = https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1` y como `api_key` tu llave de RunPod.
3. El endpoint escala a cero y se paga por segundo. La primera petición tras un periodo inactivo tiene **arranque en frío** de uno a varios minutos. Para la sustentación, deja 1 worker activo durante la demo.

Hay alternativas administradas: Fireworks o Together permiten entrenar y desplegar por API, pero hoy sirven los modelos ajustados en despliegues dedicados que se cobran por hora. Together suspendió su inferencia LoRA *serverless*. HF Inference Endpoints también funciona, con cobro por hora y opción de escalar a cero. Verifica precios antes de elegir.

---

## 4. Cambios en la app de Streamlit

```python
# .streamlit/secrets.toml  (local: valores de Ollama; en Streamlit Cloud: valores de RunPod)
# [generador_raft]
# base_url = "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1"
# api_key  = "..."
# model    = "tu-usuario/raft-acreditacion-8b"

import streamlit as st
from openai import OpenAI
from raft_formato import construir_mensajes, extraer_respuesta

cfg = st.secrets["generador_raft"]
cliente = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

def generar_raft(pregunta, chunks_recuperados):
    mensajes = construir_mensajes(pregunta, chunks_recuperados)   # MISMO formato del entrenamiento
    salida = cliente.chat.completions.create(model=cfg["model"], messages=mensajes,
                                             temperature=0, max_tokens=700)
    texto = salida.choices[0].message.content
    return extraer_respuesta(texto), texto      # respuesta para el usuario, razonamiento con citas
```

Conserva el generador de OpenRouter como opción seleccionable, porque es tu línea base. Para el modelo RAFT, pasa los documentos **en el orden en que los devuelve el recuperador**, ya que el modelo fue entrenado con órdenes aleatorios.

---

## 5. Evaluación (equivalente a la Tabla 1 del paper)

Usa el dataset de validación y tus métricas actuales, calculadas solo sobre el texto de `##Answer:`.

| Configuración | Equivalente en el paper |
|---|---|
| A. Modelo de OpenRouter + RAG (sistema actual) | GPT-3.5 + RAG |
| B. Llama-3.1-8B-Instruct sin ajustar + RAG | LLaMA2-7B + RAG |
| C. **Llama-3.1-8B + RAFT + RAG** | RAFT |
| D. (opcional) RAFT sin CoT (`--sin-cot`) | Tabla 2 |
| E. (opcional) C con top-k = 1, 3, 5, 10 | Fig. 6 |

D y E son baratas: D es otro entrenamiento corto y E solo requiere inferencia. Ambas fortalecen mucho la sustentación.

Ten en cuenta una limitación al escribir la tesis: el 20 % de ejemplos sin D\* entrena al modelo para responder "de memoria" cuando la recuperación falla. Esto es lo que propone el paper, pero implica que ante preguntas que no están en el documento el modelo puede responder en lugar de abstenerse.

---

## Checklist

- [x] Fijar top-k de la app (5 → `--k 4`)
- [x] Dataset de validación conectado (`datos/`)
- [ ] Piloto de 30 pares, revisión manual y ajuste del prompt o del modelo generador
- [ ] Corrida completa de `--etapa cot` y ensamblaje
- [ ] Aceptar la licencia de Llama 3.1 en HF; notebook de Unsloth en Colab o Kaggle
- [ ] Entrenar y subir el modelo fusionado (HF privado) y el GGUF
- [ ] Ollama local, probado con `raft_formato.py`
- [ ] Endpoint de RunPod y secrets en Streamlit Cloud
- [ ] Evaluación de las configuraciones A–C (y D–E si hay tiempo)
