# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Validador de Dataset para Entrenamiento de Embeddings                      ║
# ║  Uso:  python validar_dataset.py                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import json
import random
from pathlib import Path
from collections import Counter
import statistics

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
DATASET_FILE       = "dataset_acreditacion.json"  # archivo a validar
TAMANIO_MUESTRA    = 300                           # tamaño de muestra para validación manual
SEED               = 42                            # para reproducibilidad del muestreo
OUTPUT_MUESTRA     = "muestra_validacion.json"     # archivo de salida con la muestra
OUTPUT_REPORTE     = "reporte_validacion.txt"      # reporte estadístico

random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATASET
# ─────────────────────────────────────────────────────────────────────────────
print(f"📂 Cargando dataset: {DATASET_FILE}")
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"✅ Total de pares cargados: {len(dataset)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ANÁLISIS ESTADÍSTICO BÁSICO
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Análisis estadístico del dataset:")

# Distribución por fuente
por_fuente = Counter(item["fuente"] for item in dataset)
print(f"\n📚 Distribución por fuente ({len(por_fuente)} archivos):")
for fuente, n in sorted(por_fuente.items(), key=lambda x: x[1], reverse=True):
    porcentaje = (n / len(dataset)) * 100
    print(f"   - {fuente}: {n} pares ({porcentaje:.1f}%)")

# Longitud de contextos
longitudes_contexto = [len(item["contexto"]) for item in dataset]
print(f"\n📏 Longitud de contextos:")
print(f"   - Mínima: {min(longitudes_contexto)} caracteres")
print(f"   - Máxima: {max(longitudes_contexto)} caracteres")
print(f"   - Promedio: {statistics.mean(longitudes_contexto):.1f} caracteres")
print(f"   - Mediana: {statistics.median(longitudes_contexto):.1f} caracteres")

# Longitud de preguntas
longitudes_pregunta = [len(item["pregunta"]) for item in dataset]
print(f"\n❓ Longitud de preguntas:")
print(f"   - Mínima: {min(longitudes_pregunta)} caracteres")
print(f"   - Máxima: {max(longitudes_pregunta)} caracteres")
print(f"   - Promedio: {statistics.mean(longitudes_pregunta):.1f} caracteres")

# Longitud de respuestas
longitudes_respuesta = [len(item["respuesta"]) for item in dataset]
print(f"\n💬 Longitud de respuestas:")
print(f"   - Mínima: {min(longitudes_respuesta)} caracteres")
print(f"   - Máxima: {max(longitudes_respuesta)} caracteres")
print(f"   - Promedio: {statistics.mean(longitudes_respuesta):.1f} caracteres")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DETECCIÓN DE POSIBLES PROBLEMAS
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔍 Detección de posibles problemas:")

# Respuestas vacías o muy cortas
respuestas_cortas = [item for item in dataset if len(item["respuesta"]) < 10]
print(f"   ⚠️  Respuestas muy cortas (<10 chars): {len(respuestas_cortas)}")

# Preguntas vacías o muy cortas
preguntas_cortas = [item for item in dataset if len(item["pregunta"]) < 10]
print(f"   ⚠️  Preguntas muy cortas (<10 chars): {len(preguntas_cortas)}")

# Duplicados exactos (pregunta + respuesta)
pares_vistos = {}
duplicados = []
for item in dataset:
    key = (item["pregunta"], item["respuesta"])
    if key in pares_vistos:
        duplicados.append(item)
    else:
        pares_vistos[key] = item
print(f"   ⚠️  Pares duplicados exactos: {len(duplicados)}")

# Contextos duplicados
contextos_vistos = {}
contextos_duplicados = 0
for item in dataset:
    if item["contexto"] in contextos_vistos:
        contextos_duplicados += 1
    else:
        contextos_vistos[item["contexto"]] = True
print(f"   ℹ️  Contextos repetidos (esperado por ventana deslizante): {contextos_duplicados}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MUESTREO PARA VALIDACIÓN MANUAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🎲 Generando muestra aleatoria de {TAMANIO_MUESTRA} pares para validación manual...")

# Muestreo estratificado por fuente (si es posible)
if len(dataset) <= TAMANIO_MUESTRA:
    muestra = dataset.copy()
else:
    # Muestreo estratificado proporcional
    muestra = []
    for fuente, n_total in por_fuente.items():
        n_muestra_fuente = max(1, int((n_total / len(dataset)) * TAMANIO_MUESTRA))
        items_fuente = [item for item in dataset if item["fuente"] == fuente]
        muestra_fuente = random.sample(items_fuente, min(n_muestra_fuente, len(items_fuente)))
        muestra.extend(muestra_fuente)
    
    # Si quedó más corta, completar aleatoriamente
    if len(muestra) < TAMANIO_MUESTRA:
        restantes = [item for item in dataset if item not in muestra]
        muestra.extend(random.sample(restantes, min(TAMANIO_MUESTRA - len(muestra), len(restantes))))
    
    # Si quedó más larga, recortar
    muestra = muestra[:TAMANIO_MUESTRA]

print(f"✅ Muestra generada: {len(muestra)} pares")

# Guardar muestra
with open(OUTPUT_MUESTRA, 'w', encoding='utf-8') as f:
    json.dump(muestra, f, ensure_ascii=False, indent=2)
print(f"💾 Muestra guardada en: {OUTPUT_MUESTRA}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERAR REPORTE
# ─────────────────────────────────────────────────────────────────────────────
reporte_lines = []
reporte_lines.append("=" * 80)
reporte_lines.append("REPORTE DE VALIDACIÓN DE DATASET")
reporte_lines.append("=" * 80)
reporte_lines.append(f"\nArchivo: {DATASET_FILE}")
reporte_lines.append(f"Fecha de análisis: {Path(DATASET_FILE).stat().st_mtime}")
reporte_lines.append(f"\n{'=' * 80}")
reporte_lines.append("ESTADÍSTICAS GENERALES")
reporte_lines.append("=" * 80)
reporte_lines.append(f"Total de pares: {len(dataset)}")
reporte_lines.append(f"Total de fuentes: {len(por_fuente)}")
reporte_lines.append(f"\nDistribución por fuente:")
for fuente, n in sorted(por_fuente.items(), key=lambda x: x[1], reverse=True):
    porcentaje = (n / len(dataset)) * 100
    reporte_lines.append(f"  - {fuente}: {n} pares ({porcentaje:.1f}%)")

reporte_lines.append(f"\n{'=' * 80}")
reporte_lines.append("MÉTRICAS DE LONGITUD")
reporte_lines.append("=" * 80)
reporte_lines.append(f"\nContextos:")
reporte_lines.append(f"  Mínimo: {min(longitudes_contexto)} chars")
reporte_lines.append(f"  Máximo: {max(longitudes_contexto)} chars")
reporte_lines.append(f"  Promedio: {statistics.mean(longitudes_contexto):.1f} chars")
reporte_lines.append(f"  Mediana: {statistics.median(longitudes_contexto):.1f} chars")

reporte_lines.append(f"\nPreguntas:")
reporte_lines.append(f"  Mínimo: {min(longitudes_pregunta)} chars")
reporte_lines.append(f"  Máximo: {max(longitudes_pregunta)} chars")
reporte_lines.append(f"  Promedio: {statistics.mean(longitudes_pregunta):.1f} chars")

reporte_lines.append(f"\nRespuestas:")
reporte_lines.append(f"  Mínimo: {min(longitudes_respuesta)} chars")
reporte_lines.append(f"  Máximo: {max(longitudes_respuesta)} chars")
reporte_lines.append(f"  Promedio: {statistics.mean(longitudes_respuesta):.1f} chars")

reporte_lines.append(f"\n{'=' * 80}")
reporte_lines.append("PROBLEMAS DETECTADOS")
reporte_lines.append("=" * 80)
reporte_lines.append(f"Respuestas muy cortas (<10 chars): {len(respuestas_cortas)}")
reporte_lines.append(f"Preguntas muy cortas (<10 chars): {len(preguntas_cortas)}")
reporte_lines.append(f"Pares duplicados exactos: {len(duplicados)}")
reporte_lines.append(f"Contextos repetidos: {contextos_duplicados}")

reporte_lines.append(f"\n{'=' * 80}")
reporte_lines.append("VALIDACIÓN MANUAL")
reporte_lines.append("=" * 80)
reporte_lines.append(f"\nMuestra para validación: {len(muestra)} pares")
reporte_lines.append(f"Archivo de muestra: {OUTPUT_MUESTRA}")
reporte_lines.append(f"\nCriterios de validación recomendados:")
reporte_lines.append(f"  1. La respuesta está contenida literalmente en el contexto")
reporte_lines.append(f"  2. La pregunta es relevante y responde al contexto")
reporte_lines.append(f"  3. La pregunta es clara y bien formulada")
reporte_lines.append(f"  4. La respuesta es completa y precisa")
reporte_lines.append(f"\nResultados de validación manual:")
reporte_lines.append(f"  [COMPLETAR DESPUÉS DE REVISAR LA MUESTRA]")
reporte_lines.append(f"  - Pares válidos: ___ / {len(muestra)}")
reporte_lines.append(f"  - Tasa de validez: ___%")
reporte_lines.append(f"  - Observaciones: ___________________________________")

reporte_text = '\n'.join(reporte_lines)

with open(OUTPUT_REPORTE, 'w', encoding='utf-8') as f:
    f.write(reporte_text)

print(f"\n💾 Reporte guardado en: {OUTPUT_REPORTE}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MOSTRAR EJEMPLOS DE LA MUESTRA
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n📋 Primeros 3 ejemplos de la muestra:")
for i, item in enumerate(muestra[:3], 1):
    print(f"\n--- Ejemplo {i} ---")
    print(f"Fuente   : {item['fuente']}")
    print(f"Contexto : {item['contexto'][:150]}...")
    print(f"Pregunta : {item['pregunta']}")
    print(f"Respuesta: {item['respuesta']}")

print(f"\n{'=' * 80}")
print("✅ Validación inicial completada")
print(f"📄 Revisa {OUTPUT_MUESTRA} para validación manual")
print(f"📊 Revisa {OUTPUT_REPORTE} para el reporte completo")
print(f"{'=' * 80}")