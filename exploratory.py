import json

with open("dataset_acreditacion_5_tokens_mejorado.json", "r", encoding="utf-8") as f:
    datos = json.load(f)

print(len(datos))