"""Junta las predicciones de SAM 3 hechas por trozos (una GPU por trozo).

Uso:  python 03_sam3/unir_partes.py sam3_zeroshot 4
Lee resultados/predicciones/partes/<nombre>.<i>.json (+ .meta.json) y escribe
resultados/predicciones/<nombre>.json y <nombre>.meta.json, como si fuera una sola pasada.
"""
import json
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "resultados" / "predicciones"
nombre, n = sys.argv[1], int(sys.argv[2])
preds, metas = [], []
for i in range(n):
    f = DIR / "partes" / f"{nombre}.{i}.json"
    if not f.exists():
        raise SystemExit(f"Falta {f}: esa parte no termino")
    preds += json.loads(f.read_text())
    metas.append(json.loads((DIR / "partes" / f"{nombre}.{i}.meta.json").read_text()))

n_img = sum(m["num_imagenes"] for m in metas)
meta = dict(metas[0])
meta.update({
    "nombre": nombre, "num_predicciones": len(preds), "num_imagenes": n_img,
    # media ponderada por imagenes; la mediana es aproximada (media de las medianas)
    "tiempo_ms_media": sum(m["tiempo_ms_media"] * m["num_imagenes"] for m in metas) / n_img,
    "tiempo_ms_mediana": sum(m["tiempo_ms_mediana"] for m in metas) / n,
    "partes": n,
})
for k in ("vram_pico_mb", "vram_pico_asignada_mb"):  # memoria de UNA GPU: la mayor de las partes
    if all(k in m for m in metas):
        meta[k] = max(m[k] for m in metas)
(DIR / f"{nombre}.json").write_text(json.dumps(preds))
(DIR / f"{nombre}.meta.json").write_text(json.dumps(meta, indent=2))
print(f"Unido {DIR / (nombre + '.json')}: {len(preds)} predicciones de {n_img} imagenes "
      f"({meta['tiempo_ms_media']:.1f} ms/img por GPU)")
