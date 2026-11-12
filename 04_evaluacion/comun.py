"""Utilidades compartidas por los predictores de los tres modelos.

Formato comun de predicciones: JSON tipo "resultados COCO"
    [{"image_id": int, "category_id": int (1..13), "score": float,
      "segmentation": {"size": [h, w], "counts": "..."}}, ...]
mas un fichero <nombre>.meta.json con el tiempo medio de inferencia, etc.

Solo depende de numpy, pycocotools y PIL para que funcione en los tres entornos
(yolo, sam3, dinov3, rfdetr).
"""

import json
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

RAIZ = Path(__file__).resolve().parent.parent
GT_VALID = RAIZ / "datos" / "preparado" / "coco" / "valid.json"
GT_POR_CLASE = RAIZ / "datos" / "preparado" / "coco" / "valid_por_clase.json"
DIR_IMG_VALID = RAIZ / "datos" / "VehicleSeg10K" / "images" / "valid"
DIR_ANN_VALID = RAIZ / "datos" / "VehicleSeg10K" / "annotations" / "valid"
DIR_PRED = RAIZ / "resultados" / "predicciones"
CLASES = json.loads((RAIZ / "datos" / "preparado" / "clases.json").read_text()) \
    if (RAIZ / "datos" / "preparado" / "clases.json").exists() else None


def cargar_imagenes_test(limite=None):
    """Lista de dicts {id, file_name, width, height, ruta} del split de test (valid)."""
    gt = json.loads(GT_VALID.read_text())
    imgs = gt["images"][:limite] if limite else gt["images"]
    for im in imgs:
        im["ruta"] = str(DIR_IMG_VALID / im["file_name"])
    return imgs


def mascara_a_rle(m: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def contar_parametros(obj, prof=0):
    """Numero de parametros de un modelo torch (o de un envoltorio con .model dentro). None si no se puede."""
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            return int(sum(p.numel() for p in obj.parameters()))
    except ImportError:
        return None
    if prof < 4 and getattr(obj, "model", None) is not None:
        return contar_parametros(obj.model, prof + 1)
    return None


def _hardware():
    """GPU usada y memoria de GPU maxima reservada por PyTorch durante la ejecucion."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu": "CPU"}
        return {"gpu": torch.cuda.get_device_name(0),
                "vram_pico_mb": round(torch.cuda.max_memory_reserved(0) / 2**20),
                "vram_pico_asignada_mb": round(torch.cuda.max_memory_allocated(0) / 2**20)}
    except Exception:
        return {}


class Escritor:
    """Acumula predicciones y tiempos y los guarda en resultados/predicciones/."""

    def __init__(self, nombre: str, descripcion: str = ""):
        self.nombre, self.descripcion = nombre, descripcion
        self.preds, self.tiempos = [], []
        DIR_PRED.mkdir(parents=True, exist_ok=True)

    def anadir(self, image_id: int, category_id: int, mascara: np.ndarray, score: float):
        if mascara.any():
            self.preds.append({"image_id": int(image_id), "category_id": int(category_id),
                               "score": round(float(score), 5), "segmentation": mascara_a_rle(mascara)})

    def cronometro(self):
        escritor = self

        class _C:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                escritor.tiempos.append((time.perf_counter() - self.t) * 1000)
        return _C()

    def guardar(self, extra: dict | None = None):
        ruta = DIR_PRED / f"{self.nombre}.json"
        ruta.write_text(json.dumps(self.preds))
        t = np.array(self.tiempos[1:] or self.tiempos or [0])  # se ignora la 1a (calentamiento)
        meta = {"nombre": self.nombre, "descripcion": self.descripcion,
                "num_predicciones": len(self.preds), "num_imagenes": len(self.tiempos),
                "tiempo_ms_media": float(t.mean()), "tiempo_ms_mediana": float(np.median(t))}
        meta.update(_hardware())
        meta.update(extra or {})
        (DIR_PRED / f"{self.nombre}.meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Guardado {ruta} ({len(self.preds)} predicciones, {meta['tiempo_ms_media']:.1f} ms/img)")
