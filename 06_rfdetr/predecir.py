"""Inferencia RF-DETR Seg sobre el test (valid) -> resultados/predicciones/<nombre>.json

  python 06_rfdetr/predecir.py --pesos resultados/rfdetr/2xl/last_ema.pth --nombre rfdetr_2xl

Se evalua el ultimo modelo, no el "best": valid es el test y no debe elegir el checkpoint.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETR
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "04_evaluacion"))
from comun import contar_parametros, CLASES, Escritor, cargar_imagenes_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()

    modelo = RFDETR.from_checkpoint(a.pesos, trust_checkpoint=True)  # archivo propio, entrenado aqui
    esc = Escritor(a.nombre, f"RF-DETR Seg {a.pesos}")
    desconocidas = set()
    for im in tqdm(cargar_imagenes_test(a.limite)):
        bgr = cv2.imread(im["ruta"])
        h, w = bgr.shape[:2]
        imagen = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        with esc.cronometro():
            # antialias=True: el entrenamiento sin aumentos propios usa el redimensionado de torchvision
            det = modelo.predict(imagen, threshold=a.conf, include_source_image=False, antialias=True)
        if det.mask is None or len(det) == 0:
            continue
        for i in np.argsort(-det.confidence)[:a.max_det]:
            nombre = str(det.data["class_name"][i])
            if nombre not in CLASES:
                desconocidas.add(nombre)
                continue
            m = det.mask[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            esc.anadir(im["id"], CLASES.index(nombre) + 1, m, float(det.confidence[i]))
    if desconocidas:
        print("AVISO: clases que no estan en el dataset:", sorted(desconocidas))
    esc.guardar({"pesos": a.pesos, "conf": a.conf, "parametros": contar_parametros(modelo)})


if __name__ == "__main__":
    main()
