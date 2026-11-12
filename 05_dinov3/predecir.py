"""Predicciones de DINOv3 + decodificador sobre `valid`, en el formato comun.

Cada pieza es UNA mascara por clase e imagen (segmentacion semantica: no separa las
ruedas entre si). El score es la probabilidad media dentro de la mascara.

Uso: python 05_dinov3/predecir.py --pesos resultados/dinov3/vitl16/last.pt --nombre dinov3_vitl16
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_evaluacion"))
from comun import contar_parametros, Escritor, cargar_imagenes_test  # noqa: E402
from comun_dino import NUM_CLASES, Backbone, Cabeza, preparar_imagen  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", required=True)
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--imgsz", type=int, default=None, help="por defecto, la del entrenamiento")
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()

    c = torch.load(a.pesos, map_location="cpu", weights_only=False)
    lado = a.imgsz or c["args"]["imgsz"]
    if lado % 16:
        raise SystemExit("--imgsz debe ser multiplo de 16")
    disp = torch.device("cuda")
    backbone = Backbone(c["modelo"]).to(disp)
    cabeza = Cabeza(c["dim"]).to(disp).eval()
    cabeza.load_state_dict(c["cabeza"])

    # float16 solo compensa con nucleos tensor (Volta/Turing o mas nueva, capacidad >= 7). En Pascal
    # (GTX 1080 Ti, 6.1) float16 va a 1/64 de la velocidad de float32: alli se usa float32.
    fp16 = torch.cuda.get_device_capability()[0] >= 7
    print("precision:", "fp16" if fp16 else "fp32")
    esc = Escritor(a.nombre, f"{c['modelo']} congelado + decodificador FPN, imgsz={lado}")
    for im in tqdm(cargar_imagenes_test(a.limite)):
        rgb = cv2.imread(im["ruta"])[..., ::-1].copy()
        h, w = rgb.shape[:2]
        x = preparar_imagen(rgb, lado).to(disp)
        with esc.cronometro():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                s = cabeza(backbone(x))
            prob = F.interpolate(s.float(), size=(h, w), mode="bilinear", align_corners=False).softmax(1)[0]
            conf, clase = prob.max(0)
            conf, clase = conf.cpu().numpy(), clase.cpu().numpy()
            torch.cuda.synchronize()
        for cat in range(1, NUM_CLASES):
            m = clase == cat
            if m.any():
                esc.anadir(im["id"], cat, m, float(conf[m].mean()))
    esc.guardar({"pesos": a.pesos, "imgsz": lado, "epoca": c["epoca"] + 1,
                 "precision": "fp16" if fp16 else "fp32",
                 "parametros": contar_parametros(backbone) + contar_parametros(cabeza)})


if __name__ == "__main__":
    main()
