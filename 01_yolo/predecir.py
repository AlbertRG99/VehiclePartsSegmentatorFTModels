"""Inferencia YOLO-seg sobre el test (valid) -> resultados/predicciones/<nombre>.json

  python 01_yolo/predecir.py --pesos resultados/yolo/x/weights/last.pt --nombre yolo_x
  python 01_yolo/predecir.py --mitbersh --nombre yolo_mitbersh_sin_reentrenar   (referencia)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "04_evaluacion"))
from comun import contar_parametros, Escritor, cargar_imagenes_test  # noqa: E402

# Clases del modelo de mitbersh -> clases de VehicleSeg10K (para la referencia sin reentrenar)
MAPEO_MITBERSH = {
    "license-plate": "Plate", "windshield": "Front window", "back-windshield": "Back window",
    "left_front-window": "Left front window", "left_front-door": "Left front door",
    "left_back-window": "Left back window", "left_back-door": "Left back door",
    "right_front-window": "Right front window", "right_front-door": "Right front door",
    "right_back-window": "Right back window", "right_back-door": "Right back door",
    "left_front-wheel": "Wheel", "left_back-wheel": "Wheel",
    "right_front-wheel": "Wheel", "right_back-wheel": "Wheel",
}
CLASES = ["Foreground", "Plate", "Wheel", "Front window", "Back window",
          "Left front window", "Left front door", "Left back window", "Left back door",
          "Right front window", "Right front door", "Right back window", "Right back door"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos")
    ap.add_argument("--mitbersh", action="store_true")
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()
    if a.mitbersh:
        from huggingface_hub import hf_hub_download
        a.pesos = hf_hub_download("mitbersh/car-parts-segmentation-m", "parts_segmentation.pt",
                                  local_dir=str(RAIZ / "pesos" / "mitbersh_m"))
        a.imgsz = 896
    modelo = YOLO(a.pesos)
    nombres = modelo.names
    esc = Escritor(a.nombre, f"YOLO-seg {a.pesos} imgsz={a.imgsz}")
    for im in tqdm(cargar_imagenes_test(a.limite)):
        img = cv2.imread(im["ruta"])
        h, w = img.shape[:2]
        with esc.cronometro():
            r = modelo.predict(img, imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            continue
        datos = r.masks.data.cpu().numpy() > 0.5
        for m, c, s in zip(datos, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            n = nombres[int(c)]
            if a.mitbersh:
                n = MAPEO_MITBERSH.get(n)
                if n is None:
                    continue
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            esc.anadir(im["id"], CLASES.index(n) + 1, m, float(s))
    esc.guardar({"pesos": a.pesos, "imgsz": a.imgsz, "parametros": contar_parametros(modelo)})


if __name__ == "__main__":
    main()
