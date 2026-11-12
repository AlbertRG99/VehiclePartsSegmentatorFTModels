"""Crea datos/preparado/rfdetr con el formato COCO que espera RF-DETR:

    train/_annotations.coco.json + imagenes (enlaces)
    valid/_annotations.coco.json + imagenes (enlaces)

Usa los COCO por poligono de datos/preparado/coco (los genera 00_dataset/preparar_dataset.py).

Uso:  python 06_rfdetr/preparar_datos.py
"""
import json
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PREPARADO = RAIZ / "datos" / "preparado"
SALIDA = PREPARADO / "rfdetr"


def preparar(split: str):
    origen = PREPARADO / "coco" / f"{split}.json"
    if not origen.exists():
        raise SystemExit(f"Falta {origen}. Lanza antes 00_dataset/preparar_dataset.py")
    destino = SALIDA / split
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy(origen, destino / "_annotations.coco.json")
    imagenes = json.loads(origen.read_text())["images"]
    dir_img = RAIZ / "datos" / "VehicleSeg10K" / "images" / split
    for im in imagenes:
        real = dir_img / im["file_name"]
        if not real.exists():
            raise SystemExit(f"No existe la imagen {real}")
        enlace = destino / im["file_name"]
        if enlace.is_symlink() or enlace.exists():
            enlace.unlink()
        enlace.symlink_to(real.resolve())
    print(f"[{split}] {len(imagenes)} imagenes en {destino}")


if __name__ == "__main__":
    preparar("train")
    preparar("valid")
