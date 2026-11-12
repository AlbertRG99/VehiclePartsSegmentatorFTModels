"""
Prepara VehicleSeg10K para los tres experimentos (YOLO, SAV, SAM 3) y para la
evaluacion comun.

Entrada (tal cual sale del zip de SAV):
    datos/VehicleSeg10K/images/{train,valid}/*.jpg
    datos/VehicleSeg10K/annotations/{train,valid}/<id>.json   (poligonos, formato labelme)
    datos/VehicleSeg10K/annotations/{train,valid}/<id>.png    (mascara semantica, valor = clase+1)

Salida en datos/preparado/:
    clases.json                       lista de las 13 clases (orden de SAV)
    coco/train.json, coco/valid.json  COCO instancias (un poligono = una instancia)
    coco/valid_por_clase.json         COCO con UNA mascara por clase e imagen (RLE),
                                      la que usa la evaluacion comun (igual que SAV)
    yolo/images/{train,valid}         enlaces simbolicos a las imagenes
    yolo/labels/{train,valid}/*.txt   poligonos YOLO normalizados
    yolo/data.yaml
    sam3/vehicleseg10k/{train,test}/  imagenes (enlaces) + _annotations.coco.json
                                      (layout que espera el entrenador de SAM 3)

Splits: se respeta el split oficial (train = 8.596, valid = 3.069). 'valid' se usa
como TEST en todos los modelos (como hace el paper de SAV). No se usa para elegir
checkpoint: se evalua siempre el ultimo.

Uso:
    python 00_dataset/preparar_dataset.py --raiz datos/VehicleSeg10K --salida datos/preparado
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

# Orden oficial de SAV (seg/datasets/mask_dataset.py). En el PNG, valor = indice + 1.
CLASES = [
    "Foreground", "Plate", "Wheel", "Front window", "Back window",
    "Left front window", "Left front door", "Left back window", "Left back door",
    "Right front window", "Right front door", "Right back window", "Right back door",
]
SPLITS = {"train": "train", "valid": "valid"}


def forma_a_poligono(s):
    """Convierte una 'shape' de labelme en lista de puntos [[x,y],...] o None."""
    pts = np.asarray(s.get("points", []), dtype=np.float64)
    tipo = s.get("shape_type", "polygon")
    if tipo in ("polygon", "line", "linestrip") and len(pts) >= 3:  # hay un "line" de 33 pts que es un contorno
        return pts
    if tipo == "rectangle" and len(pts) >= 4:  # X-AnyLabeling guarda las 4 esquinas
        return pts[:4]
    if tipo == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if tipo == "circle" and len(pts) == 2:
        (cx, cy), (px, py) = pts
        r = math.hypot(px - cx, py - cy)
        a = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    return None  # lineas, puntos o poligonos degenerados


def buscar_imagen(dir_img: Path, ident: str, image_path: str | None):
    if image_path:
        p = dir_img / Path(image_path).name
        if p.exists():
            return p
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = dir_img / f"{ident}{ext}"
        if p.exists():
            return p
    return None


def enlazar(origen: Path, destino: Path):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.is_symlink() or destino.exists():
        destino.unlink()
    destino.symlink_to(origen.resolve())


def procesar_split(raiz: Path, salida: Path, split: str, nombre_sam3: str):
    dir_img = raiz / "images" / split
    dir_ann = raiz / "annotations" / split
    jsons = sorted(dir_ann.glob("*.json"))
    coco = {"images": [], "annotations": [], "categories": [
        {"id": i + 1, "name": c, "supercategory": "car"} for i, c in enumerate(CLASES)]}
    coco_clase = {"images": [], "annotations": [], "categories": coco["categories"]}
    ann_id = ann_id_c = 1
    sin_imagen = descartadas = 0
    desconocidas = {}

    for img_id, fj in enumerate(tqdm(jsons, desc=split), start=1):
        d = json.loads(fj.read_text(encoding="utf-8"))
        ident = fj.stem
        ruta_img = buscar_imagen(dir_img, ident, d.get("imagePath"))
        if ruta_img is None:
            sin_imagen += 1
            continue
        w, h = d.get("imageWidth"), d.get("imageHeight")
        if not w or not h:
            with Image.open(ruta_img) as im:
                w, h = im.size
        info = {"id": img_id, "file_name": ruta_img.name, "width": w, "height": h}
        coco["images"].append(info)
        coco_clase["images"].append(info)

        # --- YOLO + COCO por poligono ---
        lineas = []
        for s in d["shapes"]:
            if s["label"] not in CLASES:
                desconocidas[s["label"]] = desconocidas.get(s["label"], 0) + 1
                continue
            poly = forma_a_poligono(s)
            if poly is None:
                descartadas += 1
                continue
            poly[:, 0] = poly[:, 0].clip(0, w)
            poly[:, 1] = poly[:, 1].clip(0, h)
            cls = CLASES.index(s["label"])
            flat = poly.ravel().tolist()
            rle = mask_utils.frPyObjects([flat], h, w)
            area = float(mask_utils.area(mask_utils.merge(rle)))
            if area < 1:
                descartadas += 1
                continue
            x, y = poly.min(0)
            x2, y2 = poly.max(0)
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls + 1,
                "segmentation": [flat], "area": area,
                "bbox": [float(x), float(y), float(x2 - x), float(y2 - y)], "iscrowd": 0})
            ann_id += 1
            norm = (poly / [w, h]).ravel()
            lineas.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
        fl = salida / "yolo" / "labels" / split / f"{ruta_img.stem}.txt"
        fl.parent.mkdir(parents=True, exist_ok=True)
        fl.write_text("\n".join(lineas))
        enlazar(ruta_img, salida / "yolo" / "images" / split / ruta_img.name)
        enlazar(ruta_img, salida / "sam3" / "vehicleseg10k" / nombre_sam3 / ruta_img.name)

        # --- COCO "por clase": una mascara por clase sacada del PNG semantico ---
        fpng = dir_ann / f"{ident}.png"
        if fpng.exists():
            sem = np.array(Image.open(fpng))
            if sem.shape != (h, w):
                sem = np.array(Image.open(fpng).resize((w, h), Image.NEAREST))
            for v in np.unique(sem):
                if v == 0 or v > len(CLASES):
                    continue
                m = np.asfortranarray((sem == v).astype(np.uint8))
                rle = mask_utils.encode(m)
                area = float(mask_utils.area(rle))
                bbox = [float(b) for b in mask_utils.toBbox(rle)]
                rle["counts"] = rle["counts"].decode("ascii")
                coco_clase["annotations"].append({
                    "id": ann_id_c, "image_id": img_id, "category_id": int(v),
                    "segmentation": rle, "area": area, "bbox": bbox, "iscrowd": 0})
                ann_id_c += 1

    (salida / "coco").mkdir(parents=True, exist_ok=True)
    (salida / "coco" / f"{split}.json").write_text(json.dumps(coco))
    if split == "valid":
        (salida / "coco" / "valid_por_clase.json").write_text(json.dumps(coco_clase))
    (salida / "sam3" / "vehicleseg10k" / nombre_sam3 / "_annotations.coco.json").write_text(json.dumps(coco))
    print(f"[{split}] imagenes={len(coco['images'])} instancias={len(coco['annotations'])} "
          f"sin_imagen={sin_imagen} formas_descartadas={descartadas} etiquetas_desconocidas={desconocidas}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default="datos/VehicleSeg10K")
    ap.add_argument("--salida", default="datos/preparado")
    a = ap.parse_args()
    raiz, salida = Path(a.raiz), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "clases.json").write_text(json.dumps(CLASES, indent=2))
    procesar_split(raiz, salida, "train", "train")
    procesar_split(raiz, salida, "valid", "test")
    yaml = (f"path: {salida.resolve() / 'yolo'}\ntrain: images/train\nval: images/valid\n"
            f"test: images/valid\n\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASES)))
    (salida / "yolo" / "data.yaml").write_text(yaml)
    print(f"Listo en {salida.resolve()}")


if __name__ == "__main__":
    main()
