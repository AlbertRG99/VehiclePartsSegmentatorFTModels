"""
Evaluacion comun de todos los modelos + informe PDF.

Lee todas las predicciones de resultados/predicciones/*.json (formato comun, ver
comun.py) y las compara con las mascaras reales del test (VehicleSeg10K/valid).

Metricas (las mismas para todos los modelos):
  1. Segmentacion semantica por pixel :
       IoU por clase, mIoU (13 clases, sin fondo), Acc por clase y mAcc, Dice,
       exactitud de pixel. Se construye un mapa de etiquetas por imagen:
       primero la carroceria (Foreground) y encima las piezas, por orden de
       confianza. El real es el PNG oficial del dataset.
  2. Deteccion de piezas por imagen: una pieza cuenta como acertada si el modelo
       la predice y su mascara (union de instancias) tiene IoU >= 0.5 con la real.
       Precision, recall y F1 por clase.
  3. mAP de mascaras estilo COCO (pycocotools, iou_type='segm') contra una
       mascara por clase e imagen: mAP50, mAP75, mAP50-95.
  4. Confusion izquierda/derecha y delante/detras.
  5. Velocidad (ms por fotograma, FPS), GPU usada, parametros y memoria de GPU maxima,
       si el predictor lo guardo en el .meta.json.

SAV no entra (EXCLUIR) ni las pruebas ("prueba" en el nombre), salvo que se pidan con --modelos.

Uso:
  python 04_evaluacion/evaluar.py                       # todos los modelos
  python 04_evaluacion/evaluar.py --modelos yolo_x rfdetr_2xl sam3_ft
  python 04_evaluacion/evaluar.py --conf 0.3 --limite 300   # prueba rapida
Salida: resultados/informe/informe_segmentacion_piezas.pdf (espanol),
        car_part_segmentation_report.pdf (ingles) + metricas.json + CSVs
"""

import argparse
import contextlib
import csv
import io
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import DIR_ANN_VALID, DIR_PRED, GT_POR_CLASE, GT_VALID, RAIZ  # noqa: E402

CLASES = ["Foreground", "Plate", "Wheel", "Front window", "Back window",
          "Left front window", "Left front door", "Left back window", "Left back door",
          "Right front window", "Right front door", "Right back window", "Right back door"]
ES = ["Carrocería", "Matrícula", "Rueda", "Parabrisas", "Luneta",
      "Ventanilla del. izq.", "Puerta del. izq.", "Ventanilla tras. izq.", "Puerta tras. izq.",
      "Ventanilla del. der.", "Puerta del. der.", "Ventanilla tras. der.", "Puerta tras. der."]
NC = len(CLASES)
EXCLUIR = {"sav"}  # SAV no se pudo reproducir (el entrenamiento falla); no va al informe


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------

def cargar_predicciones(nombre):
    preds = json.loads((DIR_PRED / f"{nombre}.json").read_text())
    meta_f = DIR_PRED / f"{nombre}.meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    por_img = {}
    for p in preds:
        por_img.setdefault(p["image_id"], []).append(p)
    return por_img, meta


def mapa_real(im):
    sem = np.array(Image.open(DIR_ANN_VALID / (Path(im["file_name"]).stem + ".png")))
    if sem.shape != (im["height"], im["width"]):
        sem = np.array(Image.fromarray(sem).resize((im["width"], im["height"]), Image.NEAREST))
    sem[sem > NC] = 0
    return sem.astype(np.int64)


def mascaras_por_clase(preds, conf, h, w):
    """Union de las mascaras con score >= conf por clase -> {cid: (mask, score_max)}."""
    out = {}
    for p in preds:
        if p["score"] < conf:
            continue
        m = mask_utils.decode(p["segmentation"]).astype(bool)
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
        c = p["category_id"]
        if c in out:
            out[c] = (out[c][0] | m, max(out[c][1], p["score"]))
        else:
            out[c] = (m, p["score"])
    return out


def mapa_pred(por_clase, h, w):
    sem = np.zeros((h, w), np.int64)
    if 1 in por_clase:
        sem[por_clase[1][0]] = 1
    for c, (m, s) in sorted(((c, v) for c, v in por_clase.items() if c != 1), key=lambda x: x[1][1]):
        sem[m] = c
    return sem


# ----------------------------------------------------------------------------
# Evaluacion de un modelo
# ----------------------------------------------------------------------------

def evaluar_modelo(nombre, imagenes, conf, umbral_iou):
    por_img, meta = cargar_predicciones(nombre)
    confusion = np.zeros((NC + 1, NC + 1), np.int64)  # filas = real, columnas = pred
    det = np.zeros((NC, 3), np.int64)                  # tp, fp, fn por clase
    iou_img = []
    coco_res = []
    for im in tqdm(imagenes, desc=nombre, leave=False):
        h, w = im["height"], im["width"]
        real = mapa_real(im)
        pc = mascaras_por_clase(por_img.get(im["id"], []), conf, h, w)
        pred = mapa_pred(pc, h, w)
        confusion += np.bincount(real.ravel() * (NC + 1) + pred.ravel(),
                                 minlength=(NC + 1) ** 2).reshape(NC + 1, NC + 1)
        ious = []
        for c in range(1, NC + 1):
            g = real == c
            if c in pc:
                m = pc[c][0]
                inter, union = (g & m).sum(), (g | m).sum()
                iou = inter / union if union else 0.0
                if g.any():
                    ious.append(iou)
                    if iou >= umbral_iou:
                        det[c - 1, 0] += 1
                    else:
                        det[c - 1, 1] += 1
                        det[c - 1, 2] += 1
                else:
                    det[c - 1, 1] += 1
                coco_res.append({"image_id": im["id"], "category_id": c, "score": float(pc[c][1]),
                                 "segmentation": _rle(m)})
            elif g.any():
                det[c - 1, 2] += 1
                ious.append(0.0)
        if ious:
            iou_img.append(float(np.mean(ious)))

    tp = np.diag(confusion)[1:].astype(float)
    fp = confusion[:, 1:].sum(0) - np.diag(confusion)[1:]
    fn = confusion[1:, :].sum(1) - np.diag(confusion)[1:]
    iou = np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), np.nan)
    acc = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
    dice = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    p = det[:, 0] / np.maximum(det[:, 0] + det[:, 1], 1)
    r = det[:, 0] / np.maximum(det[:, 0] + det[:, 2], 1)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0)
    mapas = _coco_map(coco_res, [im["id"] for im in imagenes])
    return {
        "nombre": nombre, "meta": meta,
        "mIoU": float(np.nanmean(iou) * 100), "mAcc": float(np.nanmean(acc) * 100),
        "mDice": float(np.nanmean(dice) * 100),
        "pixel_acc": float(np.diag(confusion).sum() / confusion.sum() * 100),
        "iou_clase": (iou * 100).tolist(), "acc_clase": (acc * 100).tolist(), "dice_clase": (dice * 100).tolist(),
        "precision_clase": (p * 100).tolist(), "recall_clase": (r * 100).tolist(), "f1_clase": (f1 * 100).tolist(),
        "precision": float(det[:, 0].sum() / max(det[:, 0].sum() + det[:, 1].sum(), 1) * 100),
        "recall": float(det[:, 0].sum() / max(det[:, 0].sum() + det[:, 2].sum(), 1) * 100),
        "F1_macro": float(f1.mean() * 100),
        **mapas,
        "iou_por_imagen": iou_img,
        "confusion": confusion.tolist(),
        "confusion_lado": _confusion_lado(confusion),
        "ms_imagen": meta.get("tiempo_ms_media"),
    }


def _rle(m):
    r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def _coco_map(res, ids):
    if not res:
        return {"mAP50": 0.0, "mAP75": 0.0, "mAP50_95": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(GT_POR_CLASE))
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "segm")
        ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"mAP50_95": ev.stats[0] * 100, "mAP50": ev.stats[1] * 100, "mAP75": ev.stats[2] * 100}


def _confusion_lado(conf):
    """% de pixeles de cada pieza con lado que el modelo asigna a la pieza del lado contrario."""
    pares = [("Left front window", "Right front window"), ("Left front door", "Right front door"),
             ("Left back window", "Right back window"), ("Left back door", "Right back door")]
    out = {}
    for a, b in pares:
        ia, ib = CLASES.index(a) + 1, CLASES.index(b) + 1
        tot = conf[ia].sum() + conf[ib].sum()
        out[f"{a} <-> {b}"] = float((conf[ia, ib] + conf[ib, ia]) / max(tot, 1) * 100)
    fb = [("Front window", "Back window")]
    for a, b in fb:
        ia, ib = CLASES.index(a) + 1, CLASES.index(b) + 1
        tot = conf[ia].sum() + conf[ib].sum()
        out[f"{a} <-> {b}"] = float((conf[ia, ib] + conf[ib, ia]) / max(tot, 1) * 100)
    return out


# ----------------------------------------------------------------------------
# Informe PDF
# ----------------------------------------------------------------------------

# Textos del informe en los dos idiomas. Se generan dos PDFs con el mismo contenido.
NOMBRES = {  # nombre en predicciones -> (es, en)
    "yolo_x": ("YOLO26x-seg (entrenado)", "YOLO26x-seg (trained)"),
    "yolo_m": ("YOLO26m-seg (entrenado)", "YOLO26m-seg (trained)"),
    "yolo_mitbersh": ("YOLO26m mitbersh (afinado)", "YOLO26m mitbersh (fine-tuned)"),
    "yolo_mitbersh_sin_reentrenar": ("YOLO mitbersh (sin reentrenar)", "YOLO mitbersh (not retrained)"),
    "rfdetr_2xl": ("RF-DETR Seg 2XL",) * 2, "rfdetr_xl": ("RF-DETR Seg XL",) * 2,
    "rfdetr_l": ("RF-DETR Seg L",) * 2, "rfdetr_m": ("RF-DETR Seg M",) * 2,
    "dinov3_vitl16": ("DINOv3 ViT-L + decodificador", "DINOv3 ViT-L + decoder"),
    "sam3_ft": ("SAM 3 (afinado)", "SAM 3 (fine-tuned)"),
    "sam3_zeroshot": ("SAM 3 (zero-shot)",) * 2,
}
# GPUs con las que se entreno cada modelo (no queda guardado en las predicciones)
ENTRENAMIENTO = {
    "yolo_x": "4× RTX 2080 Ti", "yolo_m": "4× RTX 2080 Ti", "rfdetr_2xl": "4× RTX 2080 Ti",
    "dinov3_vitl16": "4× GTX 1080 Ti", "sam3_ft": "4× RTX 2080 Ti", "sam3_zeroshot": None,
}
# GPU de inferencia para predicciones antiguas que no la guardaron en el .meta.json
GPU_INFERENCIA = {"yolo_x": "NVIDIA GeForce RTX 2080 Ti", "rfdetr_2xl": "NVIDIA GeForce RTX 2080 Ti",
                  "dinov3_vitl16": "NVIDIA GeForce GTX 1080 Ti", "sam3_zeroshot": "NVIDIA GeForce RTX 2080 Ti"}
# Orden de los modelos en tablas y graficas (los que no esten aqui van al final, por nombre)
ORDEN = ["yolo_m", "yolo_x", "rfdetr_m", "rfdetr_l", "rfdetr_xl", "rfdetr_2xl", "dinov3_vitl16",
         "sam3_zeroshot", "sam3_ft"]

TXT = {
 "es": {
  "titulo": "Comparativa de modelos de segmentación de piezas de coche",
  "subtitulo": "Dataset VehicleSeg10K · test = split 'valid' ({n} imágenes) · {fecha}",
  "mejor": "Mejor modelo por mIoU: {m} ({v:.1f} %).",
  "umbral": "Umbral de confianza para mapas y detección: {conf}. Pieza acertada si IoU ≥ {iou}.",
  "cols_resumen": ["Modelo", "mIoU", "mAcc", "mDice", "mAP50", "mAP50-95", "F1 piezas", "ms/fotog."],
  "barras": "mIoU (barra sólida) y mAP50 (barra clara)",
  "hw_titulo": "Hardware, tiempo por fotograma y memoria",
  "hw_texto": [
    "Servidores. Dos nodos con 4 GPUs cada uno, que comparten el mismo disco: uno con 4× NVIDIA GeForce "
    "RTX 2080 Ti (11 GB, arquitectura Turing) y otro con 4× NVIDIA GeForce GTX 1080 Ti (11 GB, Pascal). "
    "Ninguna de las dos tiene bfloat16 por hardware, así que la media precisión es float16.",
    "Tiempo por fotograma. Se mide con una sola GPU y lote 1, desde la imagen ya decodificada en memoria "
    "hasta tener las máscaras: incluye preprocesado, red y postprocesado, pero no la lectura del disco. Se "
    "descarta la primera imagen (calentamiento). Las fotos del test son grandes (hasta ~12 Mpx) y las "
    "máscaras se devuelven a la resolución original, así que el postprocesado pesa. SAM 3 hace una consulta "
    "de texto por pieza (13 por foto). Si la inferencia se reparte en varias GPUs, el tiempo es el de cada "
    "GPU, no el total dividido.",
    "Memoria. 'VRAM pico' es la memoria de GPU máxima que reservó PyTorch durante toda la inferencia "
    "(modelo + activaciones); es lo que necesita como mínimo la GPU. 'Tamaño fp32' es el número de "
    "parámetros por 4 bytes. '-' indica que esas predicciones se generaron antes de medirlo.",
    "Ojo: los modelos que se ejecutaron en la GTX 1080 Ti no se pueden comparar en velocidad de tú a tú con "
    "los de la RTX 2080 Ti: la 1080 Ti es más lenta y, al no tener núcleos tensor, en float16 va mucho más despacio.",
  ],
  "cols_hw": ["Modelo", "Entrenado en", "Inferencia en", "ms/fotog.\nmedia", "ms/fotog.\nmediana",
              "FPS", "Parámetros\n(M)", "Tamaño\nfp32 (MB)", "VRAM pico\n(GB)"],
  "iou_titulo": "IoU por pieza (segmentación por píxel)",
  "tablas": {"iou_clase": "IoU por pieza (%)", "f1_clase": "F1 de detección por pieza (%)",
             "recall_clase": "Recall de detección por pieza (%)"},
  "pieza": "Pieza", "media": "MEDIA",
  "caja_y": "IoU medio de la imagen (%)",
  "caja_t": "Distribución por imagen (mitad central de las imágenes en la caja)",
  "vel_y": "ms por fotograma", "vel_t": "Tiempo de inferencia por fotograma (GPU indicada en cada barra)",
  "lado_t": "Confusión izquierda/derecha y delante/detrás",
  "lado_s": "% de píxeles de cada par de piezas que el modelo asigna a la pieza contraria. Más bajo es mejor.",
  "par": "Par", "matriz": "Matriz de confusión por píxel (% por fila) · {m}",
  "imagen": "Imagen", "real": "Real", "ejemplos": "Ejemplos (imágenes aleatorias del test)",
  "metodo_t": "Metodología y advertencias",
  "metodo": [
    "Datos. VehicleSeg10K: 8.596 imágenes de entrenamiento y 3.069 de 'valid', con 13 "
    "clases: carrocería, matrícula, rueda, parabrisas, luneta y las cuatro puertas y cuatro ventanillas con "
    "su lado. 'Izquierda' es la del coche (punto de vista del conductor), no la de la foto. Todos los modelos "
    "se entrenan con 'train' y se evalúan con 'valid', que no se usa para elegir checkpoint (se evalúa el último).",
    "Métricas por píxel. Por cada imagen se hace un mapa de etiquetas: primero la carrocería y encima las "
    "piezas, la más confiada al final. Se compara con el PNG oficial. mIoU y mAcc se calculan sobre las 13 "
    "clases sin el fondo, acumulando píxeles de todo el test.",
    "mAP. Con una máscara por clase e imagen (la unión de las instancias del modelo con confianza ≥ {conf}) "
    "contra una máscara real por clase e imagen, con pycocotools.",
    "YOLO. Ultralytics YOLO26-seg desde pesos COCO, sin volteo horizontal (un volteo cambia izquierda por "
    "derecha y estropearía las etiquetas).",
    "RF-DETR. RF-DETR Seg 2XL de Roboflow desde sus pesos preentrenados, 50 épocas, también sin volteo. Se "
    "evalúan los pesos EMA de la última época.",
    "DINOv3. Backbone ViT-L/16 de Meta congelado, con un decodificador tipo FPN entrenado encima (40 épocas). "
    "Es segmentación semántica pura: da un mapa por píxel, no instancias.",
    "SAM 3. Modelo de Meta guiado por texto, con una consulta por pieza (p. ej. 'left front door'). En "
    "zero-shot se usa tal cual, sin entrenar con este dataset; en la versión afinada se entrena con el "
    "entrenador oficial y el backbone congelado.",
  ],
 },
 "en": {
  "titulo": "Comparison of car-part segmentation models",
  "subtitulo": "VehicleSeg10K dataset · test = 'valid' split ({n} images) · {fecha}",
  "mejor": "Best model by mIoU: {m} ({v:.1f} %).",
  "umbral": "Confidence threshold for maps and detection: {conf}. A part counts as found if IoU ≥ {iou}.",
  "cols_resumen": ["Model", "mIoU", "mAcc", "mDice", "mAP50", "mAP50-95", "Part F1", "ms/frame"],
  "barras": "mIoU (solid bar) and mAP50 (light bar)",
  "hw_titulo": "Hardware, time per frame and memory",
  "hw_texto": [
    "Servers. Two nodes with 4 GPUs each, sharing the same storage: one with 4× NVIDIA GeForce RTX 2080 Ti "
    "(11 GB, Turing architecture) and one with 4× NVIDIA GeForce GTX 1080 Ti (11 GB, Pascal). Neither has "
    "hardware bfloat16, so half precision means float16.",
    "Time per frame. Measured on a single GPU with batch size 1, from the image already decoded in memory to "
    "the final masks: it includes pre-processing, the network and post-processing, but not reading from "
    "disk. The first image is discarded (warm-up). Test photos are large (up to ~12 Mpx) and masks are "
    "returned at full resolution, so post-processing matters. SAM 3 runs one text query per part (13 per "
    "photo). When inference is split across several GPUs, the time is per GPU, not the total divided.",
    "Memory. 'Peak VRAM' is the maximum GPU memory reserved by PyTorch during the whole inference run "
    "(model + activations); it is the minimum the GPU needs. 'fp32 size' is the parameter count times 4 "
    "bytes. '-' means those predictions were generated before this was measured.",
    "Note: speed of models run on the GTX 1080 Ti cannot be compared directly with those run on the "
    "RTX 2080 Ti: the 1080 Ti is slower and, having no tensor cores, runs float16 much more slowly.",
  ],
  "cols_hw": ["Model", "Trained on", "Inference on", "ms/frame\nmean", "ms/frame\nmedian",
              "FPS", "Parameters\n(M)", "fp32\nsize (MB)", "Peak\nVRAM (GB)"],
  "iou_titulo": "IoU per part (pixel segmentation)",
  "tablas": {"iou_clase": "IoU per part (%)", "f1_clase": "Detection F1 per part (%)",
             "recall_clase": "Detection recall per part (%)"},
  "pieza": "Part", "media": "MEAN",
  "caja_y": "Mean IoU of the image (%)",
  "caja_t": "Per-image distribution (the box holds the middle half of the images)",
  "vel_y": "ms per frame", "vel_t": "Inference time per frame (GPU shown on each bar)",
  "lado_t": "Left/right and front/back confusion",
  "lado_s": "% of pixels of each pair of parts that the model assigns to the opposite part. Lower is better.",
  "par": "Pair", "matriz": "Pixel confusion matrix (% per row) · {m}",
  "imagen": "Image", "real": "Ground truth", "ejemplos": "Examples (random test images)",
  "metodo_t": "Methodology and caveats",
  "metodo": [
    "Data. VehicleSeg10K: 8,596 training images and 3,069 'valid' images, with 13 "
    "classes: body, licence plate, wheel, windscreen, rear window and the four doors and four side windows "
    "with their side. 'Left' is the car's left (driver's point of view), not the left of the photo. All "
    "models are trained on 'train' and evaluated on 'valid', which is not used to pick a checkpoint (the "
    "last one is evaluated).",
    "Pixel metrics. For each image a label map is built: first the body and the parts on top, the most "
    "confident one last. It is compared with the official PNG. mIoU and mAcc are computed over the 13 "
    "classes without background, accumulating pixels over the whole test set.",
    "mAP. With one mask per class and image (the union of the model's instances with confidence ≥ {conf}) "
    "against one ground-truth mask per class and image, using pycocotools.",
    "YOLO. Ultralytics YOLO26-seg from COCO weights, without horizontal flipping (a flip swaps left and "
    "right and would corrupt the labels).",
    "RF-DETR. Roboflow RF-DETR Seg 2XL from its pre-trained weights, 50 epochs, also without flipping. The "
    "EMA weights of the last epoch are evaluated.",
    "DINOv3. Meta's ViT-L/16 backbone, frozen, with an FPN-style decoder trained on top (40 epochs). It is "
    "pure semantic segmentation: it outputs a per-pixel map, not instances.",
    "SAM 3. Meta's text-prompted model, with one query per part (e.g. 'left front door'). Zero-shot uses it "
    "as released, without training on this dataset; the fine-tuned version is trained with the official "
    "trainer and a frozen backbone.",
  ],
 },
}
CLASES_EN = ["Body", "Licence plate", "Wheel", "Windscreen", "Rear window",
             "Left front window", "Left front door", "Left rear window", "Left rear door",
             "Right front window", "Right front door", "Right rear window", "Right rear door"]


def nombre_modelo(n, lang):
    par = NOMBRES.get(n)
    return par[0 if lang == "es" else 1] if par else n


def gpu_corta(g):
    return (g or "-").replace("NVIDIA ", "").replace("GeForce ", "")


def fila_hardware(r, lang):
    m = r["meta"]
    ms, med = m.get("tiempo_ms_media"), m.get("tiempo_ms_mediana")
    params = m.get("parametros")
    vram = m.get("vram_pico_mb")
    gpu = m.get("gpu") or GPU_INFERENCIA.get(r["nombre"])
    entren = ENTRENAMIENTO.get(r["nombre"], "-")
    if entren is None:
        entren = "sin entrenar" if lang == "es" else "not trained"
    return [nombre_modelo(r["nombre"], lang), entren, gpu_corta(gpu),
            f"{ms:.0f}" if ms else "-", f"{med:.0f}" if med else "-",
            f"{1000 / ms:.1f}" if ms else "-",
            f"{params / 1e6:.0f}" if params else "-",
            f"{params * 4 / 2**20:,.0f}" if params else "-",
            f"{vram / 1024:.1f}" if vram else "-"]


def informe_pdf(resultados, imagenes, conf, umbral_iou, salida: Path, lang="es", n_ejemplos=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    T = TXT[lang]
    piezas = ES if lang == "es" else CLASES_EN
    nombres = [r["nombre"] for r in resultados]
    bonitos = [nombre_modelo(n, lang) for n in nombres]
    colores = plt.cm.tab10(np.arange(len(nombres)))
    A4 = (8.27, 11.69)
    mejor = max(resultados, key=lambda r: r["mIoU"])
    fecha = f"{date.today():%d/%m/%Y}" if lang == "es" else f"{date.today():%Y-%m-%d}"

    def parrafos(fig, bloques, y, ancho=95, fs=9.5, paso=0.019):
        for b in bloques:
            for linea in textwrap.wrap(b, ancho) or [""]:
                fig.text(0.07, y, linea, fontsize=fs, va="top")
                y -= paso
            y -= 0.012
        return y

    def tabla(ax, filas, cols, anchos=None, fs=8, alto=1.35):
        ax.axis("off")
        t = ax.table(cellText=filas, colLabels=cols, loc="upper center", cellLoc="center",
                     colWidths=anchos)
        t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, alto)
        for (i, j), c in t.get_celld().items():
            if i == 0:
                c.set_facecolor("#e8e8e8"); c.set_text_props(weight="bold")
                if any("\n" in str(x) for x in cols):
                    c.set_height(c.get_height() * 1.8)
        return t

    with PdfPages(salida) as pdf:
        # --- 1. Portada y resumen ---
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.95, T["titulo"], fontsize=15, weight="bold", va="top")
        fig.text(0.07, 0.915, T["subtitulo"].format(n=len(imagenes), fecha=fecha), fontsize=9.5, va="top", color="#555")
        y = parrafos(fig, [T["mejor"].format(m=nombre_modelo(mejor["nombre"], lang), v=mejor["mIoU"]),
                           T["umbral"].format(conf=conf, iou=umbral_iou)], 0.88, ancho=100, paso=0.018)
        ax = fig.add_axes([0.05, 0.38, 0.9, y - 0.40])
        filas = [[b, f"{r['mIoU']:.1f}", f"{r['mAcc']:.1f}", f"{r['mDice']:.1f}", f"{r['mAP50']:.1f}",
                  f"{r['mAP50_95']:.1f}", f"{r['F1_macro']:.1f}",
                  "-" if not r["ms_imagen"] else f"{r['ms_imagen']:.0f}"] for b, r in zip(bonitos, resultados)]
        tabla(ax, filas, T["cols_resumen"], [0.3, 0.09, 0.09, 0.09, 0.09, 0.11, 0.1, 0.09])
        ax2 = fig.add_axes([0.12, 0.06, 0.8, 0.28])
        x = np.arange(len(nombres))
        ax2.bar(x - 0.2, [r["mIoU"] for r in resultados], 0.4, label="mIoU", color=colores)
        ax2.bar(x + 0.2, [r["mAP50"] for r in resultados], 0.4, label="mAP50", color=colores, alpha=0.45)
        ax2.set_xticks(x, bonitos, rotation=20, ha="right", fontsize=7.5)
        ax2.set_ylabel("%"); ax2.set_ylim(0, 100); ax2.legend(fontsize=8)
        ax2.set_title(T["barras"], fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # --- 2. Hardware, tiempo por fotograma y memoria ---
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.95, T["hw_titulo"], fontsize=13, weight="bold", va="top")
        y = parrafos(fig, T["hw_texto"], 0.91, ancho=100, fs=9, paso=0.017)
        ax = fig.add_axes([0.02, y - 0.30, 0.96, 0.28])
        tabla(ax, [fila_hardware(r, lang) for r in resultados], T["cols_hw"],
              [0.21, 0.12, 0.12, 0.08, 0.08, 0.06, 0.1, 0.1, 0.09], fs=7)
        ax2 = fig.add_axes([0.12, 0.06, 0.8, max(0.15, y - 0.42)])
        ms = [r["ms_imagen"] or 0 for r in resultados]
        barras = ax2.bar(range(len(nombres)), ms, color=colores)
        for b, r in zip(barras, resultados):
            g = gpu_corta(r["meta"].get("gpu") or GPU_INFERENCIA.get(r["nombre"]))
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.0f} ms\n{g}",
                     ha="center", va="bottom", fontsize=6.5)
        ax2.set_xticks(range(len(nombres)), bonitos, rotation=20, ha="right", fontsize=7.5)
        ax2.set_ylabel(T["vel_y"]); ax2.set_title(T["vel_t"], fontsize=10)
        ax2.set_ylim(0, max(ms + [1]) * 1.25)
        pdf.savefig(fig); plt.close(fig)

        # --- 3. IoU por clase ---
        fig, ax = plt.subplots(figsize=(A4[0], 6.5))
        x = np.arange(NC); wb = 0.8 / len(nombres)
        for k, r in enumerate(resultados):
            ax.barh(x + k * wb, r["iou_clase"], wb, label=bonitos[k], color=colores[k])
        ax.set_yticks(x + 0.4 - wb / 2, piezas, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("IoU (%)"); ax.set_xlim(0, 100); ax.legend(fontsize=7, loc="lower right")
        ax.set_title(T["iou_titulo"], fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # --- 4. Tablas por clase ---
        for metrica, titulo in T["tablas"].items():
            fig = plt.figure(figsize=A4)
            fig.text(0.07, 0.95, titulo, fontsize=13, weight="bold", va="top")
            ax = fig.add_axes([0.03, 0.05, 0.94, 0.86])
            filas = [[piezas[c]] + [f"{r[metrica][c]:.1f}" for r in resultados] for c in range(NC)]
            filas.append([T["media"]] + [f"{np.nanmean(r[metrica]):.1f}" for r in resultados])
            cols = [T["pieza"]] + [textwrap.fill(b, 14) for b in bonitos]
            tabla(ax, filas, cols, [0.26] + [0.74 / len(nombres)] * len(nombres), fs=7.5)
            pdf.savefig(fig); plt.close(fig)

        # --- 5. Distribucion de IoU por imagen ---
        con_dist = [(b, r) for b, r in zip(bonitos, resultados) if r.get("iou_por_imagen")]
        if con_dist:
            fig, a1 = plt.subplots(figsize=(A4[0], A4[1] / 2))
            a1.boxplot([np.array(r["iou_por_imagen"]) * 100 for _, r in con_dist], showfliers=False)
            a1.set_xticks(range(1, len(con_dist) + 1), [b for b, _ in con_dist], rotation=20, ha="right", fontsize=7.5)
            a1.set_ylabel(T["caja_y"]); a1.set_title(T["caja_t"], fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # --- 6. Confusiones de lado ---
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.95, T["lado_t"], fontsize=13, weight="bold", va="top")
        fig.text(0.07, 0.915, T["lado_s"], fontsize=9, va="top")
        ax = fig.add_axes([0.03, 0.55, 0.94, 0.33])
        pares = list(resultados[0]["confusion_lado"].keys())

        def nombre_par(p):
            a, b = p.split(" <-> ")
            if lang == "es":
                return f"{ES[CLASES.index(a)]} ↔ {ES[CLASES.index(b)]}"
            return f"{CLASES_EN[CLASES.index(a)]} ↔ {CLASES_EN[CLASES.index(b)]}"
        filas = [[nombre_par(p)] + [f"{r['confusion_lado'][p]:.2f}" for r in resultados] for p in pares]
        tabla(ax, filas, [T["par"]] + [textwrap.fill(b, 14) for b in bonitos],
              [0.34] + [0.66 / len(nombres)] * len(nombres), fs=7.5)
        cm = np.array(mejor["confusion"], float)[1:, 1:]
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        ax = fig.add_axes([0.25, 0.05, 0.62, 0.42])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(NC), piezas, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(NC), piezas, fontsize=6.5)
        ax.set_title(T["matriz"].format(m=nombre_modelo(mejor["nombre"], lang)), fontsize=8.5)
        for i in range(NC):
            for j in range(NC):
                if cm[i, j] >= 1:
                    ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=5,
                            color="white" if cm[i, j] > 50 else "black")
        pdf.savefig(fig); plt.close(fig)

        # --- 7. Ejemplos visuales ---
        rng = np.random.default_rng(0)
        idx = rng.choice(len(imagenes), size=min(n_ejemplos, len(imagenes)), replace=False)
        paleta = (plt.cm.tab20(np.linspace(0, 1, 20))[:NC, :3] * 255).astype(np.uint8)
        preds_cache = {r["nombre"]: cargar_predicciones(r["nombre"])[0] for r in resultados}
        fig, axes = plt.subplots(len(idx), len(nombres) + 2, figsize=(A4[1] * 0.95, A4[0] * 0.95))
        axes = np.atleast_2d(axes)
        for fila, i in enumerate(idx):
            im = imagenes[i]
            img = np.array(Image.open(im["ruta"]).convert("RGB"))
            h, w = img.shape[:2]
            real = mapa_real(im)
            vistas = [(T["imagen"], img), (T["real"], _pintar(img, real, paleta))]
            for r in resultados:
                pc = mascaras_por_clase(preds_cache[r["nombre"]].get(im["id"], []), conf, h, w)
                vistas.append((nombre_modelo(r["nombre"], lang), _pintar(img, mapa_pred(pc, h, w), paleta)))
            for col, (t, v) in enumerate(vistas):
                ax = axes[fila, col]; ax.imshow(v); ax.axis("off")
                if fila == 0:
                    ax.set_title(textwrap.fill(t, 18), fontsize=7)
        fig.suptitle(T["ejemplos"], fontsize=11)
        fig.tight_layout(); pdf.savefig(fig, orientation="landscape"); plt.close(fig)

        # --- 8. Metodologia y advertencias ---
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.95, T["metodo_t"], fontsize=16, weight="bold", va="top")
        parrafos(fig, [b.format(conf=conf) for b in T["metodo"]], 0.90)
        pdf.savefig(fig); plt.close(fig)
    print(f"Informe: {salida}")


def _pintar(img, sem, paleta, alpha=0.55):
    out = img.copy().astype(np.float32)
    m = sem > 0
    col = paleta[np.clip(sem - 1, 0, len(paleta) - 1)]
    out[m] = out[m] * (1 - alpha) + col[m] * alpha
    return out.astype(np.uint8)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="*", help="nombres en resultados/predicciones (por defecto todos)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--umbral-iou", type=float, default=0.5)
    ap.add_argument("--limite", type=int, default=None, help="evaluar solo N imagenes (prueba)")
    ap.add_argument("--salida", default=str(RAIZ / "resultados" / "informe"))
    ap.add_argument("--solo-informe", action="store_true",
                    help="rehace solo los PDFs a partir de metricas.json (sin volver a evaluar)")
    a = ap.parse_args()

    modelos = a.modelos  # con --modelos se respeta el orden dado
    if not modelos:
        modelos = sorted((p.stem for p in DIR_PRED.glob("*.json")
                          if not p.name.endswith(".meta.json") and p.stem not in EXCLUIR
                          and "prueba" not in p.stem),
                         key=lambda m: (ORDEN.index(m) if m in ORDEN else len(ORDEN), m))
    if not modelos:
        raise SystemExit(f"No hay predicciones en {DIR_PRED}")
    gt = json.loads(GT_VALID.read_text())
    imagenes = gt["images"][: a.limite] if a.limite else gt["images"]
    from comun import DIR_IMG_VALID
    for im in imagenes:
        im["ruta"] = str(DIR_IMG_VALID / im["file_name"])

    salida = Path(a.salida); salida.mkdir(parents=True, exist_ok=True)
    if a.solo_informe:
        guardados = {r["nombre"]: r for r in json.loads((salida / "metricas.json").read_text())}
        resultados = [guardados[m] for m in modelos if m in guardados]
        for r in resultados:  # GPU/parametros: los del .meta.json actual, si se ha rehecho la prediccion
            f = DIR_PRED / f"{r['nombre']}.meta.json"
            if f.exists():
                r["meta"] = {**r["meta"], **json.loads(f.read_text())}
        informe_pdf(resultados, imagenes, a.conf, a.umbral_iou, salida / "informe_segmentacion_piezas.pdf", "es")
        informe_pdf(resultados, imagenes, a.conf, a.umbral_iou, salida / "car_part_segmentation_report.pdf", "en")
        return
    resultados = [evaluar_modelo(m, imagenes, a.conf, a.umbral_iou) for m in modelos]
    (salida / "metricas.json").write_text(json.dumps(resultados, indent=1))  # con iou_por_imagen
    with open(salida / "resumen.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["modelo", "mIoU", "mAcc", "mDice", "pixel_acc", "mAP50", "mAP75", "mAP50_95",
                     "precision", "recall", "F1_macro", "ms_imagen", "ms_imagen_mediana", "fps",
                     "gpu_inferencia", "gpus_entrenamiento", "parametros", "vram_pico_mb"])
        for r in resultados:
            m = r["meta"]
            cw.writerow([r["nombre"]] + [round(r[k], 2) if r[k] is not None else "" for k in
                        ("mIoU", "mAcc", "mDice", "pixel_acc", "mAP50", "mAP75", "mAP50_95",
                         "precision", "recall", "F1_macro", "ms_imagen")] +
                        [m.get("tiempo_ms_mediana", ""), round(1000 / r["ms_imagen"], 2) if r["ms_imagen"] else "",
                         m.get("gpu") or GPU_INFERENCIA.get(r["nombre"], ""), ENTRENAMIENTO.get(r["nombre"]) or "",
                         m.get("parametros", ""), m.get("vram_pico_mb", "")])
    with open(salida / "por_clase.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["clase"] + [f"{r['nombre']}_{m}" for r in resultados for m in ("iou", "acc", "f1")])
        for c in range(NC):
            cw.writerow([CLASES[c]] + [round(r[m][c], 2) for r in resultados
                                       for m in ("iou_clase", "acc_clase", "f1_clase")])
    for r in resultados:
        print(f"{r['nombre']:<32} mIoU {r['mIoU']:6.2f}  mAcc {r['mAcc']:6.2f}  mAP50 {r['mAP50']:6.2f}  "
              f"F1 {r['F1_macro']:6.2f}")
    informe_pdf(resultados, imagenes, a.conf, a.umbral_iou, salida / "informe_segmentacion_piezas.pdf", "es")
    informe_pdf(resultados, imagenes, a.conf, a.umbral_iou, salida / "car_part_segmentation_report.pdf", "en")


if __name__ == "__main__":
    main()
