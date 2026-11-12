"""Graficas PNG para el README a partir de resultados/informe/metricas.json.

Uso:  python 04_evaluacion/graficas_readme.py
Salida: resultados/informe/figuras/*.png (metricas, iou_por_pieza, precision_vs_velocidad,
        matriz_confusion, confusion_lados, distribucion) + tablas.md (tablas por pieza en Markdown)
        y ejemplos.png (pagina de ejemplos del informe en ingles, si pdftoppm esta instalado)
Textos en ingles (el README principal esta en ingles).
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
INF = RAIZ / "resultados" / "informe"
SAL = INF / "figuras"

NOMBRE = {"yolo_x": "YOLO26x-seg", "rfdetr_2xl": "RF-DETR Seg 2XL", "dinov3_vitl16": "DINOv3 ViT-L + decoder",
          "sam3_zeroshot": "SAM 3 (zero-shot)", "sam3_ft": "SAM 3 (fine-tuned)"}
ORDEN = ["yolo_x", "rfdetr_2xl", "dinov3_vitl16", "sam3_zeroshot", "sam3_ft"]
# Paleta categorica validada (daltonismo y vision normal); el color sigue al modelo, no a su posicion
COLOR = {"yolo_x": "#2a78d6", "rfdetr_2xl": "#eb6834", "dinov3_vitl16": "#1baf7a",
         "sam3_zeroshot": "#eda100", "sam3_ft": "#e87ba4"}
GPU = {"yolo_x": "RTX 2080 Ti", "rfdetr_2xl": "RTX 2080 Ti", "dinov3_vitl16": "GTX 1080 Ti",
       "sam3_zeroshot": "RTX 2080 Ti", "sam3_ft": "RTX 2080 Ti"}
PIEZAS = ["Body", "Licence plate", "Wheel", "Windscreen", "Rear window",
          "Left front window", "Left front door", "Left rear window", "Left rear door",
          "Right front window", "Right front door", "Right rear window", "Right rear door"]
FONDO, TINTA, TINTA2, REJILLA = "#fcfcfb", "#1f1f1e", "#5f5e5a", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA,
    "text.color": TINTA, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def cargar():
    r = {x["nombre"]: x for x in json.loads((INF / "metricas.json").read_text())}
    return [r[n] for n in ORDEN if n in r]


def metricas(res):
    claves = [("mIoU", "mIoU"), ("mAcc", "mAcc"), ("mDice", "mDice"), ("mAP50", "mAP50"),
              ("mAP50_95", "mAP50-95"), ("F1_macro", "Part F1")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n = len(res); alto = 0.8 / n
    y = np.arange(len(claves))
    for k, r in enumerate(res):
        v = [r[c] for c, _ in claves]
        pos = y + (k - (n - 1) / 2) * alto
        ax.barh(pos, v, alto * 0.88, color=COLOR[r["nombre"]], label=NOMBRE[r["nombre"]])
        for p, x in zip(pos, v):
            ax.text(x + 0.8, p, f"{x:.1f}", va="center", fontsize=7.5, color=TINTA2)
    ax.set_yticks(y, [t for _, t in claves]); ax.invert_yaxis()
    ax.set_xlim(0, 105); ax.set_xlabel("%")
    ax.xaxis.grid(True, color=REJILLA, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.set_title("Overall metrics on the VehicleSeg10K test set (3,069 images)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=n, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(SAL / "metricas.png", dpi=160); plt.close(fig)


def iou_por_pieza(res):
    m = np.array([r["iou_clase"] for r in res]).T   # piezas x modelos
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    from matplotlib.colors import LinearSegmentedColormap
    azul = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(m, cmap=azul, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if m[i, j] > 55 else TINTA)
    ax.set_xticks(range(len(res)), [NOMBRE[r["nombre"]].replace(" + ", "\n+ ").replace(" (", "\n(") for r in res], fontsize=8.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks(range(len(PIEZAS)), PIEZAS, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(res)), minor=True); ax.set_yticks(np.arange(-0.5, len(PIEZAS)), minor=True)
    ax.grid(which="minor", color=FONDO, lw=2); ax.tick_params(which="minor", length=0)
    ax.set_title("IoU per part (%)  ·  'Left' = the car's left", pad=42)
    fig.tight_layout(); fig.savefig(SAL / "iou_por_pieza.png", dpi=160); plt.close(fig)


def precision_vs_velocidad(res):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for r in res:
        ms = r["ms_imagen"]
        if not ms:
            continue
        ax.scatter(ms, r["mIoU"], s=90, color=COLOR[r["nombre"]], edgecolor=FONDO, linewidth=2, zorder=3)
        arriba = r["nombre"] in ("rfdetr_2xl", "dinov3_vitl16")   # evita que se pisen las etiquetas
        ax.annotate(f"{NOMBRE[r['nombre']]}\n{ms:.0f} ms · {GPU[r['nombre']]}", (ms, r["mIoU"]),
                    xytext=(8, 6 if arriba else -6), textcoords="offset points", fontsize=8.5,
                    va="bottom" if arriba else "top", color=TINTA)
    ax.set_xscale("log"); ax.set_xlim(20, 5000); ax.set_ylim(0, 105)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000], ["20", "50", "100", "200", "500", "1000", "2000", "5000"])
    ax.minorticks_off()
    ax.set_xlabel("ms per frame (log scale, 1 GPU, batch 1)"); ax.set_ylabel("mIoU (%)")
    ax.grid(True, color=REJILLA, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Accuracy vs. inference time")
    fig.text(0.01, 0.01, "DINOv3 was timed in float16 on a GTX 1080 Ti (no tensor cores), "
             "so its time is pessimistic.", fontsize=7.5, color=TINTA2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(SAL / "precision_vs_velocidad.png", dpi=160); plt.close(fig)


def matriz_confusion(res):
    mejor = max(res, key=lambda r: r["mIoU"])
    cm = np.array(mejor["confusion"], float)[1:, 1:]
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    from matplotlib.colors import LinearSegmentedColormap
    azul = LinearSegmentedColormap.from_list("azul", [FONDO, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(8.5, 7.6))
    ax.imshow(cm, cmap=azul, vmin=0, vmax=100)
    for i in range(13):
        for j in range(13):
            if cm[i, j] >= 0.5:
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > 55 else TINTA)
    ax.set_xticks(range(13), PIEZAS, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticks(range(13), PIEZAS, fontsize=8.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Pixel confusion matrix, % of each row · {NOMBRE[mejor['nombre']]}")
    fig.tight_layout(); fig.savefig(SAL / "matriz_confusion.png", dpi=160); plt.close(fig)


PARES = [("Left front window <-> Right front window", "Front windows L↔R"),
         ("Left front door <-> Right front door", "Front doors L↔R"),
         ("Left back window <-> Right back window", "Rear windows L↔R"),
         ("Left back door <-> Right back door", "Rear doors L↔R"),
         ("Front window <-> Back window", "Windscreen ↔ rear window")]


def confusion_lados(res):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    n = len(res); alto = 0.8 / n; y = np.arange(len(PARES))
    vmax = max(r["confusion_lado"][k] for r in res for k, _ in PARES)
    for k, r in enumerate(res):
        v = [r["confusion_lado"][c] for c, _ in PARES]
        pos = y + (k - (n - 1) / 2) * alto
        ax.barh(pos, v, alto * 0.88, color=COLOR[r["nombre"]], label=NOMBRE[r["nombre"]])
        for p_, x in zip(pos, v):
            ax.text(x + vmax * 0.01, p_, f"{x:.2f}", va="center", fontsize=7.5, color=TINTA2)
    ax.set_yticks(y, [t for _, t in PARES]); ax.invert_yaxis(); ax.set_xlim(0, vmax * 1.12)
    ax.set_xlabel("% of the pair's pixels assigned to the opposite part (lower is better)")
    ax.xaxis.grid(True, color=REJILLA, lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.set_title("Left/right and front/back confusion")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=n, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(SAL / "confusion_lados.png", dpi=160); plt.close(fig)


def distribucion(res):
    con = [r for r in res if r.get("iou_por_imagen")]
    if not con:
        print("metricas.json sin iou_por_imagen: no se genera distribucion.png"); return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b = ax.boxplot([np.array(r["iou_por_imagen"]) * 100 for r in con], showfliers=False, patch_artist=True,
                   medianprops={"color": TINTA})
    for caja, r in zip(b["boxes"], con):
        caja.set_facecolor(COLOR[r["nombre"]]); caja.set_alpha(0.8)
    ax.set_xticks(range(1, len(con) + 1), [NOMBRE[r["nombre"]] for r in con])
    ax.set_ylabel("Mean IoU of the image (%)"); ax.yaxis.grid(True, color=REJILLA); ax.set_axisbelow(True)
    ax.set_title("Per-image distribution (box = middle half of the images)")
    fig.tight_layout(); fig.savefig(SAL / "distribucion.png", dpi=160); plt.close(fig)


def tablas_markdown(res):
    """Tablas por pieza en Markdown (para pegar en los README)."""
    cab = "| Part | " + " | ".join(NOMBRE[r["nombre"]] for r in res) + " |\n|---|" + "---|" * len(res) + "\n"
    out = []
    for clave, titulo in (("iou_clase", "IoU per part (%)"), ("acc_clase", "Pixel accuracy per part (%)"),
                          ("f1_clase", "Detection F1 per part (%)"), ("recall_clase", "Detection recall per part (%)")):
        filas = []
        for i, pz in enumerate(PIEZAS):
            vals = [r[clave][i] for r in res]
            mx = max(vals)
            filas.append(f"| {pz} | " + " | ".join(f"**{v:.1f}**" if v == mx else f"{v:.1f}" for v in vals) + " |")
        filas.append("| **Mean** | " + " | ".join(f"{np.nanmean(r[clave]):.1f}" for r in res) + " |")
        out.append(f"#### {titulo}\n\n" + cab + "\n".join(filas) + "\n")
    filas = ["| Pair | " + " | ".join(NOMBRE[r["nombre"]] for r in res) + " |", "|---|" + "---|" * len(res)]
    for c, t in PARES:
        vals = [r["confusion_lado"][c] for r in res]
        mn = min(vals)
        filas.append(f"| {t} | " + " | ".join(f"**{v:.2f}**" if v == mn else f"{v:.2f}" for v in vals) + " |")
    out.append("#### Left/right and front/back confusion (% of pixels, lower is better)\n\n" + "\n".join(filas) + "\n")
    (SAL / "tablas.md").write_text("\n".join(out))


def ejemplos():
    pdf = INF / "car_part_segmentation_report.pdf"
    if not (shutil.which("pdftoppm") and pdf.exists()):
        print("sin pdftoppm o sin PDF: no se genera ejemplos.png"); return
    # la pagina de ejemplos es la penultima del informe
    paginas = int(next(l.split()[-1] for l in subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
                       .stdout.splitlines() if l.startswith("Pages:")))
    subprocess.run(["pdftoppm", "-png", "-r", "110", "-singlefile", "-f", str(paginas - 1), "-l", str(paginas - 1),
                    str(pdf), str(SAL / "ejemplos")], check=True)


if __name__ == "__main__":
    SAL.mkdir(parents=True, exist_ok=True)
    res = cargar()
    metricas(res); iou_por_pieza(res); precision_vs_velocidad(res); matriz_confusion(res)
    confusion_lados(res); distribucion(res); tablas_markdown(res); ejemplos()
    print("Figuras en", SAL)
