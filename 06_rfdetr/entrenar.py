"""Entrena RF-DETR Seg con VehicleSeg10K. Se lanza desde 06_rfdetr/entrenar.sh.

Sin volteo horizontal: convertiria una puerta izquierda en derecha sin cambiar la etiqueta.
Parte de los pesos preentrenados en COCO que publica Roboflow.
"""
import argparse
from pathlib import Path

from rfdetr import RFDETRSeg2XLarge, RFDETRSegLarge, RFDETRSegMedium, RFDETRSegXLarge
from rfdetr.evaluation import matching

# La validacion calcula el IoU de mascaras a la resolucion original, en bloques de 32 mascaras.
# Con las fotos de ~12 Mpx de valid, un bloque ocupa 1,5 GB y agota una GPU de 11 GB.
# Con 4 ocupa ~190 MB; el resultado no cambia, solo cuantas mascaras se procesan a la vez.
matching._MASK_IOU_CHUNK = 4

RAIZ = Path(__file__).resolve().parent.parent
MODELOS = {"2xl": RFDETRSeg2XLarge, "xl": RFDETRSegXLarge, "l": RFDETRSegLarge, "m": RFDETRSegMedium}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=MODELOS, default="2xl")
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--imgsz", type=int, help="multiplo de 24; por defecto, la del modelo (2xl: 768)")
    ap.add_argument("--batch", type=int, default=1, help="imagenes POR GPU")
    ap.add_argument("--acum", type=int, default=4, help="iteraciones que acumulan gradiente")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--workers", type=int, default=4, help="por GPU")
    ap.add_argument("--val-cada", type=int, default=5)
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--solo-descargar", action="store_true", help="descarga los pesos base y termina")
    a = ap.parse_args()

    # gradient_checkpointing: menos memoria a cambio de algo de velocidad (GPUs de 11 GB)
    modelo = MODELOS[a.modelo](gradient_checkpointing=True)
    if a.solo_descargar:
        return

    salida = RAIZ / "resultados" / "rfdetr" / a.modelo
    extra = {}
    if a.imgsz:
        extra["resolution"] = a.imgsz
    if a.reanudar:
        extra["resume"] = str(salida / "last.ckpt")

    modelo.train(
        dataset_dir=str(RAIZ / "datos" / "preparado" / "rfdetr"),
        output_dir=str(salida),
        epochs=a.epocas,
        batch_size=a.batch,
        eval_batch_size=a.batch,
        grad_accum_steps=a.acum,
        lr=a.lr,
        lr_encoder=a.lr_encoder,
        num_workers=a.workers,
        eval_interval=a.val_cada,
        aug_config={},          # {} = sin volteo; se mantiene el cambio de escala
        expanded_scales=False,  # la escala maxima baja de 888 a 864 px
        devices="auto",         # sin esto, usa una sola GPU aunque se lance con torchrun
        **extra,
    )


if __name__ == "__main__":
    main()
