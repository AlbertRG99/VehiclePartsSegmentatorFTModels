#!/usr/bin/env bash
# Entrena YOLO26-seg sobre VehicleSeg10K.
#
# Variantes (primer argumento):
#   m        YOLO26m-seg desde COCO       (misma arquitectura que el modelo de mitbersh)
#   x        YOLO26x-seg desde COCO       (el mas preciso de la familia; mas lento)
#   mitbersh YOLO26m-seg partiendo de los pesos de mitbersh/car-parts-segmentation-m
#
# IMPORTANTE: fliplr=0. El dataset tiene clases con lado (Left/Right front door...).
# Un volteo horizontal convierte una puerta izquierda en derecha sin cambiar la
# etiqueta y estropea el entrenamiento. SAV tambien entrena sin volteo.
#
# Uso:  bash 01_yolo/entrenar.sh x            (otros parametros: EPOCHS=150 BATCH=8 bash ...)
#
# Varias GPUs (DDP):  DEVICE=0,1,2,3 bash 01_yolo/entrenar.sh x
#   Con varias GPUs el autobatch (BATCH=-1) no funciona; si no se da BATCH se usa
#   un lote total pensado para GPUs de 11 GB (2080 Ti): x -> 2 por GPU, m -> 4 por GPU.
#   En 11 GB, x a 1024 solo cabe con 1 imagen por GPU: usa IMGSZ=768 con x.
set -euo pipefail
cd "$(dirname "$0")/.."
VAR=${1:-x}
EPOCHS=${EPOCHS:-100}
IMGSZ=${IMGSZ:-1024}
BATCH=${BATCH:--1}          # -1 = autobatch (usa ~60% de la VRAM; solo con 1 GPU)
DEVICE=${DEVICE:-0}         # 0 = una GPU; 0,1,2,3 = cuatro GPUs
DATA=datos/preparado/yolo/data.yaml

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate yolo

case $VAR in
  m) MODEL=yolo26m-seg.pt ;;
  x) MODEL=yolo26x-seg.pt ;;
  mitbersh)
    MODEL=$(python -c "from huggingface_hub import hf_hub_download as d; print(d('mitbersh/car-parts-segmentation-m','parts_segmentation.pt', local_dir='pesos/mitbersh_m'))") ;;
  *) echo "variante desconocida: $VAR"; exit 1 ;;
esac

NGPU=$(awk -F, '{print NF}' <<< "$DEVICE")
if [ "$NGPU" -gt 1 ] && [ "$BATCH" = "-1" ]; then
  case $VAR in x) POR_GPU=2 ;; *) POR_GPU=4 ;; esac
  BATCH=$((POR_GPU * NGPU))
  echo ">> $NGPU GPUs ($DEVICE): lote total $BATCH ($POR_GPU por GPU). Cambialo con BATCH=..."
fi

yolo segment train model="$MODEL" data=$DATA epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH device=$DEVICE \
  fliplr=0.0 mosaic=1.0 close_mosaic=10 cos_lr=True patience=0 \
  project="$PWD/resultados/yolo" name=$VAR exist_ok=True seed=0 deterministic=False \
  workers=8 plots=True

echo ">> Pesos finales: resultados/yolo/$VAR/weights/last.pt"
echo ">> Siguiente: python 01_yolo/predecir.py --pesos resultados/yolo/$VAR/weights/last.pt --nombre yolo_$VAR"
