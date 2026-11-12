#!/usr/bin/env bash
# Entrena RF-DETR Seg con VehicleSeg10K.
#
# Uso:         bash 06_rfdetr/entrenar.sh
# Varias GPUs: DEVICE=0,1,2,3 bash 06_rfdetr/entrenar.sh
# Variables:   MODELO=2xl (xl, l, m)  EPOCHS=50  IMGSZ=<multiplo de 24>
#              BATCH=1 (imagenes POR GPU)  ACUM=<por defecto, lote efectivo cercano a 16>
#              LR=1e-4  WORKERS=4
# Si se corta: REANUDAR=1 con las mismas variables. Otros argumentos pasan a entrenar.py.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate rfdetr

DEVICE=${DEVICE:-0}
NGPU=$(awk -F, '{print NF}' <<< "$DEVICE")
MODELO=${MODELO:-2xl}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-1}
ACUM=${ACUM:-$(( (16 + BATCH * NGPU - 1) / (BATCH * NGPU) ))}
LR=${LR:-1e-4}
WORKERS=${WORKERS:-4}

export RF_HOME=$PWD/pesos/rfdetr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

[ -f datos/preparado/rfdetr/train/_annotations.coco.json ] || python 06_rfdetr/preparar_datos.py

ARGS=(--modelo "$MODELO" --epocas "$EPOCHS" --batch "$BATCH" --acum "$ACUM" --lr "$LR" --workers "$WORKERS")
[ -n "${IMGSZ:-}" ] && ARGS+=(--imgsz "$IMGSZ")
[ -n "${REANUDAR:-}" ] && ARGS+=(--reanudar)
ARGS+=("$@")

# Descarga en un solo proceso, para que las GPUs no bajen el mismo archivo a la vez
CUDA_VISIBLE_DEVICES= python 06_rfdetr/entrenar.py --modelo "$MODELO" --solo-descargar

echo ">> RF-DETR Seg $MODELO: GPUs $DEVICE, $BATCH por GPU, acumulacion $ACUM, lote efectivo $((BATCH * NGPU * ACUM))"
export CUDA_VISIBLE_DEVICES=$DEVICE
if [ "$NGPU" -gt 1 ]; then
  python -m torch.distributed.run --nproc_per_node "$NGPU" --master_port $((20000 + RANDOM % 20000)) 06_rfdetr/entrenar.py "${ARGS[@]}"
else
  python 06_rfdetr/entrenar.py "${ARGS[@]}"
fi
echo ">> Siguiente: python 06_rfdetr/predecir.py --pesos resultados/rfdetr/$MODELO/last_ema.pth --nombre rfdetr_$MODELO"
