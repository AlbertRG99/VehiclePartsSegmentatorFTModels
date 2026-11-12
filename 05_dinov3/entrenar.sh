#!/usr/bin/env bash
# Entrena un decodificador sobre DINOv3 congelado con VehicleSeg10K.
#
# Uso:         bash 05_dinov3/entrenar.sh
# Varias GPUs: DEVICE=0,1,2,3 bash 05_dinov3/entrenar.sh
# Variables:   MODELO=vitl16 (vitb16, vith16plus, vit7b16...)  EPOCHS=40  IMGSZ=768
#              BATCH=<lote total; por defecto 8 por GPU>  LR=5e-4  WORKERS=8
# Si se corta: REANUDAR=1 con las mismas variables. Otros argumentos pasan a entrenar.py.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate dinov3

DEVICE=${DEVICE:-0}
NGPU=$(awk -F, '{print NF}' <<< "$DEVICE")
MODELO=${MODELO:-vitl16}
EPOCHS=${EPOCHS:-40}
IMGSZ=${IMGSZ:-768}
BATCH=${BATCH:-$((8 * NGPU))}
LR=${LR:-5e-4}
WORKERS=${WORKERS:-8}
EXTRA=(); [ -n "${REANUDAR:-}" ] && EXTRA+=(--reanudar)

export CUDA_VISIBLE_DEVICES=$DEVICE
ARGS=(--modelo "$MODELO" --epocas "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --lr "$LR" --workers "$WORKERS" "${EXTRA[@]}" "$@")
if [ "$NGPU" -gt 1 ]; then
  python -m torch.distributed.run --nproc_per_node "$NGPU" --master_port $((20000 + RANDOM % 20000)) 05_dinov3/entrenar.py "${ARGS[@]}"
else
  python 05_dinov3/entrenar.py "${ARGS[@]}"
fi
echo ">> Siguiente: python 05_dinov3/predecir.py --pesos resultados/dinov3/$MODELO/last.pt --nombre dinov3_$MODELO"
