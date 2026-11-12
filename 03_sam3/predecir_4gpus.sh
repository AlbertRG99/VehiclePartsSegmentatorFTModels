#!/usr/bin/env bash
# Inferencia de SAM 3 repartida entre varias GPUs (un trozo del test por GPU) y union final.
#
#   bash 03_sam3/predecir_4gpus.sh                       # zero-shot en las GPUs 0,1,2,3
#   bash 03_sam3/predecir_4gpus.sh --pesos resultados/sam3/entrenamiento/checkpoints/checkpoint.pt --nombre sam3_ft
#   GPUS=0,1 bash 03_sam3/predecir_4gpus.sh --limite 40  # prueba corta en 2 GPUs
#
# Log de cada GPU: resultados/predicciones/partes/<nombre>.<i>.log
set -euo pipefail
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
cd "$RAIZ"
IFS=, read -ra LISTA <<< "${GPUS:-0,1,2,3}"
N=${#LISTA[@]}
NOMBRE=sam3_zeroshot
args=("$@")
for ((j = 0; j < ${#args[@]}; j++)); do
  [[ ${args[j]} == --nombre ]] && NOMBRE=${args[j + 1]}
  [[ ${args[j]} == --pesos ]] && [[ ! " ${args[*]} " == *" --nombre "* ]] && NOMBRE=sam3_ft
done
mkdir -p resultados/predicciones/partes
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate sam3

echo ">> $NOMBRE en $N GPUs (${GPUS:-0,1,2,3})"
pids=()
for i in "${!LISTA[@]}"; do
  CUDA_VISIBLE_DEVICES=${LISTA[i]} python 03_sam3/predecir.py --nombre "$NOMBRE" "$@" \
    --parte "$i" --partes "$N" > "resultados/predicciones/partes/$NOMBRE.$i.log" 2>&1 &
  pids+=($!)
done
fallos=0
for i in "${!pids[@]}"; do
  if wait "${pids[i]}"; then echo ">> parte $i OK"; else echo ">> parte $i FALLO (ver resultados/predicciones/partes/$NOMBRE.$i.log)"; fallos=1; fi
done
[[ $fallos == 0 ]] || exit 1
python 03_sam3/unir_partes.py "$NOMBRE" "$N"
