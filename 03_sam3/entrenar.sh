#!/usr/bin/env bash
# Experimento 3: fine-tuning de SAM 3 sobre VehicleSeg10K con el entrenador oficial.
#
#   bash 03_sam3/entrenar.sh                 # congela backbone, 10 pasadas, batch efectivo 4
#   SAM3_CONGELAR=nada SAM3_EPOCAS=20 bash 03_sam3/entrenar.sh   # FT completo (mas memoria)
#   SAM3_GPUS=4 bash 03_sam3/entrenar.sh     # 4 GPUs; la acumulacion baja para que el batch efectivo siga en 4
#   SAM3_AMP=bfloat16 ...                    # solo en GPUs Ampere o mas nuevas (por defecto float16)
#
# Antes: bash 03_sam3/instalar.sh  y  python 03_sam3/preparar_prompts.py
set -euo pipefail
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
export EXP_RAIZ=$RAIZ
export SAM3_EPOCAS=${SAM3_EPOCAS:-10}
export SAM3_GPUS=${SAM3_GPUS:-1}
export SAM3_ACUM=${SAM3_ACUM:-$(( 4 / SAM3_GPUS > 0 ? 4 / SAM3_GPUS : 1 ))}
export SAM3_AMP=${SAM3_AMP:-float16}
export TORCH_BLAS_PREFER_CUBLASLT=1  # GEMM float16 por cuBLAS falla en las 2080 Ti (ver probar_fp16.py)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export SAM3_CONGELAR=${SAM3_CONGELAR:-backbone}
export PYTHONPATH=$RAIZ/03_sam3:${PYTHONPATH:-}
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate sam3

DEST=$RAIZ/repos/sam3/sam3/train/configs/vehicleseg10k
mkdir -p "$DEST"; cp "$RAIZ/03_sam3/sam3_vehicleseg10k.yaml" "$DEST/"
mkdir -p "$RAIZ/resultados/sam3"
echo ">> GPUs=$SAM3_GPUS acum=$SAM3_ACUM amp=$SAM3_AMP congelar=$SAM3_CONGELAR epocas=$SAM3_EPOCAS"
cd "$RAIZ/repos/sam3"
python sam3/train/train.py -c configs/vehicleseg10k/sam3_vehicleseg10k.yaml \
  --use-cluster 0 --num-gpus "$SAM3_GPUS" 2>&1 | tee "$RAIZ/resultados/sam3/entrenamiento.log"
echo ">> Checkpoint: resultados/sam3/entrenamiento/checkpoints/checkpoint.pt"
echo ">> Siguiente: python 03_sam3/predecir.py --pesos resultados/sam3/entrenamiento/checkpoints/checkpoint.pt"
