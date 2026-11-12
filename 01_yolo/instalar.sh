#!/usr/bin/env bash
# Entorno conda "yolo" con Ultralytics (YOLO26-seg) para el experimento 1.
# Uso: bash 01_yolo/instalar.sh
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^yolo " || conda create -y -n yolo python=3.12
conda activate yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "ultralytics>=8.4" huggingface_hub pycocotools opencv-python tqdm matplotlib reportlab
python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)"
echo ">> Entorno 'yolo' listo"
