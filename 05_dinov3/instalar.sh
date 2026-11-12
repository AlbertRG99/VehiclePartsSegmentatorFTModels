#!/usr/bin/env bash
# Entorno conda "dinov3". Los pesos son "gated": antes pide acceso en
#   https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
# Uso: bash 05_dinov3/instalar.sh
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^dinov3 " || conda create -y -n dinov3 python=3.12
conda activate dinov3
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install "transformers==5.22.2" huggingface_hub pycocotools opencv-python-headless pillow tqdm matplotlib
python -c "import torch, transformers; print('torch', torch.__version__, 'GPUs', torch.cuda.device_count(), 'transformers', transformers.__version__)"
hf auth whoami >/dev/null 2>&1 || hf auth login
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("facebook/dinov3-vitl16-pretrain-lvd1689m", "config.json")
print(">> Acceso a los pesos de DINOv3 correcto")
PY
echo ">> Entorno 'dinov3' listo"
