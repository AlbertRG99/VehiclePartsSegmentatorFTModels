#!/usr/bin/env bash
# Entorno conda "rfdetr" para RF-DETR Seg (version fijada para que el experimento sea repetible).
# Los pesos preentrenados se descargan solos al entrenar; no hace falta cuenta.
# Uso: bash 06_rfdetr/instalar.sh
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^rfdetr " || conda create -y -n rfdetr python=3.12
conda activate rfdetr
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install "rfdetr[train,loggers]==1.12.0" pycocotools tqdm pillow
# opencv sin interfaz grafica: no necesita libGL (servidores y contenedores Docker)
python -m pip uninstall -y opencv-python || true
python -m pip install opencv-python-headless
python - <<'PY'
import cv2, pytorch_lightning, rfdetr, torch
from importlib.metadata import version
from rfdetr import RFDETRSeg2XLarge  # noqa: F401
print("torch", torch.__version__, "GPUs", torch.cuda.device_count(), "rfdetr", version("rfdetr"),
      "lightning", pytorch_lightning.__version__, "opencv", cv2.__version__)
PY
echo ">> Entorno 'rfdetr' listo"
