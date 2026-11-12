#!/usr/bin/env bash
# Entorno conda "sam3" (Python 3.12, PyTorch 2.10 + CUDA 12.8, como el README de SAM 3).
# ANTES: pide acceso a los pesos en https://huggingface.co/facebook/sam3 (son "gated")
#        y crea un token de lectura en https://huggingface.co/settings/tokens
# Uso: bash 03_sam3/instalar.sh
set -euo pipefail
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^sam3 " || conda create -y -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# Repo oficial de SAM 3 en la version con la que se hicieron los experimentos
if [ ! -d "$RAIZ/repos/sam3" ]; then
  git clone https://github.com/facebookresearch/sam3.git "$RAIZ/repos/sam3"
  git -C "$RAIZ/repos/sam3" checkout -q f1647e8d0c1d280991f4d1b7d974c2704537fe66
fi
pip install -e "$RAIZ/repos/sam3[train]"
# sam3 exige numpy<2; opencv-python 5 arrastraria numpy 2
# setuptools 81+ ya no trae pkg_resources, que usa model_builder.py
# einops, psutil, pandas y matplotlib los importa el codigo pero no estan en el extra [train]
pip install pycocotools "opencv-python<4.11" "numpy>=1.26,<2" "setuptools<81" einops psutil pandas matplotlib tqdm huggingface_hub tensorboard
echo ">> Inicia sesion en Hugging Face (pega tu token cuando lo pida):"
hf auth login || huggingface-cli login
python -c "from huggingface_hub import hf_hub_download as d; print('acceso OK:', d('facebook/sam3','config.json'))"
python "$RAIZ/03_sam3/parchear_sam3.py"
echo ">> Entorno 'sam3' listo"
