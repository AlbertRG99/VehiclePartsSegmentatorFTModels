#!/usr/bin/env bash
# Genera las predicciones de todos los modelos entrenados que existan y el informe PDF.
#   bash 04_evaluacion/predecir_todo.sh
set -uo pipefail
RAIZ=$(cd "$(dirname "$0")/.." && pwd); cd "$RAIZ"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate yolo
for v in m x mitbersh; do
  P=resultados/yolo/$v/weights/last.pt
  [ -f "$P" ] || continue
  # misma resolucion que en el entrenamiento (args.yaml lo guarda Ultralytics)
  IMG=$(awk '/^imgsz:/{print $2}' resultados/yolo/$v/args.yaml 2>/dev/null); IMG=${IMG:-1024}
  python 01_yolo/predecir.py --pesos "$P" --nombre yolo_$v --imgsz "$IMG"
done
python 01_yolo/predecir.py --mitbersh --nombre yolo_mitbersh_sin_reentrenar --imgsz 896   # referencia (entrenado a 896)


conda activate sam3
python 03_sam3/predecir.py --nombre sam3_zeroshot                               # referencia
P=resultados/sam3/entrenamiento/checkpoints/checkpoint.pt
[ -f "$P" ] && python 03_sam3/predecir.py --pesos "$P" --nombre sam3_ft

if ls resultados/dinov3/*/last.pt >/dev/null 2>&1; then
  conda activate dinov3
  for P in resultados/dinov3/*/last.pt; do
    python 05_dinov3/predecir.py --pesos "$P" --nombre "dinov3_$(basename "$(dirname "$P")")"
  done
fi

if ls -d resultados/rfdetr/*/ >/dev/null 2>&1; then
  conda activate rfdetr
  for D in resultados/rfdetr/*/; do
    P=${D}last_ema.pth; [ -f "$P" ] || P=${D}last.ckpt   # last.ckpt si el entrenamiento no llego al final
    [ -f "$P" ] && python 06_rfdetr/predecir.py --pesos "$P" --nombre "rfdetr_$(basename "$D")"
  done
fi

conda activate yolo
python 04_evaluacion/evaluar.py
echo ">> Informe: resultados/informe/informe_segmentacion_piezas.pdf"
