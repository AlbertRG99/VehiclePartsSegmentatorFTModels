"""Inferencia de SAM 3 sobre el test (valid) con un prompt de texto por clase.

Sirve para dos cosas:
  - SAM 3 tal cual (zero-shot):      python 03_sam3/predecir.py --nombre sam3_zeroshot
  - SAM 3 afinado (este experimento): python 03_sam3/predecir.py \
        --pesos resultados/sam3/entrenamiento/checkpoints/checkpoint.pt --nombre sam3_ft

Escribe resultados/predicciones/<nombre>.json en el formato comun.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "04_evaluacion"))
sys.path.insert(0, str(RAIZ / "03_sam3"))
from comun import contar_parametros, Escritor, cargar_imagenes_test  # noqa: E402
from preparar_prompts import PROMPTS  # noqa: E402
from sam3.model import box_ops  # noqa: E402
from sam3.model.data_misc import interpolate  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402

CLASES = list(PROMPTS)  # mismo orden que el dataset (id = indice + 1)


class ProcesadorLigero(Sam3Processor):
    """Como Sam3Processor, pero sin agotar la GPU con las fotos de ~12 Mpx del test.

    El original lleva a resolucion original TODAS las mascaras sobre el umbral, en float32
    y en la GPU (>1,5 GB en una foto grande). Aqui: solo las max_det mejores por prompt,
    ampliadas de 4 en 4 y guardadas en la CPU como binarias (logit > 0 == sigmoid > 0.5).
    """

    def __init__(self, *args, max_det=20, bloque=4, **kw):
        super().__init__(*args, **kw)
        self.max_det, self.bloque = max_det, bloque

    @torch.inference_mode()
    def _forward_grounding(self, state):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"], find_target=None)
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        probs = (outputs["pred_logits"].sigmoid() * presence).squeeze(-1)
        keep = probs > self.confidence_threshold
        probs, masks, bbox = probs[keep], outputs["pred_masks"][keep], outputs["pred_boxes"][keep]
        if len(probs) > self.max_det:
            idx = probs.topk(self.max_det).indices
            probs, masks, bbox = probs[idx], masks[idx], bbox[idx]
        h, w = state["original_height"], state["original_width"]
        boxes = box_ops.box_cxcywh_to_xyxy(bbox) * torch.tensor([w, h, w, h], device=bbox.device)[None]
        binarias = [interpolate(masks[i:i + self.bloque].unsqueeze(1), (h, w), mode="bilinear",
                                align_corners=False).gt(0).cpu()
                    for i in range(0, len(masks), self.bloque)]
        state["masks"] = torch.cat(binarias) if binarias else torch.zeros((0, 1, h, w), dtype=torch.bool)
        state["boxes"], state["scores"] = boxes, probs
        return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", default=None, help="checkpoint.pt del entrenamiento (sin el: zero-shot)")
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--conf", type=float, default=0.05, help="umbral bajo; la evaluacion aplica el suyo")
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--parte", type=int, default=0, help="con --partes N: que trozo del test procesa (0..N-1)")
    ap.add_argument("--partes", type=int, default=1, help="reparte el test en N trozos (uno por GPU)")
    ap.add_argument("--max-det", type=int, default=20, help="detecciones como maximo por prompt y foto")
    ap.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto",
                    help="auto: bf16 en Ampere o mas nueva, fp16 en las anteriores")
    a = ap.parse_args()
    nombre = a.nombre or ("sam3_ft" if a.pesos else "sam3_zeroshot")

    # En las 2080 Ti con torch cu128, todo GEMM float16 por cuBLAS da CUBLAS_STATUS_INVALID_VALUE;
    # por cuBLASLt funciona (03_sam3/probar_fp16.py)
    torch.backends.cuda.preferred_blas_library("cublaslt")
    modelo = build_sam3_image_model(enable_segmentation=True)
    if a.pesos:
        ck = torch.load(a.pesos, map_location="cpu", weights_only=False)
        ck = ck.get("model", ck)
        falt, sobr = modelo.load_state_dict(ck, strict=False)
        print(f"Pesos cargados: {len(falt)} claves faltan, {len(sobr)} sobran")
    modelo = modelo.cuda().eval()
    # bfloat16 solo es rapido desde Ampere (8.x); en las 2080 Ti (7.5) se usa float16
    if a.precision == "auto":
        a.precision = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.precision]
    print("precision:", a.precision)
    proc = ProcesadorLigero(modelo, confidence_threshold=a.conf, max_det=a.max_det)

    if a.partes > 1:  # los trozos van a predicciones/partes/ y los junta 03_sam3/unir_partes.py
        (RAIZ / "resultados" / "predicciones" / "partes").mkdir(parents=True, exist_ok=True)
        nombre = f"partes/{nombre}.{a.parte}"
    imagenes = cargar_imagenes_test(a.limite)[a.parte::a.partes]
    esc = Escritor(nombre, f"SAM 3 {'afinado' if a.pesos else 'zero-shot'}, prompts: {PROMPTS}")
    for im in tqdm(imagenes, desc=f"parte {a.parte}/{a.partes}"):
        img = Image.open(im["ruta"]).convert("RGB")
        with esc.cronometro(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            estado = proc.set_image(img)
            salidas = []
            for i, c in enumerate(CLASES):
                proc.reset_all_prompts(estado)
                out = proc.set_text_prompt(prompt=PROMPTS[c], state=estado)
                salidas.append((i + 1, out["masks"].numpy(), out["scores"].float().cpu().numpy()))
        for cid, masks, scores in salidas:
            for m, s in zip(masks, scores):
                esc.anadir(im["id"], cid, np.squeeze(m).astype(bool), float(s))
    esc.guardar({"pesos": a.pesos, "prompts": PROMPTS, "precision": a.precision,
                 "parametros": contar_parametros(modelo)})


if __name__ == "__main__":
    main()
