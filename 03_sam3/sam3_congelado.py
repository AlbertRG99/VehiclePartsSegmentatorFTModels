"""Construye SAM 3 para el entrenador oficial con la opcion de congelar partes.

congelar:
  'backbone' -> congela vision + texto (solo se entrena detector/decodificador de
                mascaras). Recomendado en una GPU de 24 GB.
  'vision'   -> congela solo el backbone de vision.
  'nada'     -> fine-tuning completo (848M parametros; puede no caber en 24 GB).
Los parametros congelados no tienen gradientes ni estado de AdamW, asi que se
ahorra memoria de verdad (poner lr=0 no la ahorra).
"""
import torch
from sam3.model_builder import build_sam3_image_model

PATRONES = {
    "backbone": ("backbone.vision_backbone.", "backbone.language_backbone."),
    "vision": ("backbone.vision_backbone.",),
    "nada": (),
}


def construir(bpe_path=None, enable_segmentation=True, congelar="backbone", **kw):
    # Se llama en cada proceso de entrenamiento. En las 2080 Ti con torch cu128 los GEMM
    # float16 por cuBLAS fallan (CUBLAS_STATUS_INVALID_VALUE); por cuBLASLt funcionan.
    torch.backends.cuda.preferred_blas_library("cublaslt")
    modelo = build_sam3_image_model(bpe_path=bpe_path, device="cpu", eval_mode=False,
                                    enable_segmentation=enable_segmentation, load_from_HF=True)
    pats = PATRONES[congelar]
    n_cong = n_tot = 0
    for nombre, p in modelo.named_parameters():
        n_tot += p.numel()
        if nombre.startswith(pats):
            p.requires_grad_(False)
            n_cong += p.numel()
    print(f"[sam3_congelado] congelar={congelar}: {n_cong/1e6:.0f}M de {n_tot/1e6:.0f}M parametros congelados")
    return modelo
