"""Prueba aislada de matmul float16 en la GPU (para el CUBLAS_STATUS_INVALID_VALUE de SAM 3).

Uso (entorno 'sam3'):  CUDA_VISIBLE_DEVICES=0 python 03_sam3/probar_fp16.py
"""
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0), "| cuBLAS", torch.backends.cuda.preferred_blas_library())


def prueba(nombre, m, k=1024, n=512, dtype=torch.float16, biblioteca=None):
    if biblioteca:
        torch.backends.cuda.preferred_blas_library(biblioteca)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    try:
        c = a @ b
        torch.cuda.synchronize()
        err = (c.float() - a.float() @ b.float()).abs().max().item()
        print(f"  OK    {nombre:38s} -> {tuple(c.shape)}  (dif. max {err:.3f})")
    except RuntimeError as e:
        print(f"  FALLA {nombre:38s} -> {str(e).splitlines()[0][:90]}")


for blas in ("cublas", "cublaslt"):
    print(f"[{blas}]")
    prueba("fp16 m=1", 1, biblioteca=blas)
    prueba("fp16 m=2", 2, biblioteca=blas)
    prueba("fp16 m=8", 8, biblioteca=blas)
    prueba("fp16 m=1024", 1024, biblioteca=blas)
    prueba("fp32 m=1", 1, dtype=torch.float32, biblioteca=blas)
