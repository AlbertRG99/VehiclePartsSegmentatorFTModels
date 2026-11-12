"""Parches para usar SAM 3 en GPUs sin bfloat16 (2080 Ti / 1080 Ti). Se puede repetir.

perflib/fused.py: el MLP del backbone de vision convierte siempre entrada y pesos a
bfloat16 (pensado para A100/H100) y falla si hay gradientes. Se cambia para que use el
tipo de autocast (o el de la entrada si no hay autocast) y, con gradientes activos,
la ruta normal Linear + activacion. El resultado numerico es el mismo calculo.

model/text_encoder_ve.py: la proyeccion final del texto ([1, 1024] @ [1024, 512]) da
CUBLAS_STATUS_INVALID_VALUE en float16 en las 2080 Ti. VETextEncoder descarta ese
resultado (solo usa los tokens), asi que se calcula en float32 sin autocast.

Uso:  python 03_sam3/parchear_sam3.py
"""
from pathlib import Path

SAM3 = Path(__file__).resolve().parent.parent / "repos" / "sam3" / "sam3"
f = SAM3 / "perflib" / "fused.py"

VIEJO = """def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
"""
NUEVO = """def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        # [parche experimentos_piezas] con gradientes: ruta normal en vez de error
        return activation()(linear(mat1)) if isinstance(activation, type) else activation(linear(mat1))
    # [parche experimentos_piezas] tipo de autocast o de la entrada, no bfloat16 fijo
    if torch.is_autocast_enabled("cuda") and mat1.is_cuda:
        dtype = torch.get_autocast_dtype("cuda")
    else:
        dtype = mat1.dtype
    self = linear.bias.detach().to(dtype)
    mat1 = mat1.to(dtype)
    mat2 = linear.weight.detach().to(dtype)
"""

def parchear(f, viejo, nuevo):
    s = f.read_text(encoding="utf-8")
    if "[parche experimentos_piezas]" in s:
        print("ya parcheado:", f.relative_to(SAM3.parent))
    elif viejo in s:
        f.write_text(s.replace(viejo, nuevo), encoding="utf-8")
        print("parcheado:", f.relative_to(SAM3.parent))
    else:
        raise SystemExit(f"No encuentro el bloque esperado en {f}; revisa la version del repo")


parchear(f, VIEJO, NUEVO)
parchear(SAM3 / "model" / "text_encoder_ve.py",
         """                pooled = pooled @ self.text_projection
""",
         """                # [parche experimentos_piezas] en float32: en float16 falla cuBLAS en Turing
                with torch.autocast("cuda", enabled=False):
                    pooled = pooled.float() @ self.text_projection.float()
""")
