"""Modelo y datos del experimento DINOv3.

DINOv3 es solo un extractor de caracteristicas: no segmenta por si mismo. Aqui se
deja congelado y encima se entrena un decodificador convolucional (pirámide tipo
FPN) que predice, por pixel, una de 14 clases: fondo (0) y las 13 piezas (1..13,
los mismos valores que los PNG semanticos y que category_id en la evaluacion).
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos" / "VehicleSeg10K"
NUM_CLASES = 14       # fondo + 13 piezas
IGNORAR = 255         # relleno de los recortes
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # los pesos LVD-1689M usan ImageNet
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PARCHE = 16
cv2.setNumThreads(0)  # los procesos del DataLoader ya van en paralelo


def _tam_multiplo(h, w, lado):
    """Reduce/amplia para que el lado largo sea `lado`, con ambos lados multiplos del parche."""
    f = lado / max(h, w)
    return (max(PARCHE, round(h * f / PARCHE) * PARCHE), max(PARCHE, round(w * f / PARCHE) * PARCHE))


def preparar_imagen(rgb: np.ndarray, lado: int) -> torch.Tensor:
    """Imagen RGB uint8 -> tensor normalizado 1x3xHxW listo para el modelo (sin recorte)."""
    h, w = _tam_multiplo(*rgb.shape[:2], lado)
    x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return torch.from_numpy(((x - MEDIA) / STD).transpose(2, 0, 1)).unsqueeze(0)


def _color(rgb: np.ndarray) -> np.ndarray:
    """Variacion de brillo, contraste, saturacion y tono."""
    x = rgb.astype(np.float32)
    x = np.clip(x * random.uniform(0.75, 1.25), 0, 255)                     # brillo
    x = np.clip((x - x.mean()) * random.uniform(0.75, 1.25) + x.mean(), 0, 255)  # contraste
    hsv = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.75, 1.25), 0, 255)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-9, 9)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class VehicleSeg(Dataset):
    """Imagen + PNG semantico. SIN volteo horizontal: las clases tienen lado."""

    def __init__(self, split: str, lado: int, entrenar: bool, limite: int | None = None):
        self.lado, self.entrenar = lado, entrenar
        imgs = {p.stem: p for p in (DIR_DATOS / "images" / split).iterdir()}
        pngs = sorted(p for p in (DIR_DATOS / "annotations" / split).glob("*.png")
                      if not p.stem.endswith("_color") and p.stem in imgs)
        if limite:
            pngs = pngs[:limite]
        self.pares = [(imgs[p.stem], p) for p in pngs]
        if not self.pares:
            raise SystemExit(f"No hay datos en {DIR_DATOS} para el split {split}")

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, i):
        fimg, fpng = self.pares[i]
        rgb = cv2.imread(str(fimg))[..., ::-1].copy()
        sem = np.array(Image.open(fpng), dtype=np.uint8)
        h, w = rgb.shape[:2]
        if sem.shape != (h, w):
            sem = cv2.resize(sem, (w, h), interpolation=cv2.INTER_NEAREST)
        sem[sem >= NUM_CLASES] = IGNORAR
        if not self.entrenar:
            x = preparar_imagen(rgb, self.lado)[0]
            sem = cv2.resize(sem, (x.shape[2], x.shape[1]), interpolation=cv2.INTER_NEAREST)
            return x, torch.from_numpy(sem).long()

        # escala aleatoria -> relleno -> recorte cuadrado
        f = self.lado / max(h, w) * random.uniform(0.5, 1.5)
        nw, nh = max(1, round(w * f)), max(1, round(h * f))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        sem = cv2.resize(sem, (nw, nh), interpolation=cv2.INTER_NEAREST)
        px, py = max(0, self.lado - nw), max(0, self.lado - nh)
        if px or py:
            arr, izq = random.randint(0, py), random.randint(0, px)
            media = tuple(int(v * 255) for v in MEDIA)
            rgb = cv2.copyMakeBorder(rgb, arr, py - arr, izq, px - izq, cv2.BORDER_CONSTANT, value=media)
            sem = cv2.copyMakeBorder(sem, arr, py - arr, izq, px - izq, cv2.BORDER_CONSTANT, value=IGNORAR)
        y = random.randint(0, sem.shape[0] - self.lado)
        x = random.randint(0, sem.shape[1] - self.lado)
        rgb, sem = rgb[y:y + self.lado, x:x + self.lado], sem[y:y + self.lado, x:x + self.lado]
        rgb = _color(rgb).astype(np.float32) / 255
        return (torch.from_numpy(((rgb - MEDIA) / STD).transpose(2, 0, 1)),
                torch.from_numpy(sem.copy()).long())


def _bloque(c):
    return nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GroupNorm(32, c), nn.GELU(),
                         nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GroupNorm(32, c), nn.GELU())


def _subir(c):
    return nn.Sequential(nn.ConvTranspose2d(c, c, 2, 2, bias=False), nn.GroupNorm(32, c), nn.GELU())


class Cabeza(nn.Module):
    """Decodificador tipo FPN sobre 4 capas del ViT (todas a 1/16).

    Usa GroupNorm porque con pocas imagenes por GPU BatchNorm funciona mal.
    Devuelve logits a 1/4 de la resolucion de entrada.
    """

    def __init__(self, dim: int, canales: int = 256, n: int = NUM_CLASES):
        super().__init__()
        self.proy = nn.ModuleList(nn.Sequential(nn.Conv2d(dim, canales, 1, bias=False),
                                                nn.GroupNorm(32, canales), nn.GELU()) for _ in range(4))
        self.a_escala = nn.ModuleList([nn.Sequential(_subir(canales), _subir(canales)), _subir(canales),
                                       nn.Identity(), nn.MaxPool2d(2)])
        self.fusion = nn.ModuleList(_bloque(canales) for _ in range(3))
        self.salida = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(canales, n, 1))

    def forward(self, caract):
        p = [e(pr(c)) for c, pr, e in zip(caract, self.proy, self.a_escala)]
        x = p[3]
        for bloque, nivel in zip(self.fusion, (p[2], p[1], p[0])):
            x = bloque(F.interpolate(x, size=nivel.shape[-2:], mode="bilinear", align_corners=False) + nivel)
        return self.salida(x)


class Backbone(nn.Module):
    """DINOv3 congelado que devuelve 4 capas intermedias como mapas 2D."""

    def __init__(self, nombre: str):
        super().__init__()
        from transformers import AutoModel
        self.vit = AutoModel.from_pretrained(nombre)   # pesos preentrenados oficiales de Meta
        self.vit.requires_grad_(False).eval()
        cfg = self.vit.config
        self.dim, self.salto = cfg.hidden_size, 1 + cfg.num_register_tokens  # CLS + registros
        n = cfg.num_hidden_layers
        self.capas = [n // 4 * k for k in (1, 2, 3, 4)]   # indices en hidden_states (0 = embeddings)

    def train(self, mode=True):  # siempre en eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[-2] // PARCHE, x.shape[-1] // PARCHE
        estados = self.vit(pixel_values=x, output_hidden_states=True).hidden_states
        return [self.vit.norm(estados[i])[:, self.salto:].permute(0, 2, 1).reshape(x.shape[0], self.dim, h, w)
                for i in self.capas]
