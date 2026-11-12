"""Entrena el decodificador sobre DINOv3 congelado (lo lanza 05_dinov3/entrenar.sh).

Con varias GPUs usa DistributedDataParallel. --batch es el lote TOTAL, como en YOLO.
El mIoU que se imprime es solo para seguir el progreso, sobre una parte de `valid`:
no se usa para elegir checkpoint. La evaluacion final se hace sobre last.pt.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from comun_dino import IGNORAR, NUM_CLASES, RAIZ, Backbone, Cabeza, VehicleSeg


def perdida(logits, y):
    """Entropia cruzada + Dice (solo clases presentes en el lote)."""
    logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, y, ignore_index=IGNORAR)
    validos = y != IGNORAR
    prob = logits.softmax(1).permute(0, 2, 3, 1)[validos]        # N x clases
    obj = F.one_hot(y[validos], NUM_CLASES).float()
    presentes = obj.sum(0) > 0
    inter, total = (prob * obj).sum(0), (prob + obj).sum(0)
    return ce + (1 - (2 * inter + 1) / (total + 1))[presentes].mean()


@torch.no_grad()
def validar(backbone, cabeza, datos, disp, amp, workers):
    cabeza.eval()
    m = torch.zeros(NUM_CLASES, NUM_CLASES, dtype=torch.int64, device=disp)
    for x, y in DataLoader(datos, batch_size=1, num_workers=min(4, workers)):  # WORKERS=0 no usa /dev/shm
        x, y = x.to(disp), y.to(disp)
        with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
            s = cabeza(backbone(x))
        p = F.interpolate(s.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        v = y != IGNORAR
        m += torch.bincount(NUM_CLASES * y[v] + p[v], minlength=NUM_CLASES ** 2).reshape(NUM_CLASES, NUM_CLASES)
    cabeza.train()
    d, g, q = m.diag().float(), m.sum(1).float(), m.sum(0).float()
    iou = d / (g + q - d).clamp(min=1)
    return 100 * iou[1:][g[1:] > 0].mean().item()   # sin la clase fondo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="vitl16")
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-cada", type=int, default=5)
    ap.add_argument("--val-imagenes", type=int, default=300)
    ap.add_argument("--sin-amp", action="store_true")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--limite", type=int, default=None, help="usar solo N imagenes (prueba)")
    a = ap.parse_args()
    if a.imgsz % 16:
        raise SystemExit("--imgsz debe ser multiplo de 16")

    n_proc = int(os.environ.get("WORLD_SIZE", 1))
    rango = int(os.environ.get("RANK", 0))
    if n_proc > 1:
        dist.init_process_group("nccl")
    disp = torch.device("cuda", rango % torch.cuda.device_count())
    torch.cuda.set_device(disp)
    torch.manual_seed(rango)
    es_0 = rango == 0
    amp = not a.sin_amp
    nombre = f"facebook/dinov3-{a.modelo}-pretrain-lvd1689m"
    salida = Path(a.salida or RAIZ / "resultados" / "dinov3" / a.modelo)
    salida.mkdir(parents=True, exist_ok=True)

    if n_proc > 1 and not es_0:   # el proceso 0 descarga los pesos; el resto los reutiliza
        dist.barrier()
    backbone = Backbone(nombre).to(disp)
    if n_proc > 1 and es_0:
        dist.barrier()
    cabeza_base = Cabeza(backbone.dim).to(disp)
    cabeza = DDP(cabeza_base, device_ids=[disp.index]) if n_proc > 1 else cabeza_base

    datos = VehicleSeg("train", a.imgsz, True, a.limite)
    sampler = DistributedSampler(datos, n_proc, rango, shuffle=True, drop_last=True) if n_proc > 1 else None
    carga = DataLoader(datos, batch_size=max(1, a.batch // n_proc), shuffle=sampler is None, sampler=sampler,
                       num_workers=a.workers, pin_memory=True, drop_last=True,
                       persistent_workers=a.workers > 0)
    val = VehicleSeg("valid", a.imgsz, False, a.limite or a.val_imagenes)

    total_it, calent = a.epocas * len(carga), min(500, len(carga))
    opt = torch.optim.AdamW(cabeza_base.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / calent) * 0.5 * (
        1 + math.cos(math.pi * min(t, total_it) / max(1, total_it))))
    escalador = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)

    inicio, ckpt_ruta, csv_ruta = 0, salida / "last.pt", salida / "results.csv"
    if a.reanudar and ckpt_ruta.exists():
        c = torch.load(ckpt_ruta, map_location=disp, weights_only=False)
        cabeza_base.load_state_dict(c["cabeza"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); escalador.load_state_dict(c["escalador"])
        inicio = c["epoca"] + 1
        if es_0:
            print(f">> Reanudando desde la epoca {inicio + 1}", flush=True)
    if es_0:
        print(f">> {nombre} | {n_proc} GPU(s) | {len(datos)} imagenes | lote total {a.batch} "
              f"({carga.batch_size} por GPU) | {len(carga)} iter/epoca | imgsz {a.imgsz} | "
              f"parametros cabeza {sum(p.numel() for p in cabeza_base.parameters()) / 1e6:.1f}M", flush=True)
        if not (a.reanudar and csv_ruta.exists()):
            with open(csv_ruta, "w", newline="") as f:
                csv.writer(f).writerow(["epoca", "perdida", "lr", "miou_parcial", "segundos"])

    for ep in range(inicio, a.epocas):
        if sampler:
            sampler.set_epoch(ep)
        cabeza.train()
        t0, acum = time.time(), 0.0
        for it, (x, y) in enumerate(carga):
            x, y = x.to(disp, non_blocking=True), y.to(disp, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
                caract = backbone(x)
                s = cabeza(caract)
            loss = perdida(s, y)
            opt.zero_grad(set_to_none=True)
            escalador.scale(loss).backward()
            escalador.step(opt)
            escalador.update()
            sched.step()
            acum += loss.item()
            if es_0 and (it % 20 == 0 or it == len(carga) - 1):
                mem = torch.cuda.max_memory_allocated(disp) / 2 ** 30
                resto = (time.time() - t0) / (it + 1) * (len(carga) - it - 1)
                print(f"epoca {ep + 1}/{a.epocas}  {it + 1}/{len(carga)}  perdida {acum / (it + 1):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  mem {mem:.2f}G  resta {resto / 60:.1f} min", flush=True)
        if not es_0:
            if n_proc > 1:
                dist.barrier()
            continue
        miou = None
        if (ep + 1) % a.val_cada == 0 or ep + 1 == a.epocas:
            miou = validar(backbone, cabeza_base, val, disp, amp, a.workers)
            print(f">> epoca {ep + 1}: mIoU en {len(val)} imagenes de valid = {miou:.2f} %", flush=True)
        tmp = ckpt_ruta.with_suffix(".tmp")
        torch.save({"cabeza": cabeza_base.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "escalador": escalador.state_dict(), "epoca": ep, "args": vars(a), "modelo": nombre,
                    "dim": backbone.dim}, tmp)
        tmp.replace(ckpt_ruta)
        with open(csv_ruta, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, f"{acum / len(carga):.4f}", f"{sched.get_last_lr()[0]:.2e}",
                                    "" if miou is None else f"{miou:.2f}", f"{time.time() - t0:.0f}"])
        if n_proc > 1:
            dist.barrier()
    if es_0:
        print(f">> Pesos finales: {ckpt_ruta}", flush=True)
    if n_proc > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
