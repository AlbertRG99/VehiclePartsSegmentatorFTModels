"""Cambia los nombres de categoria de los JSON de SAM 3 por frases en ingles.

SAM 3 usa el nombre de cada categoria como prompt de texto ("left front door"...).
Los ids de categoria no cambian, asi la evaluacion comun sigue valiendo.
Uso: python 03_sam3/preparar_prompts.py
"""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PROMPTS = {
    "Foreground": "car body",
    "Plate": "license plate",
    "Wheel": "car wheel",
    "Front window": "windshield",
    "Back window": "rear windshield",
    "Left front window": "left front side window",
    "Left front door": "left front door",
    "Left back window": "left rear side window",
    "Left back door": "left rear door",
    "Right front window": "right front side window",
    "Right front door": "right front door",
    "Right back window": "right rear side window",
    "Right back door": "right rear door",
}

if __name__ == "__main__":
    for split in ("train", "test"):
        f = RAIZ / "datos" / "preparado" / "sam3" / "vehicleseg10k" / split / "_annotations.coco.json"
        d = json.loads(f.read_text())
        for c in d["categories"]:
            c["name"] = PROMPTS.get(c["name"], c["name"])
        f.write_text(json.dumps(d))
        print(f, [c["name"] for c in d["categories"]])
