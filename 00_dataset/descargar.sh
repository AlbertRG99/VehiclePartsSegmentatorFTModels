#!/usr/bin/env bash
# Descarga VehicleSeg10K (1,6 GB, enlace de Dropbox del README de SAV) y lo descomprime
# en datos/VehicleSeg10K. Solo hace falta si empiezas en otra maquina.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p datos
[ -d datos/VehicleSeg10K/images ] && { echo "ya existe datos/VehicleSeg10K"; exit 0; }
curl -L --fail -o datos/VehicleSeg10K.zip \
  "https://www.dropbox.com/scl/fi/tgtdboa4pxa7rtfg9ag52/VehicleSeg10K.zip?rlkey=ex4ssmgb21agqyvg0rysbl9pv&st=o14a1fo3&dl=1"
unzip -q datos/VehicleSeg10K.zip -d datos/VehicleSeg10K
echo "Listo: datos/VehicleSeg10K (images/ y annotations/ con train y valid)"
