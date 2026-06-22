#!/usr/bin/env bash
# Dumpt die exakten Code-Stellen, die für die Verdrahtung von moebius_backend.py
# nötig sind (Pipeline-Bau, VAE-/Modell-Laden, Dataset-Format, Maskenpolarität,
# Output-Größe). Ausführen im Projekt-Root:
#   bash scripts/moebius_api_probe2.sh
# Gesamte Ausgabe zurückgeben.
set -uo pipefail
M="${MOEBIUS_SRC:-./Moebius}"
cd "$(dirname "$0")/.."
sec() { printf '\n========== %s ==========\n' "$1"; }

sec "infer/utils.py (vollständig)"
cat "$M/infer/utils.py"

sec "infer/utils_dataset.py (vollständig)"
cat "$M/infer/utils_dataset.py"

sec "config/model_cfg/moebius.yaml (vollständig)"
cat "$M/config/model_cfg/moebius.yaml" 2>/dev/null || echo "(fehlt)"

sec "Wo werden load_removal_model / load_cfg / vae definiert?"
grep -rn --include=*.py -E "def load_removal_model|def load_cfg|vae" "$M" | grep -viE "train|chinese" | head -40

sec "removal/v1_2/pipeline.py: Klasse + __call__ (Zeilen 120-200, 320-520)"
sed -n '120,200p' "$M/removal/v1_2/pipeline.py"
echo "----- __call__ -----"
sed -n '320,520p' "$M/removal/v1_2/pipeline.py"

sec "removal/v1_2/ Dateien"
ls -1 "$M/removal/v1_2" 2>/dev/null

sec "FERTIG"
