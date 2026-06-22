#!/usr/bin/env bash
# Liest die echte Inferenz-API der installierten Moebius-Quelle aus, damit der
# Integrationspunkt in server/moebius_backend.py korrekt verdrahtet werden kann.
# Ausführen im Projekt-Root (Moebius/ muss daneben liegen):
#   bash scripts/moebius_api_probe.sh
# Die GESAMTE Ausgabe zurückgeben.
set -uo pipefail

M="${MOEBIUS_SRC:-./Moebius}"
sec() { printf '\n========== %s ==========\n' "$1"; }

cd "$(dirname "$0")/.."
[[ -d "$M" ]] || { echo "Moebius-Quelle nicht gefunden unter $M"; exit 1; }

sec "Dateibaum (oberste Ebene + infer/)"
ls -1 "$M"
echo "--- infer/ ---"; ls -1 "$M/infer" 2>/dev/null || echo "(kein infer/)"

sec "Wo ist build_pipeline / get_batch_infer_args definiert?"
grep -rn --include=*.py -E "def (build_pipeline|get_batch_infer_args)" "$M" || echo "NICHT als def gefunden"
echo "--- Vorkommen insgesamt ---"
grep -rn --include=*.py -E "build_pipeline|get_batch_infer_args" "$M" | head -40

sec "Inferenz-Einstieg: infer/infer_moebius.py (vollständig)"
cat "$M/infer/infer_moebius.py" 2>/dev/null || echo "(Datei fehlt)"

sec "Pipeline-Klasse / Aufruf-Signaturen"
grep -rn --include=*.py -E "class .*Pipeline|def __call__|def (infer|inpaint|sample|generate|run)\b" "$M" | head -40

sec "Mögliche Definitionsdatei von build_pipeline – Kontext"
DEF_FILE="$(grep -rln --include=*.py -E "def build_pipeline" "$M" | head -1)"
if [[ -n "${DEF_FILE:-}" ]]; then
    echo "Datei: $DEF_FILE"
    grep -n -E "def build_pipeline|def get_batch_infer_args|return|class |def __call__|def (infer|inpaint|forward)" "$DEF_FILE" | head -60
else
    echo "keine build_pipeline-Definition gefunden – siehe Vorkommen oben"
fi

sec "README: Inferenz-Aufruf"
grep -n -A3 -B1 -iE "infer_moebius|python -m infer|build_pipeline|inference" "$M/README.md" 2>/dev/null | head -40

sec "argparse-Argumente (cfg/pst/cps/num_step/model-weight/vae …)"
grep -rn --include=*.py -E "add_argument\(|model_weight|model_config|--vae|args\.(cfg|pst|cps|num_step|noise_offset)" "$M" | head -50

sec "FERTIG – bitte gesamte Ausgabe zurückgeben"
