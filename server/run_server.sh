#!/usr/bin/env bash
# Startet den Moebius-Inpainting-API-Server.
#
# Verwendung:
#   ./run_server.sh                # echter Modus (braucht Moebius + Gewichte)
#   ./run_server.sh --mock         # Mock-Modus (ohne torch/GPU, nur Pfad-Test)
#   PORT=9000 ./run_server.sh      # anderer Port
#
# Relevante Umgebungsvariablen (siehe moebius_backend.py):
#   MOEBIUS_SRC      Pfad zur geklonten Moebius-Quelle (Default: ../Moebius)
#   MOEBIUS_WEIGHTS  Pfad zu den Gewichten (Default: $MOEBIUS_SRC/weight)
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

# venv aktivieren, falls vorhanden (von install_backend.sh angelegt).
if [[ -f "../.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "../.venv/bin/activate"
fi

if [[ "${1:-}" == "--mock" ]]; then
    export MOEBIUS_MOCK=1
    echo "[moebius] Starte im MOCK-Modus (kein torch/GPU) auf ${HOST}:${PORT}"
else
    echo "[moebius] Starte echten Modus auf ${HOST}:${PORT}"
    echo "[moebius]   MOEBIUS_SRC=${MOEBIUS_SRC:-../Moebius}"
fi

exec uvicorn server:app --host "${HOST}" --port "${PORT}"
