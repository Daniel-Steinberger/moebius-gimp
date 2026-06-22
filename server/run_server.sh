#!/usr/bin/env bash
# Startet den Moebius-Inpainting-API-Server (uv-basiert).
#
# Verwendung (vom Projekt-Root oder aus server/ heraus):
#   ./server/run_server.sh                # echter Modus (braucht Moebius + Gewichte)
#   ./server/run_server.sh --mock         # Mock-Modus (ohne torch/GPU, nur Pfad-Test)
#   PORT=9000 ./server/run_server.sh      # anderer Port
#
# Relevante Umgebungsvariablen (siehe moebius_backend.py):
#   MOEBIUS_SRC      Pfad zur geklonten Moebius-Quelle (Default: ./Moebius)
#   MOEBIUS_WEIGHTS  Pfad zu den Gewichten (Default: $MOEBIUS_SRC/weight)
set -euo pipefail

# Projekt-Root (Verzeichnis über server/).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"

command -v uv >/dev/null 2>&1 || { echo "FEHLER: uv nicht installiert." >&2; exit 1; }

ARGS=(--host "${HOST}" --port "${PORT}")
if [[ "${1:-}" == "--mock" ]]; then
    ARGS+=(--mock)
    echo "[moebius] Starte im MOCK-Modus (kein torch/GPU) auf ${HOST}:${PORT}"
else
    echo "[moebius] Starte echten Modus auf ${HOST}:${PORT}"
    echo "[moebius]   MOEBIUS_SRC=${MOEBIUS_SRC:-./Moebius}"
fi

# --no-sync: NICHT erneut auflösen/prunen – sonst würden die per 'uv pip install'
# zusätzlich installierten Moebius-Deps (torch, diffusers) entfernt.
exec uv run --no-sync python server/server.py "${ARGS[@]}"
