#!/usr/bin/env bash
# Diagnose des GPU-Rechners für das Moebius-Backend.
# Ausführen auf dem Remote-Rechner (z. B. pc-ai-gpu), Ausgabe hierher zurückgeben:
#   bash scripts/remote_probe.sh
set -uo pipefail

line() { printf '\n===== %s =====\n' "$1"; }

line "HOST / OS"
hostname; uname -a; cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION)=' || true

line "uv"
command -v uv >/dev/null 2>&1 && uv --version || echo "uv nicht gefunden"

line "Python-Interpreter (System)"
for p in python3.10 python3.11 python3.12 python3.13 python3.14 python3; do
    command -v "$p" >/dev/null 2>&1 && echo "$p -> $($p --version 2>&1)"
done

line "uv python list"
command -v uv >/dev/null 2>&1 && uv python list 2>/dev/null | head -20 || true

line "NVIDIA GPU / Treiber"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "--- nvidia-smi (CUDA-Treiberversion) ---"
    nvidia-smi | grep -i "CUDA Version" || true
else
    echo "nvidia-smi NICHT gefunden -> keine NVIDIA-GPU oder Treiber fehlt"
fi

line "CUDA-Toolkit (nvcc)"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "nvcc nicht gefunden (nur Toolkit-Info; für PyTorch-Wheels nicht zwingend nötig)"

line "Moebius/requirements.txt (falls geklont)"
if [[ -f Moebius/requirements.txt ]]; then
    grep -niE "torch|cu1|index|extra|diffusers|transformers|flash" Moebius/requirements.txt || true
    echo "--- Zeilenzahl ---"; wc -l Moebius/requirements.txt
else
    echo "Moebius/ noch nicht geklont (ok – install_backend.sh holt es)"
fi

line "Plattenplatz im Projektverzeichnis"
df -h . | tail -1

line "FERTIG"
echo "Bitte die GESAMTE Ausgabe oben zurückgeben."
