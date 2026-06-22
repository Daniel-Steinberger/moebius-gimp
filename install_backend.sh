#!/usr/bin/env bash
# Richtet das Backend (Moebius-API-Server) mit uv ein.
# Gedacht für den GPU-Rechner (z. B. pc-ai-gpu).
#
# Was dieses Skript tut:
#   1. stellt mit uv eine MOEBIUS-KOMPATIBLE Python-Version bereit (Default 3.12)
#      -> torch 2.7.x hat KEINE Wheels für Python 3.13/3.14!
#   2. erzeugt das .venv und installiert die Server-Deps aus pyproject.toml (uv sync)
#   3. klont das Moebius-Repo nach ./Moebius
#   4. installiert die Moebius-Deps (torch, diffusers …) in DASSELBE venv,
#      mit passendem Torch-Backend (CUDA automatisch erkannt)
#
# Was dieses Skript NICHT tut (bewusst – mehrere GB):
#   - die Modellgewichte von Hugging Face herunterladen -> Hinweis am Ende
#
# Konfigurierbar per Umgebungsvariablen:
#   PYVER          Python-Version fürs venv          (Default 3.12)
#   TORCH_BACKEND  uv --torch-backend: auto|cpu|cu121|cu124|cu128|cu130 (Default auto)
#   MOEBIUS_REPO   Git-URL des Moebius-Repos
set -euo pipefail
cd "$(dirname "$0")"

# Python 3.12: torch 2.7.x hat KEINE Wheels für 3.13/3.14.
PYVER="${PYVER:-3.12}"
# cu128 ist der richtige Build für Moebius (torch 2.7.1) auf modernen NVIDIA-GPUs
# inkl. Blackwell (RTX 50xx). NICHT 'auto' nehmen: bei sehr neuem Treiber (CUDA 13.x)
# würde uv cu130 wählen, das es für torch 2.7.1 gar nicht gibt.
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"
MOEBIUS_REPO="${MOEBIUS_REPO:-https://github.com/hustvl/Moebius.git}"

command -v uv >/dev/null 2>&1 || {
    echo "FEHLER: uv ist nicht installiert. Siehe https://docs.astral.sh/uv/" >&2
    exit 1
}

echo "[1/4] Python ${PYVER} via uv bereitstellen …"
uv python install "${PYVER}"

echo "[2/4] venv + Server-Deps (pyproject.toml) installieren …"
uv sync --python "${PYVER}"
VENV_PY="$(pwd)/.venv/bin/python"

echo "[3/4] Moebius-Quelle bereitstellen …"
if [[ ! -d Moebius ]]; then
    git clone "${MOEBIUS_REPO}" Moebius
else
    echo "      ./Moebius existiert bereits – überspringe Klonen."
fi

echo "[4/4] Moebius-Deps (torch/diffusers …) ins venv installieren (Backend: ${TORCH_BACKEND}) …"
if [[ -f Moebius/requirements.txt ]]; then
    # Moebius pinnt 'torch==2.7.1+cu130' – diesen Build gibt es NICHT (cu130 erst ab
    # torch 2.9). Wir strippen den +cuXXX-Suffix; den passenden CUDA-Build wählt dann
    # uv über --torch-backend (Default cu128, Blackwell-tauglich).
    SAN="Moebius/requirements.uv.txt"
    sed -E 's/(^torch==[0-9][0-9.]*)\+cu[0-9]+/\1/' Moebius/requirements.txt > "${SAN}"
    echo "      bereinigte torch-Zeile: $(grep -E '^torch' "${SAN}")"
    uv pip install --python "${VENV_PY}" \
        --torch-backend="${TORCH_BACKEND}" \
        -r "${SAN}"
else
    echo "      WARNUNG: Moebius/requirements.txt nicht gefunden – bitte manuell prüfen."
fi

cat <<'EOF'

============================================================
Fertig mit der Software-Installation.

NOCH ZU TUN – Modellgewichte (manuell, mehrere GB):
  Von https://huggingface.co/hustvl/Moebius herunterladen und ablegen unter:
    ./Moebius/weight/Moebius/ft_places2/diffusion_pytorch_model.bin
    ./Moebius/weight/Moebius/ft_celebahq/diffusion_pytorch_model.bin
    ./Moebius/weight/Moebius/ft_ffhq/diffusion_pytorch_model.bin
  VAE von https://huggingface.co/hustvl/PixelHacker/tree/main/vae nach:
    ./Moebius/weight/vae/

  Beispiel:
    uvx --from "huggingface_hub[cli]" hf download hustvl/Moebius \
        --local-dir ./Moebius/weight/Moebius

Server starten:
    ./server/run_server.sh              # echter Modus
    ./server/run_server.sh --mock       # ohne Modell, nur zum Testen
============================================================
EOF
