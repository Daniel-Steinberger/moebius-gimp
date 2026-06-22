#!/usr/bin/env bash
# Richtet das Backend (Moebius-API-Server) ein.
#
# Was dieses Skript tut:
#   1. legt ein venv (.venv) an und installiert die SERVER-Abhängigkeiten
#      (fastapi/uvicorn/pillow – siehe server/requirements.txt)
#   2. klont das Moebius-Repo (falls nicht vorhanden) nach ./Moebius
#   3. installiert die MOEBIUS-eigenen Abhängigkeiten (torch, diffusers, …)
#
# Was dieses Skript NICHT tut (bewusst – mehrere GB):
#   - die Modellgewichte von Hugging Face herunterladen  -> siehe Hinweis unten
#
# Für GPU-Betrieb auf diesem oder einem entfernten Rechner ausführen.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
MOEBIUS_REPO="${MOEBIUS_REPO:-https://github.com/hustvl/Moebius.git}"

echo "[1/3] venv anlegen + Server-Deps installieren …"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r server/requirements.txt

echo "[2/3] Moebius-Quelle bereitstellen …"
if [[ ! -d Moebius ]]; then
    git clone "$MOEBIUS_REPO" Moebius
else
    echo "      ./Moebius existiert bereits – überspringe Klonen."
fi

echo "[3/3] Moebius-Abhängigkeiten installieren (torch, diffusers …) …"
if [[ -f Moebius/requirements.txt ]]; then
    echo "      Hinweis: Für CUDA ggf. den passenden torch-Build von pytorch.org wählen."
    pip install -r Moebius/requirements.txt
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

  Beispiel mit huggingface-cli:
    pip install -U "huggingface_hub[cli]"
    huggingface-cli download hustvl/Moebius --local-dir ./Moebius/weight/Moebius

Server starten:
    ./server/run_server.sh              # echter Modus
    ./server/run_server.sh --mock       # ohne Modell, nur zum Testen
============================================================
EOF
