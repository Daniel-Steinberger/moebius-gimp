#!/usr/bin/env bash
# Installiert das GIMP-Plugin in das versionsrichtige User-Plugin-Verzeichnis.
#
# Wichtig: GIMP legt sein User-Verzeichnis nach der STABILEN Version an
# (z. B. ~/.config/GIMP/3.2 für GIMP 3.2.x), NICHT nach der GIR-API-Version 3.0.
# Dieses Skript ermittelt die Version aus dem gimp-Binary und wählt das passende
# Verzeichnis. Mit GIMP_PLUGIN_DIR lässt sich das Ziel komplett überschreiben.
set -euo pipefail
cd "$(dirname "$0")"

# gimp-Binary finden.
GIMP_BIN="${GIMP_BIN:-}"
if [[ -z "$GIMP_BIN" ]]; then
    for cand in gimp-3.2 gimp-3.0 gimp; do
        if command -v "$cand" >/dev/null 2>&1; then GIMP_BIN="$cand"; break; fi
    done
fi
if [[ -z "$GIMP_BIN" ]]; then
    echo "FEHLER: kein gimp-Binary gefunden (gimp-3.2/gimp-3.0/gimp)." >&2
    exit 1
fi

CONFIG_BASE="${XDG_CONFIG_HOME:-$HOME/.config}"

if [[ -n "${GIMP_PLUGIN_DIR:-}" ]]; then
    DEST_BASE="$GIMP_PLUGIN_DIR"
else
    # Version "3.2.2" -> "3.2"
    VER="$("$GIMP_BIN" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    MAJMIN="${VER%.*}"
    if [[ -z "$MAJMIN" ]]; then
        echo "FEHLER: GIMP-Version nicht erkannt." >&2
        exit 1
    fi
    DEST_BASE="$CONFIG_BASE/GIMP/$MAJMIN/plug-ins"
fi

DEST="$DEST_BASE/moebius_inpaint"
mkdir -p "$DEST"
cp plug-in/moebius_inpaint/moebius_inpaint.py "$DEST/"
chmod +x "$DEST/moebius_inpaint.py"

echo "[moebius] GIMP-Binary : $GIMP_BIN ($("$GIMP_BIN" --version 2>/dev/null | head -1))"
echo "[moebius] Plugin nach : $DEST"
echo "[moebius] GIMP neu starten. Menü:  Filter ▸ Moebius ▸ Moebius Inpainting…"
