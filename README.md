# Moebius Inpainting für GIMP

Ein GIMP-3.x-Plugin, das die aktuelle Auswahl mit dem Inpainting-Modell
[**Moebius**](https://github.com/hustvl/Moebius) (ECCV 2026, Apache-2.0) neu füllt.

> Es gab bisher **kein** GIMP-Plugin für Moebius. Dieses Projekt liefert es –
> als Client-Server-Lösung, damit das schwere Modell (PyTorch/CUDA) **nicht** im
> GIMP-Prozess laufen muss.

## Architektur

```
GIMP 3.2 (lokal)                     Moebius-API (lokal oder Remote-GPU)
┌──────────────────────┐  HTTP/JSON  ┌────────────────────────────────┐
│ moebius_inpaint.py   │ ──image───▶ │ server.py (FastAPI/uvicorn)    │
│ (GIMP-GI-Plugin)     │ ──mask────▶ │  └─ moebius_backend.py         │
│  • Auswahl → Maske   │ ◀─result─── │      └─ Moebius build_pipeline │
│  • Ergebnis = Ebene  │   (PNG b64) │          (CUDA wenn vorhanden) │
└──────────────────────┘             └────────────────────────────────┘
```

Das Plugin braucht **nur** GIMP (Python-Standardbibliothek). Server und Modell
laufen getrennt – ideal: GIMP auf dem Arbeitsrechner, der Server auf einer
GPU-Maschine.

## Komponenten

| Pfad | Zweck |
|------|-------|
| `plug-in/moebius_inpaint/moebius_inpaint.py` | Das GIMP-Plugin |
| `server/server.py` | FastAPI-Server (`/inpaint`, `/health`, `/models`) |
| `server/moebius_backend.py` | Adapter zur Moebius-Inferenz (+ `--mock`) |
| `pyproject.toml` / `uv.lock` | Server-Abhängigkeiten (von **uv** verwaltet) |
| `install_plugin.sh` | Plugin ins GIMP-User-Verzeichnis kopieren |
| `install_backend.sh` | venv (uv) + Moebius + Deps einrichten |
| `server/run_server.sh` | Server starten (`uv run`) |
| `scripts/remote_probe.sh` | GPU-Rechner diagnostizieren (CUDA/Python/uv) |

> **Abhängigkeiten:** Das Projekt nutzt [**uv**](https://docs.astral.sh/uv/).
> Die Server-Deps stehen in `pyproject.toml` (kein `requirements.txt` mehr).
> Wichtig: torch 2.7.x hat **keine** Wheels für Python 3.13/3.14 – das Backend-venv
> wird deshalb mit Python **3.12** erzeugt (uv lädt es bei Bedarf selbst).

## Schnellstart

### 1. Plugin installieren (auf dem GIMP-Rechner)

```bash
./install_plugin.sh
```

Installiert ins versionsrichtige Verzeichnis (z. B. `~/.config/GIMP/3.2/plug-ins/`,
abgeleitet aus `gimp --version`). GIMP neu starten.

### 2a. Schnelltest ohne GPU/Modell (Mock)

Braucht nur die leichten Server-Deps (kein torch) – läuft auf jedem Python ≥3.10:

```bash
uv run --no-sync python server/server.py --mock
# oder gleichwertig:
./server/run_server.sh --mock
```

Der Mock-Server liefert das Originalbild mit rot markiertem Auswahlbereich
zurück – damit lässt sich der gesamte Pfad GIMP → Server → GIMP prüfen, bevor
das echte Modell installiert ist.

### 2b. Echtes Inpainting (auf dem GPU-Rechner)

```bash
git clone https://github.com/Daniel-Steinberger/moebius-gimp.git
cd moebius-gimp
./install_backend.sh        # uv: Python 3.12 + Server-Deps + Moebius + torch/diffusers
# danach Modellgewichte von Hugging Face laden (siehe Skript-Ausgabe)
./server/run_server.sh
```

Das Skript setzt standardmäßig `TORCH_BACKEND=cu128` – das ist der korrekte
CUDA-Build für Moebius (torch 2.7.1) auf modernen NVIDIA-GPUs **inkl. Blackwell
(RTX 50xx)**. Moebius pinnt im Repo fälschlich `torch==2.7.1+cu130` (diesen Build
gibt es nicht – `cu130` erst ab torch 2.9); das Skript entfernt den Suffix
automatisch. Für eine andere CUDA-Version gezielt überschreiben, z. B.:

```bash
TORCH_BACKEND=cu126 ./install_backend.sh
```

### 3. In GIMP benutzen

1. Bild öffnen.
2. Mit einem Auswahlwerkzeug (Lasso, Zauberstab, Rechteck …) die zu füllende
   Region markieren.
3. **Filter ▸ Moebius ▸ Moebius Inpainting…**
4. Server-URL, Modell und Parameter prüfen, **OK**.
5. Das Ergebnis erscheint als neue Ebene **Moebius Inpaint** über dem Original.

## Remote-GPU nutzen (GIMP lokal, Modell auf GPU-Server)

Auf dem GPU-Server den Backend einrichten und starten (`--host 0.0.0.0`, Default).
Dann zwei Möglichkeiten:

**Empfohlen – SSH-Tunnel** (kein offener Port nötig):

```bash
ssh -L 8765:localhost:8765 user@gpu-host
# Server auf gpu-host lauscht auf 8765; lokal ist er als 127.0.0.1:8765 sichtbar.
```

Im Plugin dann `http://127.0.0.1:8765` als Server-URL belassen.

**Direkt im LAN:** Im Plugin `http://gpu-host:8765` eintragen.

> ⚠️ **Sicherheit:** Die API ist **nicht** authentifiziert und nicht
> verschlüsselt. Nur in einem vertrauenswürdigen Netz oder über einen
> SSH-Tunnel betreiben. Den Port nicht ins offene Internet stellen.

## Parameter im Dialog

| Parameter | Bedeutung |
|-----------|-----------|
| Server-URL | Adresse der Moebius-API |
| Modell | `places2` (Szenen), `celebahq`/`ffhq` (Gesichter), `pretrained` |
| Guidance (cfg) | Classifier-free-Guidance-Skala |
| Schritte | Diffusionsschritte (mehr = langsamer, ggf. besser) |
| Paste | Pixel außerhalb der Auswahl unverändert lassen |
| Compensate | Farb-/Helligkeitsangleich am Rand |
| Timeout | HTTP-Timeout (CPU-Inferenz kann lange dauern) |

## Hinweise / Annahmen

- Die exakte Aufrufsignatur von Moebius (`pipe(...)`, Eingabeauflösung) ist nicht
  vollständig öffentlich dokumentiert. Die gesamte Kopplung steckt deshalb in
  **einer** Funktion `run_inpaint(...)` in `server/moebius_backend.py`; falls die
  installierte Moebius-Version abweicht, ist das die einzige anzupassende Stelle.
- Ohne CUDA-GPU läuft Moebius auf CPU – funktioniert, ist aber deutlich langsamer.
- Das Plugin zielt auf das API-Verzeichnis `GIMP/3.0` und läuft mit GIMP 3.0 und
  3.2 (und voraussichtlich späteren 3.x).

## Lizenz

Dieses Glue-Projekt: Apache-2.0 (kompatibel zu Moebius). Modell und Gewichte
unterliegen den Bedingungen von hustvl/Moebius auf Hugging Face.
