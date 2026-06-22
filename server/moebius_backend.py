"""
moebius_backend.py
==================

Dünner Adapter zwischen dem HTTP-Server (server.py) und der eigentlichen
Moebius-Inferenz (https://github.com/hustvl/Moebius, Apache-2.0).

Designentscheidung
-------------------
Die Vorverarbeitung (Resize, Normalisierung) und der Pipeline-Aufruf von
Moebius liegen in Modulen (``utils_infer.py``, ``infer/infer_moebius.py``,
``utils_dataset``), deren exakte Signatur nicht vollständig öffentlich
dokumentiert ist. Deshalb wird die *gesamte* Kopplung an Moebius hier in einer
einzigen Funktion ``run_inpaint(image, mask, params)`` gekapselt. Sollte die
echte ``pipe(...)``-Signatur der installierten Moebius-Version abweichen, ist
das die einzige Stelle, die angepasst werden muss.

Die Logik spiegelt ``infer/infer_moebius.py``:

    pipe = build_pipeline(args)
    pipe = functools.partial(
        pipe,
        guidance_scale=args.cfg,
        paste=args.pst,
        compensate=args.cps,
        num_steps=args.num_step,
        noise_offset=args.noise_offset,
    )
    out = pipe([image], [mask])[0]

Mock-Modus
----------
Wird der Server mit ``--mock`` gestartet (Umgebungsvariable
``MOEBIUS_MOCK=1``), wird Moebius gar nicht importiert. Stattdessen liefert
``run_inpaint`` ein Platzhalter-Ergebnis (Originalbild mit halbtransparent
eingefärbtem Maskenbereich). Damit lässt sich der komplette Pfad
GIMP -> HTTP -> Server -> GIMP ohne torch/GPU testen.
"""

from __future__ import annotations

import functools
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

def is_mock() -> bool:
    return os.environ.get("MOEBIUS_MOCK", "0") == "1"


# Pfad zur installierten Moebius-Quelle (geklontes Repo). Kann per Env-Var
# überschrieben werden; Default ist ein Geschwister-Verzeichnis "Moebius".
MOEBIUS_SRC = os.environ.get(
    "MOEBIUS_SRC",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Moebius"),
)

# Verzeichnis mit den heruntergeladenen Gewichten (HF). Erwartete Struktur laut
# README von Moebius:  <WEIGHT_ROOT>/Moebius/{pretrained,ft_places2,ft_celebahq,
# ft_ffhq}/diffusion_pytorch_model.bin  und  <WEIGHT_ROOT>/vae
WEIGHT_ROOT = os.environ.get("MOEBIUS_WEIGHTS", os.path.join(MOEBIUS_SRC, "weight"))

# Verfügbare, mit Moebius mitgelieferte Modell-Varianten -> Unterverzeichnis.
MODEL_VARIANTS = {
    "places2": "ft_places2",       # natürliche Szenen (Default)
    "celebahq": "ft_celebahq",     # Porträts / Gesichter
    "ffhq": "ft_ffhq",             # Gesichter
    "pretrained": "pretrained",    # generisches Vortraining
}
DEFAULT_VARIANT = "places2"


@dataclass
class InpaintParams:
    """Vom Client (GIMP-Plugin) übergebene Inferenz-Parameter."""
    model: str = DEFAULT_VARIANT
    cfg: float = 1.0          # guidance_scale
    num_steps: int = 20       # Anzahl Diffusionsschritte
    paste: bool = True        # Original außerhalb der Maske wieder einsetzen
    compensate: bool = True   # Farb-/Helligkeitskompensation
    noise_offset: float = 0.0


# ---------------------------------------------------------------------------
# Lazy-geladene, gecachte Pipeline (echter Modus)
# ---------------------------------------------------------------------------

class _PipelineCache:
    """Hält genau eine geladene Moebius-Pipeline. Bei Variantenwechsel wird neu
    geladen. Thread-sicher, da der Server ggf. parallel Anfragen annimmt."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._variant: Optional[str] = None
        self._pipe = None
        self.device: str = "cpu"

    def get(self, variant: str):
        with self._lock:
            if self._pipe is not None and self._variant == variant:
                return self._pipe
            self._pipe = self._build(variant)
            self._variant = variant
            return self._pipe

    def _build(self, variant: str):
        if variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unbekannte Modell-Variante '{variant}'. "
                f"Verfügbar: {', '.join(MODEL_VARIANTS)}"
            )

        # Moebius-Quelle importierbar machen.
        if MOEBIUS_SRC not in sys.path:
            sys.path.insert(0, MOEBIUS_SRC)

        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - umgebungsabhängig
            raise RuntimeError(
                "torch ist nicht installiert. Bitte die Moebius-Abhängigkeiten "
                "im Backend-venv installieren (siehe install_backend.sh) oder den "
                "Server mit --mock starten."
            ) from exc

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            sys.stderr.write(
                "[moebius] WARNUNG: Keine CUDA-GPU gefunden -> CPU-Inferenz. "
                "Das ist deutlich langsamer (Sekunden bis Minuten pro Bild).\n"
            )

        # --- Integrationspunkt zu Moebius -----------------------------------
        # Diese Importe/Aufrufe spiegeln infer/infer_moebius.py. Falls die
        # installierte Moebius-Version andere Namen/Signaturen nutzt, hier
        # anpassen (und sonst nirgends).
        from utils_infer import build_pipeline, get_batch_infer_args  # type: ignore

        weight_path = os.path.join(
            WEIGHT_ROOT, "Moebius", MODEL_VARIANTS[variant],
            "diffusion_pytorch_model.bin",
        )
        model_cfg = os.path.join(MOEBIUS_SRC, "config", "model_cfg", "moebius.yaml")

        args = get_batch_infer_args()
        # Pflichtfelder analog zur CLI in infer_moebius.py setzen:
        args.model_config = model_cfg
        args.model_weight = weight_path
        if hasattr(args, "vae"):
            args.vae = os.path.join(WEIGHT_ROOT, "vae")
        if hasattr(args, "device"):
            args.device = self.device

        pipe = build_pipeline(args)
        pipe = functools.partial(
            pipe,
            guidance_scale=getattr(args, "cfg", 1.0),
            paste=getattr(args, "pst", True),
            compensate=getattr(args, "cps", True),
            num_steps=getattr(args, "num_step", 20),
            noise_offset=getattr(args, "noise_offset", 0.0),
        )
        return pipe


_CACHE = _PipelineCache()


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

def available_models() -> list[str]:
    return list(MODEL_VARIANTS.keys())


def device_info() -> str:
    if is_mock():
        return "mock"
    return _CACHE.device


def run_inpaint(image: Image.Image, mask: Image.Image, params: InpaintParams) -> Image.Image:
    """Führt das Inpainting aus und liefert das Ergebnisbild (RGB).

    image : RGB-Bild
    mask  : 8-bit-Graustufenmaske, weiß (255) = neu füllen, schwarz = behalten
    """
    image = image.convert("RGB")
    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    if is_mock():
        return _mock_inpaint(image, mask)

    # --- echter Moebius-Aufruf ------------------------------------------------
    pipe = _CACHE.get(params.model)
    # Parameter pro Aufruf überschreiben (functools.partial-Defaults).
    out = pipe(
        [image],
        [mask],
        guidance_scale=params.cfg,
        paste=params.paste,
        compensate=params.compensate,
        num_steps=params.num_steps,
        noise_offset=params.noise_offset,
    )
    result = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(result, Image.Image):
        # Moebius könnte einen Tensor/ndarray zurückgeben -> nach PIL wandeln.
        result = _to_pil(result)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _to_pil(obj) -> Image.Image:
    """Best-effort-Konvertierung Tensor/ndarray -> PIL.Image."""
    arr = obj
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().float().numpy()
    except ImportError:
        pass
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _mock_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Platzhalter-Ergebnis ohne ML: maskierter Bereich wird halbtransparent
    rot überlagert, damit man im GIMP klar sieht, dass die Maske korrekt
    übertragen wurde."""
    base = image.copy()
    overlay = Image.new("RGB", image.size, (220, 40, 40))
    # mask als Alpha (weiß=voll überlagern)
    blended = Image.composite(
        Image.blend(base, overlay, 0.5), base, mask
    )
    return blended
