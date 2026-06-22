"""
moebius_backend.py
==================

Dünner Adapter zwischen dem HTTP-Server (server.py) und der eigentlichen
Moebius-Inferenz (https://github.com/hustvl/Moebius, Apache-2.0).

Designentscheidung
-------------------
Die gesamte Kopplung an Moebius steckt in ``_PipelineCache._build`` (Modell-Bau)
und ``run_inpaint`` (Aufruf). Verifiziert gegen Moebius (Stand Juni 2026):

* ``build_pipeline`` liegt in ``infer/utils.py`` und braucht nur
  ``model_config``, ``model_weight`` und ``device``.
* Das Ergebnis ist eine ``removal.v1_2.pipeline.RemovalSDXLPipeline_BatchMode``.
  Deren ``__call__(input_image_list, input_mask_list, image_size=512,
  num_steps, guidance_scale, paste, compensate, noise_offset, …)`` nimmt
  Listen von **PIL-Bildern** und liefert eine Liste von **PIL-Ergebnissen**
  (kein Dataloader nötig).
* Maskenpolarität: weiß (255) = neu füllen (intern ``masked = image*(1-mask)``).
* ``moebius.yaml`` referenziert das VAE relativ (``./weight/vae``) -> beim
  Modell-Bau wird ins Moebius-Verzeichnis gewechselt.

    args = SimpleNamespace(model_config=..., model_weight=..., device=...)
    pipe = build_pipeline(args)               # einmal, gecacht
    out  = pipe([image], [mask], image_size=512, num_steps=..., guidance_scale=...,
                paste=..., compensate=..., noise_offset=...)[0]

Mock-Modus
----------
Wird der Server mit ``--mock`` gestartet (Umgebungsvariable
``MOEBIUS_MOCK=1``), wird Moebius gar nicht importiert. Stattdessen liefert
``run_inpaint`` ein Platzhalter-Ergebnis (Originalbild mit halbtransparent
eingefärbtem Maskenbereich). Damit lässt sich der komplette Pfad
GIMP -> HTTP -> Server -> GIMP ohne torch/GPU testen.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace
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
    # Defaults entsprechen den Moebius-Inferenz-Defaults (infer/utils.py).
    model: str = DEFAULT_VARIANT
    cfg: float = 2.5          # guidance_scale
    num_steps: int = 20       # Anzahl Diffusionsschritte
    paste: bool = True        # Original außerhalb der Maske wieder einsetzen
    compensate: bool = False  # Farb-/Helligkeitskompensation
    noise_offset: float = 0.0357


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
        else:
            sys.stderr.write("[moebius] " + gpu_mem_report() + "\n")

        weight_path = os.path.join(
            WEIGHT_ROOT, "Moebius", MODEL_VARIANTS[variant],
            "diffusion_pytorch_model.bin",
        )
        model_cfg = os.path.join(MOEBIUS_SRC, "config", "model_cfg", "moebius.yaml")
        vae_path = os.path.join(WEIGHT_ROOT, "vae")

        # Klar verständliche Fehler, BEVOR Moebius mit kryptischen Meldungen abbricht.
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Modellgewicht nicht gefunden: {weight_path}\n"
                f"Bitte die Gewichte von Hugging Face laden (siehe install_backend.sh). "
                f"MOEBIUS_WEIGHTS={WEIGHT_ROOT}"
            )
        if not os.path.isdir(vae_path):
            raise FileNotFoundError(
                f"VAE-Verzeichnis nicht gefunden: {vae_path}\n"
                f"VAE von huggingface.co/hustvl/PixelHacker/tree/main/vae dorthin laden."
            )
        if not os.path.isfile(model_cfg):
            raise FileNotFoundError(f"Model-Config nicht gefunden: {model_cfg}")

        sys.stderr.write(
            f"[moebius] Lade Variante '{variant}' (device={self.device})\n"
            f"[moebius]   weight={weight_path}\n[moebius]   vae={vae_path}\n"
        )

        # --- Integrationspunkt zu Moebius -----------------------------------
        # build_pipeline liegt in infer/utils.py (Paket 'infer') und braucht nur
        # model_config, model_weight und device. Das Ergebnis ist eine
        # RemovalSDXLPipeline_BatchMode, deren __call__ direkt Listen von
        # PIL-Bildern annimmt (siehe run_inpaint). Die Sampling-Parameter
        # (cfg, num_steps, paste, …) werden NICHT hier, sondern pro Request im
        # pipe(...)-Aufruf übergeben.
        from infer.utils import build_pipeline  # type: ignore

        args = SimpleNamespace(
            model_config=model_cfg,
            model_weight=weight_path,
            device=self.device,
        )

        # moebius.yaml referenziert das VAE relativ als './weight/vae'. Für den
        # Modell-Bau deshalb ins Moebius-Verzeichnis wechseln (sonst FileNotFound).
        # Danach cwd wiederherstellen – die Inferenz selbst braucht keinen cwd.
        prev_cwd = os.getcwd()
        os.chdir(MOEBIUS_SRC)
        try:
            pipe = build_pipeline(args)
        finally:
            os.chdir(prev_cwd)

        sys.stderr.write("[moebius] Modell geladen. " + gpu_mem_report() + "\n")
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


def gpu_mem_report() -> str:
    """Kompakter VRAM-Bericht (für /gpu-Endpoint und Logs)."""
    if is_mock():
        return "mock – kein GPU-Speicher belegt"
    try:
        import torch
    except ImportError:
        return "torch nicht installiert"
    if not torch.cuda.is_available():
        return "keine CUDA-GPU sichtbar"
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)        # tatsächlich freier/gesamter VRAM
    reserved = torch.cuda.memory_reserved(i)
    allocated = torch.cuda.memory_allocated(i)
    gb = 1024 ** 3
    return (
        f"GPU{i} {name}: frei {free/gb:.2f}/{total/gb:.2f} GiB | "
        f"von diesem Prozess belegt: allocated {allocated/gb:.2f}, "
        f"reserved {reserved/gb:.2f} GiB"
    )


def gpu_mem_dict() -> dict:
    if is_mock():
        return {"mock": True}
    try:
        import torch
    except ImportError:
        return {"torch": False}
    if not torch.cuda.is_available():
        return {"cuda": False}
    i = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(i)
    gb = 1024 ** 3
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(i),
        "free_gib": round(free / gb, 2),
        "total_gib": round(total / gb, 2),
        "process_allocated_gib": round(torch.cuda.memory_allocated(i) / gb, 2),
        "process_reserved_gib": round(torch.cuda.memory_reserved(i) / gb, 2),
        "model_loaded": _CACHE._pipe is not None,
        "loaded_variant": _CACHE._variant,
    }


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
    # RemovalSDXLPipeline_BatchMode.__call__(input_image_list, input_mask_list,
    #   image_size=512, num_steps, guidance_scale, paste, compensate, noise_offset, …)
    # nimmt Listen von PIL-Bildern und liefert eine Liste von PIL-Ergebnissen.
    # Die Maske wird intern auf image_size skaliert/binarisiert; weiß = füllen.
    # Die Inferenz wird serialisiert (eine GPU, ein Modell im Cache).
    pipe = _CACHE.get(params.model)
    with _CACHE._lock:
        out = pipe(
            [image],
            [mask],
            image_size=512,
            num_steps=params.num_steps,
            guidance_scale=params.cfg,
            paste=params.paste,
            compensate=params.compensate,
            noise_offset=params.noise_offset,
        )
    result = out[0] if isinstance(out, (list, tuple)) else out
    if not isinstance(result, Image.Image):
        # Falls Moebius einen Tensor/ndarray zurückgibt -> nach PIL wandeln.
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
