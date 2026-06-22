"""
server.py
=========

Schlanker HTTP-Server (FastAPI), der die Moebius-Inpainting-Inferenz als API
bereitstellt. Wird vom GIMP-Plugin (moebius_inpaint.py) angesprochen.

Endpoints
---------
GET  /health  -> {"status": "ok", "device": "cuda|cpu|mock", "mock": bool}
GET  /models  -> {"models": [...], "default": "places2"}
POST /inpaint -> Inpainting; Body und Antwort siehe Modelle unten.

Start
-----
    # echter Modus (braucht Moebius + torch + Gewichte):
    uvicorn server:app --host 0.0.0.0 --port 8765
    # oder via run_server.sh

    # Mock-Modus (ohne torch/GPU, nur zum Testen des Pfades):
    MOEBIUS_MOCK=1 uvicorn server:app --port 8765
    # oder:  python server.py --mock

Sicherheit
----------
Die API ist NICHT authentifiziert. Sie nur in einem vertrauenswürdigen Netz
oder über einen SSH-Tunnel betreiben (siehe README).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import traceback

from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "FastAPI/pydantic fehlen. Bitte 'pip install -r requirements.txt' im "
        "Backend-venv ausführen.\n"
    )
    raise

import moebius_backend as mb


# ---------------------------------------------------------------------------
# Request-/Response-Modelle
# ---------------------------------------------------------------------------

class InpaintRequest(BaseModel):
    image_png_b64: str = Field(..., description="Quellbild als base64-kodiertes PNG")
    mask_png_b64: str = Field(..., description="Maske als base64-PNG, weiß=füllen")
    model: str = Field(mb.DEFAULT_VARIANT, description="Modell-Variante")
    cfg: float = Field(2.5, description="Guidance scale")
    num_steps: int = Field(20, ge=1, le=200)
    paste: bool = True
    compensate: bool = False
    noise_offset: float = 0.0357


class InpaintResponse(BaseModel):
    image_png_b64: str
    device: str


app = FastAPI(title="Moebius Inpainting API", version="1.0")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _decode_png(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Ungültiges PNG: {exc}") from exc


def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "device": mb.device_info(),
        "mock": mb.is_mock(),
    })


@app.get("/models")
def models() -> JSONResponse:
    return JSONResponse({
        "models": mb.available_models(),
        "default": mb.DEFAULT_VARIANT,
    })


@app.get("/gpu")
def gpu() -> JSONResponse:
    """VRAM-Status (frei/gesamt + von diesem Prozess belegt)."""
    return JSONResponse(mb.gpu_mem_dict())


@app.post("/inpaint", response_model=InpaintResponse)
def inpaint(req: InpaintRequest) -> InpaintResponse:
    image = _decode_png(req.image_png_b64)
    mask = _decode_png(req.mask_png_b64)

    params = mb.InpaintParams(
        model=req.model,
        cfg=req.cfg,
        num_steps=req.num_steps,
        paste=req.paste,
        compensate=req.compensate,
        noise_offset=req.noise_offset,
    )
    try:
        result = mb.run_inpaint(image, mask, params)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # Erwartbare, gut erklärbare Fehler (fehlende Gewichte, falsche Config …).
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - alles andere als 500 melden
        # Volles Traceback auf die Server-Konsole, damit die echte Ursache sichtbar ist.
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        sys.stderr.flush()
        raise HTTPException(
            status_code=500,
            detail=f"Inferenzfehler: {type(exc).__name__}: {exc}",
        ) from exc

    return InpaintResponse(image_png_b64=_encode_png(result), device=mb.device_info())


# ---------------------------------------------------------------------------
# Direktstart:  python server.py [--mock] [--host ...] [--port ...]
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Moebius Inpainting API server")
    parser.add_argument("--mock", action="store_true",
                        help="ohne torch/GPU starten (nur Pfad-Test)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.mock:
        os.environ["MOEBIUS_MOCK"] = "1"

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
