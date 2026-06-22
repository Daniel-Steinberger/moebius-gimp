#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moebius Inpainting – GIMP-3.x-Plugin
====================================

Füllt die aktuelle Auswahl eines Bildes mit dem Moebius-Inpainting-Modell
(https://github.com/hustvl/Moebius). Das Plugin selbst rechnet NICHT – es
schickt Bild + Maske per HTTP an einen Moebius-API-Server (siehe ../server/),
der lokal oder auf einem entfernten GPU-Rechner laufen kann.

Workflow im GIMP
----------------
1. Bild öffnen, mit einem Auswahlwerkzeug die zu füllende Region markieren.
2. Filter ▸ Moebius ▸ Moebius Inpainting…
3. Server-URL/Modell/Parameter prüfen, OK.
4. Das Ergebnis erscheint als neue Ebene über dem Original.

Abhängigkeiten: nur Python-Standardbibliothek + GObject-Introspection (GIMP).
Getestet gegen GIMP 3.2 (API-Verzeichnis GIMP/3.0); 3.0-kompatibel.
"""

import base64
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

import gi
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gegl", "0.4")
from gi.repository import Gimp, GimpUi, Gegl, GObject, GLib, Gio  # noqa: E402


PROC_NAME = "plug-in-moebius-inpaint"
DEFAULT_URL = "http://127.0.0.1:8765"
MODELS = [
    ("places2", "Places2 – natürliche Szenen (Standard)"),
    ("celebahq", "CelebA-HQ – Porträts/Gesichter"),
    ("ffhq", "FFHQ – Gesichter"),
    ("pretrained", "Pretrained – generisch"),
]


# ---------------------------------------------------------------------------
# HTTP-Aufruf an den Moebius-Server (nur Stdlib)
# ---------------------------------------------------------------------------

def _post_inpaint(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/inpaint",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# GIMP-Bild/Maske <-> PNG-Bytes
# ---------------------------------------------------------------------------

def _export_png(image, path):
    """Flacht eine Kopie des Bildes ab und exportiert sie als PNG."""
    dup = image.duplicate()
    dup.flatten()
    gfile = Gio.File.new_for_path(path)
    ok = Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, dup, gfile, None)
    dup.delete()
    if not ok:
        raise RuntimeError("Konnte das Bild nicht als PNG exportieren.")


def _export_mask_png(image, path):
    """Rendert die aktuelle Auswahl als schwarz/weiße Maske (weiß = füllen)
    und exportiert sie als PNG. Die Auswahl wird über image.duplicate()
    mitkopiert, das Originalbild bleibt unangetastet."""
    dup = image.duplicate()
    w, h = dup.get_width(), dup.get_height()

    layer = Gimp.Layer.new(
        dup, "mask", w, h,
        Gimp.ImageType.RGB_IMAGE, 100.0, Gimp.LayerMode.NORMAL,
    )
    dup.insert_layer(layer, None, 0)

    # Ganze Ebene schwarz (fill ignoriert die Auswahl) ...
    Gimp.context_set_background(Gegl.Color.new("black"))
    layer.fill(Gimp.FillType.BACKGROUND)
    # ... dann den Auswahlbereich weiß (edit_fill respektiert die Auswahl).
    Gimp.context_set_foreground(Gegl.Color.new("white"))
    layer.edit_fill(Gimp.FillType.FOREGROUND)

    dup.flatten()
    gfile = Gio.File.new_for_path(path)
    ok = Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, dup, gfile, None)
    dup.delete()
    if not ok:
        raise RuntimeError("Konnte die Maske nicht als PNG exportieren.")


def _load_result_as_layer(image, path, name):
    """Lädt das Ergebnis-PNG und fügt es als neue Ebene in das Originalbild ein."""
    gfile = Gio.File.new_for_path(path)
    res_img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, gfile)
    src_layer = res_img.get_layers()[0]
    new_layer = Gimp.Layer.new_from_drawable(src_layer, image)
    new_layer.set_name(name)
    image.insert_layer(new_layer, None, 0)
    res_img.delete()


# ---------------------------------------------------------------------------
# Hauptlogik
# ---------------------------------------------------------------------------

def _do_run(procedure, image, config):
    # --- Auswahl prüfen ----------------------------------------------------
    _ok, non_empty, _x1, _y1, _x2, _y2 = Gimp.Selection.bounds(image)
    if not non_empty:
        Gimp.message(
            "Moebius Inpainting: Es gibt keine Auswahl.\n"
            "Bitte zuerst mit einem Auswahlwerkzeug die zu füllende Region markieren."
        )
        return False

    # --- Parameter aus dem Dialog ------------------------------------------
    server_url = config.get_property("server-url") or DEFAULT_URL
    model = config.get_property("model")
    cfg = float(config.get_property("cfg"))
    num_steps = int(config.get_property("num-steps"))
    paste = bool(config.get_property("paste"))
    compensate = bool(config.get_property("compensate"))
    timeout = int(config.get_property("timeout"))

    tmpdir = tempfile.mkdtemp(prefix="moebius_")
    img_path = os.path.join(tmpdir, "image.png")
    mask_path = os.path.join(tmpdir, "mask.png")
    out_path = os.path.join(tmpdir, "result.png")

    try:
        Gimp.progress_init("Moebius: Bild und Maske exportieren …")
        _export_png(image, img_path)
        _export_mask_png(image, mask_path)

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        with open(mask_path, "rb") as f:
            mask_b64 = base64.b64encode(f.read()).decode("ascii")

        payload = {
            "image_png_b64": img_b64,
            "mask_png_b64": mask_b64,
            "model": model,
            "cfg": cfg,
            "num_steps": num_steps,
            "paste": paste,
            "compensate": compensate,
        }

        Gimp.progress_init("Moebius: Inpainting auf dem Server …")
        try:
            result = _post_inpaint(server_url, payload, timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"Server-Fehler {e.code}: {detail}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Server nicht erreichbar unter {server_url}\n({e.reason}).\n"
                "Läuft der Moebius-API-Server? Stimmt die URL?"
            )

        with open(out_path, "wb") as f:
            f.write(base64.b64decode(result["image_png_b64"]))

        _load_result_as_layer(image, out_path, "Moebius Inpaint")
        Gimp.displays_flush()

        dev = result.get("device", "?")
        Gimp.message(f"Moebius Inpainting fertig (Gerät: {dev}).")
        return True

    finally:
        for p in (img_path, mask_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Plugin-Registrierung
# ---------------------------------------------------------------------------

class MoebiusInpaint(Gimp.PlugIn):

    def do_query_procedures(self):
        return [PROC_NAME]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self, name, Gimp.PDBProcType.PLUGIN, self.run, None
        )
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_menu_label("Moebius Inpainting…")
        procedure.add_menu_path("<Image>/Filters/Moebius")
        procedure.set_documentation(
            "Inpainting der aktuellen Auswahl via Moebius-API",
            "Schickt Bild und Auswahlmaske an einen Moebius-Inpainting-Server "
            "und fügt das Ergebnis als neue Ebene ein.",
            name,
        )
        procedure.set_attribution("Moebius-GIMP", "Apache-2.0", "2026")

        rw = GObject.ParamFlags.READWRITE

        procedure.add_string_argument(
            "server-url", "Server-_URL",
            "URL des Moebius-API-Servers (lokal oder Remote-GPU)",
            DEFAULT_URL, rw,
        )

        choice = Gimp.Choice.new()
        for i, (nick, label) in enumerate(MODELS):
            choice.add(nick, i, label, label)
        procedure.add_choice_argument(
            "model", "_Modell",
            "Modell-Variante / Gewichte",
            choice, MODELS[0][0], rw,
        )

        procedure.add_double_argument(
            "cfg", "_Guidance (cfg)",
            "Classifier-free-Guidance-Skala (Moebius-Default 2.5)",
            0.0, 20.0, 2.5, rw,
        )
        procedure.add_int_argument(
            "num-steps", "_Schritte",
            "Anzahl der Diffusionsschritte",
            1, 200, 20, rw,
        )
        procedure.add_boolean_argument(
            "paste", "_Paste (Original außerhalb der Maske behalten)",
            "Pixel außerhalb der Auswahl unverändert lassen",
            True, rw,
        )
        procedure.add_boolean_argument(
            "compensate", "_Compensate (Farbangleich)",
            "Farb-/Helligkeitskompensation aktivieren",
            False, rw,
        )
        procedure.add_int_argument(
            "timeout", "_Timeout (s)",
            "HTTP-Timeout in Sekunden (CPU-Inferenz kann lange dauern)",
            5, 3600, 600, rw,
        )
        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init(PROC_NAME)
            dialog = GimpUi.ProcedureDialog.new(procedure, config, "Moebius Inpainting")
            dialog.fill(None)
            if not dialog.run():
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )
            dialog.destroy()

        try:
            _do_run(procedure, image, config)
        except Exception as e:  # noqa: BLE001
            Gimp.message(f"Moebius Inpainting: Fehler – {e}")
            return procedure.new_return_values(
                Gimp.PDBStatusType.EXECUTION_ERROR,
                GLib.Error(str(e)),
            )

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


Gimp.main(MoebiusInpaint.__gtype__, sys.argv)
