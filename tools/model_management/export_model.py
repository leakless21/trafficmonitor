#!/usr/bin/env python3
"""
export_model.py
~~~~~~~~~~~~~~~~
Generic exporter for YOLO models saved in the **.pt** format.

The script leverages Ultralytics' native ``model.export`` method to convert a
YOLOv8 (and later) model checkpoint to one or multiple target formats in a
single run.

Supported formats depend on the local runtime environment and the Ultralytics
version, but typically include::

    torchscript, onnx, openvino, engine (TensorRT), xml, tflite, \
    coreml, paddle, mlpackage, ncnn, 

See Ultralytics documentation for the full list.

Examples
--------
Export a model to **ONNX** and **TensorRT** engines::

    python tools/export_model.py --pt_model_path yolov8n.pt \
                                 --formats onnx,engine --imgsz 640

Export to all available formats with FP16, dynamic batch size and NMS embedded::

    python tools/export_model.py --pt_model_path runs/train/exp/weights/best.pt \
                                 --formats all --half --dynamic --nms
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party
import click

# Ultralytics base model import
from ultralytics import YOLO  # type: ignore

# Ultralytics has changed the location/naming of the helper that returns supported
# export formats across versions. We attempt multiple fallbacks so the script works
# regardless of the specific version in the environment.

# pylint: disable=import-error,unused-import
try:  # UL >= 8.2.0
    from ultralytics.engine.exporter import export_formats  # type: ignore
except Exception:  # pragma: no cover
    try:  # UL <= 8.1.0
        from ultralytics.yolo.engine.exporter import export_formats  # type: ignore
    except Exception:
        # Final fallback: derive formats from registry if available, else empty
        def export_formats():  # type: ignore
            return [
                "torchscript",
                "onnx",
                "openvino",
                "engine",
                "coreml",
                "tflite",
                "ncnn",
                "mlpackage",
            ]


# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def _parse_formats(fmt_arg: str) -> List[str]:
    """Return a list of lower-case, stripped export format names.

    If the argument equals "all", return a list of **all** formats supported by
    the current Ultralytics installation.
    """
    if fmt_arg.lower() == "all":
        return sorted(export_formats())  # type: ignore[arg-type]

    parts = [f.strip().lower() for f in fmt_arg.split(",") if f.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one export format must be specified")
    return parts


def _build_export_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct keyword arguments for ``model.export`` from parsed *args*."""
    kw: Dict[str, Any] = {}

    if args.imgsz is not None:
        kw["imgsz"] = args.imgsz
    if args.opset is not None:
        kw["opset"] = args.opset
    if args.device is not None:
        kw["device"] = args.device

    # Boolean flags
    if args.half:
        kw["half"] = True
    if args.int8:
        kw["int8"] = True
    if args.dynamic:
        kw["dynamic"] = True
    if args.simplify:
        kw["simplify"] = True
    if args.nms:
        kw["nms"] = True

    # path is handled per-format below
    return kw


# ----------------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------------


def run_export(
    pt_model_path: str,
    formats: List[str],
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
    opset: Optional[int] = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = False,
    nms: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Core export routine shared by both argparse and Click CLIs."""

    pt_path = Path(pt_model_path)
    if not pt_path.is_file():
        sys.exit(f"[ERROR] Input model not found: {pt_path}")

    # Determine output directory
    out_dir_path = Path(output_dir) if output_dir else pt_path.parent
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = YOLO(str(pt_path))

    # Build kwargs dict
    kw_args = {
        k: v
        for k, v in {
            "imgsz": imgsz,
            "device": device,
            "opset": opset,
            "half": half or None,
            "int8": int8 or None,
            "dynamic": dynamic or None,
            "simplify": simplify or None,
            "nms": nms or None,
        }.items()
        if v is not None
    }

    # Execute exports
    for fmt in formats:
        try:
            click.echo(f"\n[INFO] Exporting to format: {fmt} ...", err=False)
            out_path = model.export(format=fmt, path=str(out_dir_path), **kw_args)
            click.secho(f"[SUCCESS] Exported: {out_path}", fg="green")
        except Exception as exc:
            click.secho(f"[ERROR] Failed to export format '{fmt}': {exc}", fg="red", err=True)


# -----------------------------------------------------------------------------
# Click-based CLI (preferred)
# -----------------------------------------------------------------------------


def _parse_formats_click(_ctx, _param, value: str):  # type: ignore[override]
    """Click callback to split/validate the --formats option."""
    if value.lower() == "all":
        return sorted(export_formats())
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    if not parts:
        raise click.BadParameter("At least one format must be specified")
    return parts


@click.command(name="export-model")
@click.option("--pt-model-path", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to the .pt checkpoint")
@click.option("--formats", default="onnx", callback=_parse_formats_click, help="Comma-separated list of target formats or 'all'.")
@click.option("--imgsz", type=int, default=None, show_default="model default", help="Input image size")
@click.option("--device", type=str, default=None, help="Device for export, e.g. 'cpu' or '0'")
@click.option("--opset", type=int, default=None, help="ONNX opset version")
@click.option("--half", is_flag=True, help="Export using FP16 precision")
@click.option("--int8", is_flag=True, help="INT8 quantization where supported")
@click.option("--dynamic", is_flag=True, help="Enable dynamic batch axes")
@click.option("--simplify", is_flag=True, help="Simplify ONNX graph (requires onnx-simplifier)")
@click.option("--nms", is_flag=True, help="Embed NMS in exported graph where supported")
@click.option("--output-dir", type=click.Path(file_okay=False), default=None, help="Directory to store exported files")
def cli(
    pt_model_path: str,
    formats: List[str],
    imgsz: Optional[int],
    device: Optional[str],
    opset: Optional[int],
    half: bool,
    int8: bool,
    dynamic: bool,
    simplify: bool,
    nms: bool,
    output_dir: Optional[str],
) -> None:
    """Click command wrapper that forwards to ``run_export``."""

    run_export(
        pt_model_path=pt_model_path,
        formats=formats,
        imgsz=imgsz,
        device=device,
        opset=opset,
        half=half,
        int8=int8,
        dynamic=dynamic,
        simplify=simplify,
        nms=nms,
        output_dir=output_dir,
    )


# -----------------------------------------------------------------------------
# Legacy argparse CLI (kept for backwards compatibility)
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="export_model",
        description="Export a YOLO .pt model checkpoint to various formats using Ultralytics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pt_model_path",
        type=str,
        required=True,
        help="Path to the input .pt model (e.g. 'yolov8n.pt' or 'runs/train/exp/weights/best.pt').",
    )
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default="onnx",
        help="Comma-separated list of target formats or the word 'all' for every supported format.",
    )

    # Common export options
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size for export.")
    parser.add_argument("--device", type=str, default=None, help="Device for export (e.g. 'cpu', '0').")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version.")
    parser.add_argument("--half", action="store_true", help="Export the model with FP16 precision.")
    parser.add_argument("--int8", action="store_true", help="Export the model with INT8 quantization (if supported).")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size axes (where supported).")
    parser.add_argument("--simplify", action="store_true", help="Simplify the exported ONNX model (requires onnx-sim).")
    parser.add_argument("--nms", action="store_true", help="Embed NMS into the exported model (where supported).")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save exported models. Defaults to the directory of the .pt file.",
    )

    args = parser.parse_args()

    run_export(
        pt_model_path=args.pt_model_path,
        formats=args.formats,
        imgsz=args.imgsz,
        device=args.device,
        opset=args.opset,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        nms=args.nms,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    # If the script is executed directly, default to the Click CLI for nicer UX.
    # Users can still access the argparse version via ``python -m tools.export_model``.
    cli() 