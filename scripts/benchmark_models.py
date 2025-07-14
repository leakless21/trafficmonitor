#!/usr/bin/env python3
"""
benchmark_models.py
~~~~~~~~~~~~~~~~~~~
Run Ultralytics benchmark for both vehicle- and plate-detection
models stored under the `Results/` directory.

Folder layout expected::

    Results/
    ├─ datasets/
    │  ├─ vehicles/data.yaml
    │  └─ plates/data.yaml
    ├─ Vehicles/           # vehicle detection model groups
    │  ├─ YOLOv8n/
    │  │   └─ runs/detect/train/weights/best.pt
    │  └─ ...
    └─ License Plate/      # plate detection model groups
       ├─ YOLOv8n/
       │   └─ runs/detect/train/weights/best.pt
       └─ ...

For every architecture folder found under `Vehicles/` or
`License Plate/`, the script looks for a `best.pt` or any `.pt` weight
file and benchmarks it against the corresponding dataset using
`ultralytics.utils.benchmarks.benchmark`.

Benchmark results are written as JSON next to the weight folder, e.g.::

    Results/Vehicles/YOLOv8n/benchmark_vehicles.json

Usage::

    python scripts/benchmark_models.py \
        --results_root Results --device 0 --imgsz 640 --half

Requirements:

    pip install ultralytics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import csv
import re
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

# Ultralytics benchmark utility
from ultralytics.utils.benchmarks import benchmark

import contextlib
import io

# Ultralytics model import for params/FLOPs extraction
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # if not available we will skip param/FLOPs

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # we will degrade gracefully


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def find_weight_file(model_dir: Path) -> Optional[Path]:
    """Return the first weight file (.pt) found inside *model_dir*.

    Preference order:
        1. any file named exactly 'best.pt' (deep search)
        2. any file named exactly 'last.pt' (deep search)
        3. the first *.pt file found during traversal
    """
    best, last, fallback = None, None, None
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if not f.endswith(".pt"):
                continue
            p = Path(root) / f
            if f == "best.pt" and best is None:
                best = p
            elif f == "last.pt" and last is None:
                last = p
            elif fallback is None:
                fallback = p
    return best or last or fallback


def list_arch_dirs(root: Path) -> Dict[str, Path]:
    """Return mapping of architecture name -> directory under *root* (non-files)."""
    dirs = {}
    if not root.is_dir():
        return dirs
    for p in root.iterdir():
        if p.is_dir():
            dirs[p.name] = p
    return dirs


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_path(base_dir: Path, p: str | Path) -> Path:
    """Resolve *p* relative to *base_dir* if not absolute."""
    if not isinstance(p, Path):
        p = Path(p)
    # store is_absolute as a local variable for faster access
    if p.is_absolute():
        return p
    return base_dir / p


def _has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for file in directory.iterdir():
        if file.suffix.lower() in IMG_EXTS:
            return True
    return False


def dataset_ready(data_yaml: Path, verbose: bool = True) -> bool:
    """Return True if dataset YAML points to existing image folders with content."""
    if yaml is None:
        # can't parse; assume ready
        return True

    try:
        ydata = yaml.safe_load(data_yaml.read_text())
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[WARN] Failed to parse YAML {data_yaml}: {e}\n")
        return False

    base_dir = data_yaml.parent
    keys: List[str] = []
    for k in ("train", "val", "test"):
        if k in ydata:
            keys.append(k)

    ok = True
    for k in keys:
        resolved = _resolve_path(base_dir, ydata[k])
        img_dir = Path(resolved)
        if not _has_images(img_dir):
            if verbose:
                sys.stderr.write(f"[WARN] Dataset missing images: {k} -> {img_dir}\n")
            ok = False
    return ok


# -----------------------------------------------------------------------------
# Benchmark execution
# -----------------------------------------------------------------------------

def run_benchmarks(
    results_root: Path,
    device: str | int,
    imgsz: int,
    half: bool,
    verbose: bool = True,
    override_datasets: Optional[Dict[str, Optional[Path]]] = None,
    run_log_stream: Optional[io.TextIOBase] = None,
) -> None:
    """Execute benchmarks for vehicle and plate models.

    Parameters
    ----------
    results_root : Path
        Root directory containing `Vehicles/`, `License Plate/`, and optionally `datasets/`.
    device : str | int
        Device identifier passed to Ultralytics (e.g. "0", "cpu").
    imgsz : int
        Inference image size.
    half : bool
        Whether to enable FP16.
    verbose : bool, default True
        Whether to print progress information.
    override_datasets : dict, optional
        Mapping `{"vehicles": Path|None, "plates": Path|None}` allowing callers to
        explicitly specify dataset YAML paths. If a value is ``None``, default search
        behaviour is used for that task.
    """

    def _parse_stdout_table(stdout: str) -> List[Dict[str, Any]]:
        """Parse benchmark stdout and return list of rows for **all export formats**.

        Each returned dict contains:
            format, status, size_mb, map50_95, latency_ms, fps
        """

        rows: List[Dict[str, Any]] = []
        # matches: idx   Format   status ✓   size   mAP   latency   fps
        pattern = re.compile(r"^\s*\d+\s+(.+?)\s+(✅|❌|❎)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)")

        for line in stdout.splitlines():
            m = pattern.match(line)
            if not m:
                continue
            fmt, status, size, mAP, latency, fps = m.groups()
            # Convert dashes to NaNs/None
            def _float_or_none(s: str) -> Optional[float]:
                return None if s in {"-", ""} else float(s)

            rows.append({
                "format": fmt.strip(),
                "status": status,
                "size_mb": _float_or_none(size),
                "map50_95": _float_or_none(mAP),
                "latency_ms": _float_or_none(latency),
                "fps": _float_or_none(fps),
            })
        return rows

    override_datasets = override_datasets or {}

    # Fallback dataset locations
    datasets = {
        "vehicles": results_root / "datasets" / "vehicles" / "data.yaml",
        "plates": results_root / "datasets" / "plates" / "data.yaml",
    }

    # Apply overrides when given
    for k, v in override_datasets.items():
        if v is not None:
            datasets[k] = v

    tasks = [
        ("vehicles", results_root / "Vehicles"),
        ("plates", results_root / "License Plate"),
    ]

    summary_rows: List[Dict[str, Any]] = []  # accumulate per-architecture all-format metrics (plus model info)

    for task_name, models_root in tasks:
        data_yaml = datasets[task_name]
        if not data_yaml.exists():
            sys.stderr.write(f"[WARN] Dataset YAML not found for {task_name}: {data_yaml}\n")
            continue

        if not dataset_ready(data_yaml, verbose):
            sys.stderr.write(f"[WARN] Skipping {task_name} benchmarks due to incomplete dataset ({data_yaml})\n")
            continue

        arch_dirs = list_arch_dirs(models_root)
        if not arch_dirs:
            sys.stderr.write(f"[WARN] No architectures found under {models_root}\n")
            continue

        for arch, arch_dir in arch_dirs.items():
            weight_path = find_weight_file(arch_dir)
            if weight_path is None:
                if verbose:
                    print(f"[SKIP] {arch} ({task_name}): no .pt weights found in {arch_dir}")
                continue

            if verbose:
                print(f"[BENCH] {arch} ({task_name}) → {weight_path.relative_to(results_root)}")

            # Capture benchmark console output.
            # We need three copies:
            #   1. normal console (handled by outer Tee)
            #   2. global run_log_file (also handled by outer Tee)
            #   3. per-architecture in-memory buffer so we can parse + save later.
            # Therefore our inner capture only needs to duplicate to the *previous* stdout/stderr
            # (which is already Tee-d to console + run log) and to the StringIO buffer.

            stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()

            prev_stdout, prev_stderr = sys.stdout, sys.stderr  # save handles prior to redirect

            class _TeeCapture(io.TextIOBase):
                def __init__(self, buffer: io.StringIO, passthrough: io.TextIOBase):
                    self._buf = buffer
                    self._passthrough = passthrough

                def write(self, s: str):  # type: ignore[override]
                    self._buf.write(s)
                    self._passthrough.write(s)
                    return len(s)

                def flush(self):
                    self._buf.flush()
                    self._passthrough.flush()

            tee_out = _TeeCapture(stdout_buffer, prev_stdout)
            tee_err = _TeeCapture(stderr_buffer, prev_stderr)

            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                metrics: Any = benchmark(
                    model=str(weight_path),
                    data=str(data_yaml),
                    imgsz=imgsz,
                    half=half,
                    device=device,
                )

            # Write full log
            log_path = arch_dir / f"benchmark_{task_name}.log"
            try:
                with log_path.open("w") as f:
                    std_txt = stdout_buffer.getvalue()
                    err_txt = stderr_buffer.getvalue()
                    f.write("STDOUT:\n" + std_txt)
                    f.write("\nSTDERR:\n" + err_txt)

                # Header already streamed in real time; ensure section delimiter recorded (in case buffer didn't include newline)
                if run_log_stream is not None:
                    run_log_stream.write(f"\n===== End of {arch} ({task_name}) =====\n\n")
                    run_log_stream.flush()
                if verbose:
                    print(f"      ↳ saved logs to {log_path.relative_to(results_root)}")
            except Exception as e:  # pragma: no cover
                sys.stderr.write(f"[ERROR] Failed to write log {log_path}: {e}\n")

            # Persist metrics JSON (robust dump)
            json_path = arch_dir / f"benchmark_{task_name}.json"
            try:
                with json_path.open("w") as f:
                    json.dump(metrics, f, indent=2, default=str)
                if verbose:
                    print(f"      ↳ saved {json_path.relative_to(results_root)}")
            except Exception as e:  # pragma: no cover
                sys.stderr.write(f"[ERROR] Failed to save JSON {json_path}: {e}\n")

            # Compute parameter count & FLOPs once per architecture (fallback to None if ultralytics missing)
            params_m, flops_b = None, None
            if YOLO is not None:
                try:
                    yolo_model = YOLO(str(weight_path))
                    _n_l, n_p, _n_g, flops = yolo_model.info(detailed=False, verbose=False)
                    params_m = n_p / 1e6 if n_p is not None else None
                    flops_b = flops if flops is not None else None
                    # ensure freeing GPU memory
                    del yolo_model
                except Exception:
                    pass

            # Extract metrics for all formats
            all_rows = _parse_stdout_table(stdout_buffer.getvalue())
            for r in all_rows:
                summary_rows.append({
                    "task": task_name,
                    "architecture": arch,
                    **r,
                    "params_m": params_m,
                    "flops_b": flops_b,
                })

    # After all benchmarks complete, write aggregated CSV table
    if summary_rows:
        csv_path = results_root / "benchmark_summary.csv"
        try:
            with csv_path.open("w", newline="") as csvfile:
                fieldnames = [
                    "task",
                    "architecture",
                    "format",
                    "status",
                    "size_mb",
                    "map50_95",
                    "latency_ms",
                    "fps",
                    "params_m",
                    "flops_b",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
            if verbose:
                print(f"[DONE] Aggregated summary written to {csv_path.relative_to(results_root)}")
        except Exception as e:  # pragma: no cover
            sys.stderr.write(f"[ERROR] Failed to write summary CSV {csv_path}: {e}\n")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Benchmark Ultralytics models for vehicle and plate detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_root", default="Results", type=Path, help="Root folder containing model and dataset sub-directories")
    parser.add_argument("--device", default="cuda:0", help="Computation device id or string, e.g. 'cuda:0', 'cpu', 'mps'")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for benchmarking")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference where supported")
    parser.add_argument("--vehicles-data", type=Path, dest="vehicles_data", help="Path to vehicles dataset YAML (overrides default search)")
    parser.add_argument("--plates-data", type=Path, dest="plates_data", help="Path to license-plate dataset YAML (overrides default search)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    parser.add_argument("--run-log", type=Path, dest="run_log", help="Path to save full terminal output log (default: <results_root>/benchmark_run.log)")
    args = parser.parse_args()

    override_datasets = {
        "vehicles": args.vehicles_data,
        "plates": args.plates_data,
    }

    # ------------------------------------------------------------------
    # Capture full terminal output to log file
    # ------------------------------------------------------------------

    run_log_path: Path = args.run_log if args.run_log else (args.results_root / "benchmark_run.log")
    run_log_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee(io.TextIOBase):
        """Simple tee stream to duplicate writes to multiple streams."""

        def __init__(self, *streams):
            self.streams = streams

        def write(self, s: str):  # type: ignore[override]
            for st in self.streams:
                st.write(s)
            for st in self.streams:
                st.flush()

        def flush(self):
            for st in self.streams:
                st.flush()

    # Only capture if user wants console output; otherwise just log file
    std_out_stream = sys.stdout if not args.quiet else io.StringIO()
    std_err_stream = sys.stderr if not args.quiet else io.StringIO()

    with run_log_path.open("w") as _run_log_file, \
            contextlib.redirect_stdout(_Tee(std_out_stream, _run_log_file)), \
            contextlib.redirect_stderr(_Tee(std_err_stream, _run_log_file)):
        run_benchmarks(
            results_root=args.results_root,
            device=args.device,
            imgsz=args.imgsz,
            half=args.half,
            verbose=not args.quiet,
            override_datasets=override_datasets,
            run_log_stream=_run_log_file,
        )

    if not args.quiet:
        print(f"[INFO] Full terminal output saved to {run_log_path.relative_to(args.results_root)}")


if __name__ == "__main__":  # pragma: no cover
    main() 