#!/usr/bin/env python3
"""Batch runner for Traffic Monitor.

This utility reads a YAML configuration that specifies which video files to
process and, for each video, optional counting‐line coordinates to pass to the
`traffic_monitor.cli` entry-point.  It then launches the CLI once per video and
waits for it to finish before moving on to the next file.

Example YAML configuration (save as ``batch_config.yaml``):

```yaml
# Path to the regular Traffic Monitor YAML settings. Optional.
traffic_monitor_config: config/settings.yaml

# Counting line(s) that should be applied to every video unless explicitly
# overridden inside the ``videos`` list.  Use the same format accepted by the
# CLI: "x1,y1,x2,y2" strings. Optional.
default_count_lines:
  - "100,200,400,200"

# Global CLI log level (DEBUG, INFO, WARNING, ERROR) – optional.
log_level: INFO

# "videos" can be a list of strings (path only) or objects with per-file
# options.  When the object form is used you can supply a custom set of
# counting lines (or an empty list to disable them for that specific clip).
videos:
  # Inherits default_count_lines
  - data/videos/cam_01.mp4

  # Override counting lines for this file
  - file: data/videos/cam_02.mp4
    count_lines:
      - "50,100,600,100"
      - "50,500,600,500"

  # Disable counting lines entirely for this file
  - file: data/videos/no_lines.mkv
    count_lines: []
```

Usage::

    python tools/batch_run_traffic_monitor.py -c batch_config.yaml

The script will print each command before running it so you can see exactly
what is being executed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover – dependency missing
    print("PyYAML is required to run this script (pip install pyyaml).", file=sys.stderr)
    raise SystemExit(1) from exc


def _build_cli_command(
    python_exe: str,
    traffic_cfg: str | None,
    video_path: Path,
    count_lines: List[str] | None,
    log_level: str | None,
) -> List[str]:
    """Compose the command list that will be passed to *subprocess*."""
    cmd: list[str] = [
        python_exe,
        "-m",
        "traffic_monitor.cli",
        "--mode",
        "offline",
        "--source",
        str(video_path),
    ]

    if traffic_cfg:
        cmd.extend(["--config", traffic_cfg])

    # Attach counting line parameters
    if count_lines is not None:
        if not count_lines:  # Empty list disables counting lines
            cmd.extend(["--count-line", "none"])
        else:
            for line in count_lines:
                cmd.extend(["--count-line", line])

    # Logging verbosity: prefer explicit level; fall back to quiet for less noise
    if log_level:
        cmd.extend(["--log-level", log_level.upper()])
    else:
        cmd.append("--quiet")

    return cmd


def _load_batch_config(path: Path) -> Dict[str, Any]:
    """Read YAML config and return a dictionary with sane defaults."""
    with path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    cfg.setdefault("videos", [])
    cfg.setdefault("default_count_lines", [])

    return cfg


def main() -> None:  # noqa: C901 – function is fine for a small script
    parser = argparse.ArgumentParser(
        description="Run Traffic Monitor sequentially on a collection of videos using a YAML batch configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Path to the batch YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )

    args = parser.parse_args()
    batch_cfg_path: Path = args.config.expanduser().resolve()

    if not batch_cfg_path.is_file():
        parser.error(f"Batch configuration file not found: {batch_cfg_path}")

    cfg = _load_batch_config(batch_cfg_path)

    traffic_cfg_path: str | None = cfg.get("traffic_monitor_config")
    default_lines: List[str] = cfg.get("default_count_lines", [])
    log_level: str | None = cfg.get("log_level")

    videos = cfg.get("videos", [])
    if not videos:
        print("No videos defined in the batch configuration.", file=sys.stderr)
        sys.exit(1)

    python_exe = sys.executable

    output_root = Path("data/videos/output").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for entry in videos:
        # Allow both the shorthand string form and the full mapping form
        if isinstance(entry, str):
            video_path = Path(entry).expanduser().resolve()
            count_lines = default_lines
        elif isinstance(entry, dict):
            video_path = Path(entry.get("file", "")).expanduser().resolve()
            count_lines = entry.get("count_lines", default_lines)
        else:
            print(f"Unsupported video entry format: {entry}", file=sys.stderr)
            continue

        if not video_path.is_file():
            print(f"⚠️  Skipping – file not found: {video_path}", file=sys.stderr)
            continue

        cmd = _build_cli_command(
            python_exe=python_exe,
            traffic_cfg=str(traffic_cfg_path) if traffic_cfg_path else None,
            video_path=video_path,
            count_lines=count_lines,
            log_level=log_level,
        )

        print("\n──────────────────────────────────────────────")
        print("$", " ".join(cmd))

        if args.dry_run:
            continue

        # ------------------------------------------------------------------
        # Capture existing output directories so we can detect the one created
        # by this run.  The CLI always creates a timestamped subfolder under
        # data/videos/output.
        # ------------------------------------------------------------------
        pre_existing_dirs = {d.resolve() for d in output_root.iterdir() if d.is_dir()}

        log_temp_path = output_root / f"{video_path.stem}_tmp.log"

        with log_temp_path.open("w", encoding="utf-8") as log_file:
            try:
                subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                print(
                    f"❌ Traffic Monitor exited with code {exc.returncode} for {video_path}",
                    file=sys.stderr,
                )
                # Proceed with log relocation even on failure.

        # ------------------------------------------------------------------
        # Determine which output directory was created by this execution.
        # Strategy: list dirs again and pick the new one; fallback to latest
        # modified directory.
        # ------------------------------------------------------------------
        post_dirs = {d.resolve() for d in output_root.iterdir() if d.is_dir()}
        new_dirs = post_dirs - pre_existing_dirs

        if len(new_dirs) == 1:
            run_output_dir = new_dirs.pop()
        else:
            # Fallback – choose the most recently modified directory
            run_output_dir = max(post_dirs, key=lambda d: d.stat().st_mtime)

        # ------------------------------------------------------------------
        # Guarantee that each input gets its own subfolder named after the
        # video file (e.g. cam_01) so users can easily identify results.
        # If a directory with that name already exists we append _1, _2, …
        # ------------------------------------------------------------------
        from datetime import datetime

        dest_dir_base = output_root / video_path.stem
        dest_dir = dest_dir_base
        counter = 1
        while dest_dir.exists():
            dest_dir = output_root / f"{video_path.stem}_{counter}"
            counter += 1

        try:
            run_output_dir.rename(dest_dir)
            run_output_dir = dest_dir  # Update reference for log placement
        except Exception as rename_err:
            # If rename fails (e.g. cross-device), move the contents instead.
            import shutil

            try:
                shutil.move(str(run_output_dir), str(dest_dir))
                run_output_dir = dest_dir
            except Exception as move_err:
                print(
                    f"⚠️  Could not reorganise output folder: {rename_err or move_err}",
                    file=sys.stderr,
                )

        final_log_path = run_output_dir / "run.log"
        try:
            log_temp_path.rename(final_log_path)
        except Exception as e:  # pragma: no cover – rename could fail on cross-device moves
            import shutil

            shutil.move(str(log_temp_path), str(final_log_path))

        print(
            f"📝 Log saved to {final_log_path.relative_to(Path.cwd())} | "
            f"Output folder: {run_output_dir.relative_to(Path.cwd())}"
        )

    print("\n✅ Batch processing finished.")


if __name__ == "__main__":
    main() 