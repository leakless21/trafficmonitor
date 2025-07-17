#!/usr/bin/env python3
"""
Generate character frequency report for OCR evaluation.

Given a ground-truth annotations CSV and a predictions CSV (same format as
produced by ocr_dataset_processor.py), this script counts how many times each
alphanumeric character appears in the ground-truth plate strings and how many
 times it appears in the predicted strings.  It writes a CSV with
columns: character, gt_count, pred_count.

Usage:
    python tools/benchmarking/char_frequency_report.py \
        --predictions data/outputs/ocr/fast_plate_all.csv \
        --ground_truth data/merged_dataset/all_annotations.csv \
        --output char_counts_fast_plate.csv
"""

import argparse
import csv
from pathlib import Path
from collections import Counter
from typing import Dict

CHAR_SET = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def count_characters(csv_path: Path) -> Counter:
    """Return frequency Counter of characters in plate_text column."""
    counts: Counter = Counter()
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("plate_text", "")
            # Normalise to uppercase and strip spaces/hyphens etc.
            for ch in text.upper():
                if ch in CHAR_SET:
                    counts[ch] += 1
    return counts


def generate_report(gt_csv: Path, pred_csv: Path, out_csv: Path):
    gt_counts = count_characters(gt_csv)
    pred_counts = count_characters(pred_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["character", "gt_count", "pred_count"])
        for ch in CHAR_SET:
            writer.writerow([ch, gt_counts.get(ch, 0), pred_counts.get(ch, 0)])

    print(f"Character frequency report saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate character frequency report")
    parser.add_argument("--predictions", required=True, type=Path, help="Predictions CSV path")
    parser.add_argument("--ground_truth", required=True, type=Path, help="Ground-truth annotations CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    generate_report(args.ground_truth, args.predictions, args.output) 