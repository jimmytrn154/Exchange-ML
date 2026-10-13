#!/usr/bin/env python3
"""Gate G1: Stage A per-anatomy Dice from nnU-Net's end-of-training validation.

Reads ``fold_0/validation/summary.json`` written by an Anatomy training run and
scores each class only on cases whose ground truth contains it (``n_ref > 0``):
pelvic and femur scans label disjoint anatomies, so unlabelled visible bones
would otherwise count as Dice 0. Read-only and lightweight (agent-typed).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

CLASSES = {"1": "Sacrum", "2": "LeftHip", "3": "RightHip", "4": "Femur"}
GATED = ("LeftHip", "RightHip")


def labelled_case_dice(summary: dict) -> dict[str, list[float]]:
    per_class: dict[str, list[float]] = {name: [] for name in CLASSES.values()}
    for row in summary["metric_per_case"]:
        for key, name in CLASSES.items():
            metrics = row["metrics"][key]
            if metrics["n_ref"] > 0:
                per_class[name].append(float(metrics["Dice"]))
    return per_class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="+", help="Anatomy validation summary.json")
    parser.add_argument("--min-median-dice", type=float, default=0.90)
    parser.add_argument("--max-cases-below-half", type=int, default=2)
    args = parser.parse_args()
    passed = True
    for path in args.summary:
        dice = labelled_case_dice(json.loads(path.read_text(encoding="utf-8")))
        print(path)
        for name, values in dice.items():
            if not values:
                raise RuntimeError(f"{path}: no labelled cases for {name}")
            median = statistics.median(values)
            below = sum(value < 0.5 for value in values)
            print(f"  {name:8s} n={len(values):2d} median={median:.3f} below_0.5={below}")
            if name in GATED and (median < args.min_median_dice or below > args.max_cases_below_half):
                passed = False
    print("G1:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
