#!/usr/bin/env python3
"""Evaluate frozen Stage 2 validation predictions under the internal contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exchange_ml.instances import instance_metrics  # noqa: E402
from exchange_ml.stage2_backend import discover_case_directories, select_stage2_case_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage2_backend.json")
    )
    return parser.parse_args()


def geometry_matches(left: sitk.Image, right: sitk.Image) -> bool:
    return left.GetSize() == right.GetSize() and all(
        np.allclose(a, b, atol=1e-6, rtol=1e-6)
        for a, b in (
            (left.GetSpacing(), right.GetSpacing()),
            (left.GetOrigin(), right.GetOrigin()),
            (left.GetDirection(), right.GetDirection()),
        )
    )


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = json.loads(Path(config["split"]["manifest"]).read_text(encoding="utf-8"))
    selected = select_stage2_case_ids(split, int(config["split"]["trajectory_seed"]))
    cases = discover_case_directories(Path(config["paths"]["dataset_root"]))
    root = Path(config["paths"]["workspace_root"])
    prediction_root = root / "validation_predictions"
    evaluation = config["evaluation"]
    primary_connectivity = int(evaluation["primary_connectivity"])
    sensitivity_connectivities = [
        int(value) for value in evaluation["connectivity_sensitivity"]
    ]
    if primary_connectivity not in sensitivity_connectivities:
        raise ValueError(
            "primary connectivity must be included in sensitivity analysis"
        )
    if evaluation.get("preserve_all_gt_instance_ids") is not True:
        raise ValueError("Stage 1 contract requires preserving every GT instance ID")
    rows = []
    for case_id in selected["validation"]:
        prediction_path = prediction_root / f"{case_id}.mha"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"missing validation prediction: {prediction_path}")
        prediction_image = sitk.ReadImage(str(prediction_path))
        reference_image = sitk.ReadImage(str(cases[case_id] / "label.mha"))
        if not geometry_matches(prediction_image, reference_image):
            raise RuntimeError(f"{case_id}: prediction/reference geometry mismatch")
        prediction = sitk.GetArrayFromImage(prediction_image)
        reference = sitk.GetArrayFromImage(reference_image)
        spacing_zyx = tuple(
            float(value) for value in reference_image.GetSpacing()[::-1]
        )
        for connectivity in sensitivity_connectivities:
            result = instance_metrics(
                prediction,
                reference,
                spacing_zyx,
                connectivity=int(connectivity),
                prediction_component_prune_min_volume_mm3=float(
                    evaluation["prediction_component_prune_min_volume_mm3"]
                ),
                iou_match_threshold=float(evaluation["iou_match_threshold"]),
            )
            rows.append(
                {
                    "case_id": case_id,
                    "connectivity": int(connectivity),
                    "binary_dice": result["binary_dice"],
                    "instance_precision": result["instance_precision"],
                    "instance_recall": result["instance_recall"],
                    "instance_f1": result["instance_f1"],
                    "merge_errors": result["merge_errors"],
                    "split_errors": result["split_errors"],
                    "retained_gt_instance_count": result["retained_gt_instance_count"],
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "validation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for connectivity in sensitivity_connectivities:
        subset = [row for row in rows if row["connectivity"] == int(connectivity)]
        summary[str(connectivity)] = {
            "n_cases": len(subset),
            "mean_binary_dice": mean(row["binary_dice"] for row in subset),
            "mean_instance_f1": mean(row["instance_f1"] for row in subset),
            "total_merge_errors": sum(row["merge_errors"] for row in subset),
            "total_split_errors": sum(row["split_errors"] for row in subset),
            "cases_with_incomplete_instance_recall": sum(
                row["instance_recall"] < 1.0 for row in subset
            ),
        }
    summary_path = root / "validation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "primary_connectivity": primary_connectivity,
                "preserve_all_gt_instance_ids": True,
                "prediction_component_prune_min_volume_mm3": float(
                    evaluation["prediction_component_prune_min_volume_mm3"]
                ),
                "summary_by_connectivity": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
