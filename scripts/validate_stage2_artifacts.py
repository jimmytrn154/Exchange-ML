#!/usr/bin/env python3
"""Validate the completed Stage 2 internal-research artifact contract."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    root = Path("outputs/stage2_backend")
    config_path = Path("configs/stage2_backend.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(Path(config["split"]["manifest"]).read_text(encoding="utf-8"))
    expected_ids = {str(value).zfill(3) for value in split["split"]["validation"]}
    if len(expected_ids) != 40:
        raise RuntimeError(f"expected 40 validation IDs, found {len(expected_ids)}")

    prediction_ids = {path.stem for path in (root / "validation_predictions").glob("*.mha")}
    if prediction_ids != expected_ids:
        raise RuntimeError(
            f"prediction ID mismatch: missing={sorted(expected_ids - prediction_ids)} "
            f"extra={sorted(prediction_ids - expected_ids)}"
        )

    manifest_path = root / "validation_inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = manifest["rows"]
    manifest_ids = {str(row["case_id"]).zfill(3) for row in manifest_rows}
    if len(manifest_rows) != 40 or manifest_ids != expected_ids:
        raise RuntimeError("inference manifest does not cover the frozen 40-case validation set")
    statuses = Counter(str(row["status"]) for row in manifest_rows)
    if set(statuses) - {"COMPLETE", "VALID_RESUME"}:
        raise RuntimeError(f"invalid inference statuses: {dict(statuses)}")
    for row in manifest_rows:
        if not Path(row["prediction"]).is_file():
            raise FileNotFoundError(row["prediction"])

    metrics_path = root / "validation_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    if len(metric_rows) != 120:
        raise RuntimeError(f"expected 120 metric rows, found {len(metric_rows)}")
    metric_keys = {(row["case_id"], int(row["connectivity"])) for row in metric_rows}
    expected_keys = {(case_id, connectivity) for case_id in expected_ids for connectivity in (6, 18, 26)}
    if metric_keys != expected_keys:
        raise RuntimeError("metric case/connectivity grid is incomplete or duplicated")

    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["primary_connectivity"]) != 26:
        raise RuntimeError("primary connectivity must remain 26")
    if summary["preserve_all_gt_instance_ids"] is not True:
        raise RuntimeError("Stage 2 must preserve every GT instance ID")
    if float(summary["prediction_component_prune_min_volume_mm3"]) != 1000.0:
        raise RuntimeError("Stage 2 prediction cleanup threshold drifted")

    for connectivity in (6, 18, 26):
        subset = [row for row in metric_rows if int(row["connectivity"]) == connectivity]
        observed = summary["summary_by_connectivity"][str(connectivity)]
        expected = {
            "n_cases": len(subset),
            "mean_binary_dice": mean(float(row["binary_dice"]) for row in subset),
            "mean_instance_f1": mean(float(row["instance_f1"]) for row in subset),
            "total_merge_errors": sum(int(row["merge_errors"]) for row in subset),
            "total_split_errors": sum(int(row["split_errors"]) for row in subset),
            "cases_with_incomplete_instance_recall": sum(
                float(row["instance_recall"]) < 1.0 for row in subset
            ),
        }
        for key, expected_value in expected.items():
            observed_value = observed[key]
            if isinstance(expected_value, float):
                if not close(observed_value, expected_value):
                    raise RuntimeError(
                        f"summary mismatch connectivity={connectivity} {key}: "
                        f"observed={observed_value} expected={expected_value}"
                    )
            elif int(observed_value) != int(expected_value):
                raise RuntimeError(
                    f"summary mismatch connectivity={connectivity} {key}: "
                    f"observed={observed_value} expected={expected_value}"
                )

    checkpoints = sorted((root / "nnUNet_results").glob("**/checkpoint_best.pth"))
    if len(checkpoints) != 2 or any(path.stat().st_size == 0 for path in checkpoints):
        raise RuntimeError(f"expected two non-empty best checkpoints, found {checkpoints}")

    primary = summary["summary_by_connectivity"]["26"]
    failures = sum(
        int(row["merge_errors"]) + int(row["split_errors"]) > 0
        for row in metric_rows
        if int(row["connectivity"]) == 26
    )
    result = {
        "status": "PASS_INTERNAL_RESEARCH_BACKEND",
        "validation_cases": 40,
        "metric_rows": 120,
        "manifest_statuses": dict(sorted(statuses.items())),
        "primary_connectivity": 26,
        "mean_binary_dice": primary["mean_binary_dice"],
        "mean_instance_f1": primary["mean_instance_f1"],
        "total_merge_errors": primary["total_merge_errors"],
        "total_split_errors": primary["total_split_errors"],
        "cases_with_merge_or_split": failures,
        "sha256": {
            str(path): sha256(path)
            for path in (config_path, manifest_path, metrics_path, summary_path)
        },
        "checkpoint_bytes": {str(path): path.stat().st_size for path in checkpoints},
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
