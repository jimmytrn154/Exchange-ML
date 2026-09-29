#!/usr/bin/env python3
"""Compute frozen Stage 3 acquisition scores without reading ground truth.

This dataset-wide command is user-typed under __docs__/rule.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from exchange_ml.instances import prune_small_components  # noqa: E402
from exchange_ml.stage2_backend import select_stage2_case_ids  # noqa: E402
from exchange_ml.stage3_h1 import case_uncertainty_scores  # noqa: E402
from stage3_h1 import load_config, member_root, validate_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_h1.json"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    config = load_config(args.config)
    validate_config(config)
    stage2 = json.loads(Path(config["stage2_config"]).read_text(encoding="utf-8"))
    split = json.loads(Path(stage2["split"]["manifest"]).read_text(encoding="utf-8"))
    selected = select_stage2_case_ids(split, int(config["trajectory_seed"]))
    case_ids = selected[config["analysis_role"]]
    member_records = {row["id"]: row for row in config["ensemble"]["members"]}
    members = list(member_records)
    member_provenance: dict[str, dict[str, object]] = {}
    for member_id in members:
        root = member_root(config, member_id)
        manifest_path = root / "validation_inference_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{member_id}: missing inference manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("member_id") != member_id:
            raise RuntimeError(f"{member_id}: inference manifest member mismatch")
        if int(manifest.get("model_seed", -1)) != int(
            member_records[member_id]["model_seed"]
        ):
            raise RuntimeError(f"{member_id}: inference manifest seed mismatch")
        if manifest.get("probability_semantics") != config["probability_contract"]:
            raise RuntimeError(f"{member_id}: probability semantics mismatch")
        manifest_rows = manifest.get("rows", [])
        rows_by_case = {str(row["case_id"]): row for row in manifest_rows}
        if len(manifest_rows) != len(case_ids) or set(rows_by_case) != set(case_ids):
            raise RuntimeError(f"{member_id}: inference manifest must cover 40 cases")
        if any(
            row.get("status") not in {"COMPLETE", "VALID_RESUME"}
            for row in manifest_rows
        ):
            raise RuntimeError(f"{member_id}: invalid inference manifest status")
        for case_id in case_ids:
            row = rows_by_case[case_id]
            expected_prediction = root / "validation_predictions" / f"{case_id}.mha"
            expected_probability = root / "validation_probabilities" / f"{case_id}.mha"
            if Path(row["prediction"]).resolve() != expected_prediction.resolve():
                raise RuntimeError(f"{member_id}/{case_id}: manifest prediction path drift")
            if Path(row["foreground_probability"]).resolve() != expected_probability.resolve():
                raise RuntimeError(f"{member_id}/{case_id}: manifest probability path drift")
        checkpoint_hashes = {
            "anatomy": str(manifest.get("anatomy_checkpoint_sha256", "")),
            "fragment": str(manifest.get("fragment_checkpoint_sha256", "")),
        }
        if any(len(value) != 64 for value in checkpoint_hashes.values()):
            raise RuntimeError(f"{member_id}: invalid checkpoint fingerprint")
        member_provenance[member_id] = {
            "model_seed": int(member_records[member_id]["model_seed"]),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "checkpoint_sha256": checkpoint_hashes,
        }

    contract = config["score_contract"]
    connectivity = int(contract["prediction_connectivity"])
    min_volume = float(contract["prediction_component_prune_min_volume_mm3"])

    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        partitions = []
        probabilities = []
        reference_image = None
        for member_id in members:
            root = member_root(config, member_id)
            prediction_path = root / "validation_predictions" / f"{case_id}.mha"
            probability_path = root / "validation_probabilities" / f"{case_id}.mha"
            if not prediction_path.is_file() or not probability_path.is_file():
                raise FileNotFoundError(
                    f"{member_id}/{case_id}: missing prediction or probability"
                )
            prediction_image = sitk.ReadImage(str(prediction_path))
            probability_image = sitk.ReadImage(str(probability_path))
            if reference_image is None:
                reference_image = prediction_image
            if not geometry_matches(prediction_image, reference_image):
                raise RuntimeError(f"{member_id}/{case_id}: prediction geometry mismatch")
            if not geometry_matches(probability_image, reference_image):
                raise RuntimeError(f"{member_id}/{case_id}: probability geometry mismatch")
            spacing_zyx = tuple(float(value) for value in prediction_image.GetSpacing()[::-1])
            prediction = sitk.GetArrayFromImage(prediction_image)
            probability = sitk.GetArrayFromImage(probability_image).astype(
                np.float32, copy=False
            )
            if not np.isfinite(probability).all():
                raise RuntimeError(f"{member_id}/{case_id}: non-finite probability")
            if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
                raise RuntimeError(f"{member_id}/{case_id}: probability outside [0,1]")
            cleaned = prune_small_components(
                prediction,
                spacing_zyx,
                min_volume_mm3=min_volume,
                connectivity=connectivity,
            )
            partitions.append(cleaned)
            probabilities.append(probability)

        roi = np.logical_or.reduce([partition != 0 for partition in partitions])
        if not roi.any():
            raise RuntimeError(f"{case_id}: ensemble union foreground is empty")
        scores = case_uncertainty_scores(
            partitions, probabilities, roi=roi
        )
        rows.append({"case_id": case_id, **scores})

    output_root = Path(config["workspace_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "uncertainty_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = output_root / "uncertainty_scoring_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_count": len(rows),
                "member_ids": members,
                "member_provenance": member_provenance,
                "config": str(args.config),
                "config_sha256": sha256(args.config),
                "ground_truth_loaded": False,
                "probability_contract": config["probability_contract"],
                "score_contract": contract,
                "output": str(csv_path),
                "output_sha256": sha256(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote GT-free Stage 3 scores: {csv_path} ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
