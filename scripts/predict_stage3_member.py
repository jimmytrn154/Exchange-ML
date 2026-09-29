#!/usr/bin/env python3
"""Run one frozen Stage 3 ensemble member on the 40 validation cases.

This is a user-typed full GPU inference command. It writes both deterministic
instance maps and genuine Stage-B foreground probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from exchange_ml.stage2_backend import (  # noqa: E402
    discover_case_directories,
    select_stage2_case_ids,
)
from stage2_backend import (  # noqa: E402
    check,
    environment,
    patch_inference_trainer_discovery,
    stage2_paths,
)
from stage3_h1 import (  # noqa: E402
    checkpoint_path,
    load_config,
    member_record,
    member_root,
    member_stage2_config,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_h1.json"))
    parser.add_argument("--member", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-limit", type=int, default=None)
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


def validate_outputs(
    prediction_path: Path, probability_path: Path, reference: sitk.Image
) -> None:
    prediction_image = sitk.ReadImage(str(prediction_path))
    probability_image = sitk.ReadImage(str(probability_path))
    if not geometry_matches(prediction_image, reference):
        raise RuntimeError(f"prediction geometry mismatch: {prediction_path}")
    if not geometry_matches(probability_image, reference):
        raise RuntimeError(f"probability geometry mismatch: {probability_path}")
    prediction = sitk.GetArrayViewFromImage(prediction_image)
    values = np.unique(prediction)
    if values.size == 1 and int(values[0]) == 0:
        raise RuntimeError(f"all-zero prediction is not accepted: {prediction_path}")
    if int(values.min()) < 0 or int(values.max()) > 200:
        raise RuntimeError(f"prediction labels outside 0..200: {prediction_path}")
    probability = sitk.GetArrayViewFromImage(probability_image)
    if not np.isfinite(probability).all():
        raise RuntimeError(f"non-finite probability: {probability_path}")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise RuntimeError(f"probability outside [0,1]: {probability_path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    member = member_record(config, args.member)
    stage2 = member_stage2_config(config, args.member)
    check(stage2)

    for stage in ("anatomy", "fragment"):
        path = checkpoint_path(stage2, stage)
        if not path.is_file():
            raise FileNotFoundError(f"{args.member} missing {stage} checkpoint: {path}")

    env = environment(stage2)
    env.update(
        {
            "PENGWIN_TARGET_ROUTER": "0",
            "PENGWIN_DS539_TRAINER": stage2["anatomy_stage"]["trainer"],
            "PENGWIN_DS538_TRAINER": stage2["fragment_stage"]["trainer"],
            "PENGWIN_DS538_OUT_CH": str(stage2["fragment_stage"]["output_channels"]),
            "PENGWIN_AFFINITY_DECODE": "1",
            "PENGWIN_AGGLO_T": str(
                stage2["reconstruction"]["primary_affinity_threshold"]
            ),
            "PENGWIN_FEMUR_ADAPTIVE_T": str(
                stage2["reconstruction"]["femur_retry_threshold"]
            ),
            "PENGWIN_FEMUR_ADAPTIVE_MINVOX": str(
                stage2["reconstruction"][
                    "femur_retry_large_single_instance_voxels"
                ]
            ),
            "PENGWIN_CLICK_INJECT": "0",
        }
    )
    os.environ.update(env)
    patch_inference_trainer_discovery(stage2)

    inference_dir = Path(stage2["paths"]["baseline_root"]) / "inference"
    sys.path.insert(0, str(inference_dir.resolve()))
    backend = importlib.import_module("inference")
    backend.DS539_DATASET = stage2["anatomy_stage"]["dataset_name"]
    backend.DS538_DATASET = stage2["fragment_stage"]["dataset_name"]
    backend.DS539_TRAINER = stage2["anatomy_stage"]["trainer"]
    backend.DS538_TRAINER = stage2["fragment_stage"]["trainer"]
    backend.DS538_EXPERT_TRAINERS = {}
    backend.NN_RES = stage2_paths(stage2)["results"].resolve()

    split = json.loads(Path(stage2["split"]["manifest"]).read_text(encoding="utf-8"))
    selected = select_stage2_case_ids(split, int(config["trajectory_seed"]))
    case_ids = selected[config["analysis_role"]]
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("--case-limit must be positive")
        case_ids = case_ids[: args.case_limit]
    cases = discover_case_directories(Path(stage2["paths"]["dataset_root"]))

    root = member_root(config, args.member)
    prediction_root = root / "validation_predictions"
    probability_root = root / "validation_probabilities"
    prediction_root.mkdir(parents=True, exist_ok=True)
    probability_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "validation_inference_manifest.json"
    anatomy_checkpoint_sha256 = sha256(checkpoint_path(stage2, "anatomy"))
    fragment_checkpoint_sha256 = sha256(checkpoint_path(stage2, "fragment"))
    rows = []
    for case_id in case_ids:
        source = cases[case_id] / "image.mha"
        prediction_path = prediction_root / f"{case_id}.mha"
        probability_path = probability_root / f"{case_id}.mha"
        reference = sitk.ReadImage(str(source))
        if args.resume and prediction_path.is_file() and probability_path.is_file():
            validate_outputs(prediction_path, probability_path, reference)
            status = "VALID_RESUME"
        else:
            prediction, foreground_probability = backend.run_per_anatomy(
                source,
                reference,
                prerouted_bone_masks=None,
                anatomies=None,
                return_foreground_probability=True,
            )
            if not np.any(prediction):
                raise RuntimeError(f"{case_id}: backend returned an all-zero prediction")
            backend.write_label_map(prediction, reference, prediction_path)
            probability_image = sitk.GetImageFromArray(
                np.asarray(foreground_probability, dtype=np.float32)
            )
            probability_image.CopyInformation(reference)
            sitk.WriteImage(probability_image, str(probability_path), True)
            validate_outputs(prediction_path, probability_path, reference)
            status = "COMPLETE"
        rows.append(
            {
                "case_id": case_id,
                "prediction": str(prediction_path),
                "foreground_probability": str(probability_path),
                "status": status,
            }
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "member_id": args.member,
                    "model_seed": int(member["model_seed"]),
                    "anatomy_checkpoint_sha256": anatomy_checkpoint_sha256,
                    "fragment_checkpoint_sha256": fragment_checkpoint_sha256,
                    "probability_semantics": config["probability_contract"],
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"{args.member} validation predictions complete: {len(rows)}/{len(case_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
