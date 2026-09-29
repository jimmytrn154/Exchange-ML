#!/usr/bin/env python3
"""Run the frozen two-stage ABBC backend on the 40 validation cases.

This is a user-typed full GPU inference command. Unlike the challenge
entrypoint, this wrapper never writes an all-zero fallback after an exception.
"""

from __future__ import annotations

import argparse
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

from exchange_ml.stage2_backend import discover_case_directories, select_stage2_case_ids  # noqa: E402
from stage2_backend import (  # noqa: E402
    check,
    environment,
    load_config,
    patch_inference_trainer_discovery,
    stage2_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_backend.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-limit", type=int, default=None)
    return parser.parse_args()


def validate_prediction(path: Path, reference: sitk.Image) -> None:
    prediction = sitk.ReadImage(str(path))
    if prediction.GetSize() != reference.GetSize():
        raise RuntimeError(f"prediction size mismatch: {path}")
    for name, left, right in (
        ("spacing", prediction.GetSpacing(), reference.GetSpacing()),
        ("origin", prediction.GetOrigin(), reference.GetOrigin()),
        ("direction", prediction.GetDirection(), reference.GetDirection()),
    ):
        if not np.allclose(left, right, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"prediction {name} mismatch: {path}")
    values = np.unique(sitk.GetArrayViewFromImage(prediction))
    if values.size == 1 and int(values[0]) == 0:
        raise RuntimeError(f"all-zero prediction is not accepted: {path}")
    if int(values.min()) < 0 or int(values.max()) > 200:
        raise RuntimeError(f"prediction labels outside 0..200: {path}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    check(config)
    env = environment(config)
    paths = stage2_paths(config)
    env.update(
        {
            "PENGWIN_TARGET_ROUTER": "0",
            "PENGWIN_DS539_TRAINER": config["anatomy_stage"]["trainer"],
            "PENGWIN_DS538_TRAINER": config["fragment_stage"]["trainer"],
            "PENGWIN_DS538_OUT_CH": str(config["fragment_stage"]["output_channels"]),
            "PENGWIN_AFFINITY_DECODE": "1",
            "PENGWIN_AGGLO_T": str(config["reconstruction"]["primary_affinity_threshold"]),
            "PENGWIN_FEMUR_ADAPTIVE_T": str(config["reconstruction"]["femur_retry_threshold"]),
            "PENGWIN_FEMUR_ADAPTIVE_MINVOX": str(
                config["reconstruction"]["femur_retry_large_single_instance_voxels"]
            ),
            "PENGWIN_CLICK_INJECT": "0",
        }
    )
    os.environ.update(env)
    patch_inference_trainer_discovery(config)
    inference_dir = Path(config["paths"]["baseline_root"]) / "inference"
    sys.path.insert(0, str(inference_dir.resolve()))
    backend = importlib.import_module("inference")
    backend.DS539_DATASET = config["anatomy_stage"]["dataset_name"]
    backend.DS538_DATASET = config["fragment_stage"]["dataset_name"]
    backend.DS539_TRAINER = config["anatomy_stage"]["trainer"]
    backend.DS538_TRAINER = config["fragment_stage"]["trainer"]
    backend.DS538_EXPERT_TRAINERS = {}
    backend.NN_RES = paths["results"].resolve()

    split = json.loads(Path(config["split"]["manifest"]).read_text(encoding="utf-8"))
    selected = select_stage2_case_ids(split, int(config["split"]["trajectory_seed"]))
    case_ids = selected["validation"]
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("--case-limit must be positive")
        case_ids = case_ids[: args.case_limit]
    cases = discover_case_directories(Path(config["paths"]["dataset_root"]))
    output_root = paths["root"] / "validation_predictions"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in case_ids:
        source = cases[case_id] / "image.mha"
        output = output_root / f"{case_id}.mha"
        reference = sitk.ReadImage(str(source))
        if args.resume and output.is_file():
            validate_prediction(output, reference)
            rows.append({"case_id": case_id, "prediction": str(output), "status": "VALID_RESUME"})
            continue
        prediction = backend.run_per_anatomy(
            source, reference, prerouted_bone_masks=None, anatomies=None
        )
        if not np.any(prediction):
            raise RuntimeError(f"{case_id}: backend returned an all-zero prediction")
        backend.write_label_map(prediction, reference, output)
        validate_prediction(output, reference)
        rows.append({"case_id": case_id, "prediction": str(output), "status": "COMPLETE"})
        (paths["root"] / "validation_inference_manifest.json").write_text(
            json.dumps({"schema_version": 1, "rows": rows}, indent=2), encoding="utf-8"
        )
    print(f"validation predictions complete: {len(rows)}/{len(case_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
