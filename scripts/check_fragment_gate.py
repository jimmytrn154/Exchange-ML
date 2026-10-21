#!/usr/bin/env python3
"""Per-bone Fragment (Stage B) check on the Dataset702 validation crops.

Runs one or more Fragment checkpoints on the fold-0 validation crops exactly as
nnU-Net trained them (bone-LUT CT crop around the GT anatomy, 24-voxel pad) and
scores foreground (``1 - p(bg) > 0.5``) against the crop's label, split by bone.
The pooled instance F1 logged during training cannot show a model that fails
on one bone only (D6: sacrum crops segmented as the hips).

User-typed (single GPU, model inference). Writes only under ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from predict_stage2_backend import load_configured_backend  # noqa: E402
from stage2_backend import (  # noqa: E402
    check,
    fragment_inference_checkpoint,
    load_config,
    stage2_paths,
)

BONES = ("Sacrum", "LeftHip", "RightHip", "Femur")
FIELDS = ("model", "crop", "bone", "dice", "recall", "precision", "gt_voxels", "pred_voxels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_backend.json"))
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME=RESULTS_ROOT",
        help="nnUNet_results root holding the Fragment model; repeatable "
             "(default: retrained=<config results root>)",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint file inside fold_0 "
                             "(default: fragment_stage.inference_checkpoint, F8)")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage3_h1/diagnostics/fragment_gate"))
    parser.add_argument("--min-median-dice", type=float, default=0.90)
    parser.add_argument("--max-crops-below-half", type=int, default=2)
    return parser.parse_args()


def crop_bone(crop: str) -> str:
    bone = crop.rsplit("_", 1)[-1]
    if bone not in BONES:
        raise ValueError(f"unknown bone in crop name: {crop}")
    return bone


def validation_crops(preprocessed: Path, dataset_name: str) -> list[str]:
    splits = json.loads((preprocessed / dataset_name / "splits_final.json").read_text(encoding="utf-8"))
    crops = sorted(splits[0]["val"])
    if not crops:
        raise RuntimeError("fold 0 has no validation crops")
    return crops


def foreground_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    if pred.shape != label.shape:
        raise RuntimeError(f"prediction {pred.shape} and label {label.shape} differ")
    pred, gt = pred.astype(bool), label > 0
    overlap, n_pred, n_gt = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
    return {
        "dice": 2.0 * overlap / (n_pred + n_gt) if n_pred + n_gt else 1.0,
        "recall": overlap / n_gt if n_gt else float("nan"),
        "precision": overlap / n_pred if n_pred else float("nan"),
        "gt_voxels": n_gt,
        "pred_voxels": n_pred,
    }


def gate(rows: list[dict], min_median_dice: float, max_below_half: int) -> dict:
    """Every bone present must reach the median Dice and have few failed crops."""
    result = {}
    for bone in BONES:
        dice = [float(r["dice"]) for r in rows if r["bone"] == bone]
        if not dice:
            continue
        median = statistics.median(dice)
        below = sum(d < 0.5 for d in dice)
        result[bone] = {"n": len(dice), "median_dice": median, "crops_below_0.5": below,
                        "pass": median >= min_median_dice and below <= max_below_half}
    if not result:
        raise RuntimeError("no crops scored")
    result["PASS"] = all(v["pass"] for v in result.values())
    return result


def predict_foreground(backend, predictor, image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    props = {"spacing": list(image.GetSpacing())[::-1]}
    logits, data_props = backend.predict_logits_full(
        predictor, array[None], props, output_channels=backend.DS538_OUTPUT_CHANNELS
    )
    abbc = backend.softmax_axis0(logits[:4])
    fg_pp = ((1.0 - abbc[0]) > 0.5).astype(np.uint8)
    return backend.resample_label_map_to_original(fg_pp, data_props, predictor) > 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    check(config)
    args.checkpoint = args.checkpoint or fragment_inference_checkpoint(config)
    paths = stage2_paths(config)
    models = args.model or [f"retrained={paths['results']}"]
    backend = load_configured_backend(config)
    dataset = config["fragment_stage"]["dataset_name"]
    raw = paths["raw"] / dataset
    crops = validation_crops(paths["preprocessed"], dataset)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    summary = {}
    for spec in models:
        name, _, root = spec.partition("=")
        if not name or not root:
            raise ValueError(f"--model must be NAME=RESULTS_ROOT, got {spec!r}")
        backend.NN_RES = Path(root).resolve()
        predictor = backend.build_predictor(
            dataset, config["fragment_stage"]["trainer"], backend.DS538_PLANS,
            backend.DS538_CONFIG, backend.DS538_FOLD, checkpoint_name=args.checkpoint,
        )
        rows = []
        with (args.out / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for crop in crops:
                image = sitk.ReadImage(str(raw / "imagesTr" / f"{crop}_0000.mha"))
                label = sitk.GetArrayFromImage(sitk.ReadImage(str(raw / "labelsTr" / f"{crop}.mha")))
                row = {"model": name, "crop": crop, "bone": crop_bone(crop),
                       **foreground_metrics(predict_foreground(backend, predictor, image), label)}
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(f"[fragment-gate] {name} {crop} dice={row['dice']:.3f} "
                      f"recall={row['recall']:.3f} precision={row['precision']:.3f}", flush=True)
        summary[name] = {"results_root": str(backend.NN_RES), "checkpoint": args.checkpoint,
                         **gate(rows, args.min_median_dice, args.max_crops_below_half)}
        del predictor
        torch.cuda.empty_cache()
        (args.out / f"summary_{Path(args.checkpoint).stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, result in summary.items():
        print(name)
        for bone in BONES:
            if bone in result:
                r = result[bone]
                print(f"  {bone:8s} n={r['n']:2d} median={r['median_dice']:.3f} "
                      f"below_0.5={r['crops_below_0.5']} {'ok' if r['pass'] else 'FAIL'}")
        print("  Fragment gate:", "PASS" if result["PASS"] else "FAIL")
    return 0 if all(r["PASS"] for r in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
