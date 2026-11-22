"""E1 volume I/O and evaluation. GT is loaded only after GT-free scoring."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .e1_scoring import score_unit
from .instances import ANATOMY_RANGES, instance_metrics, prune_small_components
from .metrics import binary_dice

MEMBERS = ("member_01", "member_02", "member_03")
RISKS = ("merge_split", "one_minus_f1", "missing", "merge", "split", "one_minus_dice", "missing_volume")
COMPONENTS = ("vi", "joint_entropy", "q_log_n", "q_joint", "overlap_weight", "presence",
              "weighted_q_log_n", "weighted_q_joint", "fa_v1", "fa_v2")


def geometry(image: sitk.Image) -> dict:
    return {"size": image.GetSize(), "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(), "direction": image.GetDirection()}


def read_volume(path: Path, expected: dict | None, *, probability: bool = False) -> tuple[np.ndarray, dict]:
    image = sitk.ReadImage(str(path))
    observed = geometry(image)
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"expected scalar 3D volume: {path}")
    if expected and any(not np.allclose(observed[k], expected[k], atol=1e-6, rtol=1e-6)
                        for k in observed):
        raise ValueError(f"geometry mismatch: {path}")
    array = sitk.GetArrayFromImage(image)  # [Z,Y,X], CPU
    if probability:
        if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError(f"probabilities must be finite in [0,1]: {path}")
        array = array.astype(np.float32, copy=False)
    else:
        if not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() > 200:
            raise ValueError(f"instance IDs must be integers in 0..200: {path}")
        array = array.astype(np.uint8, copy=False)
    return array, observed


def clean_prediction(labels: np.ndarray, spacing: tuple, min_volume: float, connectivity: int) -> np.ndarray:
    """Exact existing per-ID cleanup, restricted to each ID's bounding box.

    Cropping removes only zero padding, so it cannot connect/disconnect voxels
    of that ID. It avoids allocating a whole-CT component map for every ID.
    """
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 200:
        raise ValueError("expected integer 3D instance IDs in 0..200")
    output = labels.copy()
    for label_id, box in enumerate(ndi.find_objects(labels, max_label=int(labels.max())), 1):
        if box is None:
            continue
        original = labels[box]
        selected = original == label_id
        cleaned = prune_small_components(np.where(selected, label_id, 0), spacing,
                                          min_volume_mm3=min_volume, connectivity=connectivity)
        output[box][selected & (cleaned == 0)] = 0
    return output


def mask_bone(labels: np.ndarray, bone: str) -> np.ndarray:
    low, high = ANATOMY_RANGES[bone]
    return (labels >= low) & (labels <= high)


def bone_risk(pred: np.ndarray, gt: np.ndarray, spacing: tuple, bone: str, threshold: float) -> dict:
    """Reuse unchanged instance matching; inputs have already been cleaned."""
    pm, gm = mask_bone(pred, bone), mask_bone(gt, bone)
    active = pm | gm
    if active.any():
        box = ndi.find_objects(active.astype(np.uint8), max_label=1)[0]
        p = np.where(pm[box], pred[box], 0)
        g = np.where(gm[box], gt[box], 0)
    else:
        p = g = np.zeros((1, 1, 1), dtype=np.uint8)
    result = instance_metrics(p, g, spacing, connectivity=26,
                              prediction_component_prune_min_volume_mm3=0,
                              iou_match_threshold=threshold)["per_anatomy"][bone]
    _, counts = np.unique(g[g > 0], return_counts=True)
    n = result["gt_instance_count"]
    volume = float(np.prod(spacing))
    covered = int(((g > 0) & (p > 0)).sum())
    gt_voxels = int(counts.sum())
    return {"n_GT": n, "n_pred": result["pred_instance_count"],
            "gt_volume_mm3": gt_voxels * volume,
            "min_gt_fragment_volume_mm3": float(counts.min() * volume) if n else None,
            "merge_errors": result["merge_errors"], "split_errors": result["split_errors"],
            "merge_split": (result["merge_errors"] + result["split_errors"]) / max(n, 1),
            "one_minus_f1": 1 - result["instance_f1"], "missing": 1 - result["instance_recall"],
            "merge": result["merge_errors"] / max(n, 1), "split": result["split_errors"] / max(n, 1),
            "one_minus_dice": 1 - binary_dice(p > 0, g > 0),
            "missing_volume": 1 - covered / gt_voxels if gt_voxels else None}


def process_case(case_id: str, scan_type: str, paths: dict[str, Path], config: dict) -> dict:
    """One case at a time: score three models, then load GT and evaluate four."""
    bones = ("sacrum", "left_hip", "right_hip") if scan_type == "pelvic" else ("femur",)
    maps, probabilities, expected = [], [], None
    for member in MEMBERS:
        array, observed = read_volume(paths[member + "_prediction"], expected)
        expected = observed
        spacing = tuple(observed["spacing"][::-1])
        maps.append(clean_prediction(array, spacing, config["prune_mm3"], config["connectivity"]))
        probability, _ = read_volume(paths[member + "_probability"], expected, probability=True)
        probabilities.append(probability)
    scores, components = [], []
    for bone in bones:
        roi = np.logical_or.reduce([mask_bone(m, bone) for m in maps])
        parts = [np.where(mask_bone(m, bone)[roi], m[roi], 0) for m in maps]
        scores_bone, pairs = score_unit(parts, [p[roi] for p in probabilities])
        common = {"case_id": case_id, "scan_type": scan_type, "bone": bone}
        averaged = {key: float(np.mean([p[key] for p in pairs])) for key in COMPONENTS}
        scores.append({**common, **scores_bone, **{"component_" + k: v for k, v in averaged.items()},
                       "partition_share_v1": averaged["weighted_q_log_n"] / averaged["fa_v1"] if averaged["fa_v1"] else None,
                       "partition_share_v2": averaged["weighted_q_joint"] / averaged["fa_v2"] if averaged["fa_v2"] else None})
        for (i, j), pair in zip(combinations(MEMBERS, 2), pairs):
            components.append({**common, "member_i": i, "member_j": j, **pair})
    roi = np.logical_or.reduce([m != 0 for m in maps])
    case_scores, _ = score_unit([m[roi] for m in maps], [p[roi] for p in probabilities])
    del probabilities

    # The scoring path above never reads GT; it uses only declared scan type.
    gt, _ = read_volume(paths["gt"], expected)
    stage2, _ = read_volume(paths["stage2"], expected)
    stage2 = clean_prediction(stage2, spacing, config["prune_mm3"], config["connectivity"])
    models = dict(zip(MEMBERS, maps)) | {"stage2": stage2}
    model_risks, risks, case_risks = [], [], []
    by_bone = {}
    for bone in bones:
        common = {"case_id": case_id, "scan_type": scan_type, "bone": bone}
        rows = {model: bone_risk(pred, gt, spacing, bone, config["iou_threshold"])
                for model, pred in models.items()}
        if rows["stage2"]["n_GT"] == 0:
            raise ValueError(f"{case_id}/{bone}: metadata declares annotation but GT has no fragment")
        by_bone[bone] = rows
        model_risks.extend({**common, "model_id": model, **row} for model, row in rows.items())
        risk = {**common, **{k: rows["stage2"][k] for k in ("n_GT", "gt_volume_mm3", "min_gt_fragment_volume_mm3")}}
        for name in RISKS:
            risk["member_mean_" + name] = float(np.mean([rows[m][name] for m in MEMBERS]))
            risk["stage2_" + name] = rows["stage2"][name]
        risks.append(risk)
    # Legacy case-macro F1 and case total merge/split risk, preserved separately.
    for model in models:
        n_gt = sum(by_bone[b][model]["n_GT"] for b in bones)
        case_risks.append({"case_id": case_id, "scan_type": scan_type, "model_id": model,
                           "n_GT": n_gt,
                           "merge_split": sum(by_bone[b][model]["merge_errors"] + by_bone[b][model]["split_errors"] for b in bones) / max(n_gt, 1),
                           "one_minus_f1": float(np.mean([by_bone[b][model]["one_minus_f1"] for b in bones]))})
    case_aggregate = {"case_id": case_id, "scan_type": scan_type}
    for risk in ("merge_split", "one_minus_f1"):
        case_aggregate["member_mean_" + risk] = float(np.mean([r[risk] for r in case_risks if r["model_id"] in MEMBERS]))
        case_aggregate["stage2_" + risk] = next(r[risk] for r in case_risks if r["model_id"] == "stage2")
    return {"scores": scores, "pair_components": components, "model_risks": model_risks, "risks": risks,
            "case_scores": [{"case_id": case_id, "scan_type": scan_type, **case_scores}],
            "case_risks": [case_aggregate], "case_model_risks": case_risks, "geometry": expected}
