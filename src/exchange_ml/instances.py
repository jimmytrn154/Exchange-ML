"""PENGWIN-aligned instance cleanup and metric primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .metrics import binary_dice


ANATOMY_RANGES = {
    "sacrum": (1, 50),
    "left_hip": (51, 100),
    "right_hip": (101, 150),
    "femur": (151, 200),
}


def connectivity_structure(connectivity: int) -> np.ndarray:
    rank = {6: 1, 18: 2, 26: 3}.get(int(connectivity))
    if rank is None:
        raise ValueError(f"connectivity must be one of 6, 18, 26; got {connectivity}")
    return ndi.generate_binary_structure(3, rank)


def prune_small_components(
    instance_map: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    min_volume_mm3: float,
    connectivity: int,
) -> np.ndarray:
    """Remove per-ID connected components smaller than a physical-volume threshold."""
    labels = np.asarray(instance_map)
    if labels.ndim != 3:
        raise ValueError(f"expected a 3D instance map, got shape {labels.shape}")
    voxel_volume = float(np.prod(spacing_zyx))
    if voxel_volume <= 0:
        raise ValueError(f"invalid spacing: {spacing_zyx}")
    if min_volume_mm3 <= 0:
        return labels.copy()
    threshold_voxels = max(1, int(np.ceil(float(min_volume_mm3) / voxel_volume)))
    structure = connectivity_structure(connectivity)
    output = labels.copy()
    for label_id in np.unique(labels):
        label_id = int(label_id)
        if label_id <= 0:
            continue
        components, count = ndi.label(labels == label_id, structure=structure)
        sizes = np.bincount(components.ravel())
        for component_id in range(1, int(count) + 1):
            if int(sizes[component_id]) < threshold_voxels:
                output[components == component_id] = 0
    return output


def _anatomy_match(
    gt_part: np.ndarray, pred_part: np.ndarray, iou_threshold: float
) -> dict[str, Any]:
    gt_ids = sorted(int(value) for value in np.unique(gt_part) if int(value) > 0)
    pred_ids = sorted(int(value) for value in np.unique(pred_part) if int(value) > 0)
    matches: list[dict[str, float | int | None]] = []
    for gt_id in gt_ids:
        gt_mask = gt_part == gt_id
        best_pred: int | None = None
        best_iou = 0.0
        for pred_id in pred_ids:
            pred_mask = pred_part == pred_id
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou = float(intersection / union) if union else 0.0
            if iou > best_iou or (
                iou == best_iou and best_pred is not None and pred_id < best_pred
            ):
                best_iou = iou
                best_pred = pred_id
        matches.append(
            {
                "gt_id": gt_id,
                "pred_id": best_pred if best_iou >= iou_threshold else None,
                "iou": best_iou,
            }
        )

    matched_gt = {int(row["gt_id"]) for row in matches if row["pred_id"] is not None}
    matched_pred = {
        int(row["pred_id"]) for row in matches if row["pred_id"] is not None
    }
    precision = len(matched_pred) / max(len(pred_ids), 1)
    recall = len(matched_gt) / max(len(gt_ids), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    assignments = {pred_id: 0 for pred_id in pred_ids}
    for row in matches:
        if row["pred_id"] is not None:
            assignments[int(row["pred_id"])] += 1
    merge_errors = sum(max(count - 1, 0) for count in assignments.values())
    split_errors = 0
    for gt_id in gt_ids:
        overlapping = {
            int(value)
            for value in np.unique(pred_part[gt_part == gt_id])
            if int(value) > 0
        }
        split_errors += max(len(overlapping) - 1, 0)
    return {
        "gt_instance_count": len(gt_ids),
        "pred_instance_count": len(pred_ids),
        "instance_precision": float(precision),
        "instance_recall": float(recall),
        "instance_f1": float(f1),
        "merge_errors": int(merge_errors),
        "split_errors": int(split_errors),
        "matches": matches,
    }


def instance_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    connectivity: int,
    prediction_component_prune_min_volume_mm3: float = 1000.0,
    iou_match_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute the internal PENGWIN instance metric contract.

    Ground-truth instance IDs are preserved exactly after range validation.
    Connectivity-based component pruning is applied only to predictions; using
    prediction cleanup on GT would silently delete annotated small fragments.
    """
    pred = np.asarray(prediction)
    gt = np.asarray(reference)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(
            f"expected equal 3D shapes, got pred={pred.shape}, gt={gt.shape}"
        )
    if not 0 <= iou_match_threshold <= 1:
        raise ValueError("iou_match_threshold must be in [0, 1]")

    invalid_gt = sorted(
        int(value) for value in np.unique(gt) if not 0 <= int(value) <= 200
    )
    invalid_pred = sorted(
        int(value) for value in np.unique(pred) if not 0 <= int(value) <= 200
    )
    if invalid_gt:
        raise ValueError(f"GT instance IDs must be in 0..200; got {invalid_gt}")
    if invalid_pred:
        raise ValueError(
            f"prediction instance IDs must be in 0..200; got {invalid_pred}"
        )
    pred = prune_small_components(
        pred,
        spacing_zyx,
        min_volume_mm3=prediction_component_prune_min_volume_mm3,
        connectivity=connectivity,
    )

    per_anatomy: dict[str, dict[str, Any]] = {}
    present: list[str] = []
    for name, (lower, upper) in ANATOMY_RANGES.items():
        gt_part = np.where((gt >= lower) & (gt <= upper), gt, 0)
        pred_part = np.where((pred >= lower) & (pred <= upper), pred, 0)
        result = _anatomy_match(gt_part, pred_part, iou_match_threshold)
        per_anatomy[name] = result
        if result["gt_instance_count"] > 0:
            present.append(name)

    if present:
        precision = float(
            np.mean([per_anatomy[name]["instance_precision"] for name in present])
        )
        recall = float(
            np.mean([per_anatomy[name]["instance_recall"] for name in present])
        )
        f1 = float(np.mean([per_anatomy[name]["instance_f1"] for name in present]))
    else:
        precision = recall = f1 = 1.0 if not np.any(pred) else 0.0
    retained_gt = int(sum(per_anatomy[name]["gt_instance_count"] for name in present))
    merges = int(sum(per_anatomy[name]["merge_errors"] for name in present))
    splits = int(sum(per_anatomy[name]["split_errors"] for name in present))
    normalized_error = float((merges + splits) / max(retained_gt, 1))
    return {
        "binary_dice": binary_dice(pred > 0, gt > 0),
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "merge_errors": merges,
        "split_errors": splits,
        "normalized_merge_split_error": normalized_error,
        "retained_gt_instance_count": retained_gt,
        "present_anatomies": present,
        "per_anatomy": per_anatomy,
    }
