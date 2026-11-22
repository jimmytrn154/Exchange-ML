"""E4 volumes and fragment coverage, reusing unchanged E1 cleanup.

The 2x2 scorer takes predictions and declared scan type only. GT enters after
scoring to measure outside-GT disagreement and fragment-volume coverage.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi

from .e1_data import MEMBERS, clean_prediction, mask_bone, read_volume
from .e1_scoring import pair_components
from .instances import ANATOMY_RANGES


def annotated_bones(scan_type: str) -> tuple[str, ...]:
    if scan_type not in ("pelvic", "femur"):
        raise ValueError(f"unknown scan type: {scan_type}")
    return ("sacrum", "left_hip", "right_hip") if scan_type == "pelvic" else ("femur",)


def scope_scores(maps: list[np.ndarray], scan_type: str) -> tuple[dict, list[dict]]:
    """Mean of three pair scores on whole-case vs annotated-anatomy union.

    These are case scores, not averages over E3's per-bone score rows. Anatomy
    IDs and disconnected components are retained exactly as in E1/E3.
    """
    if len(maps) != 3 or any(m.shape != maps[0].shape for m in maps):
        raise ValueError("three equal-shape member maps required")
    bones = annotated_bones(scan_type)
    output, pairs = {}, []
    roi = np.logical_or.reduce([m > 0 for m in maps])
    flat = [m[roi] for m in maps]
    allowed = np.array([False] + [any(low <= value <= high for low, high in
                       (ANATOMY_RANGES[b] for b in bones)) for value in range(1, 201)])
    scoped = [np.where(allowed[m], m, 0) for m in flat]
    for scope, parts in (("all", flat), ("annotated", scoped)):
        components = []
        for i, j in combinations(range(3), 2):
            row = pair_components(parts[i], parts[j])
            components.append(row)
            pairs.append({"scope": scope, "member_i": MEMBERS[i], "member_j": MEMBERS[j], **row})
        for norm, quantity, weighted in (("log_n", "fa_v1", "weighted_q_log_n"),
                                         ("joint", "fa_v2", "weighted_q_joint")):
            value = float(np.mean([r[quantity] for r in components]))
            contribution = float(np.mean([r[weighted] for r in components]))
            output[scope + "_" + norm] = value
            output[scope + "_" + norm + "_partition_share"] = contribution / value if value else None
    fractions = []
    for i, j in combinations(range(3), 2):
        disagreement = (flat[i] > 0) != (flat[j] > 0)
        outside = ((flat[i] > 0) & ~allowed[flat[i]]) | ((flat[j] > 0) & ~allowed[flat[j]])
        denominator = int(disagreement.sum())
        fraction = int((disagreement & outside).sum()) / denominator if denominator else None
        fractions.append(fraction)
        pairs.append({"scope": "outside_annotation", "member_i": MEMBERS[i], "member_j": MEMBERS[j],
                      "n_disagreement_voxels": denominator,
                      "n_outside_scope_disagreement_voxels": int((disagreement & outside).sum()),
                      "fraction_outside_scope": fraction})
    defined = [v for v in fractions if v is not None]
    # Undefined pair ratios remain undefined; the mean explicitly reports its denominator.
    output.update(fraction_outside_scope=float(np.mean(defined)) if defined else None,
                  n_defined_outside_scope_pairs=len(defined))
    return output, pairs


def fragment_coverage(pred: np.ndarray, gt: np.ndarray, bone: str,
                      spacing: tuple, threshold: float) -> tuple[list[dict], dict]:
    """Exact E1 best-IoU matching plus coverage by ANY same-anatomy prediction.

    One contingency table [GT ID, prediction ID] suffices. Matching uses the
    lowest prediction ID on a positive IoU tie, as instances._anatomy_match.
    GT is never pruned, relabelled, or split into connected components.
    """
    if pred.shape != gt.shape or pred.ndim != 3 or not 0 <= threshold <= 1:
        raise ValueError("equal 3D maps and IoU threshold in [0,1] required")
    pm, gm = mask_bone(pred, bone), mask_bone(gt, bone)
    roi = pm | gm
    p = np.where(pm[roi], pred[roi], 0).astype(np.int32)
    g = np.where(gm[roi], gt[roi], 0).astype(np.int32)
    joint = np.bincount(g * 201 + p, minlength=201 * 201).reshape(201, 201)
    gt_counts, pred_counts = joint.sum(1), joint.sum(0)
    gt_ids, pred_ids = np.flatnonzero(gt_counts[1:]) + 1, np.flatnonzero(pred_counts[1:]) + 1
    matches, records = [], []
    split_errors = 0
    for gid in gt_ids:
        overlaps = joint[gid, pred_ids]
        ious = overlaps / (gt_counts[gid] + pred_counts[pred_ids] - overlaps)
        best_index = int(np.argmax(ious)) if len(ious) else None
        best_iou = float(ious[best_index]) if best_index is not None else 0.0
        # The original evaluator never chooses an ID when every IoU is zero.
        pid = int(pred_ids[best_index]) if best_iou > 0 and best_iou >= threshold else None
        matches.append(pid)
        split_errors += max(int(np.count_nonzero(overlaps)) - 1, 0)
        records.append({"gt_id": int(gid), "matched_pred_id": pid, "matched_iou": best_iou,
                        "gt_volume_mm3": float(gt_counts[gid] * np.prod(spacing)),
                        "gt_fraction_covered": float(joint[gid, 1:].sum() / gt_counts[gid]),
                        "n_overlapping_pred_instances": int(np.count_nonzero(overlaps))})
    matched_pred = set(pid for pid in matches if pid is not None)
    precision = len(matched_pred) / max(len(pred_ids), 1)
    recall = sum(pid is not None for pid in matches) / max(len(gt_ids), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    n_gt_voxels = gt_counts[1:].sum()
    risk = {"n_GT": len(gt_ids), "n_pred": len(pred_ids), "one_minus_f1": 1 - f1,
            "missing_volume": float(1 - joint[1:, 1:].sum() / n_gt_voxels) if n_gt_voxels else None,
            "merge_errors": sum(max(matches.count(pid) - 1, 0) for pid in matched_pred),
            "split_errors": split_errors}
    return records, risk


def load_clean_maps(paths: dict[str, Path], config: dict) -> tuple[dict, dict, dict]:
    maps, expected = {}, None
    elapsed = {"read_seconds": 0.0, "cleanup_seconds": 0.0}
    for model in (*MEMBERS, "stage2"):
        start = time.monotonic()
        labels, observed = read_volume(paths[model], expected)
        expected = observed
        elapsed["read_seconds"] += time.monotonic() - start
        start = time.monotonic()
        maps[model] = clean_prediction(labels, tuple(observed["spacing"][::-1]),
                                       config["prune_mm3"], config["connectivity"])
        elapsed["cleanup_seconds"] += time.monotonic() - start
    return maps, expected, elapsed


def process_case(case: dict, paths: dict[str, Path], config: dict,
                 expected_risks: list[dict]) -> dict:
    maps, geom, elapsed = load_clean_maps(paths, config)
    start = time.monotonic()
    scores, pairs = scope_scores([maps[m] for m in MEMBERS], case["scan_type"])
    elapsed["components_seconds"] = time.monotonic() - start
    # GT is accessed only after the prediction-only score computation above.
    start = time.monotonic()
    gt, _ = read_volume(paths["gt"], geom)
    elapsed["read_seconds"] += time.monotonic() - start
    roi = np.logical_or.reduce([maps[m] > 0 for m in MEMBERS])
    parts, truth = [maps[m][roi] for m in MEMBERS], gt[roi]
    outside_gt = []
    for i, j in combinations(range(3), 2):
        different = (parts[i] > 0) != (parts[j] > 0)
        n = int(different.sum())
        outside_gt.append({**case, "member_i": MEMBERS[i], "member_j": MEMBERS[j],
                           "n_disagreement_voxels": n,
                           "n_disagreement_outside_gt_foreground": int((different & (truth == 0)).sum()),
                           "fraction_outside_gt_foreground": float((different & (truth == 0)).sum() / n) if n else None})
    del roi, parts, truth
    fragments = []
    lookup = {(r["bone"], r["model_id"]): r for r in expected_risks}
    start = time.monotonic()
    for bone in annotated_bones(case["scan_type"]):
        for model, pred in maps.items():
            rows, risks = fragment_coverage(pred, gt, bone, tuple(geom["spacing"][::-1]), config["iou_threshold"])
            expected = lookup[bone, model]
            for key, value in risks.items():
                if value is None or not np.isclose(value, expected[key], rtol=1e-10, atol=1e-12):
                    raise ValueError(f"E1/E3 risk mismatch: {case['case_id']}/{bone}/{model}/{key}: {value} != {expected[key]}")
            fragments.extend({**case, "bone": bone, "model_id": model, **r} for r in rows)
    elapsed["fragment_seconds"] = time.monotonic() - start
    return {"scope_scores": [{**case, **scores}], "scope_pairs": [{**case, **p} for p in pairs],
            "outside_gt": outside_gt, "fragments": fragments, "runtime": [{**case, **elapsed}]}


def bounding_box(active: np.ndarray, margin: int = 0) -> tuple[slice, ...]:
    if not active.any():
        raise ValueError("empty qualitative target")
    box = ndi.find_objects(active.astype(np.uint8), max_label=1)[0]
    return tuple(slice(max(0, s.start - margin), min(size, s.stop + margin)) for s, size in zip(box, active.shape))
