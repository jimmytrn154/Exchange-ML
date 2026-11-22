"""E2 controlled interventions; originals and frozen E1 primitives stay intact."""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np
from scipy import ndimage as ndi

from .e1_scoring import information_from_counts
from .instances import ANATOMY_RANGES
from .metrics import binary_dice
from .perturbations import matched_boundary_perturbation, relabel_instances

METRICS = ("presence", "vi", "joint_entropy", "q_log_n", "q_joint", "overlap_weight",
           "weighted_q_log_n", "weighted_q_joint", "fa_v1", "fa_v2")


def contingency(first: np.ndarray, second: np.ndarray, bone: str) -> np.ndarray:
    """51x51 counts: background at 0, up to 50 subject-local bone IDs.

    Count (0,0) is removed, so summing the table gives pairwise foreground U.
    This counts-only representation also permits replication without images.
    """
    left, right = np.asarray(first), np.asarray(second)
    low, high = ANATOMY_RANGES[bone]
    if left.shape != right.shape:
        raise ValueError("different partition shapes")
    arrays = []
    for labels in (left, right):
        if not np.issubdtype(labels.dtype, np.integer) or ((labels != 0) & ((labels < low) | (labels > high))).any():
            raise ValueError(f"IDs outside {bone} range")
        arrays.append(np.where(labels > 0, labels.astype(np.int64) - low + 1, 0))
    joint = np.bincount((arrays[0] * 51 + arrays[1]).ravel(), minlength=51**2).reshape(51, 51)
    joint[0, 0] = 0
    return joint


def components(joint: np.ndarray) -> dict:
    counts = np.asarray(joint)
    if counts.ndim != 2 or counts.shape[0] < 1 or counts.shape[1] < 1 or counts[0, 0] != 0:
        raise ValueError("expected foreground-union table with zero background/background count")
    info = information_from_counts(counts[1:, 1:])
    union, shared = int(counts.sum()), int(counts[1:, 1:].sum())
    presence = (union - shared) / union if union else 0.0
    weight = shared / union if union else 0.0
    return {"n_shared_voxels": shared, "n_union_voxels": union,
            **info, "presence": presence, "overlap_weight": weight,
            "weighted_q_log_n": weight * info["q_log_n"],
            "weighted_q_joint": weight * info["q_joint"],
            "fa_v1": presence + weight * info["q_log_n"],
            "fa_v2": presence + weight * info["q_joint"]}


def replication(joint: np.ndarray, factors: list[int], tolerance: float) -> list[dict]:
    """Multiply ALL union counts, including foreground mismatch counts."""
    baseline = components(joint)
    rows = []
    for factor in factors:
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("replication factor must be a positive integer")
        observed = components(joint * factor)
        for name in ("vi", "joint_entropy", "presence", "overlap_weight", "q_joint"):
            if abs(observed[name] - baseline[name]) > tolerance:
                raise ArithmeticError(f"replication changed invariant {name}")
        n = baseline["n_shared_voxels"] * factor
        expected = baseline["vi"] / np.log(n) if n > 1 else 0.0
        if abs(observed["q_log_n"] - expected) > tolerance:
            raise ArithmeticError("VI/log(rN) replication identity failed")
        rows.append({"r": factor, "N": observed["n_shared_voxels"], "VI": observed["vi"],
                     "H": observed["joint_entropy"], "Q1": observed["q_log_n"],
                     "Q2": observed["q_joint"], "A": observed["presence"], "w": observed["overlap_weight"],
                     "Q1_expected": float(expected), "Q1_residual": observed["q_log_n"] - expected,
                     "VI_delta": observed["vi"] - baseline["vi"], "H_delta": observed["joint_entropy"] - baseline["joint_entropy"],
                     "Q2_delta": observed["q_joint"] - baseline["q_joint"],
                     "A_delta": observed["presence"] - baseline["presence"], "w_delta": observed["overlap_weight"] - baseline["overlap_weight"],
                     "positive_vi": baseline["vi"] > 1e-12})
    return rows


def phantoms(tolerance: float) -> tuple[list[dict], np.ndarray]:
    original = np.repeat([1, 2], 32).reshape(4, 4, 4)
    merged, missing, empty = np.ones_like(original), original.copy(), np.zeros_like(original)
    missing[missing == 2] = 0
    rows = []
    for name, first, second, expected in (
        ("identity", original, original, (0, 0, 0, 0)),
        ("permutation", original, relabel_instances(original, {1: 2, 2: 1}), (0, 0, 0, 0)),
        ("merge", original, merged, (0, 1 / 6, 1, 1)),
        ("split_reverse", merged, original, (0, 1 / 6, 1, 1)),
        ("missing", original, missing, (.5, 0, 0, .5)),
        ("one_empty", original, empty, (1, 0, 0, 1)),
        ("one_empty_reverse", empty, original, (1, 0, 0, 1)),
        ("both_empty", empty, empty, (0, 0, 0, 0)),
    ):
        observed = components(contingency(first, second, "sacrum"))
        for key, target in zip(("presence", "q_log_n", "q_joint", "fa_v2"), expected):
            error = abs(observed[key] - target)
            if error > tolerance:
                raise ArithmeticError(f"phantom {name}/{key}: {observed[key]} != {target}")
            rows.append({"phantom": name, "quantity": key, "expected": target,
                         "observed": observed[key], "absolute_error": error, "passed": True})
    return rows, contingency(original, merged, "sacrum")


def extract_bone(gt: np.ndarray, bone: str, margin: int) -> np.ndarray:
    """Keep all bone GT, with morphology padding clipped at original image edges."""
    low, high = ANATOMY_RANGES[bone]
    active = (gt >= low) & (gt <= high)
    if not active.any():
        raise ValueError(f"annotated bone {bone} has no GT fragments")
    box = ndi.find_objects(active.astype(np.uint8), max_label=1)[0]
    padded = tuple(slice(max(0, s.start - margin), min(n, s.stop + margin)) for s, n in zip(box, gt.shape))
    return np.where(active[padded], gt[padded], 0)


def fragment_order(source: np.ndarray) -> tuple[list[int], dict[int, int]]:
    ids, counts = np.unique(source[source > 0], return_counts=True)
    sizes = dict(zip(map(int, ids), map(int, counts)))
    return sorted(sizes, key=lambda label: (sizes[label], label)), sizes


def merge_pairs(source: np.ndarray, spacing: tuple[float, ...]) -> list[tuple[int, int]]:
    ids, _ = fragment_order(source)
    centroids = {label: np.argwhere(source == label).mean(0) * spacing for label in ids}
    ordered = sorted((float(np.linalg.norm(centroids[i] - centroids[j])), min(i, j), max(i, j))
                     for i, j in combinations(ids, 2))
    used, chosen = set(), []
    for _, first, second in ordered:
        if first not in used and second not in used:
            chosen.append((first, second))
            used.update((first, second))
    return chosen


def split_largest(source: np.ndarray, spacing: tuple, bone: str, parts: int) -> tuple[np.ndarray | None, dict]:
    ids, sizes = fragment_order(source)
    chosen = min(ids, key=lambda label: (-sizes[label], label))
    coords = np.argwhere(source == chosen)  # [N,3], z/y/x integer indices
    low, high = ANATOMY_RANGES[bone]
    free = [i for i in range(low, high + 1) if i not in sizes]
    if len(coords) < parts or len(free) < parts - 1:
        return None, {"reason": "insufficient_voxels_or_unused_anatomy_ids", "affected_ids": [chosen]}
    # Physical voxel-cell extent (max-min+1)*spacing; np.argmax breaks ties z,y,x.
    extent = (coords.max(0) - coords.min(0) + 1) * spacing
    axis = int(np.argmax(extent))
    priority = [axis] + [i for i in range(3) if i != axis]
    order = np.lexsort(tuple(coords[:, i] for i in reversed(priority)))
    output = source.copy()
    new_ids = [chosen, *free[:parts - 1]]
    for index, label in zip(np.array_split(order, parts), new_ids):
        output[tuple(coords[index].T)] = label
    return output, {"reason": "", "affected_ids": [chosen], "new_ids": new_ids,
                    "split_axis_zyx": axis, "physical_extent_mm": extent.tolist()}


def responses(source: np.ndarray, spacing: tuple, bone: str, config: dict) -> list[dict]:
    """Each operation/severity starts from the same unpruned GT copy."""
    ids, sizes = fragment_order(source)
    if not ids:
        raise ValueError("empty annotated GT")
    original_foreground = int((source > 0).sum())
    voxel_volume = float(np.prod(spacing))
    output_rows = []
    def add(operation, severity, changed, metadata):
        affected = metadata.get("affected_ids", [])
        row = {"bone": bone, "operation": operation, "severity": str(severity),
               "eligible": changed is not None, "reason": metadata.get("reason", ""),
               "affected_ids": json.dumps(affected), "metadata": json.dumps(metadata, sort_keys=True),
               "gt_fragment_count": len(ids), "gt_volume_mm3": original_foreground * voxel_volume,
               "affected_volume_mm3": sum(sizes.get(i, 0) for i in affected) * voxel_volume,
               "changed_voxels": None, "changed_volume_fraction": None, "foreground_changed_volume_fraction": None,
               "achieved_dice": None, "target_reached": None, **{k: None for k in METRICS}}
        if changed is not None:
            foreground_change = int(((source > 0) != (changed > 0)).sum())
            row.update({"changed_voxels": int((source != changed).sum()),
                        "foreground_changed_volume_fraction": foreground_change / original_foreground,
                        "changed_volume_fraction": float((source != changed).sum() / original_foreground),
                        "achieved_dice": binary_dice(source > 0, changed > 0),
                        **{k: v for k, v in components(contingency(source, changed, bone)).items() if k in METRICS}})
            tolerance = config["numeric_tolerance"]
            if operation in {"identity", "permutation", "merge", "split"} and row["presence"] > tolerance:
                raise ArithmeticError(f"{operation} changed foreground")
            if operation in {"identity", "permutation", "missing", "boundary"} and row["vi"] > tolerance:
                raise ArithmeticError(f"{operation} changed the shared partition")
            if operation == "boundary":
                reached = abs(row["achieved_dice"] - float(severity)) <= config["boundary_dice_tolerance"]
                row.update({"target_reached": reached, "reason": "" if reached else "target_not_reached"})
        output_rows.append(row)

    add("identity", 0, source.copy(), {"affected_ids": []})
    rotated = dict(zip(sorted(ids), np.roll(sorted(ids), 1).tolist()))
    add("permutation", 0, relabel_instances(source, rotated), {"affected_ids": sorted(ids), "mapping": rotated})
    pairs = merge_pairs(source, spacing)
    for count in (1, 2, 3):
        if len(pairs) < count:
            add("merge", count, None, {"reason": f"requires_{count * 2}_fragments_found_{len(ids)}"})
            continue
        changed = source.copy()
        for first, second in pairs[:count]:
            changed[source == second] = first
        add("merge", count, changed, {"affected_ids": [i for pair in pairs[:count] for i in pair], "pairs": pairs[:count]})
    for count in (2, 3, 4):
        changed, metadata = split_largest(source, spacing, bone, count)
        add("split", count, changed, metadata)
    largest = min(ids, key=lambda label: (-sizes[label], label))
    roles = {}
    for role, label in (("small", ids[0]), ("median", ids[len(ids) // 2]), ("large", largest)):
        roles.setdefault(label, []).append(role)
    for label, aliases in roles.items():
        changed = source.copy()
        changed[source == label] = 0
        add("missing", "|".join(aliases), changed, {"affected_ids": [label], "volume_rank_roles": aliases})
    for target in config["boundary_targets"]:
        try:
            changed, metadata = matched_boundary_perturbation(source, largest, target_binary_dice=target,
                                                              connectivity=26, max_iterations=config["boundary_max_iterations"])
        except ValueError as error:
            if str(error) != "no non-empty boundary perturbation could be generated":
                raise
            add("boundary", target, None, {"reason": "no_nonempty_boundary_candidate", "affected_ids": [largest]})
            continue
        add("boundary", target, changed, {**metadata, "affected_ids": [largest]})
    return output_rows
