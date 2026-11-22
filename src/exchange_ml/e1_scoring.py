"""GT-free E1 scores from fixed, already-cleaned instance maps.

VI/H(P,Q) follows Vinh, Epps & Bailey (JMLR 2010). This module is an
explicit v2 path; frozen v1 metrics and their aggregation remain unchanged.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import _entropy_from_counts

SCORE_NAMES = (
    "mean_entropy", "p95_entropy", "pairwise_dice", "presence", "union_nvi",
    "count_variance", "hungarian", "fa_v1", "partition_v1", "partition_v2",
    "fa_v2", "q_joint_unweighted",
)


def information_from_counts(joint: np.ndarray) -> dict[str, float]:
    """VI and joint entropy in nats, including deterministic/empty partitions."""
    counts = np.asarray(joint, dtype=np.float64)
    if counts.ndim != 2 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("joint counts must be a finite nonnegative matrix")
    n = float(counts.sum())
    h = _entropy_from_counts(counts.ravel())
    vi = max(0.0, 2 * h - _entropy_from_counts(counts.sum(0))
             - _entropy_from_counts(counts.sum(1)))
    if vi > h + 1e-10:
        raise ArithmeticError("VI exceeds joint entropy")
    return {"vi": vi, "joint_entropy": h,
            "q_log_n": float(np.clip(vi / np.log(n), 0, 1)) if n > 1 else 0.0,
            "q_joint": float(np.clip(vi / h, 0, 1)) if h > 0 and n > 1 else 0.0}


def pair_components(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Pairwise counts on U and information on C; inputs contain one bone.

    Inputs can be 3D maps or flattened voxels on the fixed ensemble ROI.
    Background remains a partition symbol for union-NVI only.
    """
    left, right = np.asarray(first), np.asarray(second)
    if left.shape != right.shape:
        raise ValueError("partition shapes differ")
    for labels in (left, right):
        if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
            raise ValueError("partitions must contain nonnegative integer IDs")
    # Include background at index zero even if it is absent from the input.
    left_ids, li = np.unique(np.concatenate(([0], left.ravel())), return_inverse=True)
    right_ids, ri = np.unique(np.concatenate(([0], right.ravel())), return_inverse=True)
    li, ri = li[1:], ri[1:]
    nr = int(ri.max(initial=0)) + 1
    nl = int(li.max(initial=0)) + 1
    joint = np.bincount(li * nr + ri, minlength=nl * nr).reshape(nl, nr)
    joint[0, 0] = 0  # discard the complement of U
    shared = joint[1:, 1:]
    n_union, n_shared = int(joint.sum()), int(shared.sum())
    a = (n_union - n_shared) / n_union if n_union else 0.0
    weight = n_shared / n_union if n_union else 0.0
    info = information_from_counts(shared)
    ki = int(np.count_nonzero(joint.sum(1)[1:]))
    kj = int(np.count_nonzero(joint.sum(0)[1:]))
    ni, nj = int(joint[1:, :].sum()), int(joint[:, 1:].sum())
    if ki and kj:
        intersections = shared.astype(np.float64)
        unions = joint.sum(1)[1:, None] + joint.sum(0)[None, 1:] - intersections
        iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
        # Also valid for the historical whole-case comparison: never match
        # a sacrum fragment to a hip/femur fragment.
        left_bones, right_bones = (left_ids[1:] - 1) // 50, (right_ids[1:] - 1) // 50
        iou[left_bones[:, None] != right_bones[None, :]] = 0
        rows, columns = linear_sum_assignment(-iou)
        denominator = sum(max(np.count_nonzero(left_bones == bone),
                              np.count_nonzero(right_bones == bone))
                          for bone in set(left_bones) | set(right_bones))
        similarity = float(iou[rows, columns].sum()) / denominator
    else:
        similarity = 1.0 if not (ki or kj) else 0.0
    return {
        "n_shared_voxels": n_shared, "n_union_voxels": n_union,
        "k_i": ki, "k_j": kj, **info, "overlap_weight": weight, "presence": a,
        "weighted_q_log_n": weight * info["q_log_n"],
        "weighted_q_joint": weight * info["q_joint"],
        "fa_v1": a + weight * info["q_log_n"],
        "fa_v2": a + weight * info["q_joint"],
        "pairwise_dice": 1 - 2 * n_shared / (ni + nj) if ni + nj else 0.0,
        "union_nvi": information_from_counts(joint)["q_log_n"],
        "hungarian": 1 - similarity,
    }


def score_unit(
    partitions: Sequence[np.ndarray], probabilities: Sequence[np.ndarray],
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Score a fixed one-bone ROI without GT or GT-derived volumes/counts."""
    if len(partitions) != 3 or len(probabilities) != 3:
        raise ValueError("E1 requires exactly three members")
    shape = partitions[0].shape
    if any(x.shape != shape for x in [*partitions, *probabilities]):
        raise ValueError("unit arrays have different shapes")
    for probability in probabilities:
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("probabilities must be finite in [0,1]")
    roi = np.logical_or.reduce([p != 0 for p in partitions])
    pairs = [pair_components(partitions[i][roi], partitions[j][roi])
             for i, j in combinations(range(3), 2)]
    mean = lambda name: float(np.mean([p[name] for p in pairs]))
    if roi.any():
        # Work only on foreground ROI; avoid three float64 full-volume copies.
        probability = sum(p[roi].astype(np.float64) for p in probabilities) / 3
        entropy = np.zeros_like(probability)
        active = (probability > 0) & (probability < 1)
        p = probability[active]
        entropy[active] = -(p * np.log(p) + (1 - p) * np.log1p(-p))
        entropy_mean, entropy_p95 = float(entropy.mean()), float(np.percentile(entropy, 95))
    else:
        entropy_mean = entropy_p95 = 0.0
    counts = [np.count_nonzero(np.unique(p)) for p in partitions]
    scores = {
        "mean_entropy": entropy_mean, "p95_entropy": entropy_p95,
        **{k: mean(k) for k in ("pairwise_dice", "presence", "union_nvi", "hungarian", "fa_v1", "fa_v2")},
        "count_variance": float(np.var(counts, ddof=0)),
        "partition_v1": mean("weighted_q_log_n"),
        "partition_v2": mean("weighted_q_joint"), "q_joint_unweighted": mean("q_joint"),
    }
    if not np.isfinite(list(scores.values())).all():
        raise ArithmeticError("nonfinite score")
    return scores, pairs
