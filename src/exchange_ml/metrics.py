"""Semantic, distance, and partition-disagreement metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


def binary_dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Binary Sørensen–Dice with an empty/empty score of one."""
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: prediction={pred.shape}, reference={ref.shape}"
        )
    denominator = int(pred.sum()) + int(ref.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(pred, ref).sum() / denominator)


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def normalized_variation_of_information(
    first: np.ndarray,
    second: np.ndarray,
    *,
    domain: np.ndarray | None = None,
) -> float:
    """Return VI/log(n) on a fixed domain.

    Labels, including background label zero, are treated as arbitrary partition
    symbols. If ``domain`` is omitted, the union of nonzero foreground is used.
    The score is zero for an empty or one-voxel domain.
    """
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: first={left.shape}, second={right.shape}")
    active = (
        np.logical_or(left != 0, right != 0)
        if domain is None
        else np.asarray(domain, dtype=bool)
    )
    if active.shape != left.shape:
        raise ValueError(
            f"domain shape mismatch: domain={active.shape}, partitions={left.shape}"
        )
    left_values = left[active]
    right_values = right[active]
    n_voxels = int(left_values.size)
    if n_voxels <= 1:
        return 0.0

    _, left_inverse, left_counts = np.unique(
        left_values, return_inverse=True, return_counts=True
    )
    _, right_inverse, right_counts = np.unique(
        right_values, return_inverse=True, return_counts=True
    )
    right_cardinality = int(right_counts.size)
    joint_codes = left_inverse.astype(np.int64) * right_cardinality + right_inverse
    joint_counts = np.bincount(joint_codes)

    h_left = _entropy_from_counts(left_counts)
    h_right = _entropy_from_counts(right_counts)
    h_joint = _entropy_from_counts(joint_counts)
    vi = max(0.0, 2.0 * h_joint - h_left - h_right)
    return float(np.clip(vi / np.log(n_voxels), 0.0, 1.0))


def foreground_aware_partition_disagreement(
    first: np.ndarray,
    second: np.ndarray,
    *,
    roi: np.ndarray | None = None,
) -> float:
    """Compare foreground presence and its shared instance partition.

    Let ``U`` be the pairwise foreground union and ``C`` its intersection,
    optionally restricted to ``roi``. The score is

    ``|U \\ C| / |U| + |C| / |U| * NVI(first|C, second|C)``.

    This is a bounded, symmetric, label-permutation-invariant dissimilarity,
    but it is not a mathematical metric: the pair-dependent intersection can
    violate the triangle inequality. Two empty foregrounds score zero; an
    empty/non-empty pair scores one. NVI is defined as zero when ``|C| <= 1``.
    """
    return float(foreground_aware_partition_components(first, second, roi=roi)["score"])


def foreground_aware_partition_components(
    first: np.ndarray,
    second: np.ndarray,
    *,
    roi: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return the auditable presence and partition components of pairwise FA-IPD."""
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: first={left.shape}, second={right.shape}")

    if roi is None:
        roi_mask = np.ones(left.shape, dtype=bool)
    else:
        roi_mask = np.asarray(roi, dtype=bool)
        if roi_mask.shape != left.shape:
            raise ValueError(
                f"ROI shape mismatch: roi={roi_mask.shape}, partitions={left.shape}"
            )

    left_foreground = np.logical_and(left != 0, roi_mask)
    right_foreground = np.logical_and(right != 0, roi_mask)
    union = np.logical_or(left_foreground, right_foreground)
    union_size = int(union.sum())
    if union_size == 0:
        return {
            "union_voxels": 0,
            "intersection_voxels": 0,
            "presence_disagreement": 0.0,
            "partition_disagreement": 0.0,
            "weighted_partition_disagreement": 0.0,
            "score": 0.0,
        }

    intersection = np.logical_and(left_foreground, right_foreground)
    intersection_size = int(intersection.sum())
    presence_disagreement = (union_size - intersection_size) / union_size
    partition_disagreement = normalized_variation_of_information(
        left,
        right,
        domain=intersection,
    )
    weighted_partition_disagreement = (
        intersection_size / union_size
    ) * partition_disagreement
    score = float(
        np.clip(presence_disagreement + weighted_partition_disagreement, 0.0, 1.0)
    )
    return {
        "union_voxels": union_size,
        "intersection_voxels": intersection_size,
        "presence_disagreement": float(presence_disagreement),
        "partition_disagreement": float(partition_disagreement),
        "weighted_partition_disagreement": float(weighted_partition_disagreement),
        "score": score,
    }


def foreground_jaccard_disagreement(
    first: np.ndarray,
    second: np.ndarray,
    *,
    roi: np.ndarray | None = None,
) -> float:
    """Return the foreground-presence term used by FA-IPD."""
    components = foreground_aware_partition_components(first, second, roi=roi)
    return float(components["presence_disagreement"])


def ipd(partitions: Iterable[np.ndarray], *, roi: np.ndarray | None = None) -> float:
    """Mean pairwise foreground-aware instance-partition disagreement."""
    arrays = [np.asarray(partition) for partition in partitions]
    if len(arrays) < 2:
        raise ValueError("IPD requires at least two partitions")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("all partitions must have identical shapes")
    if roi is not None:
        roi_mask = np.asarray(roi, dtype=bool)
        if roi_mask.shape != shape:
            raise ValueError(
                f"ROI shape mismatch: roi={roi_mask.shape}, partitions={shape}"
            )
    scores = [
        foreground_aware_partition_disagreement(left, right, roi=roi)
        for left, right in combinations(arrays, 2)
    ]
    return float(np.mean(scores))


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    eroded = ndi.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def surface_distances(
    prediction: np.ndarray,
    reference: np.ndarray,
    spacing: tuple[float, ...],
) -> dict[str, float]:
    """Symmetric ASSD and HD95 in physical units.

    Array and spacing order must match (for SimpleITK arrays this is z, y, x).
    Empty/non-empty pairs are undefined and raise ``ValueError``.
    """
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError(
            f"shape mismatch: prediction={pred.shape}, reference={ref.shape}"
        )
    if len(spacing) != pred.ndim or any(float(value) <= 0 for value in spacing):
        raise ValueError(f"invalid spacing {spacing} for shape {pred.shape}")
    if not pred.any() or not ref.any():
        raise ValueError("surface distances require two non-empty masks")

    pred_surface = _surface(pred)
    ref_surface = _surface(ref)
    distance_to_ref = ndi.distance_transform_edt(~ref_surface, sampling=spacing)
    distance_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_to_ref = distance_to_ref[pred_surface]
    ref_to_pred = distance_to_pred[ref_surface]
    symmetric = np.concatenate((pred_to_ref, ref_to_pred))
    return {
        "assd_mm": float((pred_to_ref.mean() + ref_to_pred.mean()) / 2.0),
        "hd95_mm": float(np.percentile(symmetric, 95)),
    }
