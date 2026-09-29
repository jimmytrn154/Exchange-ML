"""Frozen Stage 3 H1 uncertainty and failure-detection primitives."""

from __future__ import annotations

from itertools import combinations
from math import ceil
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from .instances import ANATOMY_RANGES
from .metrics import (
    binary_dice,
    foreground_aware_partition_components,
    normalized_variation_of_information,
)


def _validate_same_shape(arrays: Sequence[np.ndarray], name: str) -> tuple[int, ...]:
    if not arrays:
        raise ValueError(f"{name} must not be empty")
    shape = tuple(np.asarray(arrays[0]).shape)
    if any(tuple(np.asarray(array).shape) != shape for array in arrays[1:]):
        raise ValueError(f"all {name} arrays must have identical shapes")
    return shape


def binary_predictive_entropy(
    foreground_probabilities: Sequence[np.ndarray],
    roi: np.ndarray,
) -> dict[str, float]:
    """Aggregate binary entropy of the ensemble-mean foreground probability."""
    arrays = [np.asarray(array, dtype=np.float64) for array in foreground_probabilities]
    shape = _validate_same_shape(arrays, "probability")
    roi_mask = np.asarray(roi, dtype=bool)
    if roi_mask.shape != shape:
        raise ValueError(f"ROI shape mismatch: roi={roi_mask.shape}, probability={shape}")
    for array in arrays:
        if not np.isfinite(array).all():
            raise ValueError("probabilities must be finite")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError("probabilities must lie in [0, 1]")
    if not roi_mask.any():
        return {"mean_entropy": 0.0, "p95_entropy": 0.0}
    mean_probability = np.mean(arrays, axis=0)
    entropy = np.zeros(shape, dtype=np.float64)
    active = np.logical_and(mean_probability > 0.0, mean_probability < 1.0)
    p = mean_probability[active]
    entropy[active] = -(p * np.log(p) + (1.0 - p) * np.log1p(-p))
    values = entropy[roi_mask]
    return {
        "mean_entropy": float(values.mean()),
        "p95_entropy": float(np.percentile(values, 95)),
    }


def _anatomy_instance_ids(partition: np.ndarray, lower: int, upper: int) -> list[int]:
    return [
        int(value)
        for value in np.unique(partition)
        if lower <= int(value) <= upper
    ]


def hungarian_instance_disagreement(
    first: np.ndarray,
    second: np.ndarray,
    *,
    roi: np.ndarray | None = None,
) -> float:
    """Return one minus anatomy-restricted Hungarian IoU similarity."""
    left = np.asarray(first)
    right = np.asarray(second)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: first={left.shape}, second={right.shape}")
    roi_mask = np.ones(left.shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
    if roi_mask.shape != left.shape:
        raise ValueError(f"ROI shape mismatch: roi={roi_mask.shape}, partitions={left.shape}")
    left = np.where(roi_mask, left, 0)
    right = np.where(roi_mask, right, 0)

    matched_iou = 0.0
    denominator = 0
    for lower, upper in ANATOMY_RANGES.values():
        left_ids = _anatomy_instance_ids(left, lower, upper)
        right_ids = _anatomy_instance_ids(right, lower, upper)
        denominator += max(len(left_ids), len(right_ids))
        if not left_ids or not right_ids:
            continue
        iou = np.zeros((len(left_ids), len(right_ids)), dtype=np.float64)
        for i, left_id in enumerate(left_ids):
            left_mask = left == left_id
            for j, right_id in enumerate(right_ids):
                right_mask = right == right_id
                union = int(np.logical_or(left_mask, right_mask).sum())
                if union:
                    iou[i, j] = np.logical_and(left_mask, right_mask).sum() / union
        row_ind, col_ind = linear_sum_assignment(-iou)
        matched_iou += float(iou[row_ind, col_ind].sum())
    if denominator == 0:
        return 0.0
    return float(np.clip(1.0 - matched_iou / denominator, 0.0, 1.0))


def case_uncertainty_scores(
    partitions: Sequence[np.ndarray],
    foreground_probabilities: Sequence[np.ndarray],
    *,
    roi: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute every frozen Stage 3 score for one case without ground truth."""
    maps = [np.asarray(partition) for partition in partitions]
    shape = _validate_same_shape(maps, "partition")
    probabilities = [np.asarray(array) for array in foreground_probabilities]
    if len(maps) < 2 or len(maps) != len(probabilities):
        raise ValueError("Stage 3 requires equal partition/probability member counts >=2")
    _validate_same_shape(probabilities, "probability")
    if roi is None:
        roi_mask = np.logical_or.reduce([partition != 0 for partition in maps])
    else:
        roi_mask = np.asarray(roi, dtype=bool)
    if roi_mask.shape != shape:
        raise ValueError(f"ROI shape mismatch: roi={roi_mask.shape}, partitions={shape}")

    pair_dice: list[float] = []
    pair_jaccard: list[float] = []
    pair_union_nvi: list[float] = []
    pair_hungarian: list[float] = []
    pair_fa: list[float] = []
    pair_partition: list[float] = []
    pair_weighted_partition: list[float] = []
    for left, right in combinations(maps, 2):
        left_roi = np.where(roi_mask, left, 0)
        right_roi = np.where(roi_mask, right, 0)
        pair_dice.append(1.0 - binary_dice(left_roi != 0, right_roi != 0))
        components = foreground_aware_partition_components(left, right, roi=roi_mask)
        pair_jaccard.append(float(components["presence_disagreement"]))
        pair_partition.append(float(components["partition_disagreement"]))
        pair_weighted_partition.append(
            float(components["weighted_partition_disagreement"])
        )
        pair_fa.append(float(components["score"]))
        union = np.logical_and(roi_mask, np.logical_or(left != 0, right != 0))
        pair_union_nvi.append(
            normalized_variation_of_information(left, right, domain=union)
        )
        pair_hungarian.append(
            hungarian_instance_disagreement(left, right, roi=roi_mask)
        )

    counts = [
        len([value for value in np.unique(partition[roi_mask]) if int(value) > 0])
        for partition in maps
    ]
    result = binary_predictive_entropy(probabilities, roi_mask)
    result.update(
        {
            "pairwise_semantic_dice_disagreement": float(np.mean(pair_dice)),
            "foreground_jaccard_disagreement": float(np.mean(pair_jaccard)),
            "union_foreground_nvi": float(np.mean(pair_union_nvi)),
            "fragment_count_variance": float(np.var(counts, ddof=0)),
            "hungarian_structural_disagreement": float(np.mean(pair_hungarian)),
            "fa_ipd": float(np.mean(pair_fa)),
            "fa_partition_nvi": float(np.mean(pair_partition)),
            "fa_weighted_partition_contribution": float(
                np.mean(pair_weighted_partition)
            ),
            "roi_voxels": int(roi_mask.sum()),
            "mean_member_fragment_count": float(np.mean(counts)),
        }
    )
    return result


def risk_coverage_curve(
    uncertainties: Sequence[float],
    risks: Sequence[float],
    case_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return empirical coverage and prefix risk sorted by uncertainty ascending."""
    if not (len(uncertainties) == len(risks) == len(case_ids)) or not case_ids:
        raise ValueError("uncertainties, risks, and non-empty case_ids must align")
    order = sorted(
        range(len(case_ids)),
        key=lambda index: (float(uncertainties[index]), str(case_ids[index])),
    )
    ordered_risk = np.asarray([float(risks[index]) for index in order], dtype=np.float64)
    prefix_risk = np.cumsum(ordered_risk) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1, dtype=np.float64) / len(order)
    return coverage, prefix_risk, [str(case_ids[index]) for index in order]


def aurc(
    uncertainties: Sequence[float],
    risks: Sequence[float],
    case_ids: Sequence[str],
) -> float:
    """Frozen arithmetic-mean empirical AURC; lower is better."""
    _, prefix_risk, _ = risk_coverage_curve(uncertainties, risks, case_ids)
    return float(prefix_risk.mean())


def top_k_failure_summary(
    uncertainties: Sequence[float],
    failures: Sequence[bool],
    risks: Sequence[float],
    case_ids: Sequence[str],
    fraction: float,
) -> dict[str, float | int | list[str]]:
    """Summarize the highest-uncertainty deterministic top-k subset."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if not (
        len(uncertainties) == len(failures) == len(risks) == len(case_ids)
    ) or not case_ids:
        raise ValueError("top-k inputs must be non-empty and aligned")
    order = sorted(
        range(len(case_ids)),
        key=lambda index: (-float(uncertainties[index]), str(case_ids[index])),
    )
    count = int(ceil(float(fraction) * len(order)))
    selected = order[:count]
    base_rate = float(np.mean(np.asarray(failures, dtype=bool)))
    selected_rate = float(np.mean([bool(failures[index]) for index in selected]))
    enrichment = (
        selected_rate / base_rate
        if base_rate > 0.0
        else (1.0 if selected_rate == 0.0 else float("inf"))
    )
    return {
        "selected_count": count,
        "selected_case_ids": [str(case_ids[index]) for index in selected],
        "failure_prevalence": selected_rate,
        "base_failure_prevalence": base_rate,
        "failure_enrichment": float(enrichment),
        "mean_selected_risk": float(np.mean([float(risks[index]) for index in selected])),
    }


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Return Spearman rho, using zero for an undefined constant-input result."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must align and contain at least two values")
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0
    rho = float(spearmanr(left, right).statistic)
    return 0.0 if not np.isfinite(rho) else rho


def residualized_spearman(
    score: Sequence[float],
    risk: Sequence[float],
    confounders: np.ndarray,
) -> float:
    """Spearman association after linear residualization on fixed confounders."""
    y_score = np.asarray(score, dtype=np.float64)
    y_risk = np.asarray(risk, dtype=np.float64)
    x = np.asarray(confounders, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y_score.size or y_score.shape != y_risk.shape:
        raise ValueError("residualization inputs have incompatible shapes")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0.0] = 1.0
    design = np.column_stack([np.ones(x.shape[0]), (x - means) / scales])
    score_residual = y_score - design @ np.linalg.lstsq(
        design, y_score, rcond=None
    )[0]
    risk_residual = y_risk - design @ np.linalg.lstsq(
        design, y_risk, rcond=None
    )[0]
    return safe_spearman(score_residual, risk_residual)
