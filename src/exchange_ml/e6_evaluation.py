"""Case-level transfer of the frozen E3 scores to RibFrac fracture instances.

The target is fracture lesions, not whole rib/bone fragments. All nonzero
RibFrac label IDs, including label_code=-1, are segmentation/detection targets;
the code -1 excludes *classification*, not segmentation (see the E6 source
report). Scoring never accepts a reference or reference-derived ignore mask.

Matching deliberately preserves the internal E1 best-IoU GT assignment and
any-overlap split convention, without PENGWIN anatomy ID ranges. This is not
the official RibFrac detection/classification evaluator. In particular internal
F1 can equal one despite merge/split errors, so structural and volume endpoints
are reported separately. Ground truth is never component-pruned.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .e1_scoring import SCORE_NAMES
from .e1_statistics import aurc_batch, correlation, expected_curve
from .metrics import _entropy_from_counts, binary_dice

MEMBERS = ("member_01", "member_02", "member_03")
RISKS = ("merge_split", "merge", "split", "missing", "one_minus_f1",
         "missing_volume", "one_minus_dice", "false_positive_instances")
STRUCTURAL_RISKS = ("merge_split", "merge", "split", "missing", "missing_volume")


def _labels(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.size == 0 or not np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.bool_) or np.any(array < 0)):
        raise ValueError(f"{name} must contain nonnegative integer instance IDs")
    return array


def _contingency(first: np.ndarray, second: np.ndarray) -> dict:
    """Sparse counts with compact IDs; memory does not grow as max(label)^2."""
    left_ids, li = np.unique(first.ravel(), return_inverse=True)
    right_ids, ri = np.unique(second.ravel(), return_inverse=True)
    width = len(right_ids)
    codes, counts = np.unique(li.astype(np.int64) * width + ri, return_counts=True)
    rows, cols = codes // width, codes % width
    return {"left_ids": left_ids, "right_ids": right_ids,
            "rows": rows, "cols": cols, "counts": counts,
            "left_counts": np.bincount(li, minlength=len(left_ids)),
            "right_counts": np.bincount(ri, minlength=len(right_ids))}


def _information(counts: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> dict[str, float]:
    n = int(counts.sum())
    joint = _entropy_from_counts(counts)
    left = np.bincount(rows, weights=counts) if len(rows) else np.empty(0)
    right = np.bincount(cols, weights=counts) if len(cols) else np.empty(0)
    vi = max(0., 2 * joint - _entropy_from_counts(left) - _entropy_from_counts(right))
    if vi > joint + 1e-10:
        raise ArithmeticError("VI exceeds joint entropy")
    return {"vi": vi, "joint_entropy": joint,
            "q_log_n": float(np.clip(vi / np.log(n), 0, 1)) if n > 1 else 0.,
            "q_joint": float(np.clip(vi / joint, 0, 1)) if joint > 0 and n > 1 else 0.}


def _hungarian_similarity(table: dict) -> float:
    """Exact maximum summed IoU / max(K_i,K_j), allowing unmatched objects.

    Dummy columns give each left object an unmatched option. Cost 2-IoU on
    positive-overlap edges and 2 on dummies makes a full sparse assignment
    equivalent to maximum summed IoU; absent/zero-overlap edges need no storage.
    Unlike E1, ID values carry no anatomical meaning in this external dataset.
    """
    lids, rids = table["left_ids"], table["right_ids"]
    nl, nr = int(np.count_nonzero(lids)), int(np.count_nonzero(rids))
    if not nl or not nr:
        return 1. if not (nl or nr) else 0.
    # np.unique sorts nonnegative labels: zero, if present, is the first entry.
    rows, cols, counts = table["rows"], table["cols"], table["counts"]
    active = (lids[rows] > 0) & (rids[cols] > 0)
    rr = rows[active] - int(lids[0] == 0)
    cc = cols[active] - int(rids[0] == 0)
    intersection = counts[active].astype(np.float64)
    union = table["left_counts"][rows[active]] + table["right_counts"][cols[active]] - intersection
    iou = intersection / union
    graph = coo_matrix((np.r_[2 - iou, np.full(nl, 2.)],
                        (np.r_[rr, np.arange(nl)], np.r_[cc, nr + np.arange(nl)])),
                       shape=(nl, nr + nl)).tocsr()
    matched_rows, matched_cols = min_weight_full_bipartite_matching(graph)
    costs = np.asarray(graph[matched_rows, matched_cols]).ravel()
    return float(np.clip(np.sum(2 - costs) / max(nl, nr), 0, 1))


def pair_components(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """Presence plus pure/shared and mixed/union VI on generic instance IDs."""
    left, right = _labels(first, "first partition"), _labels(second, "second partition")
    if left.shape != right.shape:
        raise ValueError("partition shapes differ")
    table = _contingency(left, right)
    rows, cols, counts = table["rows"], table["cols"], table["counts"]
    lf, rf = table["left_ids"][rows] > 0, table["right_ids"][cols] > 0
    union, shared = lf | rf, lf & rf
    n_union, n_shared = int(counts[union].sum()), int(counts[shared].sum())
    info = _information(counts[shared], rows[shared], cols[shared])
    presence = (n_union - n_shared) / n_union if n_union else 0.
    weight = n_shared / n_union if n_union else 0.
    ni, nj = int(counts[lf].sum()), int(counts[rf].sum())
    return {"n_union_voxels": n_union, "n_shared_voxels": n_shared,
            "k_i": int(np.count_nonzero(table["left_ids"])),
            "k_j": int(np.count_nonzero(table["right_ids"])),
            **info, "presence": presence, "overlap_weight": weight,
            "weighted_q_log_n": weight * info["q_log_n"],
            "weighted_q_joint": weight * info["q_joint"],
            "fa_v1": presence + weight * info["q_log_n"],
            "fa_v2": presence + weight * info["q_joint"],
            "pairwise_dice": 1 - 2 * n_shared / (ni + nj) if ni + nj else 0.,
            "union_nvi": _information(counts[union], rows[union], cols[union])["q_log_n"],
            "hungarian": 1 - _hungarian_similarity(table)}


def score_case(partitions: Sequence[np.ndarray], probabilities: Sequence[np.ndarray]
               ) -> tuple[dict[str, float], list[dict]]:
    """Score exactly three members on their union foreground, without GT.

    Instance maps and probabilities have identical spatial shape [Z,Y,X].
    Float probabilities must be true model foreground outputs, not generated
    from hard instance maps. Empty ensemble foreground has all scores zero,
    matching the frozen E3 empty-ROI convention.
    """
    if len(partitions) != 3 or len(probabilities) != 3:
        raise ValueError("E6 requires exactly three independent members")
    maps = [_labels(p, "partition") for p in partitions]
    shape = maps[0].shape
    if len(shape) != 3 or any(p.shape != shape for p in maps):
        raise ValueError("expected equal 3D partition shapes")
    for value in probabilities:
        p = np.asarray(value)
        if (p.shape != shape or not np.issubdtype(p.dtype, np.floating)
                or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
            raise ValueError("probabilities must be floating, finite, same-shape arrays in [0,1]")
    roi = np.logical_or.reduce([p > 0 for p in maps])
    if roi.any():
        pairs = [pair_components(maps[i][roi], maps[j][roi]) for i, j in combinations(range(3), 2)]
        probability = sum(np.asarray(p)[roi].astype(np.float64) for p in probabilities) / 3
        entropy = np.zeros_like(probability)
        interior = (probability > 0) & (probability < 1)
        p = probability[interior]
        entropy[interior] = -(p * np.log(p) + (1 - p) * np.log1p(-p))
        mean_entropy, p95_entropy = float(entropy.mean()), float(np.percentile(entropy, 95))
    else:
        empty = np.zeros(1, dtype=np.uint32)
        pairs = [pair_components(empty, empty) for _ in range(3)]
        mean_entropy = p95_entropy = 0.
    mean = lambda name: float(np.mean([pair[name] for pair in pairs]))
    scores = {"mean_entropy": mean_entropy, "p95_entropy": p95_entropy,
              **{name: mean(name) for name in ("pairwise_dice", "presence", "union_nvi",
                                               "hungarian", "fa_v1", "fa_v2")},
              "count_variance": float(np.var([np.count_nonzero(np.unique(p)) for p in maps])),
              "partition_v1": mean("weighted_q_log_n"),
              "partition_v2": mean("weighted_q_joint"),
              "q_joint_unweighted": mean("q_joint")}
    if set(scores) != set(SCORE_NAMES) or not np.isfinite(list(scores.values())).all():
        raise ArithmeticError("invalid E6 score output")
    return scores, pairs


def model_risk(prediction: np.ndarray, reference: np.ndarray, spacing_zyx: Sequence[float],
               *, iou_threshold: float = .1, ignore_mask: np.ndarray | None = None) -> dict:
    """Internal case-level matching, preserving generic GT instance identities.

    Best-IoU assignment is per GT, with ties assigned to the lowest prediction
    ID, and is many-to-one. A split is each extra positive prediction touching
    a GT instance, irrespective of IoU. Optional ignored voxels affect only
    evaluation and do not split or relabel retained identities. E6's RibFrac
    default has no ignored voxels; label_code=-1 is retained upstream.
    """
    pred, gt = _labels(prediction, "prediction"), _labels(reference, "reference")
    spacing = np.asarray(spacing_zyx, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("expected equal 3D prediction/reference shapes")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three finite positive values")
    if not np.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0,1]")
    if ignore_mask is not None:
        ignored = np.asarray(ignore_mask)
        if ignored.shape != gt.shape or ignored.dtype != np.bool_:
            raise ValueError("ignore_mask must be a same-shape boolean array")
        if ignored.all():
            raise ValueError("no evaluable voxels remain after ignore masking")
        pred, gt = np.where(ignored, 0, pred), np.where(ignored, 0, gt)
    table = _contingency(gt, pred)
    gids, pids = table["left_ids"], table["right_ids"]
    rows, cols, counts = table["rows"], table["cols"], table["counts"]
    active = (gids[rows] > 0) & (pids[cols] > 0)
    rr, cc, overlap = rows[active], cols[active], counts[active]
    ious = overlap / (table["left_counts"][rr] + table["right_counts"][cc] - overlap)
    matched_pred = []
    matches = []
    splits = 0
    for index in np.flatnonzero(gids > 0):
        edge = np.flatnonzero(rr == index)
        splits += max(len(edge) - 1, 0)
        best = edge[np.argmax(ious[edge])] if len(edge) else None
        pred_id = int(pids[cc[best]]) if best is not None and ious[best] >= iou_threshold else None
        if pred_id is not None:
            matched_pred.append(pred_id)
        matches.append({"gt_id": int(gids[index]), "pred_id": pred_id,
                        "iou": float(ious[best]) if best is not None else 0.})
    n_gt, n_pred = int(np.count_nonzero(gids)), int(np.count_nonzero(pids))
    assignments = np.unique(matched_pred, return_counts=True)[1]
    merges = int(np.maximum(assignments - 1, 0).sum())
    precision = len(set(matched_pred)) / max(n_pred, 1)
    recall = len(matched_pred) / max(n_gt, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    if not n_gt:
        f1 = precision = recall = 1. if not n_pred else 0.
    gt_voxels = int(np.count_nonzero(gt))
    voxel_volume = float(np.prod(spacing))
    structural = {"merge_split": (merges + splits) / n_gt, "merge": merges / n_gt,
                  "split": splits / n_gt, "missing": 1 - recall,
                  "missing_volume": 1 - int(overlap.sum()) / gt_voxels} if n_gt else {
                      risk: None for risk in STRUCTURAL_RISKS}
    return {"n_GT": n_gt, "n_pred": n_pred, "n_ignored_voxels": int(np.count_nonzero(ignore_mask)) if ignore_mask is not None else 0,
            "gt_volume_mm3": gt_voxels * voxel_volume,
            "min_gt_fragment_volume_mm3": float(table["left_counts"][gids > 0].min() * voxel_volume) if n_gt else None,
            "structural_eligible": bool(n_gt), "instance_precision": precision,
            "instance_recall": recall, "instance_f1": f1,
            "merge_errors": merges if n_gt else None, "split_errors": int(splits) if n_gt else None,
            "missed_gt_instances": n_gt - len(matched_pred) if n_gt else None,
            "gt_instance_count": n_gt, "pred_instance_count": n_pred,
            "binary_dice": binary_dice(pred > 0, gt > 0),
            **structural, "one_minus_f1": 1 - f1,
            "one_minus_dice": 1 - binary_dice(pred > 0, gt > 0),
            "false_positive_instances": n_pred - len(set(matched_pred)),
            "matches": matches}


def evaluate_predictions(case_id: str, partitions: Sequence[np.ndarray], reference: np.ndarray,
                         spacing_zyx: Sequence[float], *, iou_threshold: float = .1,
                         ignore_mask: np.ndarray | None = None) -> dict:
    """Evaluate model risks after the caller has persisted GT-free scores."""
    if not isinstance(case_id, str) or not case_id or len(partitions) != 3:
        raise ValueError("nonempty case ID and exactly three predictions required")
    evaluated = [model_risk(p, reference, spacing_zyx, iou_threshold=iou_threshold,
                            ignore_mask=ignore_mask) for p in partitions]
    rows = [{"case_id": case_id, "model_id": member,
             **{key: value for key, value in risk.items() if key != "matches"}}
            for member, risk in zip(MEMBERS, evaluated)]
    common = {"case_id": case_id,
              **{key: evaluated[0][key] for key in ("n_GT", "gt_volume_mm3",
                 "min_gt_fragment_volume_mm3", "structural_eligible", "n_ignored_voxels")},
              "n_pred_mean": float(np.mean([r["n_pred"] for r in evaluated]))}
    for name in RISKS:
        values = [r[name] for r in evaluated]
        common["member_mean_" + name] = float(np.mean(values)) if all(v is not None for v in values) else None
    return {"model_risks": rows, "risks": [common],
            "matches": [{"case_id": case_id, "model_id": member, **match}
                        for member, risk in zip(MEMBERS, evaluated) for match in risk["matches"]]}


def _number(value) -> float | None:
    if value is None or value == "":
        return None
    observed = float(value)
    if not np.isfinite(observed):
        raise ValueError("nonfinite score or risk")
    return observed


def report_tables(score_rows: list[dict], risk_rows: list[dict], *,
                  bootstrap_resamples: int = 10000, bootstrap_seed: int = 20261121) -> dict:
    """Paired case bootstrap with the frozen E3 exact-tie AURC convention.

    AURC integrates coverage 1/n..1, without an artificial coverage=0 point.
    Negatives are excluded only for undefined structural endpoints; F1, Dice
    and false-positive endpoints retain them. Case-bootstrap draws are shared
    by all scores for each endpoint. No effect-size gate is invented for E6.
    Constant-risk endpoints are explicitly uninformative, even if AURC exists.
    """
    if not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    scores = {str(row["case_id"]): row for row in score_rows}
    risks = {str(row["case_id"]): row for row in risk_rows}
    if (len(scores) != len(score_rows) or len(risks) != len(risk_rows)
            or not scores or scores.keys() != risks.keys()):
        raise ValueError("score/risk tables need unique and identical nonempty case membership")
    cases = sorted(scores)
    for case in cases:
        count = _number(risks[case]["n_GT"])
        if count is None or count < 0 or not count.is_integer():
            raise ValueError(f"n_GT must be a nonnegative integer: {case}")
    n = len(cases)
    values = np.asarray([[_number(scores[c][name]) for name in SCORE_NAMES] for c in cases], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("all 12 scores must be finite for every case")
    for j, name in enumerate(SCORE_NAMES):
        if np.any(values[:, j] < 0) or (name != "count_variance" and np.any(values[:, j] > (np.log(2) if name in ("mean_entropy", "p95_entropy") else 1) + 1e-10)):
            raise ValueError(f"score outside its expected range: {name}")
    rng = np.random.default_rng(bootstrap_seed)
    comparison, contrasts, curves, endpoint_summary = [], [], [], {}
    score_index = SCORE_NAMES.index("fa_v2")
    for risk in RISKS:
        column = [_number(risks[c]["member_mean_" + risk]) for c in cases]
        if any(v is None for v in column) and risk not in STRUCTURAL_RISKS:
            raise ValueError(f"risk must be defined for all cases: {risk}")
        eligible = np.array([v is not None for v in column])
        if risk in STRUCTURAL_RISKS:
            for c, value in zip(cases, column):
                if (int(risks[c]["n_GT"]) > 0) != (value is not None):
                    raise ValueError(f"structural endpoint eligibility differs from n_GT: {c}/{risk}")
        y = np.asarray([v for v in column if v is not None], dtype=float)
        x = values[eligible]
        if np.any(y < 0) or (risk in ("missing", "missing_volume", "one_minus_f1", "one_minus_dice") and np.any(y > 1 + 1e-10)):
            raise ValueError(f"risk outside its expected range: {risk}")
        m = len(y)
        informative = m >= 2 and bool(np.ptp(y) > 0)
        status = "estimated" if informative else "insufficient_cases" if m < 2 else "constant_risk_uninformative"
        endpoint_summary[risk] = {"n_cases": m, "excluded_undefined_cases": n - m,
                                  "n_positive_risk": int(np.count_nonzero(y)),
                                  "risk_min": float(y.min()) if m else None,
                                  "risk_max": float(y.max()) if m else None,
                                  "risk_mean": float(y.mean()) if m else None,
                                  "status": status}
        point = np.full(len(SCORE_NAMES), np.nan)
        boot = np.full((bootstrap_resamples, len(SCORE_NAMES)), np.nan)
        if m >= 2:
            # Shared random choices are paired across every score. Each row is
            # a complete case; repeated case IDs are retained with multiplicity.
            draws = rng.integers(m, size=(bootstrap_resamples, m))
            for j, name in enumerate(SCORE_NAMES):
                point[j] = aurc_batch(x[:, j][None], y[None, :, None])[0, 0]
                if informative:
                    for start in range(0, bootstrap_resamples, 256):
                        take = draws[start:start + 256]
                        boot[start:start + len(take), j] = aurc_batch(x[take, j], y[take, None])[..., 0]
                curve = expected_curve(x[:, j], y)
                curves.extend({"risk_source": "member_mean", "risk": risk, "score": name,
                               "n_cases": m, "coverage": (k + 1) / m, "retained_risk": float(v)}
                              for k, v in enumerate(curve))
        for j, name in enumerate(SCORE_NAMES):
            low, high = np.quantile(boot[:, j], [.025, .975]) if informative else (None, None)
            comparison.append({"risk_source": "member_mean", "risk": risk, "score": name,
                "n_cases": m, "excluded_undefined_cases": n - m,
                "aurc": float(point[j]) if m >= 2 else None,
                "ci_low": float(low) if low is not None else None,
                "ci_high": float(high) if high is not None else None,
                "spearman": correlation(x[:, j], y) if m >= 2 else None,
                "status": status, "n_distinct_scores": len(np.unique(x[:, j])) if m else 0})
            if j != score_index:
                difference = boot[:, score_index] - boot[:, j]
                lo, hi = np.quantile(difference, [.025, .975]) if informative else (None, None)
                contrasts.append({"risk_source": "member_mean", "risk": risk,
                    "score": "fa_v2", "baseline": name, "n_cases": m,
                    "delta_aurc": float(point[score_index] - point[j]) if m >= 2 else None,
                    "ci_low": float(lo) if lo is not None else None,
                    "ci_high": float(hi) if hi is not None else None,
                    "direction": "lower_is_better", "status": status,
                    "favorable_marginal_ci": bool(hi < 0) if hi is not None else None})
    summary = {"analysis_role": "external_transfer_secondary_no_new_primary_gate",
        "unit": "case", "n_cases": n,
        "n_positive_gt_cases": sum(int(risks[c]["n_GT"]) > 0 for c in cases),
        "n_negative_gt_cases": sum(int(risks[c]["n_GT"]) == 0 for c in cases),
        "bootstrap_resamples": bootstrap_resamples, "bootstrap_seed": bootstrap_seed,
        "confidence_level": .95, "multiplicity": "marginal_CIs_no_familywise_adjustment",
        "endpoints": endpoint_summary,
        "limitations": ["fracture_lesion_instances_are_not_whole_bone_fragments",
                        "internal_matching_not_official_RibFrac_evaluator",
                        "constant_structural_risk_cannot_establish_failure_detection",
                        "no_PENGWIN_RibFrac_AURC_pooling"]}
    return {"comparison": comparison, "paired_differences": contrasts,
            "risk_coverage": curves, "summary": summary}
