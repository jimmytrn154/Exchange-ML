"""Prospective v4 fold-macro ranking evaluation and OOF scale diagnostics.

The primary estimand is the arithmetic mean of outer-fold bone-level AURCs.
Pooled OOF scores mix distinct learned score scales and remain diagnostics.
Bootstrap draws resample whole CTs within each fixed outer fold and scan type;
intervals condition on the fixed folds, models and predictions, without refits.
No independent-fold standard error or CT review utility is inferred here.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .e1_scoring import SCORE_NAMES
from .e1_statistics import aurc_batch, cluster_draws, expected_curve

FINAL_HEAD = "pairwise_p1_g"
RAW_SCORES = (*SCORE_NAMES, "graph_degree")
ALL_SCORES = (*RAW_SCORES, FINAL_HEAD)
SOURCES = ("member_mean", "stage2")
RISKS = ("merge_split", "one_minus_f1", "missing_volume")
ENDPOINTS = tuple((source, risk) for source in SOURCES for risk in RISKS)
CI_ROLE = "conditional_fixed_outer_folds_models_predictions_no_refit"
PRIMARY_ESTIMAND = "arithmetic_mean_outer_fold_bone_AURC"
POOLED_ESTIMAND = "pooled_OOF_AURC_score_scale_mixing_diagnostic_only"


def _settings(config: dict) -> tuple[int, int, int | None]:
    settings = config.get("reporting", config)
    count = settings.get("bootstrap_resamples", 10000)
    seed = settings.get("bootstrap_seed", 20261122)
    folds = config.get("cv", {}).get("outer_folds")
    for name, value, minimum in (("bootstrap_resamples", count, 1), ("bootstrap_seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if folds is not None and (isinstance(folds, bool) or not isinstance(folds, int) or folds < 2):
        raise ValueError("outer_folds must be an integer >=2")
    return count, seed, folds


def validate_rows(rows: Sequence[dict], expected_folds: int | None = None) -> list[dict]:
    if not rows:
        raise ValueError("nonempty OOF rows required")
    seen, cases = set(), {}
    for row in rows:
        cid, bone, scan, fold = (row[k] for k in ("case_id", "bone", "scan_type", "outer_fold"))
        if not isinstance(cid, str) or not cid or not isinstance(bone, str) or not bone:
            raise ValueError("nonempty case_id and bone strings required")
        if scan not in ("pelvic", "femur") or isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) or fold < 0:
            raise ValueError("declared scan type and nonnegative integer outer fold required")
        if (cid, bone) in seen:
            raise ValueError("duplicate case/bone")
        seen.add((cid, bone))
        if cid in cases and cases[cid] != (scan, int(fold)):
            raise ValueError("all bones of a CT must share scan type and outer fold")
        cases[cid] = (scan, int(fold))
        if not np.isfinite([row[name] for name in ALL_SCORES]).all():
            raise ValueError("all raw and candidate scores must be finite")
        if any(row[name] < 0 for name in RAW_SCORES) or row["partition_v1"] > 1 + 1e-10 or row["presence"] > 1 + 1e-10:
            raise ValueError("invalid raw score range")
        for source, risk in ENDPOINTS:
            value = row[f"{source}_{risk}"]
            if not np.isfinite(value) or value < 0 or (risk != "merge_split" and value > 1 + 1e-10):
                raise ValueError(f"invalid evaluation risk: {source}/{risk}")
    for cid, (scan, _) in cases.items():
        expected = {"sacrum", "left_hip", "right_hip"} if scan == "pelvic" else {"femur"}
        if {bone for case, bone in seen if case == cid} != expected:
            raise ValueError("complete declared anatomy membership required for each CT")
    folds = {fold for _, fold in cases.values()}
    count = expected_folds if expected_folds is not None else max(folds) + 1
    if count < 2 or folds != set(range(count)):
        raise ValueError("missing or noncontiguous outer fold")
    return sorted((dict(row) for row in rows), key=lambda r: (r["case_id"], r["bone"]))


def fold_cluster_draws(rows: list[dict], count: int, seed: int) -> dict[int, np.ndarray]:
    """Return global row indices [B,N_f], preserving each CT's whole bone block.

    Independent RNG seeds per fold are fixed before any outcome is inspected.
    Inside each fold, the existing cluster_draws stratifies by scan type.
    """
    result = {}
    for fold in sorted({r["outer_fold"] for r in rows}):
        indices = np.array([i for i, r in enumerate(rows) if r["outer_fold"] == fold], int)
        local_rows = [rows[i] for i in indices]
        local = cluster_draws(local_rows, count, seed + 1009 * int(fold))
        result[int(fold)] = indices[local]
    return result


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _interval(samples: np.ndarray | None) -> tuple[float | None, float | None]:
    if samples is None or not np.isfinite(samples).all():
        return None, None
    low, high = np.quantile(samples, [.025, .975])
    return float(low), float(high)


def _points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.full((len(ALL_SCORES), len(ENDPOINTS)), np.nan)
    return np.stack([aurc_batch(x[:, s][None], y[None])[0] for s in range(len(ALL_SCORES))])


def _bootstrap(x: np.ndarray, y: np.ndarray, take: np.ndarray) -> np.ndarray:
    result = np.empty((len(take), len(ALL_SCORES), len(ENDPOINTS)), float)
    if take.shape[1] < 2:
        result.fill(np.nan)
        return result
    for start in range(0, len(take), 128):
        draw = take[start:start + 128]
        for s in range(len(ALL_SCORES)):
            result[start:start + len(draw), s] = aurc_batch(x[draw, s], y[draw])
    return result


def _score_diagnostics(rows: list[dict], x: np.ndarray) -> list[dict]:
    folds = np.array([r["outer_fold"] for r in rows])
    p1 = x[:, ALL_SCORES.index("partition_v1")]
    delta = p1[:, None] - p1
    eligible = np.triu(np.ones((len(rows), len(rows)), bool), 1) & (delta != 0)
    result = []
    for j, score in enumerate(ALL_SCORES):
        score_delta = x[:, j, None] - x[:, j]
        for scope, scope_mask in (("within_fold", folds[:, None] == folds),
                                  ("between_folds", folds[:, None] != folds)):
            mask = eligible & scope_mask
            result.append({"diagnostic": "order_relative_to_raw_P1", "scope": scope, "score": score,
                           "outer_fold": None, "n_non_tied_P1_pairs": int(mask.sum()),
                           "P1_order_inversions": int(np.count_nonzero(mask & (delta * score_delta < 0))),
                           "P1_order_collapses_to_ties": int(np.count_nonzero(mask & (score_delta == 0))),
                           "role": "descriptive_not_error_count_or_evaluation_endpoint"})
        for fold in sorted(set(folds)):
            values = x[folds == fold, j]
            result.append({"diagnostic": "score_distribution", "scope": "outer_fold", "score": score,
                           "outer_fold": int(fold), "score_min": float(values.min()),
                           "score_max": float(values.max()), "score_mean": float(values.mean()),
                           "score_std": float(values.std()),
                           "role": "descriptive_rank_score_scale_not_probability_calibration"})
    return result


def report(rows: Sequence[dict], config: dict) -> dict:
    """Evaluate one fixed candidate without choosing a new winning family."""
    count, seed, requested_folds = _settings(config)
    rows = validate_rows(rows, requested_folds)
    folds = sorted({r["outer_fold"] for r in rows})
    x = np.array([[r[s] for s in ALL_SCORES] for r in rows], float)  # [N,S]
    y = np.array([[r[f"{source}_{risk}"] for source, risk in ENDPOINTS] for r in rows], float)  # [N,E]
    scan_types = np.array([r["scan_type"] for r in rows])
    assignment = np.array([r["outer_fold"] for r in rows])
    draws = fold_cluster_draws(rows, count, seed)
    comparison, paired, fold_points, curves, pooled, pooled_paired = [], [], [], [], [], []
    candidate_index = ALL_SCORES.index(FINAL_HEAD)
    for stratum in ("pooled", "pelvic", "femur"):
        selected = np.ones(len(rows), bool) if stratum == "pooled" else scan_types == stratum
        if not selected.any():
            continue
        points, bootstraps, selected_draws, min_cases = [], [], [], []
        for fold in folds:
            indices = np.flatnonzero(selected & (assignment == fold))
            sub = [rows[i] for i in indices]
            n_cases = len({r["case_id"] for r in sub})
            min_cases.append(n_cases)
            point = _points(x[indices], y[indices])
            points.append(point)
            take = draws[fold]
            if stratum != "pooled":
                # Stratification fixes scan type at each bootstrap column.
                take = take[:, scan_types[take[0]] == stratum]
            selected_draws.append(take)
            bootstraps.append(_bootstrap(x, y, take))
            common = {"stratum": stratum, "outer_fold": int(fold), "n_cases": n_cases,
                      "n_units": len(indices), "estimand": "outer_fold_bone_AURC",
                      "analysis_role": "development_only_not_fresh_confirmation"}
            for s, score in enumerate(ALL_SCORES):
                for e, (source, risk) in enumerate(ENDPOINTS):
                    status = "insufficient_units" if len(indices) < 2 else "insufficient_cases" if n_cases < 2 else "constant_risk_uninformative" if np.ptp(y[indices, e]) == 0 else "estimated"
                    fold_points.append({**common, "score": score, "risk_source": source, "risk": risk,
                                        "aurc": _number(point[s, e]), "status": status})
                    if len(indices):
                        curves.extend({**common, "score": score, "risk_source": source, "risk": risk,
                                       "coverage": (k + 1) / len(indices), "retained_mean_risk": float(value)}
                                      for k, value in enumerate(expected_curve(x[indices, s], y[indices, e])))
        point_mean = np.mean(points, axis=0)
        bootstrap_mean = np.mean(bootstraps, axis=0)
        can_bootstrap = min(min_cases) >= 2
        point_pooled = _points(x[selected], y[selected])
        take_pooled = np.concatenate(selected_draws, axis=1)
        boot_pooled = _bootstrap(x, y, take_pooled)
        common = {"stratum": stratum, "n_cases": len({r["case_id"] for i, r in enumerate(rows) if selected[i]}),
                  "n_units": int(selected.sum()), "n_outer_folds": len(folds), "ci_role": CI_ROLE,
                  "bootstrap_resamples": count, "bootstrap_unit": "CT_within_fixed_outer_fold_and_scan_type",
                  "analysis_role": "development_only_not_fresh_confirmation"}
        for estimand, point, boot, output, contrasts in (
                (PRIMARY_ESTIMAND, point_mean, bootstrap_mean, comparison, paired),
                (POOLED_ESTIMAND, point_pooled, boot_pooled, pooled, pooled_paired)):
            for e, (source, risk) in enumerate(ENDPOINTS):
                constant = np.ptp(y[selected, e]) == 0
                every_fold_constant = all(len(np.flatnonzero(selected & (assignment == fold))) > 0
                                          and np.ptp(y[selected & (assignment == fold), e]) == 0 for fold in folds)
                for s, score in enumerate(ALL_SCORES):
                    status = ("insufficient_units_in_outer_fold" if not np.isfinite(point[s, e])
                              else "insufficient_cases_in_outer_fold" if not can_bootstrap
                              else "constant_risk_uninformative" if constant
                              else "constant_within_every_fold_uninformative" if every_fold_constant and estimand == PRIMARY_ESTIMAND
                              else "estimated")
                    samples = boot[:, s, e] if can_bootstrap else None
                    lo, hi = _interval(samples)
                    ci_status = ("insufficient_cases_in_outer_fold" if not can_bootstrap
                                 else "undefined_bootstrap_draws" if not np.isfinite(samples).all() else "estimated")
                    output.append({**common, "estimand": estimand, "score": score, "risk_source": source,
                                   "risk": risk, "aurc": _number(point[s, e]), "ci_low": lo, "ci_high": hi,
                                   "status": status, "ci_status": ci_status,
                                   "evaluation_role": "primary_fold_macro" if estimand == PRIMARY_ESTIMAND else "secondary_scale_mixing_diagnostic"})
                    if score != FINAL_HEAD:
                        continue
                    for baseline in RAW_SCORES:
                        b = ALL_SCORES.index(baseline)
                        difference = boot[:, candidate_index, e] - boot[:, b, e] if can_bootstrap else None
                        dlo, dhi = _interval(difference)
                        role = ("v4_primary" if estimand == PRIMARY_ESTIMAND and stratum == "pooled"
                                and source == "member_mean" and risk == "merge_split" and baseline == "partition_v1"
                                else "secondary" if estimand == PRIMARY_ESTIMAND else "diagnostic_only")
                        contrasts.append({**common, "estimand": estimand, "score": FINAL_HEAD,
                                          "baseline": baseline, "risk_source": source, "risk": risk,
                                          "delta_aurc": _number(point[candidate_index, e] - point[b, e]),
                                          "ci_low": dlo, "ci_high": dhi, "status": status,
                                          "ci_status": ci_status, "contrast_role": role})
    return {"comparison": comparison, "paired_differences": paired, "fold_comparison": fold_points,
            "fold_risk_coverage": curves, "pooled_comparison": pooled,
            "pooled_paired_differences": pooled_paired, "score_diagnostics": _score_diagnostics(rows, x),
            "summary": {"n_cases": len({r["case_id"] for r in rows}), "n_units": len(rows),
                        "n_outer_folds": len(folds), "candidate": FINAL_HEAD, "scores": list(ALL_SCORES),
                        "primary_estimand": PRIMARY_ESTIMAND,
                        "primary_contrast": "pairwise_p1_g-partition_v1/pooled/member_mean/merge_split",
                        "pooled_OOF_role": "secondary_score_scale_mixing_diagnostic_only",
                        "ci_role": CI_ROLE, "bootstrap_resamples": count,
                        "bootstrap_seed": seed, "fold_standard_errors_used": False,
                        "CT_review_utility_tested": False,
                        "confirmation_ready": False,
                        "analysis_role": "prospective_v4_development_after_v3_inspection_not_fresh_confirmation"}}
