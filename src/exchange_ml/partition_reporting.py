"""Development-only ranking and CT-cost-matched review for partition heads.

Bootstrap intervals condition on the fixed out-of-fold predictions. They do
not refit the nested CV procedure, quantify training uncertainty, or establish
fresh confirmation. CT review means withholding a case from automatic use;
perfect human correction or measured reading-time savings are not assumed.
"""
from __future__ import annotations

from functools import lru_cache
from math import ceil

import numpy as np
from scipy.stats import hypergeom

from .e1_scoring import SCORE_NAMES
from .e1_statistics import aurc_batch, cluster_draws, correlation, expected_curve
from .partition_cv import FINAL_HEAD, HEAD_FEATURES, validate_rows

NEW_SCORES = ("graph_degree", *HEAD_FEATURES)
ALL_SCORES = (*SCORE_NAMES, *NEW_SCORES)
SOURCES = ("member_mean", "stage2")
RISKS = ("merge_split", "one_minus_f1", "missing_volume")
ENDPOINTS = tuple((source, risk) for source in SOURCES for risk in RISKS)
CI_ROLE = "descriptive_conditional_fixed_OOF_scores_not_refitted_training_uncertainty"


def _settings(config: dict) -> tuple[int, int, tuple[float, ...]]:
    settings = config.get("reporting", config)
    resamples = settings.get("bootstrap_resamples", 10000)
    seed = settings.get("bootstrap_seed", 20261122)
    fractions = tuple(settings.get("ct_review_fractions", (.1, .2)))
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("positive integer bootstrap_resamples required")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("nonnegative integer bootstrap_seed required")
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0 < v < 1 for v in fractions):
        raise ValueError("distinct CT review fractions strictly between zero and one required")
    return resamples, seed, fractions


def _validate(rows: list[dict]) -> None:
    validate_rows(rows)
    for row in rows:
        if not np.isfinite([row[s] for s in ALL_SCORES]).all():
            raise ValueError("all baseline, graph and OOF head scores must be finite")
        for source, risk in ENDPOINTS:
            value = row[f"{source}_{risk}"]
            if not np.isfinite(value) or value < -1e-12 or (risk != "merge_split" and value > 1 + 1e-12):
                raise ValueError(f"invalid risk: {source}/{risk}")
        flag = row.get("all_members_empty")
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError("all_members_empty must be an explicit boolean")


def _interval(values: np.ndarray | None) -> tuple[float | None, float | None]:
    if values is None or not np.isfinite(values).all():
        return None, None
    low, high = np.quantile(values, [.025, .975])
    return float(low), float(high)


@lru_cache(maxsize=4096)
def _composition_weights(n_first: int, n_second: int, selected: int) -> tuple[tuple[int, float], ...]:
    low, high = max(0, selected - n_second), min(n_first, selected)
    return tuple((k, float(hypergeom.pmf(k, n_first + n_second, n_first, selected)))
                 for k in range(low, high + 1))


def retained_ct_expectation(scores: np.ndarray, risk_sums: np.ndarray, bone_counts: np.ndarray,
                            retained: int, eligible: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Expected case-macro and bone-macro retained risks at an exact CT budget.

    scores [C], risk_sums [C,E], bone_counts [C]. Exact ties are selected
    uniformly without replacement. Bone-macro is E[sum risk/sum bones], NOT
    the generally different ratio of expected numerator and denominator.
    Hypergeometric conditioning handles the two PENGWIN case-size groups.
    """
    x, y, counts = np.asarray(scores, float), np.asarray(risk_sums, float), np.asarray(bone_counts)
    if (x.ndim != 1 or y.ndim != 2 or y.shape[0] != len(x) or counts.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 1)):
        raise ValueError("finite scores [C], risk sums [C,E], positive integer bone counts [C] required")
    active = np.ones(len(x), bool) if eligible is None else np.asarray(eligible)
    if active.shape != x.shape or active.dtype != np.bool_:
        raise ValueError("eligible must be a case-level boolean mask")
    if not isinstance(retained, (int, np.integer)) or not 0 <= retained <= len(x):
        raise ValueError("retained CT count is out of range")
    undefined = np.full(y.shape[1], np.nan)
    if retained == 0 or int(active.sum()) < retained:
        return undefined.copy(), undefined.copy()
    cutoff = np.partition(x[active], retained - 1)[retained - 1]
    below, tied = active & (x < cutoff), active & (x == cutoff)
    remaining = retained - int(below.sum())
    weights = below.astype(float) + tied * (remaining / int(tied.sum()))
    case_macro = np.sum(weights[:, None] * y / counts[:, None], axis=0) / retained
    sizes = np.unique(counts[tied])
    if len(sizes) > 2:
        raise ValueError("CT tie expectation supports at most two distinct bone counts")
    fixed_sum, fixed_count = y[below].sum(0), int(counts[below].sum())
    first = tied & (counts == sizes[0])
    second = tied & ~first
    if len(sizes) == 1:
        bone_macro = (fixed_sum + remaining * y[first].mean(0)) / (fixed_count + remaining * sizes[0])
    else:
        bone_macro = np.zeros(y.shape[1], float)
        for k, probability in _composition_weights(int(first.sum()), int(second.sum()), remaining):
            numerator = fixed_sum + k * y[first].mean(0) + (remaining - k) * y[second].mean(0)
            denominator = fixed_count + k * sizes[0] + (remaining - k) * sizes[1]
            bone_macro += probability * numerator / denominator
    return case_macro, bone_macro


def _case_arrays(rows: list[dict], names: tuple[str, ...]) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cases = sorted({r["case_id"] for r in rows})
    metadata, score_values, risk_sums, sizes, flags = [], [], [], [], []
    for cid in cases:
        group = [r for r in rows if r["case_id"] == cid]
        metadata.append({"case_id": cid, "scan_type": group[0]["scan_type"]})
        score_values.append([max(float(r[name]) for r in group) for name in names])
        risk_sums.append([sum(float(r[f"{source}_{risk}"]) for r in group) for source, risk in ENDPOINTS])
        sizes.append(len(group))
        flags.append(any(r["all_members_empty"] for r in group))
    return metadata, np.asarray(score_values), np.asarray(risk_sums), np.asarray(sizes, int), np.asarray(flags, bool)


def _ct_tables(rows: list[dict], resamples: int, seed: int, fractions: tuple[float, ...],
               include_guard: bool) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    policies = (*ALL_SCORES, "random", *(("guard_only", "guard_" + FINAL_HEAD) if include_guard else ()))
    points, contrasts, omission_points, omission_contrasts = [], [], [], []
    for stratum in ("pooled", "pelvic", "femur"):
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        if not sub:
            continue
        metadata, x, y, sizes, flags = _case_arrays(sub, ALL_SCORES)
        n = len(metadata)
        # These GT-derived labels are used only after scoring, for evaluation.
        # A nonempty prediction at the wrong location can also completely miss
        # an expected bone; all_members_empty is a narrower prediction-only flag.
        positive = np.array([[any(r[f"{source}_missing_volume"] == 1.
                                  for r in sub if r["case_id"] == case["case_id"])
                              for source in SOURCES] for case in metadata], bool)
        positive_counts = positive.sum(0)
        # The case-macro columns then equal the fraction of retained CTs that
        # are positive, because division by each CT's bone count cancels here.
        # Their bone-macro columns are unused; the six original risks stay intact.
        y = np.column_stack((y, positive * sizes[:, None]))
        n_metrics = len(ENDPOINTS) + len(SOURCES)
        can_bootstrap = n >= 2
        draws = cluster_draws(metadata, resamples, seed) if can_bootstrap else None
        bootstrap_positive_counts = positive[draws].sum(1) if can_bootstrap else None
        for fraction in fractions:
            reviewed, retained = int(ceil(fraction * n)), n - int(ceil(fraction * n))
            point = np.full((len(policies), 2, n_metrics), np.nan)
            boot = np.full((resamples, len(policies), 2, n_metrics), np.nan) if can_bootstrap else None
            statuses = []
            for policy_index, policy in enumerate(policies):
                guard = policy.startswith("guard_")
                score = x[:, ALL_SCORES.index(FINAL_HEAD if policy == "guard_" + FINAL_HEAD else policy)] if policy in ALL_SCORES or policy == "guard_" + FINAL_HEAD else np.zeros(n)
                active = ~flags if guard else np.ones(n, bool)
                status = "zero_retained_cases" if retained == 0 else "guard_budget_infeasible" if int(active.sum()) < retained else "estimated"
                statuses.append(status)
                point[policy_index] = retained_ct_expectation(score, y, sizes, retained, active)
                if can_bootstrap and status != "zero_retained_cases":
                    for b, take in enumerate(draws):
                        boot[b, policy_index] = retained_ct_expectation(score[take], y[take], sizes[take], retained, active[take])
            common = {"stratum": stratum, "n_cases": n, "n_bone_units": len(sub),
                      "review_fraction": fraction, "reviewed_cts": reviewed,
                      "retained_cts": retained, "actual_review_fraction": reviewed / n,
                      "budget_unit": "CT", "ct_score_aggregation": "maximum_annotated_bone_score",
                      "ci_role": CI_ROLE}
            for i, policy in enumerate(policies):
                for estimand_index, estimand in enumerate(("case_macro", "bone_macro")):
                    for j, (source, risk) in enumerate(ENDPOINTS):
                        samples = boot[:, i, estimand_index, j] if boot is not None else None
                        lo, hi = _interval(samples) if statuses[i] == "estimated" else (None, None)
                        finite_fraction = float(np.mean(np.isfinite(samples))) if samples is not None else None
                        ci_status = statuses[i] if statuses[i] != "estimated" else "insufficient_cases" if not can_bootstrap else "guard_budget_infeasible_in_bootstrap" if finite_fraction != 1 else "estimated"
                        value = point[i, estimand_index, j]
                        points.append({**common, "policy": policy, "estimand": estimand,
                                       "risk_source": source, "risk": risk,
                                       "retained_mean_risk": float(value) if np.isfinite(value) else None,
                                       "ci_low": lo, "ci_high": hi, "status": statuses[i],
                                       "ci_status": ci_status, "bootstrap_feasible_fraction": finite_fraction,
                                       "flagged_cts": int(flags.sum()) if policy.startswith("guard_") else 0})
                        if policy in ("partition_v1", "random"):
                            continue
                        for baseline in ("partition_v1", "random"):
                            k = policies.index(baseline)
                            delta = value - point[k, estimand_index, j]
                            diff = boot[:, i, estimand_index, j] - boot[:, k, estimand_index, j] if boot is not None else None
                            dlo, dhi = _interval(diff) if statuses[i] == statuses[k] == "estimated" else (None, None)
                            contrasts.append({**common, "policy": policy, "baseline": baseline,
                                              "estimand": estimand, "risk_source": source, "risk": risk,
                                              "delta_retained_risk": float(delta) if np.isfinite(delta) else None,
                                              "ci_low": dlo, "ci_high": dhi, "ci_status": ci_status})
            omission = np.full((len(policies), len(SOURCES)), np.nan)
            omission_boot = np.full((resamples, len(policies), len(SOURCES)), np.nan) if can_bootstrap else None
            for i, policy in enumerate(policies):
                if statuses[i] != "guard_budget_infeasible":
                    positive_retained = retained * point[i, 0, len(ENDPOINTS):] if retained else np.zeros(len(SOURCES))
                    np.divide(positive_counts - positive_retained, positive_counts,
                              out=omission[i], where=positive_counts > 0)
                if can_bootstrap:
                    positive_retained = retained * boot[:, i, 0, len(ENDPOINTS):] if retained else np.zeros_like(bootstrap_positive_counts)
                    np.divide(bootstrap_positive_counts - positive_retained, bootstrap_positive_counts,
                              out=omission_boot[:, i], where=bootstrap_positive_counts > 0)
            omission = np.clip(omission, 0, 1)
            if omission_boot is not None:
                omission_boot = np.clip(omission_boot, 0, 1)
            for i, policy in enumerate(policies):
                for j, source in enumerate(SOURCES):
                    status = ("guard_budget_infeasible" if statuses[i] == "guard_budget_infeasible"
                              else "no_positive_cts" if positive_counts[j] == 0 else "estimated")
                    samples = omission_boot[:, i, j] if omission_boot is not None else None
                    defined_fraction = float(np.mean(np.isfinite(samples))) if samples is not None else None
                    ci_status = (status if status != "estimated" else "insufficient_cases" if not can_bootstrap
                                 else "no_positive_or_infeasible_bootstrap_draws" if defined_fraction != 1 else "estimated")
                    lo, hi = _interval(samples) if status == "estimated" else (None, None)
                    value = omission[i, j]
                    omission_points.append({**common, "policy": policy, "risk_source": source,
                                            "positive_definition": "any_annotated_bone_missing_volume_equals_one",
                                            "positive_label_role": "evaluation_only_not_scoring_feature",
                                            "positive_cts": int(positive_counts[j]),
                                            "expected_reviewed_positive_cts": float(value * positive_counts[j]) if np.isfinite(value) else 0. if status == "no_positive_cts" else None,
                                            "complete_omission_sensitivity": float(value) if np.isfinite(value) else None,
                                            "status": status, "ci_low": lo, "ci_high": hi,
                                            "ci_status": ci_status, "bootstrap_defined_fraction": defined_fraction})
                    if policy in ("partition_v1", "random"):
                        continue
                    for baseline in ("partition_v1", "random"):
                        k = policies.index(baseline)
                        delta = value - omission[k, j]
                        diff = omission_boot[:, i, j] - omission_boot[:, k, j] if can_bootstrap else None
                        dlo, dhi = _interval(diff) if status == "estimated" else (None, None)
                        omission_contrasts.append({**common, "policy": policy, "baseline": baseline,
                                                   "risk_source": source, "positive_cts": int(positive_counts[j]),
                                                   "delta_complete_omission_sensitivity": float(delta) if np.isfinite(delta) else None,
                                                   "ci_low": dlo, "ci_high": dhi, "status": status,
                                                   "ci_status": ci_status})
    return points, contrasts, omission_points, omission_contrasts


def report(rows: list[dict], config: dict) -> dict:
    """Report frozen raw baselines and all OOF heads, without picking a winner."""
    _validate(rows)
    resamples, seed, fractions = _settings(config)
    comparison, paired, curves = [], [], []
    for stratum in ("pooled", "pelvic", "femur"):
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        if not sub:
            continue
        n_cases, n_units = len({r["case_id"] for r in sub}), len(sub)
        x = np.array([[r[s] for s in ALL_SCORES] for r in sub], float)
        y = np.array([[r[f"{source}_{risk}"] for source, risk in ENDPOINTS] for r in sub], float)
        point = np.stack([aurc_batch(x[:, i][None], y[None])[0] for i in range(len(ALL_SCORES))])
        boot = None
        if n_cases >= 2:
            take = cluster_draws(sub, resamples, seed)
            boot = np.empty((resamples, len(ALL_SCORES), len(ENDPOINTS)))
            for start in range(0, resamples, 128):
                draw = take[start:start + 128]
                for i in range(len(ALL_SCORES)):
                    boot[start:start + len(draw), i] = aurc_batch(x[draw, i], y[draw])
        common = {"stratum": stratum, "n_cases": n_cases, "n_units": n_units,
                  "ci_role": CI_ROLE, "bootstrap_resamples": resamples if boot is not None else 0}
        for i, score in enumerate(ALL_SCORES):
            for j, (source, risk) in enumerate(ENDPOINTS):
                lo, hi = _interval(boot[:, i, j] if boot is not None else None)
                comparison.append({**common, "score": score, "risk_source": source, "risk": risk,
                                   "aurc": float(point[i, j]) if np.isfinite(point[i, j]) else None,
                                   "ci_low": lo, "ci_high": hi, "rho": correlation(x[:, i], y[:, j]),
                                   "status": "insufficient_cases" if n_cases < 2 else "constant_risk_uninformative" if np.ptp(y[:, j]) == 0 else "estimated"})
                if n_units >= 2:
                    curves.extend({**common, "score": score, "risk_source": source, "risk": risk,
                                   "coverage": (k + 1) / n_units, "retained_mean_risk": float(v)}
                                  for k, v in enumerate(expected_curve(x[:, i], y[:, j])))
                if score not in NEW_SCORES:
                    continue
                for baseline in (*SCORE_NAMES, *(("ridge_p1", "ridge_p1_a") if score == FINAL_HEAD else ())):
                    k = ALL_SCORES.index(baseline)
                    delta = point[i, j] - point[k, j]
                    dlo, dhi = _interval(boot[:, i, j] - boot[:, k, j] if boot is not None else None)
                    paired.append({**common, "score": score, "baseline": baseline, "risk_source": source,
                                   "risk": risk, "delta_aurc": float(delta) if np.isfinite(delta) else None,
                                   "ci_low": dlo, "ci_high": dhi,
                                   "contrast_role": "extension_main" if score == FINAL_HEAD and baseline == "partition_v1" and source == "member_mean" and risk == "merge_split" and stratum == "pooled" else "secondary"})
    include_guard = config.get("reporting", config).get("include_omission_guard", True)
    points, contrasts, omission_points, omission_contrasts = _ct_tables(rows, resamples, seed, fractions, include_guard)
    return {"comparison": comparison, "paired_differences": paired, "risk_coverage": curves,
            "ct_review_budget": points, "ct_review_differences": contrasts,
            "ct_omission_sensitivity": omission_points, "ct_omission_differences": omission_contrasts,
            "summary": {"n_cases": len({r["case_id"] for r in rows}), "n_units": len(rows),
                        "scores": list(ALL_SCORES), "final_head": FINAL_HEAD,
                        "analysis_role": "development_only_not_confirmation", "ci_role": CI_ROLE,
                        "primary_contrast": f"{FINAL_HEAD}-partition_v1/member_mean/merge_split/pooled",
                        "ct_review_fractions": list(fractions), "ct_score_aggregation": "max",
                        "omission_positive_definition": "CT_with_any_annotated_bone_missing_volume_equals_one_per_risk_source",
                        "omission_labels": "frozen_E1_evaluation_only_not_predictor_features",
                        "guard_scope": "PENGWIN_expected_anatomy_only_not_RibFrac_lesions",
                        "human_correction_assumed": False, "measured_reading_time": False}}


report_tables = report
