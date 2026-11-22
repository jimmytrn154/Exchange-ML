"""Post-heldout E5 fixed-budget uncertainty, omission risks and CT cost.

This supplements the frozen E3/E5 analysis. Scores and risk definitions are
unchanged. Intervals are marginal paired case-bootstrap percentile intervals;
they are descriptive supplementary uncertainty, not new primary tests.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from math import floor

import numpy as np

from .e1_scoring import SCORE_NAMES
from .e1_statistics import cluster_draws, expected_curve
from .e4_e5_analysis import expected_reviewed_cases

RISKS = ("merge_split", "one_minus_f1", "missing_volume")
SOURCES = ("member_mean", "stage2")
STRATA = ("pooled", "pelvic", "femur")
POLICIES = (*SCORE_NAMES, "random")
ENDPOINTS = tuple((source, risk) for source in SOURCES for risk in RISKS)


def retained_risk_batch(scores: np.ndarray, risks: np.ndarray,
                        retained_counts: Sequence[int]) -> np.ndarray:
    """Expected retained risk under exact cutoff ties.

    scores [B,N], risks [B,N,K] -> means [B,C,K], where C is the number
    of requested counts. A cutoff group is sampled uniformly without
    replacement. Expected total risk is linear in its marginal weights.
    Counts and cutoff groups are recomputed from every resampled vector.
    A zero retained count is undefined and returns NaN for internal use.
    """
    x, y = np.asarray(scores, float), np.asarray(risks, float)
    if x.ndim != 2 or y.ndim != 3 or x.shape != y.shape[:2] or not x.shape[1]:
        raise ValueError("expected nonempty scores [B,N] and risks [B,N,K]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("scores and risks must be finite")
    if any(not isinstance(n, (int, np.integer)) or not 0 <= n <= x.shape[1] for n in retained_counts):
        raise ValueError("retained counts must be integers in [0,N]")
    result = np.full((len(x), len(retained_counts), y.shape[-1]), np.nan)
    for i, retained in enumerate(retained_counts):
        if retained == 0:
            continue
        cutoff = np.partition(x, retained - 1, axis=1)[:, retained - 1]
        below = x < cutoff[:, None]
        tied = x == cutoff[:, None]
        fraction = (retained - below.sum(axis=1)) / tied.sum(axis=1)
        weights = below.astype(float) + tied * fraction[:, None]
        result[:, i] = np.einsum("bn,bnk->bk", weights, y) / retained
    return result


def _validate_rows(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("at least one score/risk row is required")
    seen, scans = set(), {}
    for row in rows:
        key = row["case_id"], row["bone"]
        if key in seen:
            raise ValueError(f"duplicate case/bone unit: {key}")
        seen.add(key)
        scan = row["scan_type"]
        if scan not in ("pelvic", "femur") or scans.get(row["case_id"], scan) != scan:
            raise ValueError("invalid/inconsistent case scan type")
        scans[row["case_id"]] = scan
        if not np.isfinite([row[s] for s in SCORE_NAMES]).all():
            raise ValueError("all twelve scores must be finite")
        for source, risk in ENDPOINTS:
            value = row[f"{source}_{risk}"]
            if value is None or not np.isfinite(value) or value < -1e-12:
                raise ValueError(f"invalid {source}/{risk} risk; units must not be silently dropped")
            if risk != "merge_split" and value > 1 + 1e-12:
                raise ValueError(f"{risk} must be in [0,1] within floating point tolerance")


def _interval(values: np.ndarray | None) -> tuple[float | None, float | None]:
    if values is None:
        return None, None
    low, high = np.quantile(values, [.025, .975])
    return float(low), float(high)


def budget_analysis(rows: list[dict], *, resamples: int, seed: int,
                    retained_fractions: Sequence[float] = (.8, .9), batch_size: int = 128,
                    progress: Callable[[str], None] | None = None,
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """One split, all strata/policies/endpoints; caller adds dataset metadata.

    Pooled points retain equal bone-unit weighting. All bootstrap draws are
    shared over budgets, policies, risk sources and endpoints within a
    stratum. Cases are sampled within scan type and every bone/multiplicity
    is carried. Expected reviewed CTs are point estimates only; no inference
    about equal CT-cost policies is made. Analytical random means uniformly
    selecting a fixed number of bones, not independent Bernoulli selection.
    """
    _validate_rows(rows)
    if not isinstance(resamples, int) or resamples < 1 or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("positive integer resamples and batch_size required")
    fractions = tuple(float(v) for v in retained_fractions)
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0 < v < 1 for v in fractions):
        raise ValueError("distinct retained fractions strictly between zero and one required")
    points, contrasts, curves = [], [], []
    presence_index = POLICIES.index("presence")
    random_index = POLICIES.index("random")
    for stratum in STRATA:
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        if not sub:
            continue
        n = len(sub)
        cases = [r["case_id"] for r in sub]
        common = {"stratum": stratum, "n_cases": len(set(cases)), "n_units": n}
        x = np.column_stack([[r[name] for r in sub] for name in SCORE_NAMES])
        x = np.column_stack((x, np.zeros(n)))  # [N,13], analytical random at the last column
        y = np.array([[r[f"{source}_{risk}"] for source, risk in ENDPOINTS] for r in sub], float)  # [N,6]
        counts = tuple(floor(fraction * n) for fraction in fractions)
        point = np.stack([retained_risk_batch(x[:, i][None], y[None], counts)[0]
                          for i in range(len(POLICIES))])  # [13,C,6]
        point[random_index] = np.stack([y.mean(0) if count else np.full(y.shape[1], np.nan)
                                       for count in counts])
        # No confidence claim is made from a single independent case.
        can_bootstrap = common["n_cases"] > 1
        boot = None
        if can_bootstrap:
            draws = cluster_draws(sub, resamples, seed)
            boot = np.empty((resamples, len(POLICIES), len(counts), len(ENDPOINTS)))
            for start in range(0, resamples, batch_size):
                take = draws[start:start + batch_size]
                sampled_y = y[take]
                for i in range(len(POLICIES) - 1):
                    boot[start:start + len(take), i] = retained_risk_batch(x[take, i], sampled_y, counts)
                sampled_mean = sampled_y.mean(1)
                for c, count in enumerate(counts):
                    boot[start:start + len(take), random_index, c] = sampled_mean if count else np.nan
                if progress:
                    progress(f"{stratum}: paired bootstrap {start + len(take)}/{resamples}")
        for i, score in enumerate(POLICIES):
            for j, (source, risk) in enumerate(ENDPOINTS):
                curve = y[:, j].mean() * np.ones(n) if score == "random" else expected_curve(x[:, i], y[:, j])
                curves.extend({**common, "risk_source": source, "risk": risk, "score": score,
                               "coverage": (k + 1) / n, "prefix_mean_risk": float(value)}
                              for k, value in enumerate(curve))
            for c, (fraction, retained) in enumerate(zip(fractions, counts)):
                reviewed = n - retained
                cost = expected_reviewed_cases(x[:, i], cases, reviewed)
                ci_status = "zero_retained_units" if retained == 0 else "insufficient_cases" if not can_bootstrap else "estimated"
                valid_ci = ci_status == "estimated"
                for j, (source, risk) in enumerate(ENDPOINTS):
                    value = float(point[i, c, j]) if retained else None
                    lo, hi = _interval(boot[:, i, c, j] if valid_ci else None)
                    random = float(y[:, j].mean())
                    points.append({**common, "risk_source": source, "risk": risk, "score": score,
                                   "requested_coverage": fraction, "coverage": retained / n,
                                   "retained_units": retained, "reviewed_units": reviewed,
                                   "expected_reviewed_cases": cost, "retained_mean_risk": value,
                                   "ci_low": lo, "ci_high": hi, "random_mean_risk": random,
                                   "delta_vs_random": value - random if retained else None,
                                   "delta_vs_presence": value - float(point[presence_index, c, j]) if retained else None,
                                   "bootstrap_resamples": resamples if valid_ci else 0,
                                   "ci_status": ci_status, "budget_unit": "bone"})
                    if score == "random":
                        continue
                    for baseline, k in (("presence", presence_index), ("random", random_index)):
                        delta = float(point[i, c, j] - point[k, c, j]) if retained else None
                        lo, hi = _interval(boot[:, i, c, j] - boot[:, k, c, j] if valid_ci else None)
                        contrasts.append({**common, "risk_source": source, "risk": risk,
                                          "score": score, "baseline": baseline,
                                          "requested_coverage": fraction, "coverage": retained / n,
                                          "delta_retained_risk": delta, "ci_low": lo, "ci_high": hi,
                                          "bootstrap_resamples": resamples if valid_ci else 0,
                                          "ci_status": ci_status})
    return points, contrasts, curves
