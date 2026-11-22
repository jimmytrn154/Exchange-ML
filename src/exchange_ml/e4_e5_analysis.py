"""E4/E5 statistics on fixed E1/E3 rows; no model or primary-score changes.

All secondary contrasts use the E1 trapezoid/tie convention. Bootstrap units
are cases within scan type, including every bone and repeated sampled cases.
"""
from __future__ import annotations

from math import ceil, comb, floor

import numpy as np
from scipy.stats import rankdata

from .e1_data import MEMBERS
from .e1_scoring import SCORE_NAMES
from .e1_statistics import aurc_batch, cluster_draws

SOURCES = ("member_mean", "stage2")
STRATA = ("pooled", "pelvic", "femur")
CATEGORIES = ("partition_detects_structure", "presence_detects_missing", "shared_error_low_disagreement")


def top_weights(values: np.ndarray, count: float) -> np.ndarray:
    """Uniform cutoff-tie selection, including fractional severity labels."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= count <= len(values):
        raise ValueError("finite vector and count in [0,n] required")
    if count == 0:
        return np.zeros(len(values))
    cutoff = np.sort(values)[-ceil(count)]
    higher, tied = values > cutoff, values == cutoff
    return higher.astype(float) + tied * ((count - higher.sum()) / tied.sum())


def expected_reviewed_cases(scores: np.ndarray, cases: list[str], count: int) -> float:
    """Expected distinct CTs when exactly count bones are selected for review.

    A cutoff tie is sampled without replacement. Marginal bone probabilities
    cannot be treated as independent when computing the probability of a CT.
    """
    scores = np.asarray(scores, float)
    if len(cases) != len(scores) or not 0 <= count <= len(scores):
        raise ValueError("case vector/count mismatch")
    weights = top_weights(scores, count)
    certain = {case for case, weight in zip(cases, weights) if weight == 1}
    tied = (weights > 0) & (weights < 1)
    t = int(tied.sum())
    q = count - int(np.count_nonzero(weights == 1))
    total = float(len(certain))
    if t:
        for case in sorted(set(cases) - certain):
            tc = sum(c == case and is_tied for c, is_tied in zip(cases, tied))
            total += 1 - (comb(t - tc, q) / comb(t, q) if t - tc >= q else 0)
    return total


def aurc(scores: np.ndarray, risks: np.ndarray) -> float | None:
    if len(scores) < 2:
        return None
    return float(aurc_batch(np.asarray(scores)[None], np.asarray(risks)[None, :, None])[0, 0])


def normalization_scope(rows: list[dict], draws_n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Case-level 2x2, holding both the risk and the cases fixed in each contrast."""
    estimates, contrasts = [], []
    variants = ("all_log_n", "all_joint", "annotated_log_n", "annotated_joint")
    changes = (("normalization_all", "all_joint", "all_log_n"),
               ("normalization_annotated", "annotated_joint", "annotated_log_n"),
               ("scope_log_n", "annotated_log_n", "all_log_n"),
               ("scope_joint", "annotated_joint", "all_joint"))
    for stratum in STRATA:
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        if not sub:
            continue
        common = {"stratum": stratum, "n_cases": len(sub), "unit": "case"}
        x = np.array([[r[v] for v in variants] for r in sub], float)  # [N,4]
        y = np.array([[r[s + "_merge_split"] for s in SOURCES] for r in sub], float)  # [N,2]
        point = np.stack([aurc_batch(x[:, i][None], y[None])[0] for i in range(4)])
        for i, variant in enumerate(variants):
            scope, norm = variant.split("_", 1)
            shares = [r[variant + "_partition_share"] for r in sub
                      if r[variant + "_partition_share"] is not None]
            for j, source in enumerate(SOURCES):
                estimates.append({**common, "risk_source": source, "variant": variant,
                                  "scope": scope, "normalization": norm,
                                  "aurc": float(point[i, j]) if len(sub) > 1 else None,
                                  "mean_score": float(x[:, i].mean()),
                                  "median_partition_share": float(np.median(shares)) if shares else None,
                                  "n_defined_partition_shares": len(shares)})
        boot = np.empty((draws_n, 4, 2))
        if len(sub) > 1:
            draws = cluster_draws(sub, draws_n, seed)
            for start in range(0, draws_n, 128):
                take = draws[start:start + 128]
                for i in range(4):
                    boot[start:start + len(take), i] = aurc_batch(x[take, i], y[take])
        for change, name, baseline in changes:
            i, k = variants.index(name), variants.index(baseline)
            for j, source in enumerate(SOURCES):
                lo, hi = (np.quantile(boot[:, i, j] - boot[:, k, j], [.025, .975])
                          if len(sub) > 1 else (None, None))
                contrasts.append({**common, "risk_source": source, "contrast": change,
                                  "variant": name, "baseline": baseline,
                                  "delta_aurc": float(point[i, j] - point[k, j]) if len(sub) > 1 else None,
                                  "ci_low": float(lo) if lo is not None else None,
                                  "ci_high": float(hi) if hi is not None else None,
                                  "bootstrap_resamples": draws_n, "secondary_analysis": True})
    return estimates, contrasts


def risk_source_table(comparison: list[dict], paired: list[dict]) -> list[dict]:
    output = []
    for stratum in STRATA:
        sub = [r for r in comparison if r["stratum"] == stratum and r["risk"] == "merge_split"]
        if not sub:
            continue
        lookup = {(r["score"], r["risk_source"]): r for r in sub}
        ranks = {}
        for source in SOURCES:
            values = [lookup[name, source]["aurc"] for name in SCORE_NAMES]
            ranks[source] = rankdata(values) if all(v is not None for v in values) else [None] * len(values)
        for i, name in enumerate(SCORE_NAMES):
            row = {"stratum": stratum, "score": name,
                   "n_cases": lookup[name, SOURCES[0]]["n_cases"],
                   "n_units": lookup[name, SOURCES[0]]["n_units"]}
            for source in SOURCES:
                value = lookup[name, source]
                contrast = next((r for r in paired if r["stratum"] == stratum and r["risk"] == "merge_split"
                                 and r["risk_source"] == source and r["score"] == "fa_v2"
                                 and r["baseline"] == name), {})
                row.update({source + "_aurc": value["aurc"], source + "_rank": ranks[source][i],
                            source + "_delta_fa_v2_minus_score": contrast.get("delta_aurc", 0.0 if name == "fa_v2" else None),
                            source + "_ci_low": contrast.get("ci_low"), source + "_ci_high": contrast.get("ci_high")})
            output.append(row)
    return output


def enrichment(rows: list[dict], config: dict) -> list[dict]:
    output = []
    for stratum in STRATA:
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        if not sub:
            continue
        n = len(sub)
        for source in SOURCES:
            y = np.array([r[source + "_merge_split"] for r in sub], float)
            labels = {"any_merge_split": (y > 0).astype(float),
                      "top_severity_quartile": top_weights(y, config["severity_fraction"] * n)}
            for label, truth in labels.items():
                prevalence = float(truth.mean())
                status = "zero_prevalence" if prevalence == 0 else "constant_risk" if np.ptp(y) == 0 else "defined"
                for score in SCORE_NAMES:
                    x = np.array([r[score] for r in sub], float)
                    for fraction in config["review_fractions"]:
                        k = ceil(fraction * n)
                        selected = top_weights(x, k)
                        output.append({"stratum": stratum, "risk_source": source, "score": score,
                                       "label": label, "n_cases": len({r['case_id'] for r in sub}), "n_units": n,
                                       "review_fraction": fraction, "reviewed_units": k,
                                       "label_mass": float(truth.sum()), "label_prevalence": prevalence,
                                       "ceiling_one_over_prevalence": 1 / prevalence if prevalence else None,
                                       "enrichment": float(selected @ truth / k / prevalence) if status == "defined" else None,
                                       "status": status})
    return output


def review_budget(rows: list[dict], config: dict) -> list[dict]:
    output = []
    for stratum in STRATA:
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        if not sub:
            continue
        n = len(sub)
        cases = [r["case_id"] for r in sub]
        presence = np.array([r["presence"] for r in sub])
        for source in SOURCES:
            y = np.array([r[source + "_merge_split"] for r in sub])
            for fraction in config["retained_fractions"]:
                retained = floor(fraction * n)
                reviewed = n - retained
                presence_keep = 1 - top_weights(presence, reviewed)
                base = float(presence_keep @ y / retained) if retained else None
                for score in SCORE_NAMES:
                    x = np.array([r[score] for r in sub])
                    keep = 1 - top_weights(x, reviewed)
                    risk = float(keep @ y / retained) if retained else None
                    output.append({"stratum": stratum, "score": score, "risk_source": source,
                                   "n_cases": len(set(cases)), "n_units": n, "requested_coverage": fraction,
                                   "coverage": retained / n, "retained_units": retained, "reviewed_units": reviewed,
                                   "expected_reviewed_cases": expected_reviewed_cases(x, cases, reviewed),
                                   "retained_mean_risk": risk, "random_mean_risk": float(y.mean()),
                                   "delta_vs_random": risk - float(y.mean()) if risk is not None else None,
                                   "delta_vs_presence": risk - base if risk is not None else None,
                                   "budget_unit": "bone"})
    return output


def qualitative_candidates(rows: list[dict], model_risks: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """Declare all eligible units and select at most one per frozen category."""
    settings = config["qualitative"]
    model_lookup = {(r["case_id"], r["bone"], r["model_id"]): r for r in model_risks}
    ranked = []
    for scan in ("pelvic", "femur"):
        sub = [r for r in rows if r["scan_type"] == scan]
        if not sub:
            continue
        percentiles = {s: rankdata([r[s] for r in sub], method="average") / len(sub)
                       for s in ("partition_v2", "presence", "mean_entropy", "fa_v2")}
        for i, r in enumerate(sub):
            ranked.append({**r, **{s + "_percentile": float(v[i]) for s, v in percentiles.items()}})
    candidates, selections = [], []
    for category in CATEGORIES:
        eligible = []
        for r in ranked:
            partition, presence, entropy, fa = (r[s + "_percentile"] for s in ("partition_v2", "presence", "mean_entropy", "fa_v2"))
            if category == CATEGORIES[0]:
                passed = (r["member_mean_merge_split"] > 0 and partition >= settings["high_percentile"]
                          and presence <= settings["low_percentile"] and entropy <= settings["low_percentile"])
                priority = partition - max(presence, entropy)
            elif category == CATEGORIES[1]:
                passed = (r["member_mean_missing"] > 0 and presence >= settings["high_percentile"]
                          and partition <= settings["low_percentile"])
                priority = r["member_mean_missing_volume"]
            else:
                passed = (all(model_lookup[r['case_id'], r['bone'], m]["one_minus_f1"] > 0 for m in MEMBERS)
                          and r["member_mean_one_minus_f1"] >= settings["shared_error_min_mean_one_minus_f1"]
                          and fa <= settings["shared_error_max_fa_percentile"])
                priority = r["member_mean_one_minus_f1"]
            if passed:
                eligible.append({"category": category, "case_id": r["case_id"], "scan_type": r["scan_type"],
                                 "bone": r["bone"], "priority": float(priority),
                                 **{s: r[s] for s in ("partition_v2", "presence", "mean_entropy", "fa_v2")},
                                 **{s + "_percentile": r[s + "_percentile"] for s in ("partition_v2", "presence", "mean_entropy", "fa_v2")},
                                 "merge_split": r["member_mean_merge_split"], "missing": r["member_mean_missing"],
                                 "missing_volume": r["member_mean_missing_volume"],
                                 "one_minus_f1": r["member_mean_one_minus_f1"]})
        eligible.sort(key=lambda r: (-r["priority"], r["case_id"], r["bone"]))
        for i, r in enumerate(eligible):
            candidates.append({**r, "selected": i == 0})
        selections.append({"category": category, "status": "selected" if eligible else "no_eligible_case",
                           "n_eligible": len(eligible), "case_id": eligible[0]["case_id"] if eligible else None,
                           "bone": eligible[0]["bone"] if eligible else None})
    return candidates, selections


def runtime_summary(inference: list[dict], evaluation: list[dict], measured: list[dict]) -> list[dict]:
    """Summarize measured quantities only; missing historical timings stay null."""
    output = []
    first_ten = [r for model in (*MEMBERS, "stage2") for r in [v for v in inference if v["model"] == model][:10]]
    for sampling, rows in (("all_recorded", inference), ("first10_per_model", first_ten)):
        for model in (*MEMBERS, "stage2"):
            for stratum in STRATA:
                sub = [r for r in rows if r["model"] == model and (stratum == "pooled" or r["scan_type"] == stratum)]
                if not sub:
                    continue
                metrics = {k: k for k in ("read_seconds", "write_seconds", "peak_vram_bytes", "n_voxels")}
                metrics.update(inference_including_model_load_seconds="inference_seconds", model_load_seconds=None)
                for quantity, key in metrics.items():
                    output.append(_runtime_row("E3_inference", sampling, model, stratum, quantity,
                                               [r[key] for r in sub] if key else [], "not_recorded_separately_in_E3"))
    for stratum in STRATA:
        ev = [r for r in evaluation if stratum == "pooled" or r["scan_type"] == stratum]
        for quantity, key in (("scoring_and_evaluation_seconds", "scoring_and_evaluation_seconds"),
                              ("process_peak_rss_kib_cumulative", "process_peak_rss_kib"),
                              *[(k, None) for k in ("read_seconds", "cleanup_seconds", "components_seconds", "baselines_seconds")]):
            output.append(_runtime_row("E3_CPU", "all_recorded", "all_models", stratum, quantity,
                                       [r[key] for r in ev] if key else [], "not_recorded_separately_in_E3"))
        for split in sorted({r["split"] for r in measured}):
            sub = [r for r in measured if r["split"] == split and (stratum == "pooled" or r["scan_type"] == stratum)]
            for quantity in ("read_seconds", "cleanup_seconds", "components_seconds", "fragment_seconds"):
                output.append(_runtime_row("E4_CPU_" + split, "all_processed", "all_models", stratum,
                                           quantity, [r[quantity] for r in sub], "no_samples"))
    return output


def _runtime_row(stage: str, sampling: str, model: str, stratum: str, quantity: str,
                 values: list[float], missing_reason: str) -> dict:
    q25, median, q75 = (np.quantile(values, [.25, .5, .75]) if values else (None, None, None))
    return {"stage": stage, "sampling": sampling, "model": model, "stratum": stratum,
            "quantity": quantity, "n": len(values), "median": float(median) if values else None,
            "q25": float(q25) if values else None, "q75": float(q75) if values else None,
            "status": "measured" if values else missing_reason}
