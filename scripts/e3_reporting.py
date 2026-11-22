"""Held-out E3 metrics and tables using the same statistics as E1."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from exchange_ml.e1_data import RISKS
from exchange_ml.e1_reporting import write_csv
from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.e1_statistics import aurc_batch, cluster_draws, correlation, expected_curve
from run_e1 import save_json

SOURCES = ("member_mean", "stage2")
STRATA = ("pooled", "pelvic", "femur")
CURVES = ("fa_v2", "partition_v2", "presence", "mean_entropy", "union_nvi", "pairwise_dice")
ENDPOINTS = tuple((source, risk) for source in SOURCES for risk in RISKS)


def top_weights(values: np.ndarray, count: float) -> np.ndarray:
    """Expected selection weight per unit under a uniformly broken cutoff tie."""
    cutoff = np.sort(values)[-int(np.ceil(count))]
    higher, tied = values > cutoff, values == cutoff
    return higher.astype(float) + tied * ((count - higher.sum()) / tied.sum())


def primary_result(paired: list[dict], protocol: dict, smoke: bool) -> dict:
    rule = protocol["primary"]
    results = {}
    for source in SOURCES:
        baselines = {}
        for baseline in rule["baselines"]:
            row = next((r for r in paired if r["stratum"] == "pooled" and r["risk_source"] == source
                        and r["risk"] == rule["risk"] and r["baseline"] == baseline), None)
            if row is not None:
                metrics = {key: row[key] for key in ("delta_aurc", "ci_low", "ci_high")}
                baselines[baseline] = {**metrics, "meets_rule": None if smoke else (
                    metrics["delta_aurc"] <= rule["max_delta_aurc"] and metrics["ci_high"] < 0)}
        meets = None if smoke else len(baselines) == len(rule["baselines"]) and all(
            row["meets_rule"] for row in baselines.values())
        results[source] = {"meets_rule": meets, "baselines": baselines}
    results["robust_across_risk_sources"] = None if smoke else all(results[source]["meets_rule"] for source in SOURCES)
    return results


def report(root: Path, tables: dict, runtime: list[dict], scoring_times: list[dict],
           protocol: dict, smoke: bool) -> dict:
    counts = tuple(len(tables[name]) for name in ("scores", "pair_components", "model_risks", "risks"))
    expected = (4, 12, 16, 4) if smoke else (200, 600, 800, 200)
    if counts != expected:
        raise ValueError(f"unexpected E3 row counts: {counts}; expected {expected}")
    by_unit = {(r["case_id"], r["bone"]): r for r in tables["risks"]}
    score_units = {(r["case_id"], r["bone"]) for r in tables["scores"]}
    if len(score_units) != counts[0] or score_units != by_unit.keys():
        raise ValueError("score and risk unit identities differ")
    for name, rows in tables.items():
        write_csv(root / f"{name}.csv", rows)
    write_csv(root / "inference_runtime.csv", runtime)
    write_csv(root / "evaluation_runtime.csv", scoring_times)
    write_csv(root / "cases.csv", [{k: r[k] for k in ("case_id", "scan_type")} for r in scoring_times])
    rows = [{**r, **by_unit[r["case_id"], r["bone"]]} for r in tables["scores"]]
    comparison, paired, curves, enrichment, frequencies = [], [], [], [], []
    draws_n = 32 if smoke else protocol["bootstrap_resamples"]
    primary_index = SCORE_NAMES.index(protocol["primary"]["score"])
    alpha = (1 - protocol["primary"]["confidence_level"]) / 2
    for stratum in STRATA:
        subset = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        n = len(subset)
        common = {"stratum": stratum, "n_cases": len({r["case_id"] for r in subset}), "n_units": n}
        x = np.array([[r[name] for name in SCORE_NAMES] for r in subset])
        y = np.array([[r[f"{source}_{risk}"] for source, risk in ENDPOINTS] for r in subset])
        z = np.array([[r[k] for k in ("n_GT", "gt_volume_mm3", "min_gt_fragment_volume_mm3")] for r in subset])
        point = np.stack([aurc_batch(x[:, i][None], y[None])[0] for i in range(len(SCORE_NAMES))])
        for j, (source, risk) in enumerate(ENDPOINTS):
            observed = y[:, j]
            frequencies.append({**common, "risk_source": source, "risk": risk,
                                "n_positive_units": int(np.count_nonzero(observed > 0)),
                                "n_positive_cases": len({r["case_id"] for r, v in zip(subset, observed) if v > 0}),
                                "mean_risk": float(observed.mean()), "median_risk": float(np.median(observed))})
            for i, score in enumerate(SCORE_NAMES):
                comparison.append({**common, "score": score, "risk_source": source, "risk": risk,
                                   "aurc": float(point[i, j]) if n > 1 else None,
                                   "random_aurc": float((1 - 1 / n) * observed.mean()) if n > 1 else None,
                                   "rho": correlation(x[:, i], observed), "partial_rho": correlation(x[:, i], observed, z)})
                if risk == "merge_split" and n > 1:
                    truth = top_weights(observed, .25 * n)
                    for fraction in (.1, .2):
                        k = int(np.ceil(fraction * n))
                        selection = top_weights(x[:, i], k)
                        enrichment.append({**common, "score": score, "risk_source": source, "review_fraction": fraction,
                                           "n_reviewed": k, "label_prevalence": .25,
                                           "enrichment": float(selection @ truth / k / .25) if np.ptp(observed) else None})
            if risk == "merge_split" and n > 1:
                for score in (*CURVES, "random"):
                    curve = (np.full(n, observed.mean()) if score == "random"
                             else expected_curve(x[:, SCORE_NAMES.index(score)], observed))
                    curves.extend({**common, "risk_source": source, "score": score, "coverage": (k + 1) / n,
                                   "prefix_mean_risk": float(value)} for k, value in enumerate(curve))
        if n < 2:
            continue
        draws = cluster_draws(subset, draws_n, protocol["bootstrap_seed"])
        boot = np.empty((draws_n, len(SCORE_NAMES), len(ENDPOINTS)))
        for start in range(0, draws_n, 128):
            sampled = draws[start:start + 128]
            risks = y[sampled]
            for i in range(len(SCORE_NAMES)):
                boot[start:start + len(sampled), i] = aurc_batch(x[sampled, i], risks)
        for i, baseline in enumerate(SCORE_NAMES):
            if i == primary_index:
                continue
            low, high = np.quantile(boot[:, primary_index] - boot[:, i], [alpha, 1 - alpha], axis=0)
            for j, (source, risk) in enumerate(ENDPOINTS):
                paired.append({**common, "score": SCORE_NAMES[primary_index], "baseline": baseline,
                               "risk_source": source, "risk": risk,
                               "delta_aurc": float(point[primary_index, j] - point[i, j]),
                               "ci_low": float(low[j]), "ci_high": float(high[j]), "bootstrap_resamples": draws_n,
                               "primary_contrast": stratum == "pooled" and source == protocol["primary"]["risk_source"]
                               and risk == protocol["primary"]["risk"] and baseline in protocol["primary"]["baselines"]})
    for name, data in (("comparison", comparison), ("paired_differences", paired), ("risk_coverage", curves),
                       ("top_quartile_enrichment", enrichment), ("risk_frequency", frequencies)):
        write_csv(root / f"{name}.csv", data)
    if not smoke:
        with Path("outputs/stage3_h1_v2/validation/comparison.csv").open(newline="") as handle:
            validation = list(csv.DictReader(handle))
        key = lambda r: (r["score"], r["risk"])
        main_heldout = {key(r): r for r in comparison if r["stratum"] == "pooled" and r["risk_source"] == "member_mean"}
        main_validation = {key(r): r for r in validation if r["stratum"] == "pooled" and r["risk_source"] == "member_mean"}
        write_csv(root / "main_table.csv", [
            {"score": score, **{f"{split}_{risk}_aurc": lookup[score, risk]["aurc"]
                                for split, lookup in (("validation", main_validation), ("heldout", main_heldout))
                                for risk in ("merge_split", "one_minus_f1")}} for score in SCORE_NAMES
        ])
    plot_curves(root, curves)
    summary = {"n_cases": len(scoring_times), "n_units": counts[0], "pair_rows": counts[1],
               "model_risk_rows": counts[2], "inference_records": len(runtime), "bootstrap_resamples": draws_n,
               "smoke_only": smoke, "primary_result": primary_result(paired, protocol, smoke)}
    save_json(root / "summary.json", summary)
    return summary


def plot_curves(root: Path, curves: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, source in zip(axes, SOURCES):
        for score in (*CURVES, "random"):
            selected = [r for r in curves if r["stratum"] == "pooled" and r["risk_source"] == source and r["score"] == score]
            ax.plot([r["coverage"] for r in selected], [r["prefix_mean_risk"] for r in selected],
                    label=score, linestyle="--" if score == "random" else "-")
        ax.set(title="Mean member risk" if source == "member_mean" else "Stage 2 risk", xlabel="Retained coverage")
    axes[0].set_ylabel("Mean normalized merge + split risk")
    axes[1].legend(fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(root / f"risk_coverage.{ext}", dpi=180)
    plt.close(fig)
