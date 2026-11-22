"""Tables, paired case-bootstrap and static E1 scientific figures."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .e1_data import COMPONENTS, RISKS
from .e1_scoring import SCORE_NAMES
from .e1_statistics import aurc_batch, cluster_draws, correlation, expected_curve


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output: Path, data: dict[str, list[dict]], bootstrap: int, seed: int) -> dict:
    for name, rows in data.items():
        write_csv(output / "validation" / f"{name}.csv", rows)
    keyed = {(r["case_id"], r["bone"]): r for r in data["risks"]}
    rows = [{**r, **keyed[r["case_id"], r["bone"]]} for r in data["scores"]]
    comparison, paired, projection, decomposition, curves = [], [], [], [], []
    for stratum in ("pooled", "pelvic", "femur"):
        selected = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        n_cases = len({r["case_id"] for r in selected})
        n = len(selected)
        common = {"stratum": stratum, "n_cases": n_cases, "n_units": n}
        names = ["component_" + name for name in COMPONENTS] + ["partition_share_v1", "partition_share_v2"]
        for name in names:
            values = np.array([r[name] for r in selected if r[name] is not None])
            decomposition.append({**common, "quantity": name.removeprefix("component_"),
                                  "aggregation": "mean_of_three_pairs_then_summary_over_units",
                                  "n_defined": len(values), "n_undefined": n - len(values),
                                  "median": float(np.median(values)) if len(values) else None,
                                  "q25": float(np.quantile(values, .25)) if len(values) else None,
                                  "q75": float(np.quantile(values, .75)) if len(values) else None,
                                  "max": float(values.max()) if len(values) else None,
                                  "fraction_le_1e_minus12": float(np.mean(values <= 1e-12)) if len(values) else None})
        x = np.array([[r[name] for name in SCORE_NAMES] for r in selected])  # [N,12]
        risk_keys = [(source, risk) for source in ("member_mean", "stage2") for risk in RISKS]
        y = np.array([[r[source + "_" + risk] for source, risk in risk_keys] for r in selected])
        confounders = np.array([[r[k] for k in ("n_GT", "gt_volume_mm3", "min_gt_fragment_volume_mm3")] for r in selected], float)
        point = np.stack([aurc_batch(x[:, i][None, :], y[None, :, :])[0] for i in range(len(SCORE_NAMES))])
        for i, score in enumerate(SCORE_NAMES):
            for j, (source, risk) in enumerate(risk_keys):
                comparison.append({**common, "score": score, "risk_source": source, "risk": risk,
                                   "aurc": float(point[i, j]) if n >= 2 else None,
                                   "random_aurc": float((1 - 1 / n) * y[:, j].mean()) if n >= 2 else None,
                                   "rho": correlation(x[:, i], y[:, j]),
                                   "partial_rho": correlation(x[:, i], y[:, j], confounders)})
        if n < 2:
            # Only possible for the single femur in --smoke; AURC/CI undefined.
            continue
        draws = cluster_draws(selected, bootstrap, seed)
        # Identical resampled patients for every score, source and endpoint.
        boot_keys = [(source, risk) for source in ("member_mean", "stage2") for risk in ("merge_split", "one_minus_f1")]
        indices = [risk_keys.index(k) for k in boot_keys]
        boot = np.empty((bootstrap, len(SCORE_NAMES), len(boot_keys)))
        for start in range(0, bootstrap, 128):
            take = draws[start:start + 128]
            sampled_risk = y[take][:, :, indices]
            for i in range(len(SCORE_NAMES)):
                boot[start:start + len(take), i] = aurc_batch(x[take, i], sampled_risk)
        primary = SCORE_NAMES.index("fa_v2")
        for i, baseline in enumerate(SCORE_NAMES):
            if i == primary:
                continue
            for j, (source, risk) in enumerate(boot_keys):
                difference = boot[:, primary, j] - boot[:, i, j]
                lo, hi = np.quantile(difference, [.025, .975])
                delta = float(point[primary, indices[j]] - point[i, indices[j]])
                paired.append({**common, "score": "fa_v2", "baseline": baseline,
                               "risk_source": source, "risk": risk, "delta_aurc": delta,
                               "ci_low": float(lo), "ci_high": float(hi), "bootstrap_resamples": bootstrap})
                if risk == "merge_split":
                    for target in (100, 210):
                        # Same scan-type balance; E1c's explicitly rough sqrt(n) rule.
                        target_cases = target if stratum == "pooled" else target / 2
                        half_width = float((hi - lo) / 2)
                        projected = half_width * np.sqrt(n_cases / target_cases)
                        projection.append({**common, "baseline": baseline, "risk_source": source,
                                           "risk": risk, "validation_effect_baseline_minus_v2": -delta,
                                           "ci_low_v2_minus_baseline": float(lo), "ci_high_v2_minus_baseline": float(hi),
                                           "current_half_width": half_width, "target_total_cases": target,
                                           "target_stratum_cases": target_cases, "projected_half_width": float(projected),
                                           "favorable_validation_effect": delta < 0,
                                           "projected_half_width_exceeds_favorable_effect": bool(delta < 0 and projected > -delta),
                                           "interpretation": "rough_precision_projection_not_power_or_confirmation"})
        for source in ("member_mean", "stage2"):
            risk = np.array([r[source + "_merge_split"] for r in selected])
            for score in ("fa_v2", "partition_v2", "presence", "mean_entropy", "union_nvi", "pairwise_dice"):
                curve = expected_curve(np.array([r[score] for r in selected]), risk)
                curves.extend({**common, "risk_source": source, "score": score,
                               "coverage": (k + 1) / n, "prefix_mean_risk": float(value)}
                              for k, value in enumerate(curve))
    for name, table in (("comparison", comparison), ("paired_differences", paired), ("risk_coverage", curves)):
        write_csv(output / "validation" / f"{name}.csv", table)
    write_csv(output / "tables/decomposition.csv", decomposition)
    write_csv(output / "tables/precision_projection.csv", projection)
    plot_figures(output, rows, curves)
    return {"n_cases": len({r["case_id"] for r in rows}), "n_units": len(rows),
            "pair_rows": len(data["pair_components"]), "model_risk_rows": len(data["model_risks"]),
            "raw_vi_zero_fraction_units": float(np.mean([r["component_vi"] <= 1e-12 for r in rows])),
            "v2_partition_min": min(r["partition_v2"] for r in rows),
            "v2_partition_max": max(r["partition_v2"] for r in rows),
            "results_role": "validation_diagnosis_exploratory_not_heldout_confirmation"}


def plot_figures(output: Path, rows: list[dict], curves: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(folder / (name + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    axes[0].hist([r["component_vi"] for r in rows], bins=20)
    axes[0].set(xlabel="Raw shared-foreground VI (nats)", ylabel="Bone units")
    for key, label in (("component_q_log_n", "VI / log N"), ("component_q_joint", "VI / H")):
        axes[1].hist([r[key] for r in rows], bins=20, alpha=.5, label=label)
    axes[1].set(xlabel="Normalized partition disagreement", ylabel="Bone units")
    axes[1].legend()
    for key, label in (("partition_v1", "w VI / log N"), ("partition_v2", "w VI / H")):
        axes[2].hist([r[key] for r in rows], bins=20, alpha=.5, label=label)
    axes[2].set(xlabel="Weighted partition contribution", ylabel="Bone units")
    axes[2].legend()
    save(fig, "e1_decomposition")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
    for ax, scan_type in zip(axes, ("pelvic", "femur")):
        selected = [r for r in rows if r["scan_type"] == scan_type]
        points = ax.scatter([r["presence"] for r in selected], [r["partition_v2"] for r in selected],
                            c=[r["member_mean_merge_split"] for r in selected], cmap="viridis")
        ax.set(title=scan_type, xlabel="Presence disagreement", ylabel="Weighted partition (VI/H)")
        fig.colorbar(points, ax=ax, label="Member-mean merge+split risk")
    save(fig, "e1_signal_vs_risk")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for axes_row, source in zip(axes, ("member_mean", "stage2")):
        for ax, stratum in zip(axes_row, ("pooled", "pelvic", "femur")):
            for score in ("fa_v2", "partition_v2", "presence", "mean_entropy", "union_nvi", "pairwise_dice"):
                selected = [r for r in curves if r["risk_source"] == source and r["stratum"] == stratum and r["score"] == score]
                ax.plot([r["coverage"] for r in selected], [r["prefix_mean_risk"] for r in selected], label=score)
            risk = [r[source + "_merge_split"] for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
            ax.axhline(np.mean(risk), color="black", linestyle=":", label="random expectation")
            ax.set(title=f"{source} / {stratum}", xlabel="Coverage", ylabel="Mean merge+split risk")
    axes[0, 0].legend(fontsize=7)
    save(fig, "e1_risk_coverage")
