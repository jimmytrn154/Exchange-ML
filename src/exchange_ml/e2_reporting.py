"""Static E2 tables/figures; median/IQR only, no independence claim or CI."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .e2_perturbations import METRICS


def write_rows(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output: Path, responses: list[dict], replicated: list[dict], phantom_rows: list[dict], cases: list[dict]) -> dict:
    write_rows(output / "selected_cases.csv", cases)
    write_rows(output / "phantoms.csv", phantom_rows)
    write_rows(output / "responses.csv", responses)
    positive = [r for r in replicated if r["positive_vi"]]
    zero = [r for r in replicated if not r["positive_vi"]]
    columns = list(replicated[0])
    write_rows(output / "replication.csv", positive, columns)
    write_rows(output / "replication_zero_vi.csv", zero, columns)
    summary_rows = []
    for stratum in ("pooled", "pelvic", "femur"):
        subset = [r for r in responses if stratum == "pooled" or r["scan_type"] == stratum]
        for operation, severity in sorted({(r["operation"], r["severity"]) for r in subset}):
            group = [r for r in subset if r["operation"] == operation and r["severity"] == severity]
            valid = [r for r in group if r["eligible"]]
            row = {"stratum": stratum, "operation": operation, "severity": severity,
                   "n_cases": len({r["case_id"] for r in valid}), "n_units": len(valid),
                   "n_ineligible": len(group) - len(valid),
                   "n_target_not_reached": sum(r["reason"] == "target_not_reached" for r in valid)}
            for quantity in (*METRICS, "changed_volume_fraction", "achieved_dice", "affected_volume_mm3"):
                values = [r[quantity] for r in valid]
                for suffix, q in (("q25", .25), ("median", .5), ("q75", .75)):
                    row[quantity + "_" + suffix] = float(np.quantile(values, q)) if values else None
            summary_rows.append(row)
    write_rows(output / "response_summary.csv", summary_rows)
    replication_summary = []
    for group in ("phantom", "E1_positive_vi", "E1_zero_vi"):
        for factor in sorted({r["r"] for r in replicated}):
            selected = [r for r in replicated if r["r"] == factor and
                        ((group == "phantom" and r["source"] == "phantom") or
                         (group == "E1_positive_vi" and r["source"] == "E1" and r["positive_vi"]) or
                         (group == "E1_zero_vi" and r["source"] == "E1" and not r["positive_vi"]))]
            replication_summary.append({"group": group, "r": factor, "n_pairs": len(selected),
                                        "n_cases": len({r["case_id"] for r in selected if r["case_id"]}),
                                        **{k + "_max_abs": max((abs(r[k]) for r in selected), default=None)
                                           for k in ("VI_delta", "H_delta", "A_delta", "w_delta", "Q2_delta", "Q1_residual")}})
    write_rows(output / "replication_summary.csv", replication_summary)
    plot_figures(output, summary_rows, replicated, responses)
    return {"case_count": len(cases), "bone_units": len({(r["case_id"], r["bone"]) for r in responses}),
            "response_rows": len(responses), "eligible_response_rows": sum(r["eligible"] for r in responses),
            "target_not_reached_rows": sum(r["reason"] == "target_not_reached" for r in responses),
            "replication_rows_positive_vi": len(positive), "replication_rows_zero_vi": len(zero),
            "e1_pairs_reconstructed": len({(r["case_id"], r["bone"], r["member_i"], r["member_j"]) for r in replicated if r["source"] == "E1"}),
            "phantom_checks": len(phantom_rows), "phantom_checks_passed": all(r["passed"] for r in phantom_rows),
            "interpretation": "controlled_mechanism_check_not_clinical_or_heldout_validation"}


def plot_figures(output: Path, summaries: list[dict], replicated: list[dict], responses: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def save(fig, stem):
        fig.savefig(output / (stem + ".png"), dpi=180, bbox_inches="tight")
        fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    metrics = (("presence", "presence"), ("weighted_q_log_n", "weighted VI/log N"),
               ("weighted_q_joint", "weighted VI/H"), ("fa_v2", "FA-v2"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, operation in zip(axes.ravel(), ("merge", "split", "missing", "boundary")):
        candidates = [r for r in summaries if r["stratum"] == "pooled" and r["operation"] == operation]
        rows = [r for r in candidates if r["n_units"] or operation in {"merge", "split"}]
        def x_value(row):
            if operation == "missing":
                return row["changed_volume_fraction_median"]
            if operation == "boundary":
                return row["achieved_dice_median"]
            return float(row["severity"])
        rows.sort(key=x_value)
        x = [x_value(r) for r in rows]
        for metric, label in metrics:
            y = np.array([r[metric + "_median"] for r in rows], dtype=float)
            low = np.array([r[metric + "_q25"] for r in rows], dtype=float)
            high = np.array([r[metric + "_q75"] for r in rows], dtype=float)
            ax.plot(x, y, marker="o", label=label)
            ax.fill_between(x, low, high, alpha=.13)
        for location, row in zip(x, rows):
            ax.annotate(f"n={row['n_units']}", (location, .98), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="top")
        if operation in {"merge", "split"}:
            ax.set_xticks(x)
        elif any(r["n_ineligible"] for r in candidates):
            ax.text(.02, .9, f"Ineligible rows: {sum(r['n_ineligible'] for r in candidates)}", transform=ax.transAxes, fontsize=7)
        if operation == "boundary":
            samples = [r for r in responses if r["operation"] == "boundary" and r["eligible"]]
            ax.scatter([r["achieved_dice"] for r in samples], [r["presence"] for r in samples], alpha=.2, s=10, color="gray")
            missed = [r for r in samples if not r["target_reached"]]
            if missed:
                ax.scatter([r["achieved_dice"] for r in missed], [r["presence"] for r in missed], marker="x", color="red", label="target not reached")
        xlabel = {"merge": "Disjoint merged pairs", "split": "Parts in the split fragment",
                  "missing": "Removed GT volume fraction (median)", "boundary": "Achieved foreground Dice (median)"}[operation]
        ax.set(title=operation, xlabel=xlabel, ylabel="Disagreement (median / IQR)")
    axes[0, 0].legend(fontsize=8)
    save(fig, "response_curves")
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for source, label in (("phantom", "phantom merge"), ("E1", "E1 positive-VI pairs")):
        records = [r for r in replicated if r["source"] == source and r["positive_vi"]]
        bases = {(r["case_id"], r["bone"], r["member_i"], r["member_j"]): r for r in records if r["r"] == 1}
        factors = sorted({r["r"] for r in records})
        for quantity, style in (("Q1", "-"), ("Q2", "--")):
            medians, lows, highs = [], [], []
            for factor in factors:
                ratios = [r[quantity] / bases[(r["case_id"], r["bone"], r["member_i"], r["member_j"])][quantity]
                          for r in records if r["r"] == factor]
                medians.append(np.median(ratios)); lows.append(np.quantile(ratios, .25)); highs.append(np.quantile(ratios, .75))
            ax.plot(factors, medians, style, marker="o", label=label + " / " + quantity)
            ax.fill_between(factors, lows, highs, alpha=.12)
    ax.set(xscale="log", xlabel="Count replication factor r", ylabel="Normalized score / value at r=1")
    ax.legend(fontsize=8)
    save(fig, "replication_curves")
