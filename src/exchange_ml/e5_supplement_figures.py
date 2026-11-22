"""E5 supplementary presentation; frozen scores and example selection stay intact.

Every figure is labelled as a supplementary analysis added after held-out
inspection. Display arrows are deterministic ID discrepancies, not clinical
adjudications. No figure helper mutates its input arrays or score rows.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .e1_data import MEMBERS

POLICIES = ("fa_v2", "presence", "mean_entropy", "partition_v1", "partition_v2", "random")
LABELS = {"fa_v2": "FA-v2", "presence": "Presence", "mean_entropy": "Mean entropy",
          "partition_v1": "Partition-v1", "partition_v2": "Partition-v2", "random": "Random"}
COLORS = {name: color for name, color in zip(POLICIES, ("#2166ac", "#d6604d", "#b35806", "#1b7837", "#762a83", "#666666"))}
STRATA = ("pooled", "pelvic", "femur")
SOURCES = ("member_mean", "stage2")
RISK_LABELS = {"merge_split": "Normalized merge + split risk", "one_minus_f1": "1 - instance F1",
               "missing_volume": "Missing GT volume fraction"}
NOTES = "Supplementary post-heldout analysis. Frozen policies and 10% / 20% bone review budgets; no policy selection from this figure."
CATEGORY_LABELS = {
    "partition_detects_structure": "Partition flags structural disagreement",
    "presence_detects_missing": "Presence flags missing fragments",
    "shared_error_low_disagreement": "Shared error despite low disagreement",
}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finite(value) -> bool:
    return value is not None and bool(np.isfinite(float(value)))


def _save(fig, stem: Path, *, dpi: int = 190) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def plot_budget_figures(output: Path, points: list[dict], contrasts: list[dict], curves: list[dict]) -> None:
    """Show the full curve and actual CT cost at both fixed bone budgets.

    Curve panels use exact expected prefix risks. Error bars in the cost panels
    are marginal risk CIs; paired delta CIs remain in the contrast CSV. CT counts
    are descriptive point estimates, not equal-CT-budget policy comparisons.
    """
    plt = _pyplot()
    groups = sorted({(r["split"], r["risk"]) for r in points})
    for split, risk in groups:
        selected_points = [r for r in points if r["split"] == split and r["risk"] == risk]
        selected_curves = [r for r in curves if r["split"] == split and r["risk"] == risk]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), squeeze=False)
        for row, source in enumerate(SOURCES):
            for column, stratum in enumerate(STRATA):
                ax = axes[row, column]
                for policy in POLICIES:
                    values = sorted((r for r in selected_curves if r["risk_source"] == source
                                     and r["stratum"] == stratum and r["score"] == policy), key=lambda r: r["coverage"])
                    ax.plot([r["coverage"] for r in values], [r["prefix_mean_risk"] for r in values],
                            color=COLORS[policy], label=LABELS[policy], linewidth=1.4,
                            linestyle="--" if policy == "random" else "-")
                    for point in selected_points:
                        if (point["risk_source"] == source and point["stratum"] == stratum
                            and point["score"] == policy and _finite(point["retained_mean_risk"])):
                            ax.plot(point["coverage"], point["retained_mean_risk"], color=COLORS[policy],
                                    marker="s" if point["requested_coverage"] == .8 else "o", markersize=4)
                for coverage in (.8, .9):
                    ax.axvline(coverage, color="#aaaaaa", linestyle=":", linewidth=.7)
                ax.set(title=f"{stratum} | {'Mean member risk' if source == 'member_mean' else 'Stage 2 risk'}",
                       xlabel="Retained bone coverage", ylabel=RISK_LABELS.get(risk, risk), xlim=(0, 1))
                ax.grid(alpha=.15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .98), ncol=6, fontsize=8)
        fig.suptitle(f"{split}: full risk–coverage curve and fixed-budget points", y=1.015, fontsize=12)
        fig.text(.02, .005, NOTES + " Squares: 20% review; circles: 10% review.", fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, .91))
        _save(fig, output / "figures" / f"risk_coverage_{split}_{risk}")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), squeeze=False)
        for row, source in enumerate(SOURCES):
            for column, stratum in enumerate(STRATA):
                ax = axes[row, column]
                defined = 0
                for point in selected_points:
                    policy = point["score"]
                    if (policy not in POLICIES or point["risk_source"] != source or point["stratum"] != stratum
                        or not _finite(point["retained_mean_risk"])):
                        continue
                    y = float(point["retained_mean_risk"])
                    x = float(point["expected_reviewed_cases"])
                    if _finite(point.get("ci_low")) and _finite(point.get("ci_high")):
                        # A percentile interval need not contain the point estimate.
                        ax.vlines(x, float(point["ci_low"]), float(point["ci_high"]), color=COLORS[policy], linewidth=1)
                    marker = "s" if point["requested_coverage"] == .8 else "o"
                    ax.plot(x, y, marker=marker, color=COLORS[policy], markersize=5, linestyle="none")
                    defined += 1
                if not defined:
                    ax.text(.5, .5, "Retained risk undefined\nat this sample size", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=f"{stratum} | {'Mean member risk' if source == 'member_mean' else 'Stage 2 risk'}",
                       xlabel="Expected distinct CTs reviewed", ylabel=RISK_LABELS.get(risk, risk))
                ax.grid(alpha=.15)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", color=COLORS[p], label=LABELS[p], linestyle="none") for p in POLICIES]
        handles += [Line2D([], [], marker="s", color="black", label="20% bone review", linestyle="none"),
                    Line2D([], [], marker="o", color="black", label="10% bone review", linestyle="none")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .98), ncol=4, fontsize=8)
        fig.suptitle(f"{split}: fixed bone budgets, risk uncertainty and actual CT cost", y=1.015, fontsize=12)
        fig.text(.02, .005, "Supplementary post-heldout. Bars: marginal 95% bootstrap risk CI; paired Δ CIs in CSV. "
                 "CT costs are point estimates; these are not equal-CT-budget comparisons.", fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, .88))
        _save(fig, output / "figures" / f"fixed_budget_{split}_{risk}")
        plt.close(fig)


def annotation_target(panel: dict) -> tuple[int, int] | None:
    """Choose a discrepancy pixel with greatest distance to its boundary.

    Distance-to-boundary prioritizes a readable arrow target. This does not
    alter crop, slice, matching, score or clinical-risk labels.
    """
    labels = panel["labels"]
    error = np.logical_or.reduce([np.asarray(p) != np.asarray(labels[0]) for p in labels[1:]])
    if not error.any():
        return None
    distances = ndi.distance_transform_edt(error)
    return tuple(int(i) for i in np.unravel_index(np.argmax(distances), distances.shape))


def _draw_case(example: dict, candidate: dict, risks: list[dict], letter: str):
    plt = _pyplot()
    import matplotlib
    from matplotlib.colors import ListedColormap
    panels, selected = example["panels"], example["selection"]
    if not panels:
        raise ValueError("selected qualitative example has no slices")
    fig, axes = plt.subplots(len(panels), 5, figsize=(14, 3.9 * len(panels) + 1.3), squeeze=False)
    colors = [(0, 0, 0, 0)] + [(*matplotlib.colors.hsv_to_rgb(((i * .61803398875) % 1, .75, .95)), .65)
                               for i in range(1, 202)]
    cmap = ListedColormap(colors)
    annotations = []
    for index, panel in enumerate(panels):
        sy, sx = panel["spacing_yx"]
        extent = (0, panel["ct"].shape[1] * sx, panel["ct"].shape[0] * sy, 0)
        target = annotation_target(panel)
        if target is not None:
            annotations.append({"category": selected["category"], "case_id": selected["case_id"],
                                "bone": selected["bone"], "z_index": panel["z"],
                                "crop_y_index": target[0], "crop_x_index": target[1],
                                "rule": "maximum_distance_inside_display_ID_discrepancy",
                                "role": "display_only_not_clinical_adjudication"})
        for column, ax in enumerate(axes[index]):
            ax.imshow(panel["ct"], cmap="gray", vmin=-200, vmax=1200, extent=extent)
            title = ("CT", "GT", *MEMBERS)[column]
            if column:
                labels = panel["labels"][column - 1]
                ax.imshow(labels, cmap=cmap, vmin=0, vmax=201, interpolation="nearest", extent=extent)
                for label in np.unique(panel["original"][column - 1]):
                    if label:
                        yy, xx = np.nonzero(panel["original"][column - 1] == label)
                        ax.text((xx.mean() + .5) * sx, (yy.mean() + .5) * sy, str(label), color="white", fontsize=7,
                                ha="center", bbox={"facecolor": "black", "alpha": .55, "pad": .4})
                if column >= 2:
                    risk = next(r for r in risks if r["case_id"] == selected["case_id"] and r["bone"] == selected["bone"]
                                and r["model_id"] == MEMBERS[column - 2])
                    title += f"\nF1={1-risk['one_minus_f1']:.3f}  M+S={risk['merge_split']:.3f}\nMissing volume={risk['missing_volume']:.3f}"
                    if not np.any(panel["original"][column - 1]):
                        ax.text(.5, .08, "No predicted target on this slice", ha="center", transform=ax.transAxes,
                                color="white", fontsize=7, bbox={"facecolor": "black", "alpha": .6})
            if target is not None and (column == 1 or (column >= 2 and panel["labels"][column - 1][target] != panel["labels"][0][target])):
                y, x = target
                ax.annotate("", xy=((x + .5) * sx, (y + .5) * sy),
                            xytext=(min(extent[1] * .85, (x + .5) * sx + extent[1] * .17),
                                    max(extent[2] * .1, (y + .5) * sy - extent[2] * .18)),
                            arrowprops={"arrowstyle": "->", "color": "#ffd92f", "lw": 1.5})
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[index, 0].set_ylabel(f"Axial z={panel['z']}", fontsize=8)
    category = selected["category"]
    fig.suptitle(f"{letter}  {CATEGORY_LABELS.get(category, category)} — {selected['case_id']} / {selected['bone']}",
                 x=.03, ha="left", y=.99, fontsize=12)
    subtitle = (f"High score = high disagreement. FA-v2={candidate['fa_v2']:.3f}; partition-v2={candidate['partition_v2']:.3f}; "
                f"presence={candidate['presence']:.3f}; entropy={candidate['mean_entropy']:.3f}.")
    if category == "shared_error_low_disagreement":
        subtitle += " Shared low disagreement does not guarantee correctness."
    fig.text(.03, .945, subtitle, fontsize=8)
    fig.text(.03, .025, "Supplementary post-heldout presentation; original case, crop and slices retained. "
             "Yellow arrow: display-ID discrepancy, not clinical adjudication.\n"
             "Numbers: original IDs; colors: one-to-one GT overlap for display only. CT window [-200,1200] HU; array orientation retained.", fontsize=7)
    fig.subplots_adjust(left=.03, right=.99, top=.87, bottom=.09, hspace=.24, wspace=.03)
    return fig, annotations


def plot_qualitative_supplement(output: Path, examples: list[dict], selections: list[dict],
                                candidates: list[dict], model_risks: list[dict]) -> None:
    """One page per originally selected category, without reselection or recropping."""
    plt = _pyplot()
    from matplotlib.backends.backend_pdf import PdfPages
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    provided = {e["selection"]["category"]: e for e in examples}
    if len(provided) != len(examples):
        raise ValueError("duplicate qualitative category")
    notes, annotations = [], []
    with PdfPages(directory / "qualitative.pdf") as pdf:
        for index, selection in enumerate(selections):
            category = selection["category"]
            example = provided.get(category)
            if example is None:
                notes.append(f"{category}: {'not rendered in this run' if selection['status'] == 'selected' else 'no eligible case; cutoffs retained'}")
                continue
            if selection["status"] != "selected" or any(example["selection"][k] != selection[k] for k in ("case_id", "bone")):
                raise ValueError("provided example differs from frozen selection")
            matching = [r for r in candidates if r["category"] == category and r.get("selected")
                        and r["case_id"] == selection["case_id"] and r["bone"] == selection["bone"]]
            if len(matching) != 1:
                raise ValueError("selected candidate identity missing or ambiguous")
            fig, arrows = _draw_case(example, matching[0], model_risks, chr(ord("A") + index))
            annotations.extend(arrows)
            _save(fig, directory / ("qualitative_" + category), dpi=230)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        if notes or not examples:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.axis("off")
            ax.text(.03, .9, "Supplementary post-heldout qualitative rendering\n\n" + "\n".join(notes or ["No examples supplied."]),
                    va="top", transform=ax.transAxes, fontsize=10)
            pdf.savefig(fig)
            if not examples:
                fig.savefig(directory / "qualitative.png", dpi=180)
            plt.close(fig)
    with (directory / "qualitative_annotations.csv").open("w", newline="") as handle:
        fields = list(annotations[0]) if annotations else ["category", "case_id", "bone", "z_index", "crop_y_index", "crop_x_index", "rule", "role"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(annotations)
