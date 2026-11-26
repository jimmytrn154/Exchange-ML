#!/usr/bin/env python3
"""Draw the two manuscript figures from saved evidence only.

Fig. 1 (concept): the four real CT/prediction panels are cropped from the
provenance-tracked ``figures/overview.png`` (produced by ``make_overview.py``
from case 004, left hip; see ``evidence/overview.json``); the remaining panels
are schematic. Fig. 2 (evidence): every value is read from saved experiment
CSVs. No training, inference, scoring, fitting or bootstrap is run here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/exchange-paper-figures-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
OUT = PAPER / "figures"
V2 = ROOT / "outputs/stage3_h1_v2"
SOURCES = {
    "overview_png": PAPER / "figures/overview.png",
    "responses": V2 / "perturbations/responses.csv",
    "dev_scores": V2 / "validation/scores.csv",
    "heldout_rc": V2 / "heldout/risk_coverage.csv",
    "heldout_comparison": V2 / "heldout/comparison.csv",
    "heldout_scores": V2 / "heldout/scores.csv",
    "replication": V2 / "perturbations/replication.csv",
    "heldout_risks": V2 / "heldout/risks.csv",
}

# Okabe-Ito based palette, one colour per concept, reused in both figures.
C_COV = "#0072B2"     # coverage / presence
C_GRP = "#D55E00"     # grouping / partition-only
C_GRP2 = "#E69F00"    # grouping, joint-entropy weighted variant
C_COMB = "#009E73"    # combined, joint entropy
C_COMB1 = "#69A995"   # combined, voxel-count
C_VOX = "#4D4D4D"     # voxel entropy
C_RAND = "#9A9A9A"
C_TEXT = "#1F2933"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.6, "axes.linewidth": 0.65, "xtick.major.width": 0.65,
    "ytick.major.width": 0.65, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT, "axes.edgecolor": "#52606D",
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save_figure(fig: plt.Figure, basename: str) -> None:
    """Keep the IEEE full-width physical size; export vector text/paths and 450-dpi previews."""
    fig.savefig(OUT / f"{basename}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{basename}.png", dpi=450)
    plt.close(fig)



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def expected_curve(scores: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Copy of exchange_ml.e1_statistics.expected_curve (numpy only).

    Expected prefix risk over all orders within exact score ties; lowest
    scores are retained first.
    """
    order = np.argsort(scores, kind="stable")
    x, y = scores[order], risks[order].copy()
    starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(x)]):
        y[start:end] = y[start:end].mean()
    return np.cumsum(y) / np.arange(1, len(y) + 1)


def panel_label(ax, text: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=C_TEXT)


# --------------------------------------------------------------------------
# Figure 1: concept
# --------------------------------------------------------------------------
CT_BOXES = ((40, 277), (323, 560), (605, 842), (888, 1125))  # x-ranges in overview.png
CT_ROWS = (284, 499)
CT_TITLES = ("CT", "Model 1", "Model 2", "Model 3")


def draw_schematic(ax: plt.Axes, variant: str = "base") -> None:
    """Vector illustration only: same-foreground merge or unchanged-grouping removal."""
    ax.set(xlim=(-1.15, 1.15), ylim=(-.75, .75), aspect="equal")
    outline = Ellipse((0, 0), 2, 1.12, facecolor="#8C99A3", edgecolor="none")
    ax.add_patch(outline)
    if variant != "merge":
        right = Rectangle((0, -.7), 1.2, 1.4, facecolor="#CBD2D8", edgecolor="none")
        right.set_clip_path(outline)
        ax.add_patch(right)
        divider, = ax.plot([0, 0], [-.56, .56], color="white", lw=1.1)
        divider.set_clip_path(outline)
    if variant == "coverage":
        # The removed tip is shown by a dashed outline, never as predicted foreground.
        ax.add_patch(Rectangle((.53, -.7), .7, 1.4, facecolor="white", edgecolor="none"))
        ref = Ellipse((0, 0), 2, 1.12, facecolor="none", edgecolor=C_COV,
                      lw=.85, linestyle=(0, (2, 1.5)))
        ref.set_clip_path(Rectangle((.53, -.7), .7, 1.4, transform=ax.transData))
        ax.add_patch(ref)
        ax.plot([.53, .53], [-.475, .475], color=C_TEXT, lw=.6)
    border = Ellipse((0, 0), 2, 1.12, facecolor="none", edgecolor=C_TEXT, lw=.65)
    if variant == "coverage":
        border.set_clip_path(Rectangle((-1.2, -.7), 1.73, 1.4, transform=ax.transData))
    ax.add_patch(border)
    ax.axis("off")


def figure1() -> None:
    fig = plt.figure(figsize=(7.16, 1.9), facecolor="white")
    # Fixed source crops retain exactly the same real CT and instance overlays.
    src = mpimg.imread(SOURCES["overview_png"])
    positions = [(.010, .48), (.111, .48), (.010, .055), (.111, .055)]
    for (x0, x1), title, (left, bottom) in zip(CT_BOXES, CT_TITLES, positions):
        ax = fig.add_axes([left, bottom, .091, .335])
        ax.imshow(src[CT_ROWS[0]:CT_ROWS[1], x0:x1, :3], interpolation="none")
        ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")
    fig.text(.010, .97, "a  Real CT example", fontsize=8, weight="bold", va="top")

    # Panel b is deliberately marked schematic and uses smooth vector shapes.
    bx = .231
    fig.text(bx, .97, "b  Coverage vs. grouping", fontsize=8, weight="bold", va="top")
    for k, (variant, title, color, formula, note) in enumerate((
        ("coverage", "Coverage", C_COV, r"$A>0,\ \mathrm{VI}=0$", "Foreground differs"),
        ("merge", "Grouping", C_GRP, r"$A=0,\ \mathrm{VI}>0$", "Fragment division differs"),
    )):
        bottom = .48 - .425 * k
        for j, v in enumerate(("base", variant)):
            ax = fig.add_axes([bx + j * .079, bottom, .076, .30])
            draw_schematic(ax, v)
            ax.set_title("Model " + ("$i$" if j == 0 else "$j$"), fontsize=6.8, pad=1)
        fig.text(bx + .164, bottom + .24, title, fontsize=7.6, weight="bold", color=color)
        fig.text(bx + .164, bottom + .13, formula, fontsize=7.2, color=color)
        fig.text(bx + .164, bottom + .035, note, fontsize=6.6)
    fig.text(bx, .014, "Schematic examples", fontsize=6.5, color="#52606D")

    cx = .565
    fig.text(cx, .97, "c  Prediction-only scoring", fontsize=8, weight="bold", va="top")
    ax = fig.add_axes([cx, .035, .425, .82])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str,
            edge: str, face: str, size: float = 7) -> None:
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=0,rounding_size=0.018",
                     linewidth=.8, edgecolor=edge, facecolor=face))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=size, linespacing=1.25)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=7, linewidth=.8, color="#52606D",
                     shrinkA=1, shrinkB=1))

    box(.005, .69, .345, .285, "Coverage $A$\n" + r"$|F_i\,\triangle\,F_j|/|F_i\cup F_j|$",
        C_COV, "#EEF5FA", size=6.8)
    box(.385, .69, .61, .285,
        "Grouping VI$(P,Q)$\n" + r"Shared voxels $C=F_i\cap F_j$" + "\n"
        + r"Weight $w=|C|/|F_i\cup F_j|$",
        C_GRP, "#FCF2EB", size=6.8)
    box(.385, .345, .61, .265,
        r"$Q_{\rm vox}=\mathrm{VI}/\log |C|$" + "\n" + r"$Q_{\rm joint}=\mathrm{VI}/H(P,Q)$",
        C_GRP, "white", size=7)
    box(.005, .055, .99, .205,
        r"Grouping only: $wQ$   |   Combined: $A+wQ$",
        C_COMB, "#EDF7F3", size=7)
    arrow(.1775, .69, .1775, .26)
    arrow(.69, .69, .69, .61)
    arrow(.69, .345, .69, .26)
    fig.text(cx + .008, .014, "Average 3 model pairs  →  rank bones", fontsize=7)
    save_figure(fig, "fig1_concept")


# --------------------------------------------------------------------------
# Figure 2: evidence
# --------------------------------------------------------------------------
INTERVENTIONS = (("identity", "Unchanged"), ("permutation", "Permuted IDs"),
                 ("merge", "Merge"), ("split", "Split"),
                 ("missing", "Removal"), ("boundary", "Boundary"))
RC_SCORES = (("mean_entropy", "Mean entropy", C_VOX, (0, (1, 1.2))),
             ("presence", "Coverage $A$", C_COV, "-"),
             ("fa_v1", r"$D_{\rm vox}$", C_COMB1, (0, (4, 1.5))),
             ("fa_v2", r"$D_{\rm joint}$", C_COMB, "-"),
             ("partition_v1", r"$wQ_{\rm vox}$", C_GRP, "-"))
RHO_SCORES = (("mean_entropy", "Mean entropy"), ("presence", "Coverage $A$"),
              ("fa_v1", r"$D_{\rm vox}$"), ("fa_v2", r"$D_{\rm joint}$"),
              ("partition_v1", r"$wQ_{\rm vox}$"))
RHO_RISKS = (("merge_split", "Merge/\nsplit"), ("one_minus_f1", "$1-F_1$"),
             ("missing_volume", "Missing\nvolume"))


def figure2(receipt: dict) -> None:
    fig = plt.figure(figsize=(7.16, 2.0))
    axes = [fig.add_axes(bounds) for bounds in (
        (.133, .22, .072, .62), (.273, .22, .103, .62),
        (.461, .22, .251, .45), (.847, .27, .145, .53),
    )]
    headers = ((.010, "a  Controlled responses"), (.235, "b  Grouping share"),
               (.422, "c  Held-out ranking · AURC ↓"), (.765, "d  Failure association"))
    for left, title in headers:
        fig.text(left, .98, title, weight="bold", fontsize=8, va="top")

    # (a) controlled interventions: fraction of interventions with a nonzero response
    resp = [r for r in rows(SOURCES["responses"]) if r["eligible"] == "True"]
    groups = defaultdict(list)
    for r in resp:
        groups[r["operation"]].append(r)
    mat, counts = [], []
    for key, _ in INTERVENTIONS:
        g = groups[key]
        counts.append(len(g))
        mat.append([np.mean([float(r["presence"]) > 1e-12 for r in g]),
                    np.mean([float(r["weighted_q_joint"]) > 1e-12 for r in g])])
    mat = np.array(mat)
    receipt["fig2a_nonzero_fraction"] = {k: dict(zip(("coverage", "grouping"), m.tolist()))
                                         for (k, _), m in zip(INTERVENTIONS, mat)}
    receipt["fig2a_counts"] = dict(zip([k for k, _ in INTERVENTIONS], counts))
    merges = groups["merge"]
    receipt["text_merge_median_grouping"] = {
        "weighted_q_log_n": float(np.median([float(r["weighted_q_log_n"]) for r in merges])),
        "weighted_q_joint": float(np.median([float(r["weighted_q_joint"]) for r in merges]))}
    # Count replication: Q_vox ratio at r=16 and shared-voxel counts (text, Sec. Results B).
    rep = [r for r in rows(SOURCES["replication"]) if r["source"] == "E1"]
    q1 = defaultdict(dict)
    for r in rep:
        q1[(r["case_id"], r["bone"], r["member_i"], r["member_j"])][int(r["r"])] = r
    ratios = [float(v[16]["Q1"]) / float(v[1]["Q1"]) for v in q1.values() if float(v[1]["Q1"]) > 0]
    shared = [int(v[1]["N"]) for v in q1.values()]
    receipt["text_replication"] = {
        "positive_vi_tables": len(ratios),
        "q_vox_ratio_r16_median": float(np.median(ratios)),
        "max_abs_change_q_joint": max(abs(float(r["Q2_delta"])) for r in rep),
        "shared_voxels_median": float(np.median(shared)),
        "log_shared_voxels_median": float(np.log(np.median(shared)))}
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("w2k", ["#FFFFFF", "#3E4C59"])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(2):
            ax.text(j, i, f"{100 * mat[i, j]:.0f}%", ha="center", va="center", fontsize=6.8,
                    color="white" if mat[i, j] > 0.5 else C_TEXT)
    ax.set_yticks(range(len(INTERVENTIONS)))
    ax.set_yticklabels([f"{lab} ({n})" for (_, lab), n in zip(INTERVENTIONS, counts)])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$A$", "$wQ$"])
    ax.get_xticklabels()[0].set_color(C_COV); ax.get_xticklabels()[1].set_color(C_GRP)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(.133 + .036, .865, "Nonzero (%)", ha="center", fontsize=6.8)

    # (b) development: share of the grouping term in the combined score
    dev = rows(SOURCES["dev_scores"])
    s1 = np.array([float(r["partition_share_v1"]) for r in dev])
    s2 = np.array([float(r["partition_share_v2"]) for r in dev])
    receipt["fig2b_median_share"] = {"vox": float(np.median(s1)), "joint": float(np.median(s2)),
                                     "n_bones": len(dev)}
    ax = axes[1]
    rng = np.random.default_rng(0)  # jitter only
    for x, s, col in ((0, s1, C_COMB1), (1, s2, C_COMB)):
        ax.scatter(x + rng.uniform(-0.18, 0.18, s.size), 100 * s, s=5, color=col,
                   alpha=0.75, linewidths=0)
        ax.plot([x - 0.28, x + 0.28], [100 * np.median(s)] * 2, color=C_TEXT, lw=1.1)
        ax.text(x, 103, f"{100 * np.median(s):.1f}%", ha="center", va="bottom", fontsize=6.8, weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$D_{\rm vox}$", r"$D_{\rm joint}$"])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-3, 112)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.grid(True, color="#E4E9ED", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(r"Grouping share $wQ/D$ (%)", labelpad=2)
    fig.text(.3245, .865, "80 development bones", ha="center", fontsize=6.8)

    # (c) held-out risk-coverage, merge/split. Curves are recomputed from saved
    # per-bone scores and risks with the project's tie-aware convention
    # (exchange_ml.e1_statistics.expected_curve) and must reproduce the saved
    # curves and AURCs exactly; any mismatch aborts.
    rc = [r for r in rows(SOURCES["heldout_rc"]) if r["stratum"] == "pooled"
          and r["risk_source"] == "member_mean"]
    comp = {(r["score"], r["risk"]): r for r in rows(SOURCES["heldout_comparison"])
            if r["stratum"] == "pooled" and r["risk_source"] == "member_mean"}
    scores = {(r["case_id"], r["bone"]): r for r in rows(SOURCES["heldout_scores"])}
    risks = {(r["case_id"], r["bone"]): r for r in rows(SOURCES["heldout_risks"])}
    keys = sorted(scores)
    if set(keys) != set(risks) or len(keys) != 200:
        raise ValueError("held-out score/risk units do not match")
    y_ms = np.array([float(risks[k]["member_mean_merge_split"]) for k in keys])
    curves = {}
    for key, *_ in RC_SCORES:
        curve = expected_curve(np.array([float(scores[k][key]) for k in keys]), y_ms)
        aurc = float(np.sum((curve[:-1] + curve[1:]) / (2 * len(curve))))
        if abs(aurc - float(comp[(key, "merge_split")]["aurc"])) > 1e-9:
            raise ValueError(f"AURC reconstruction mismatch for {key}")
        saved = sorted((float(r["coverage"]), float(r["prefix_mean_risk"])) for r in rc
                       if r["score"] == key)
        if saved and np.max(np.abs(np.array([v for _, v in saved]) - curve)) > 1e-9:
            raise ValueError(f"curve reconstruction mismatch for {key}")
        curves[key] = curve
    receipt["fig2c_reconstruction"] = "curves and AURCs reproduce saved values within 1e-9"
    n = len(y_ms)
    receipt["text_numbers"] = {
        "heldout_bones_with_nonzero_merge_split": int(np.sum(y_ms > 0)),
        "retained_risk_at_coverage_0.30": {k: float(curves[k][int(round(0.3 * n)) - 1])
                                           for k in curves},
    }
    # Missing-volume extremes versus coverage-score rank (text, Sec. "Unanimous omission").
    mv = np.array([float(risks[k]["member_mean_missing_volume"]) for k in keys])
    cov = np.array([float(scores[k]["presence"]) for k in keys])
    top6 = np.argsort(-mv, kind="stable")[:6]
    rank_desc = {int(i): int(np.sum(cov > cov[i])) + 1 for i in top6}
    receipt["text_numbers"]["largest_missing_volume_bones"] = [
        {"bone": "/".join(keys[i]), "missing_volume": float(mv[i]),
         "coverage_score_rank_desc_of_200": rank_desc[int(i)]} for i in top6]
    ax = axes[2]
    rand = float(comp[("presence", "merge_split")]["random_aurc"])
    ax.axhline(rand, color=C_RAND, lw=0.8, ls=(0, (2, 2)), label=f"Random  {rand:.3f}")
    receipt["fig2c_aurc"] = {"random": rand}
    for key, label, col, ls in RC_SCORES:
        y = curves[key]
        x = np.arange(1, len(y) + 1) / len(y)
        aurc = float(comp[(key, "merge_split")]["aurc"])
        receipt["fig2c_aurc"][key] = aurc
        ax.plot(x, y, color=col, lw=1.25, ls=ls, label=f"{label}  {aurc:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.0))
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.yaxis.grid(True, color="#E4E9ED", lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Fraction of bones retained", labelpad=3)
    ax.set_ylabel("Retained merge/split risk", labelpad=2)
    ax.legend(loc="lower center", frameon=False, handlelength=1.6,
              labelspacing=.3, handletextpad=.45, columnspacing=.9, ncol=2,
              bbox_to_anchor=(.5, 1.025), borderaxespad=0,
              fontsize=6.6)

    # (d) held-out Spearman correlation between score and each risk
    rho = np.array([[float(comp[(s, r)]["rho"]) for r, _ in RHO_RISKS] for s, _ in RHO_SCORES])
    receipt["fig2d_rho"] = {s: dict(zip([r for r, _ in RHO_RISKS], v.tolist()))
                            for (s, _), v in zip(RHO_SCORES, rho)}
    ax = axes[3]
    div = LinearSegmentedColormap.from_list("div", ["#5E3C99", "#FFFFFF", "#E66101"])
    heatmap = ax.imshow(rho, cmap=div, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=7.1,
                    color="white" if abs(rho[i, j]) > 0.55 else C_TEXT)
    ax.set_yticks(range(len(RHO_SCORES))); ax.set_yticklabels([l for _, l in RHO_SCORES])
    for tick, (s, _) in zip(ax.get_yticklabels(), RHO_SCORES):
        tick.set_color({"presence": C_COV, "partition_v1": C_GRP, "partition_v2": C_GRP,
                        "fa_v2": C_COMB}.get(s, C_TEXT))
    ax.set_xticks(range(len(RHO_RISKS)))
    ax.set_xticklabels([l for _, l in RHO_RISKS], fontsize=6.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    colorbar_ax = fig.add_axes([.867, .135, .106, .025])
    colorbar = fig.colorbar(heatmap, cax=colorbar_ax, orientation="horizontal", ticks=[-.8, 0, .8])
    colorbar.outline.set_visible(False)
    colorbar_ax.tick_params(labelsize=6.5, length=2, pad=1)
    fig.text(.920, .025, r"Spearman $\rho$", ha="center", fontsize=7)
    save_figure(fig, "fig2_evidence")


def main() -> None:
    receipt: dict = {
        "role": "reporting_only_saved_evidence_no_new_experiment",
        "fig1_real_panels": "cropped from figures/overview.png (case 004, left hip, development); "
                            "provenance in evidence/overview.json",
        "fig1_schematic": "panels b and c are schematic illustrations, not data",
        "visual_design": {"width_inches": 7.16, "height_inches": [1.9, 2.0],
                          "preview_dpi": 450, "minimum_new_text_pt": 6.5,
                          "schematics": "vector paths; real CT overlays unchanged",
                          "heatmap_selection": "same five scores as risk-coverage panel; all 12 in Table I"},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES.values()},
        "producer": {"script": "paper/make_figures.py", "sha256": sha256(Path(__file__))},
    }
    figure1()
    figure2(receipt)
    receipt["output_sha256"] = {f"figures/{n}": sha256(OUT / n) for n in
                                ("fig1_concept.pdf", "fig1_concept.png",
                                 "fig2_evidence.pdf", "fig2_evidence.png")}
    (PAPER / "evidence/figures.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Wrote figures/fig1_concept.*, figures/fig2_evidence.*, evidence/figures.json")


if __name__ == "__main__":
    main()
