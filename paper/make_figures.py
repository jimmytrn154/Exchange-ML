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
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

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
C_COMB1 = "#7FCDB4"   # combined, voxel-count
C_VOX = "#4D4D4D"     # voxel entropy
C_RAND = "#9A9A9A"
C_TEXT = "#1F2933"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False,
})


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


def toy_masks():
    """Schematic 2D label maps: a bone with two fragments and three variants."""
    h, w = 28, 44
    yy, xx = np.mgrid[0:h, 0:w]
    bone = ((xx - 21) / 19.0) ** 2 + ((yy - 14) / 10.5) ** 2 <= 1
    base = np.zeros((h, w), int)
    base[bone & (xx < 20)] = 1
    base[bone & (xx >= 20)] = 2
    shrunk = base.copy()
    shrunk[bone & (xx >= 31)] = 0          # coverage change: tip missing
    merged = np.where(bone, 1, 0)          # grouping change: one fragment
    return base, shrunk, merged


def draw_mask(ax, lab, colors, outline_ref=None):
    cmap = ListedColormap(["white"] + colors)
    ax.imshow(lab, cmap=cmap, vmin=0, vmax=len(colors), interpolation="nearest")
    if outline_ref is not None:
        diff = (outline_ref > 0) ^ (lab > 0)
        over = np.zeros((*lab.shape, 4))
        over[diff] = (0.0, 0.447, 0.698, 0.9)
        ax.imshow(over, interpolation="nearest")
    ax.contour(lab > 0, levels=[0.5], colors=[C_TEXT], linewidths=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figure1() -> None:
    fig = plt.figure(figsize=(7.16, 1.9))
    # (a) real example, 2x2 grid of panels cropped from the provenance-tracked overview
    src = mpimg.imread(SOURCES["overview_png"])
    w, h, gap = 0.088, 0.36, 0.008
    pos = [(0.008, 0.47), (0.008 + w + gap, 0.47), (0.008, 0.035), (0.008 + w + gap, 0.035)]
    for (x0, x1), title, (px, py) in zip(CT_BOXES, CT_TITLES, pos):
        ax = fig.add_axes([px, py, w, h])
        ax.imshow(src[CT_ROWS[0]:CT_ROWS[1], x0:x1, :3], interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=6.5, pad=1.5, color=C_TEXT)
    fig.text(0.008, 0.975, "a", fontsize=8, fontweight="bold", color=C_TEXT, va="top")
    fig.text(0.026, 0.975, "Ensemble output", fontsize=7.2, color=C_TEXT, va="top")

    # (b) schematic: coverage vs grouping disagreement
    base, shrunk, merged = toy_masks()
    frag = ["#8C8C8C", "#C7C7C7"]
    bx = 0.225
    fig.text(bx, 0.975, "b", fontsize=8, fontweight="bold", color=C_TEXT, va="top")
    fig.text(bx + 0.018, 0.975, "Two separable ways to disagree", fontsize=7.2,
             color=C_TEXT, va="top")
    rows_spec = [
        ("Coverage", C_COV, shrunk, [frag[0], frag[1]], True,
         r"$A>0,\ \mathrm{VI}=0$", "missing or extra bone,\nboundary shift"),
        ("Grouping", C_GRP, merged, [frag[0]], False,
         r"$A=0,\ \mathrm{VI}>0$", "merged or split\nfragments"),
    ]
    for k, (name, color, other, ocol, outline, eq, err) in enumerate(rows_spec):
        y = 0.47 - 0.435 * k
        ax1 = fig.add_axes([bx, y, 0.075, 0.33])
        draw_mask(ax1, base, frag)
        ax1.set_title("model $i$", fontsize=6.2, pad=1, color=C_TEXT)
        ax2 = fig.add_axes([bx + 0.082, y, 0.075, 0.33])
        draw_mask(ax2, other, ocol, outline_ref=base if outline else None)
        ax2.set_title("model $j$", fontsize=6.2, pad=1, color=C_TEXT)
        fig.text(bx + 0.168, y + 0.25, name, fontsize=7, fontweight="bold", color=color, va="center")
        fig.text(bx + 0.168, y + 0.15, eq, fontsize=6.8, color=color, va="center")
        fig.text(bx + 0.168, y + 0.04, err, fontsize=6.2, color=C_TEXT, va="center",
                 linespacing=1.1)

    # (c) score construction
    cx = 0.545
    fig.text(cx, 0.975, "c", fontsize=8, fontweight="bold", color=C_TEXT, va="top")
    fig.text(cx + 0.018, 0.975, "Prediction-only scores for one bone", fontsize=7.2,
             color=C_TEXT, va="top")
    ax = fig.add_axes([cx, 0.055, 0.44, 0.80])
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")

    def box(x, y, bw, bh, text, edge, face, size=6.0):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.006,rounding_size=0.025",
                                    linewidth=0.8, edgecolor=edge, facecolor=face))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=size,
                color=C_TEXT, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.7, color="#52606D", shrinkA=0, shrinkB=0))

    box(0.00, 0.64, 0.33, 0.34,
        "Coverage, pair $(i,j)$\n" + r"$A=|F_i\,\triangle\,F_j|\,/\,|F_i\cup F_j|$",
        C_COV, "#EAF3FA")
    box(0.38, 0.64, 0.62, 0.34,
        r"Grouping on shared foreground $C=F_i\cap F_j$" + "\n"
        + r"$\mathrm{VI}(P,Q)=H(P|Q)+H(Q|P)$,  $w=|C|\,/\,|F_i\cup F_j|$",
        C_GRP, "#FBEDE4")
    box(0.38, 0.34, 0.62, 0.22,
        r"$Q_{\rm vox}=\mathrm{VI}/\log|C|$" + "   or   " + r"$Q_{\rm joint}=\mathrm{VI}/H(P,Q)$",
        C_GRP, "white")
    box(0.00, 0.00, 1.00, 0.26,
        "Mean over the 3 model pairs  →  grouping only $wQ$,  combined $D=A+wQ$\n"
        "Bones ranked by score; references define risk only",
        C_COMB, "#E6F5EF")
    arrow(0.165, 0.64, 0.165, 0.26)
    arrow(0.69, 0.64, 0.69, 0.56)
    arrow(0.69, 0.34, 0.69, 0.26)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_concept.{ext}", dpi=300)
    plt.close(fig)


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
RHO_SCORES = (("mean_entropy", "Mean entropy"), ("p95_entropy", "P95 entropy"),
              ("presence", "Coverage $A$"), ("count_variance", "Count variance"),
              ("hungarian", "Hungarian"), ("fa_v1", r"$D_{\rm vox}$"),
              ("fa_v2", r"$D_{\rm joint}$"), ("partition_v2", r"$wQ_{\rm joint}$"),
              ("partition_v1", r"$wQ_{\rm vox}$"))
RHO_RISKS = (("merge_split", "Merge/\nsplit"), ("one_minus_f1", "$1-F_1$"),
             ("missing_volume", "Missing\nvolume"))


def figure2(receipt: dict) -> None:
    fig = plt.figure(figsize=(7.16, 2.0))
    gs = fig.add_gridspec(1, 4, left=0.125, right=0.985, bottom=0.19, top=0.86,
                          wspace=0.62, width_ratios=[0.5, 0.55, 1.25, 0.85])

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
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("w2k", ["#FFFFFF", "#3E4C59"])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(2):
            ax.text(j, i, f"{100 * mat[i, j]:.0f}%", ha="center", va="center", fontsize=6.3,
                    color="white" if mat[i, j] > 0.5 else C_TEXT)
    ax.set_yticks(range(len(INTERVENTIONS)))
    ax.set_yticklabels([f"{lab} ({n})" for (_, lab), n in zip(INTERVENTIONS, counts)])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$A$", "$wQ$"])
    ax.get_xticklabels()[0].set_color(C_COV); ax.get_xticklabels()[1].set_color(C_GRP)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Nonzero response", pad=3)

    # (b) development: share of the grouping term in the combined score
    dev = rows(SOURCES["dev_scores"])
    s1 = np.array([float(r["partition_share_v1"]) for r in dev])
    s2 = np.array([float(r["partition_share_v2"]) for r in dev])
    receipt["fig2b_median_share"] = {"vox": float(np.median(s1)), "joint": float(np.median(s2)),
                                     "n_bones": len(dev)}
    ax = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)  # jitter only
    for x, s, col in ((0, s1, C_COMB1), (1, s2, C_COMB)):
        ax.scatter(x + rng.uniform(-0.18, 0.18, s.size), 100 * s, s=4, color=col,
                   alpha=0.75, linewidths=0)
        ax.plot([x - 0.28, x + 0.28], [100 * np.median(s)] * 2, color=C_TEXT, lw=1.1)
        ax.text(x, 103, f"{100 * np.median(s):.0f}%", ha="center", va="bottom", fontsize=6.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$D_{\rm vox}$", r"$D_{\rm joint}$"])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-3, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(r"Grouping share $wQ/D$ (%)")
    ax.set_title("Development bones", pad=3)

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
    ax = fig.add_subplot(gs[2])
    rand = float(comp[("presence", "merge_split")]["random_aurc"])
    ax.axhline(rand, color=C_RAND, lw=0.8, ls=(0, (2, 2)), label=f"Random  {rand:.3f}")
    receipt["fig2c_aurc"] = {"random": rand}
    for key, label, col, ls in RC_SCORES:
        y = curves[key]
        x = np.arange(1, len(y) + 1) / len(y)
        aurc = float(comp[(key, "merge_split")]["aurc"])
        receipt["fig2c_aurc"][key] = aurc
        ax.plot(x, y, color=col, lw=1.1, ls=ls, label=f"{label}  {aurc:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.spines["left"].set_bounds(0, 1.0)
    ax.set_xlabel("Coverage (fraction of bones retained)")
    ax.set_ylabel("Merge/split risk of retained bones")
    ax.legend(loc="upper center", frameon=False, handlelength=1.8, borderaxespad=0.0,
              labelspacing=0.2, handletextpad=0.4, columnspacing=1.0, ncol=2,
              bbox_to_anchor=(0.5, 1.0), title="AURC (lower is better)",
              title_fontsize=6.0, fontsize=5.9)
    ax.set_title("Held-out bones (n = 200)", pad=3)

    # (d) held-out Spearman correlation between score and each risk
    rho = np.array([[float(comp[(s, r)]["rho"]) for r, _ in RHO_RISKS] for s, _ in RHO_SCORES])
    receipt["fig2d_rho"] = {s: dict(zip([r for r, _ in RHO_RISKS], v.tolist()))
                            for (s, _), v in zip(RHO_SCORES, rho)}
    ax = fig.add_subplot(gs[3])
    div = LinearSegmentedColormap.from_list("div", ["#5E3C99", "#FFFFFF", "#E66101"])
    ax.imshow(rho, cmap=div, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(rho[i, j]) > 0.55 else C_TEXT)
    ax.set_yticks(range(len(RHO_SCORES))); ax.set_yticklabels([l for _, l in RHO_SCORES])
    for tick, (s, _) in zip(ax.get_yticklabels(), RHO_SCORES):
        tick.set_color({"presence": C_COV, "partition_v1": C_GRP, "partition_v2": C_GRP,
                        "fa_v2": C_COMB}.get(s, C_TEXT))
    ax.set_xticks(range(len(RHO_RISKS))); ax.set_xticklabels([l for _, l in RHO_RISKS])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(r"Spearman $\rho$ held-out", labelpad=3)
    for letter, axis, dx in zip("abcd", fig.axes, (0.115, 0.06, 0.05, 0.10)):
        box = axis.get_position()
        fig.text(box.x0 - dx, 0.985, letter, fontsize=8, fontweight="bold", color=C_TEXT, va="top")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_evidence.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    receipt: dict = {
        "role": "reporting_only_saved_evidence_no_new_experiment",
        "fig1_real_panels": "cropped from figures/overview.png (case 004, left hip, development); "
                            "provenance in evidence/overview.json",
        "fig1_schematic": "panels b and c are schematic illustrations, not data",
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
