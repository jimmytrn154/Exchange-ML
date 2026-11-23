#!/usr/bin/env python3
"""Render the overview from one saved CT and predictions; no inference or GT I/O.

The illustrative slice is selected using predicted anatomy and instance IDs only.
Its images illustrate the inputs, while the numerical scores refer to the whole
3D left hip. Source experiment outputs and hash-bound configurations are read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/exchange-overview-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import SimpleITK as sitk

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "src"))
from exchange_ml.e1_data import MEMBERS, clean_prediction, geometry, mask_bone, read_volume
from exchange_ml.e1_scoring import pair_components
from exchange_ml.e1_statistics import expected_curve

CASE = "004"
BONE = "left_hip"
BLUE, ORANGE, TEAL, INK = "#0077BB", "#EE7733", "#009988", "#263643"
PALETTE = [BLUE, ORANGE, TEAL, "#EE3377", "#BBBBBB", "#33BBEE"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verified_table(folder: Path, manifest: dict, relative: str) -> Path:
    path = folder / relative
    if sha(path) != manifest["output_sha256"][relative]:
        raise ValueError(f"Saved table hash mismatch: {path}")
    return path


def arrow(fig: plt.Figure, start: tuple, end: tuple, color: str = INK,
          *, dashed: bool = False) -> None:
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=8,
                                  linewidth=.85, color=color,
                                  linestyle="--" if dashed else "-"))


def main() -> None:
    e1root = ROOT / "outputs/stage3_h1_v2"
    e3root = e1root / "heldout"
    e1path, e3path = e1root / "e1_manifest.json", e3root / "e3_manifest.json"
    e1, e3 = json.loads(e1path.read_text()), json.loads(e3path.read_text())
    assert e1["status"] == e3["status"] == "complete"
    assert not e1["smoke_only"] and not e3["provenance"]["smoke"]
    assert len(e1["case_ids"]) == 40 and len(e3["case_ids"]) == 100
    assert not set(e1["case_ids"]) & set(e3["case_ids"])
    config_path = ROOT / "configs/e1.json"
    assert sha(config_path) == e1["provenance"]["config_sha256"]
    config = json.loads(config_path.read_text())
    code_sources = ["src/exchange_ml/e1_data.py", "src/exchange_ml/e1_scoring.py",
                    "src/exchange_ml/e1_statistics.py", "src/exchange_ml/instances.py"]
    for rel in code_sources:
        assert sha(ROOT / rel) == e3["provenance"]["code"][rel]

    scores_path = verified_table(e1root, e1, "validation/scores.csv")
    pairs_path = verified_table(e1root, e1, "validation/pair_components.csv")
    curve_path = verified_table(e3root, e3, "risk_coverage.csv")
    evaluation_scores_path = verified_table(e3root, e3, "scores.csv")
    evaluation_risks_path = verified_table(e3root, e3, "risks.csv")
    score = next(r for r in rows(scores_path) if r["case_id"] == CASE and r["bone"] == BONE)
    saved_pairs = [r for r in rows(pairs_path) if r["case_id"] == CASE and r["bone"] == BONE]
    curves = rows(curve_path)

    # Only this one case is loaded. GT, probability maps and other CTs are not read.
    sources = e1["sources"][CASE]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in
                     [e1path, e3path, config_path, scores_path, pairs_path, curve_path,
                      evaluation_scores_path, evaluation_risks_path]}
    source_hashes.update({rel: sha(ROOT / rel) for rel in code_sources})
    predictions, prediction_paths, expected = [], [], None
    original_ids, retained_ids = {}, {}
    for member in MEMBERS:
        key = member + "_prediction"
        path = Path(sources["states"][key]["path"])
        assert sha(path) == sources["sha256"][key]
        array, observed = read_volume(path, expected)
        expected = observed
        original_ids[member] = np.unique(array[mask_bone(array, BONE)]).tolist()
        array = clean_prediction(array, tuple(observed["spacing"][::-1]),
                                 config["prune_mm3"], config["connectivity"])
        array = np.where(mask_bone(array, BONE), array, 0).astype(np.uint8)
        retained_ids[member] = np.unique(array[array > 0]).tolist()
        predictions.append(array)
        prediction_paths.append(path)
        source_hashes[str(path.relative_to(ROOT))] = sources["sha256"][key]

    ct_path = Path(sources["states"]["gt"]["path"]).with_name("image.mha")
    ct_image = sitk.ReadImage(str(ct_path))
    assert all(np.allclose(geometry(ct_image)[k], expected[k], atol=1e-6, rtol=1e-6)
               for k in expected)
    ct = sitk.GetArrayFromImage(ct_image)  # CPU [Z,Y,X], native CT intensity.
    assert ct.ndim == 3 and np.isfinite(ct).all()
    source_hashes[str(ct_path.resolve())] = sha(ct_path)

    union = np.logical_or.reduce([p > 0 for p in predictions])
    area = union.sum(axis=(1, 2))
    # Restrict to slices containing >=2 retained IDs in any single model, then
    # maximize anatomy area. A tie chooses the smallest native axial index.
    eligible = np.array([any(np.count_nonzero(np.unique(p[z])) >= 2
                             for p in predictions) for z in range(len(area))])
    candidates = np.where(eligible, area, -1) if eligible.any() else area
    z = int(candidates.argmax())
    yy, xx = np.where(union[z])
    margin = 12  # Native voxels; identical crop and physical aspect in all panels.
    y0, y1 = max(0, int(yy.min()) - margin), min(ct.shape[1], int(yy.max()) + margin + 1)
    x0, x1 = max(0, int(xx.min()) - margin), min(ct.shape[2], int(xx.max()) + margin + 1)
    image_slice = ct[z, y0:y1, x0:x1]
    slices = [p[z, y0:y1, x0:x1] for p in predictions]

    # Verify the two reported components against the same cleaned 3D predictions.
    recomputed = [pair_components(predictions[i][union], predictions[j][union])
                  for i, j in [(0, 1), (0, 2), (1, 2)]]
    for measured, saved in zip(recomputed, saved_pairs):
        for key in ["presence", "weighted_q_joint", "fa_v2", "n_shared_voxels", "n_union_voxels"]:
            assert np.isclose(measured[key], float(saved[key]), atol=1e-12, rtol=1e-10), key
    a, partition, combined = (float(score[k]) for k in ["presence", "partition_v2", "fa_v2"])
    assert np.isclose(a + partition, combined, atol=1e-12)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(6.9, 2.6), facecolor="white")
    # Labels use reader-facing terms. Internal experiment and score IDs stay in receipt.
    fig.text(.015, .95, "a  CT and ensemble predictions", weight="bold", color=INK)
    fig.text(.477, .95, "b  Prediction-only scores", weight="bold", color=INK)
    fig.text(.761, .95, "c  Evaluate ranking", weight="bold", color=INK)
    xs, width = [.015, .132, .249, .366], .101
    title_labels = ["CT", "Model 1", "Model 2", "Model 3"]
    color_map = {}
    image_axes = []
    for k, (left, title) in enumerate(zip(xs, title_labels)):
        ax = fig.add_axes([left, .31, width, .52])
        ax.imshow(image_slice, cmap="gray", vmin=-300, vmax=1500,
                  interpolation="nearest", aspect=expected["spacing"][1] / expected["spacing"][0])
        if k:
            p = slices[k - 1]
            rgba = np.zeros((*p.shape, 4), dtype=np.float32)
            color_map[MEMBERS[k - 1]] = {}
            for index, label_id in enumerate(retained_ids[MEMBERS[k - 1]]):
                color = PALETTE[(index + 2 * (k - 1)) % len(PALETTE)]
                rgba[p == label_id] = to_rgba(color, .52)
                color_map[MEMBERS[k - 1]][str(label_id)] = color
                region = p == label_id
                if region.any() and not region.all():
                    ax.contour(region, levels=[.5], colors=[color], linewidths=.55)
            ax.imshow(rgba, interpolation="nearest")
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        image_axes.append(ax)
        for spine in ax.spines.values():
            spine.set_color("#D4DCE2"); spine.set_linewidth(.6)
        if k == 0:
            length = 20 / expected["spacing"][0]
            ax.plot([5, 5 + length], [image_slice.shape[0] - 8] * 2, color="white", lw=1.4)
            ax.text(5, image_slice.shape[0] - 12, "20 mm", color="white", fontsize=8)
    fig.text(.015, .22, "Development CT · left hip · same slice and crop", fontsize=8, color=INK)
    fig.text(.015, .135, "Colors distinguish IDs within each model;", fontsize=8, color="#536471")
    fig.text(.015, .055, "they do not match fragments between models.", fontsize=8, color="#536471")

    # Real whole-bone measurements split into the two prediction-only components.
    cards = [("Bone coverage\ndisagreement", "Foreground masks differ", a, BLUE, .66),
             ("Fragment grouping\ndisagreement", "Within shared foreground", partition, TEAL, .36)]
    for title, subtitle, value, color, bottom in cards:
        box = FancyBboxPatch((.48, bottom), .25, .23,
                             boxstyle="round,pad=.008,rounding_size=.012",
                             transform=fig.transFigure, facecolor=to_rgba(color, .055),
                             edgecolor=to_rgba(color, .55), linewidth=.7)
        fig.add_artist(box)
        fig.text(.49, bottom + .137, title, fontsize=8, weight="bold", color=color,
                 linespacing=1.0)
        fig.text(.49, bottom + .064, subtitle, fontsize=8, color=INK)
        fig.text(.49, bottom + .005, f"3D pair mean = {value:.4f}", fontsize=8, color=INK)
    # All three predictions feed the pairwise comparisons (not just model 3).
    group = fig.add_axes([0, 0, 1, 1], frameon=False)
    group.set_axis_off()
    for image_ax in image_axes[1:]:
        position = image_ax.get_position()
        x = (position.x0 + position.x1) / 2
        group.plot([x, x], [position.y0, .365], transform=fig.transFigure, color="#A1AFBA", lw=.7)
    group.plot([.1825, .47, .47], [.365, .365, .77],
               transform=fig.transFigure, color="#A1AFBA", lw=.7)
    arrow(fig, (.47, .77), (.48, .77), BLUE)
    arrow(fig, (.47, .47), (.48, .47), TEAL)
    fig.text(.48, .225, f"Combined = {combined:.4f}", color=INK, weight="bold", fontsize=8)
    fig.text(.48, .135, "Rank bones by disagreement", color=INK, fontsize=8)
    fig.text(.48, .055, "40 development CTs", color="#536471", fontsize=8)

    # Saved held-out curves: identical bone risk and case selection for both scores.
    ax = fig.add_axes([.805, .405, .18, .41])
    series_values = {}
    for key, label, color, linestyle in [("fa_v2", "Combined", TEAL, "-"),
                                         ("presence", "Coverage", BLUE, "--")]:
        series = sorted([r for r in curves if r["stratum"] == "pooled"
                         and r["risk_source"] == "member_mean" and r["score"] == key],
                        key=lambda r: float(r["coverage"]))
        assert len(series) == 200 and all(int(r["n_cases"]) == 100 for r in series)
        x = [float(r["coverage"]) for r in series]
        y = [float(r["prefix_mean_risk"]) for r in series]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.2, label=label)
        series_values[key] = {"coverage": x, "prefix_mean_risk": y}
    # The stronger voxel-normalized partition-only control uses saved table rows
    # and the original tie-aware curve function; it requires no model execution.
    evaluation_scores, evaluation_risks = rows(evaluation_scores_path), rows(evaluation_risks_path)
    risk_index = {(r["case_id"], r["bone"]): r for r in evaluation_risks}
    y_risk = np.array([float(risk_index[r["case_id"], r["bone"]]["member_mean_merge_split"])
                       for r in evaluation_scores])
    control = expected_curve(np.array([float(r["partition_v1"]) for r in evaluation_scores]), y_risk)
    x_control = np.arange(1, len(control) + 1) / len(control)
    ax.plot(x_control, control, color=ORANGE, linestyle=":", linewidth=1.2, label="Partition only")
    series_values["partition_v1"] = {"coverage": x_control.tolist(), "prefix_mean_risk": control.tolist()}
    ax.set(xlim=(0, 1), ylim=(0, None), xticks=[0, 1], xlabel="Fraction retained")
    ax.set_ylabel("Merge/split risk", labelpad=1)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=True, borderpad=.15, handlelength=1.1,
              labelspacing=.12, facecolor="white", edgecolor="none", framealpha=.95,
              handletextpad=.4)
    fig.text(.78, .855, "100 CTs · 200 bones", fontsize=8, color=INK)
    fig.text(.785, .21, "Ground truth → risk", fontsize=8, color="#536471")
    fig.text(.785, .135, "evaluation only", fontsize=8, color="#536471")
    fig.text(.785, .055, "Lower risk is better", fontsize=8, color="#536471")

    figure_dir = PAPER / "figures"
    figure_dir.mkdir(exist_ok=True)
    pdf, png = figure_dir / "overview.pdf", figure_dir / "overview.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=350)
    plt.close(fig)

    caption = ("Prediction-only failure ranking with real CT data. Case 004 (left hip, "
               "development cohort) is shown on the same prediction-selected axial "
               "slice and crop for all three models; colors identify instances within "
               "each model only. Whole-3D-bone pair-mean coverage (0.0146) and weighted "
               "grouping (0.0491) sum to 0.0637. Scores developed on 40 CTs are evaluated "
               "on 100 separate CTs (200 bones). Saved curves include the stronger "
               "voxel-normalized partition-only control. Ground truth is used only to "
               "measure evaluation risk.")
    receipt = {
        "role": "reporting_only_real_saved_data_no_inference_no_new_experiment",
        "case_id": CASE, "bone": BONE, "scan_type": "pelvic",
        "source_sha256": source_hashes,
        "geometry_xyz": expected, "array_axes": "Z,Y,X", "display_plane": "native axial",
        "slice_index_zero_based": z,
        "slice_selection_rule": "largest predicted left-hip union area among axial slices with at least two retained predicted IDs in at least one model; ties use lowest index; if none, largest area overall",
        "selection_uses_gt": False, "selection_uses_risk_or_accuracy": False,
        "eligible_slice_count": int(eligible.sum()), "selected_union_area_voxels": int(area[z]),
        "crop_zyx": {"z": z, "y": [y0, y1], "x": [x0, x1], "intervals": "half-open"},
        "crop_margin_voxels": margin, "ct_window_hu": [-300, 1500],
        "prediction_cleanup": {"prune_mm3": config["prune_mm3"], "connectivity": config["connectivity"],
                               "implementation": "exchange_ml.e1_data.clean_prediction, identical to E1"},
        "predicted_original_ids": original_ids, "predicted_retained_3d_ids": retained_ids,
        "predicted_ids_on_slice": {m: np.unique(p[p > 0]).tolist() for m, p in zip(MEMBERS, slices)},
        "instance_colors_by_model": color_map,
        "color_semantics": "arbitrary within-model instance IDs; no between-model correspondence",
        "scores_from_saved_3d_bone_row": score,
        "plotted_components": {"presence": a, "weighted_vi_over_joint_entropy": partition, "combined": combined},
        "component_recomputation_matches_saved": True,
        "plotted_curve_source": str(curve_path.relative_to(ROOT)),
        "curve_filter": {"stratum": "pooled", "risk_source": "member_mean", "scores": ["fa_v2", "presence", "partition_v1"]},
        "partition_control_curve_transformation": "tie-aware expected_curve of saved partition_v1 scores and saved member_mean_merge_split risk, on the same 200 evaluation bones",
        "plotted_curve_values": series_values,
        "cohort": {"development_cts": 40, "evaluation_cts": 100, "evaluation_bones": 200,
                   "disjoint_case_ids_verified": True},
        "gt_volumes_read": False, "gt_used_for_predictor_or_slice": False,
        "ground_truth_arrow_destination": "evaluation risk only",
        "transformation": {"script": "paper/make_overview.py", "sha256": sha(Path(__file__))},
        "output_sha256": {str(p.relative_to(PAPER)): sha(p) for p in [pdf, png]},
        "figure_size_inches": [6.9, 2.6], "png_dpi": 350,
        "visual_verification": {"method": "view_image inspection of rendered PNG at native size",
                                "text_min_font_size_pt": 8,
                                "checks": ["identical image geometry and crop", "visible real fracture and two predicted IDs per model",
                                           "branch labels and right-edge notes fit", "GT arrow terminates at evaluation risk only"]},
        "caption_draft": caption,
        "limitations": ["One illustrative development case; no case-selection claim of representative accuracy",
                        "2D images are inputs, whereas displayed components use the complete 3D bone",
                        "Curves show three of the 12 evaluated scores; complete comparisons are reported in the main table"],
        "figure_table_trace": [{"artifact_id": "fig:overview", "source_data": list(source_hashes),
                                "transformation": {"script": "paper/make_overview.py", "sha256": sha(Path(__file__))},
                                "caption_claim": caption,
                                "supported_manuscript_claims": ["Prediction-only disagreement separates coverage from fragment grouping; ground truth is used to evaluate ranking"],
                                "limitations": ["Illustrative slice and whole-3D-bone scores are distinct"]}],
    }
    (PAPER / "evidence").mkdir(exist_ok=True)
    (PAPER / "evidence/overview.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"figure": str(png), "slice": z,
                      "slice_ids": receipt["predicted_ids_on_slice"],
                      "scores": receipt["plotted_components"]}, indent=2))


if __name__ == "__main__":
    main()
