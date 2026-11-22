"""Deterministic E5 CT panels; display matching never affects scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.optimize import linear_sum_assignment

from .e1_data import MEMBERS, mask_bone, read_volume
from .e4_e5_data import bounding_box, load_clean_maps


def display_alignment(maps: list[np.ndarray], gt: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
    """One-to-one overlap matching for colors, preserving split/unmatched IDs.

    Multiple predicted pieces cannot collapse into one color. Original IDs are
    annotated on the figure and exported alongside this display-only mapping.
    """
    gt_ids = np.unique(gt[gt > 0])
    gt_display = np.zeros(201, dtype=np.int32)
    gt_display[gt_ids] = np.arange(1, len(gt_ids) + 1)
    output = [gt_display[gt]]
    records = [{"model_id": "GT", "original_id": int(g), "display_id": int(gt_display[g])} for g in gt_ids]
    next_color = len(gt_ids) + 1
    for model, pred in zip(MEMBERS, maps):
        pred_ids = np.unique(pred[pred > 0])
        counts = np.bincount(gt.ravel().astype(np.int32) * 201 + pred.ravel(), minlength=201 * 201).reshape(201, 201)
        overlap = counts[np.ix_(gt_ids, pred_ids)]
        gi, pi = linear_sum_assignment(-overlap)
        lut = np.zeros(201, dtype=np.int32)
        for g, p in zip(gi, pi):
            if overlap[g, p] > 0:
                lut[pred_ids[p]] = gt_display[gt_ids[g]]
        for pid in pred_ids:
            if lut[pid] == 0:
                lut[pid] = next_color
                next_color += 1
            records.append({"model_id": model, "original_id": int(pid), "display_id": int(lut[pid])})
        output.append(lut[pred])
    return output, records


def error_slices(display: list[np.ndarray], count: int = 2) -> list[int]:
    """Two axial slices with maximal aligned-ID discrepancy, separated in Z."""
    truth = display[0]
    error = np.logical_or.reduce([p != truth for p in display[1:]])
    mass = error.sum(axis=(1, 2))
    if not mass.any():
        mass = (truth > 0).sum(axis=(1, 2))
    order = sorted(range(len(mass)), key=lambda z: (-int(mass[z]), z))
    chosen = [order[0]]
    distance = max(1, len(mass) // 5)
    for z in order[1:]:
        if mass[z] > 0 and all(abs(z - old) >= distance for old in chosen):
            chosen.append(z)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        chosen.extend(z for z in order if mass[z] > 0 and z not in chosen)
    return sorted(chosen[:count])


def prepare_example(selection: dict, paths: dict[str, Path], config: dict) -> dict:
    maps, geom, _ = load_clean_maps(paths, config)
    gt, _ = read_volume(paths["gt"], geom)
    bone = selection["bone"]
    truth = np.where(mask_bone(gt, bone), gt, 0)
    target = [np.where(mask_bone(maps[m], bone), maps[m], 0) for m in MEMBERS]
    box = bounding_box(np.logical_or.reduce([m > 0 for m in [truth, *target]]),
                       config["qualitative"]["crop_margin_voxels"])
    del maps, gt
    truth, target = truth[box], [p[box] for p in target]
    display, mapping = display_alignment(target, truth)
    slices = error_slices(display, config["qualitative"]["slices_per_example"])
    ct_image = sitk.ReadImage(str(paths["image"]))
    from .e1_data import geometry
    if any(not np.allclose(geometry(ct_image)[k], geom[k], rtol=1e-6, atol=1e-6) for k in geom):
        raise ValueError(f"CT geometry differs for {selection['case_id']}")
    ct = sitk.GetArrayViewFromImage(ct_image)[box]
    panels, coordinates = [], []
    for z in slices:
        original_z = z + box[0].start
        center_index = ((box[2].start + box[2].stop - 1) / 2,
                        (box[1].start + box[1].stop - 1) / 2, float(original_z))
        physical = ct_image.TransformContinuousIndexToPhysicalPoint(center_index)
        coordinates.append({"case_id": selection["case_id"], "bone": bone, "category": selection["category"],
                            "z_index": original_z, "crop_z_start": box[0].start, "crop_z_stop": box[0].stop,
                            "crop_y_start": box[1].start, "crop_y_stop": box[1].stop,
                            "crop_x_start": box[2].start, "crop_x_stop": box[2].stop,
                            "center_x_mm": physical[0], "center_y_mm": physical[1], "center_z_mm": physical[2],
                            "slice_rule": "max_aligned_ID_discrepancy_two_axial_slices",
                            "n_slices_available": len(slices)})
        panels.append({"ct": np.array(ct[z], copy=True),
                       "labels": [np.array(p[z], copy=True) for p in display],
                       "original": [np.array(p[z], copy=True) for p in [truth, *target]], "z": original_z,
                       "spacing_yx": (geom["spacing"][1], geom["spacing"][0])})
    return {"selection": selection, "panels": panels, "coordinates": coordinates,
            "mapping": [{"case_id": selection["case_id"], "bone": bone, "category": selection["category"], **r} for r in mapping]}


def plot_qualitative(output: Path, examples: list[dict], selections: list[dict],
                     candidates: list[dict], model_risks: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output.parent.mkdir(parents=True, exist_ok=True)
    nrows = sum(len(e["panels"]) for e in examples)
    if not nrows:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(.05, .6, "No held-out unit meets the declared qualitative criteria.\nCutoffs were retained.", transform=ax.transAxes)
    else:
        fig, axes = plt.subplots(nrows, 5, figsize=(16, 3.4 * nrows), squeeze=False)
        # A deterministic hue sequence avoids cycling after 20 instances.
        colors = [(0, 0, 0, 0)] + [(*matplotlib.colors.hsv_to_rgb(((i * .61803398875) % 1, .75, .95)), .65)
                                   for i in range(1, 202)]
        cmap = ListedColormap(colors)
        row_index = 0
        for example in examples:
            selected = example["selection"]
            candidate = next(r for r in candidates if r["category"] == selected["category"] and r["selected"])
            for panel_index, panel in enumerate(example["panels"]):
                axes_row = axes[row_index]
                sy, sx = panel["spacing_yx"]
                extent = (0, panel["ct"].shape[1] * sx, panel["ct"].shape[0] * sy, 0)
                for col, ax in enumerate(axes_row):
                    ax.imshow(panel["ct"], cmap="gray", vmin=-200, vmax=1200, extent=extent)
                    if col:
                        ax.imshow(panel["labels"][col - 1], cmap=cmap, vmin=0, vmax=201,
                                  interpolation="nearest", extent=extent)
                        for label in np.unique(panel["original"][col - 1]):
                            if label:
                                yy, xx = np.nonzero(panel["original"][col - 1] == label)
                                ax.text((xx.mean() + .5) * sx, (yy.mean() + .5) * sy, str(label),
                                        color="white", fontsize=6, ha="center", bbox={"facecolor": "black", "alpha": .5, "pad": .3})
                    title = ("CT", "GT", *MEMBERS)[col]
                    if col >= 2:
                        risk = next(r for r in model_risks if r["case_id"] == selected["case_id"]
                                    and r["bone"] == selected["bone"] and r["model_id"] == MEMBERS[col - 2])
                        title += f" | F1={1-risk['one_minus_f1']:.3f}; M+S={risk['merge_split']:.3f}"
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                text = f"{selected['case_id']} / {selected['bone']} | axial z={panel['z']}"
                if panel_index == 0:
                    text += f"\n{selected['category']}\nFA-v2={candidate['fa_v2']:.3f}; partition={candidate['partition_v2']:.3f}\npresence={candidate['presence']:.3f}; entropy={candidate['mean_entropy']:.3f}"
                axes_row[0].set_ylabel(text, fontsize=8)
                row_index += 1
        fig.subplots_adjust(left=.18, right=.99, top=.96, bottom=.06, hspace=.2, wspace=.03)
    absent = [r["category"] for r in selections if r["status"] != "selected"]
    footer = ("High score = high disagreement. Axial index orientation; CT window [-200,1200] HU. "
              "Colors: one-to-one GT overlap for display only; numbers: original instance IDs.")
    if absent:
        footer += "\nNo eligible case: " + ", ".join(absent)
    fig.text(.02, .015, footer, fontsize=8)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
