#!/usr/bin/env python3
"""Reporting-only matched-cohort E2 merge figure from hash-verified CSVs.

No image, prediction or perturbation is generated. The original E2 artifacts
remain unchanged. Both panels hold bone membership fixed across severities.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/stage3_h1_v2/perturbations/responses.csv"
MANIFEST = SOURCE.parent / "e2_manifest.json"
STEM = ROOT / "outputs/stage3_h1_v2/reporting/e2_matched_merge"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_digest = sha256(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "complete" or manifest["smoke_only"]:
        raise ValueError("the full completed E2 experiment is required")
    expected = manifest["output_sha256"]["responses.csv"]
    if sha256(SOURCE) != expected:
        raise ValueError("E2 responses.csv does not match its recorded SHA-256")
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        if row["operation"] != "merge" or row["eligible"] != "True":
            continue
        key = row["case_id"], row["bone"]
        level = int(row["severity"])
        values = groups.setdefault(key, {})
        if level in values:
            raise ValueError(f"duplicate merge level: {key}/{level}")
        value = float(row["q_joint"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid Q2 value: {key}/{level}")
        values[level] = value
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), sharey=True)
    cohorts = []
    for ax, levels, panel, expected_n in zip(axes, ((1, 2), (1, 2, 3)), ("A", "B"), (12, 4)):
        keys = sorted(key for key, values in groups.items() if all(level in values for level in levels))
        if len(keys) != expected_n:
            raise ValueError(f"cohort size differs from audited E2: {levels}: {len(keys)}")
        values = np.array([[groups[key][level] for level in levels] for key in keys])
        medians = np.median(values, axis=0)
        n_cases = len({key[0] for key in keys})
        for trajectory in values:
            ax.plot(levels, trajectory, color="#999999", alpha=.42, linewidth=.9, marker="o", markersize=2.5)
        ax.plot(levels, medians, color="#2166ac", linewidth=2.2, marker="o", markersize=5,
                label="Median across the same bones")
        for level, median in zip(levels, medians):
            ax.annotate(f"{median:.5f}", (level, median), xytext=(0, 9), textcoords="offset points",
                        ha="center", color="#174975", fontsize=8)
        ax.set(title=f"{panel}  Fixed cohort: {len(keys)} bones / {n_cases} cases",
               xlabel="Number of disjoint merged fragment pairs", ylim=(0, 1), xticks=levels)
        ax.set_xlim(min(levels) - .2, max(levels) + .2)
        ax.grid(alpha=.15)
        cohorts.append({"panel": panel, "levels": list(levels), "n_bones": len(keys), "n_cases": n_cases,
                        "units": [{"case_id": cid, "bone": bone} for cid, bone in keys],
                        "median_q_joint": medians.tolist()})
    axes[0].set_ylabel(r"Partition disagreement $Q_2 = \mathrm{VI}/H(P,Q)$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Controlled GT merges: within-bone responses in matched cohorts", y=.98, fontsize=12)
    fig.text(.03, .025, "Thin lines: individual bones; blue: median. Each panel retains the same bones at every level.\n"
             "Descriptive controlled perturbations; no confidence intervals, held-out detection or clinical severity claim.", fontsize=8)
    fig.subplots_adjust(left=.08, right=.985, top=.84, bottom=.2, wspace=.16)
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    if sha256(SOURCE) != expected or sha256(MANIFEST) != manifest_digest:
        raise ValueError("source E2 artifacts changed during reporting")
    metadata = {"role": "reporting_only_controlled_GT_perturbations_not_heldout_or_clinical_severity",
                "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": expected,
                "source_manifest_path": str(MANIFEST.relative_to(ROOT)), "source_manifest_sha256": manifest_digest,
                "plot_script_path": str(Path(__file__).resolve().relative_to(ROOT)), "plot_script_sha256": sha256(Path(__file__)),
                "cohorts": cohorts, "y_axis": {"quantity": "q_joint", "minimum": 0, "maximum": 1},
                "uncertainty": "individual trajectories and median only; no CI",
                "outputs": {str(STEM.with_suffix(ext).relative_to(ROOT)): sha256(STEM.with_suffix(ext))
                            for ext in (".pdf", ".png")}}
    STEM.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "cohorts": cohorts, "outputs": metadata["outputs"]}), flush=True)


if __name__ == "__main__":
    main()
