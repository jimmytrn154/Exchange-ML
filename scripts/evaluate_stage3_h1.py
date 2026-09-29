#!/usr/bin/env python3
"""Evaluate Stage 3 H1 scores against frozen Stage 2 validation risks.

This retrospective dataset-wide command is user-typed under __docs__/rule.md.
It never feeds ground truth or risk values back into acquisition scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from exchange_ml.stage3_h1 import (  # noqa: E402
    aurc,
    residualized_spearman,
    risk_coverage_curve,
    safe_spearman,
    top_k_failure_summary,
)
from stage3_h1 import load_config, validate_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_h1.json"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value_quartiles(values: list[float]) -> list[int]:
    """Assign empirical quartiles without splitting equal values across strata."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("quartile values must be non-empty and finite")
    boundaries = np.quantile(array, [0.25, 0.5, 0.75], method="linear")
    return [int(np.searchsorted(boundaries, value, side="right")) for value in array]


def select_example(rows: list[dict[str, object]]) -> str | None:
    """Select a true failure with low entropy and high FA-IPD, deterministically."""
    entropy = np.asarray([float(row["mean_entropy"]) for row in rows])
    fa_ipd = np.asarray([float(row["fa_ipd"]) for row in rows])
    failures = [index for index, row in enumerate(rows) if bool(row["failure_event"])]
    if not failures:
        return None
    candidates = [
        index
        for index in failures
        if entropy[index] <= np.median(entropy) and fa_ipd[index] >= np.median(fa_ipd)
    ]
    if not candidates:
        return None

    case_ids = [str(row["case_id"]) for row in rows]
    entropy_order = sorted(
        range(len(rows)), key=lambda index: (float(entropy[index]), case_ids[index])
    )
    fa_order = sorted(
        range(len(rows)), key=lambda index: (float(fa_ipd[index]), case_ids[index])
    )
    entropy_rank = {index: rank for rank, index in enumerate(entropy_order)}
    fa_rank = {index: rank for rank, index in enumerate(fa_order)}
    selected = sorted(
        candidates,
        key=lambda index: (
            -(fa_rank[index] - entropy_rank[index]),
            case_ids[index],
        ),
    )[0]
    return case_ids[selected]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    workspace = Path(config["workspace_root"])
    scores_path = workspace / "uncertainty_scores.csv"
    scoring_manifest_path = workspace / "uncertainty_scoring_manifest.json"
    scoring_manifest = json.loads(scoring_manifest_path.read_text(encoding="utf-8"))
    if scoring_manifest.get("config_sha256") != sha256(args.config):
        raise RuntimeError("Stage 3 config changed after acquisition scoring")
    if scoring_manifest.get("ground_truth_loaded") is not False:
        raise RuntimeError("Stage 3 scoring manifest must attest ground_truth_loaded=false")
    if int(scoring_manifest.get("case_count", -1)) != 40:
        raise RuntimeError("Stage 3 scoring manifest must cover 40 cases")
    if scoring_manifest.get("score_contract") != config["score_contract"]:
        raise RuntimeError("Stage 3 score contract drifted after acquisition scoring")
    if scoring_manifest.get("probability_contract") != config["probability_contract"]:
        raise RuntimeError("Stage 3 probability contract drifted after inference")
    if scoring_manifest.get("output_sha256") != sha256(scores_path):
        raise RuntimeError("Stage 3 uncertainty score fingerprint mismatch")
    score_rows_raw = read_csv(scores_path)
    scores_by_case = {row["case_id"]: row for row in score_rows_raw}
    if len(score_rows_raw) != len(scores_by_case):
        raise RuntimeError("Stage 3 uncertainty scores contain duplicate case IDs")

    risk_contract = config["risk_contract"]
    primary_connectivity = int(risk_contract["primary_connectivity"])
    metric_rows = [
        row
        for row in read_csv(Path(risk_contract["metrics_csv"]))
        if int(row["connectivity"]) == primary_connectivity
    ]
    metrics_by_case = {row["case_id"]: row for row in metric_rows}

    stage0 = json.loads(
        Path(config["confounders"]["stage0_gt_audit"]).read_text(encoding="utf-8")
    )
    stage1 = json.loads(
        Path(config["confounders"]["stage1_connectivity_audit"]).read_text(
            encoding="utf-8"
        )
    )
    stage0_by_case = {row["case_id"]: row for row in stage0["records"]}
    stage1_by_case = {row["case_id"]: row for row in stage1["records"]}
    case_ids = sorted(scores_by_case)
    if set(case_ids) != set(metrics_by_case) or len(case_ids) != 40:
        raise RuntimeError(
            "Stage 3 scores and frozen Stage 2 validation risks must match 40 cases"
        )

    rows: list[dict[str, object]] = []
    for case_id in case_ids:
        metric = metrics_by_case[case_id]
        audit0 = stage0_by_case[case_id]
        audit1 = stage1_by_case[case_id]
        volumes = [float(value) for value in audit1["fragment_volumes_mm3"].values()]
        if not volumes:
            raise RuntimeError(f"{case_id}: missing GT fragment volumes")
        rows.append(
            {
                **{key: (case_id if key == "case_id" else value) for key, value in scores_by_case[case_id].items()},
                "one_minus_instance_f1": 1.0 - float(metric["instance_f1"]),
                "normalized_merge_split_error": (
                    int(metric["merge_errors"]) + int(metric["split_errors"])
                )
                / max(int(metric["retained_gt_instance_count"]), 1),
                "one_minus_binary_dice": 1.0 - float(metric["binary_dice"]),
                "failure_event": (
                    int(metric["merge_errors"]) + int(metric["split_errors"])
                )
                > 0,
                "retained_gt_fragment_count": int(
                    metric["retained_gt_instance_count"]
                ),
                "gt_foreground_volume_mm3": float(
                    audit0["foreground_volume_mm3"]
                ),
                "minimum_retained_gt_fragment_volume_mm3": min(volumes),
            }
        )

    score_names = list(config["score_contract"]["scores"])
    risk_names = list(risk_contract["risks"])
    for row in rows:
        values = [float(row[name]) for name in score_names + risk_names]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{row['case_id']}: non-finite score or risk")
    confounders = np.asarray(
        [
            [
                float(row["retained_gt_fragment_count"]),
                np.log1p(float(row["gt_foreground_volume_mm3"])),
                np.log1p(float(row["minimum_retained_gt_fragment_volume_mm3"])),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    failures = [bool(row["failure_event"]) for row in rows]

    comparison_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    nested: dict[str, dict[str, object]] = {}
    for score_name in score_names:
        uncertainty = [float(row[score_name]) for row in rows]
        nested[score_name] = {}
        for risk_name in risk_names:
            risks = [float(row[risk_name]) for row in rows]
            coverage, prefix_risk, ordered_ids = risk_coverage_curve(
                uncertainty, risks, case_ids
            )
            for rank, (coverage_value, risk_value, retained_case) in enumerate(
                zip(coverage, prefix_risk, ordered_ids), start=1
            ):
                curve_rows.append(
                    {
                        "score": score_name,
                        "risk": risk_name,
                        "retained_count": rank,
                        "coverage": coverage_value,
                        "prefix_mean_risk": risk_value,
                        "newly_retained_case_id": retained_case,
                    }
                )
            top_k = {
                str(fraction): top_k_failure_summary(
                    uncertainty, failures, risks, case_ids, float(fraction)
                )
                for fraction in config["evaluation"]["top_k_fractions"]
            }
            result = {
                "aurc": aurc(uncertainty, risks, case_ids),
                "spearman_rho": safe_spearman(uncertainty, risks),
                "controlled_spearman_rho": residualized_spearman(
                    uncertainty, risks, confounders
                ),
                "top_k": top_k,
            }
            nested[score_name][risk_name] = result
            comparison_rows.append(
                {
                    "score": score_name,
                    "risk": risk_name,
                    "aurc": result["aurc"],
                    "spearman_rho": result["spearman_rho"],
                    "controlled_spearman_rho": result[
                        "controlled_spearman_rho"
                    ],
                    "top10_failure_enrichment": top_k["0.1"][
                        "failure_enrichment"
                    ],
                    "top20_failure_enrichment": top_k["0.2"][
                        "failure_enrichment"
                    ],
                    "top10_mean_risk": top_k["0.1"]["mean_selected_risk"],
                    "top20_mean_risk": top_k["0.2"]["mean_selected_risk"],
                }
            )

    comparison_path = workspace / "h1_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    curve_path = workspace / "risk_coverage.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    stratified_rows: list[dict[str, object]] = []
    for field in config["confounders"]["fields"]:
        values = [float(row[field]) for row in rows]
        bins = value_quartiles(values)
        for bin_id in range(4):
            indices = [index for index, value in enumerate(bins) if value == bin_id]
            if not indices:
                continue
            for score_name in score_names:
                for risk_name in risk_names:
                    stratified_rows.append(
                        {
                            "confounder": field,
                            "quartile": bin_id + 1,
                            "n_cases": len(indices),
                            "score": score_name,
                            "risk": risk_name,
                            "mean_score": mean(
                                float(rows[index][score_name]) for index in indices
                            ),
                            "mean_risk": mean(
                                float(rows[index][risk_name]) for index in indices
                            ),
                            "spearman_rho": (
                                safe_spearman(
                                    [float(rows[index][score_name]) for index in indices],
                                    [float(rows[index][risk_name]) for index in indices],
                                )
                                if len(indices) >= 2
                                else 0.0
                            ),
                        }
                    )
    strata_path = workspace / "stratified_analysis.csv"
    with strata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stratified_rows[0]))
        writer.writeheader()
        writer.writerows(stratified_rows)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/exchange-ml-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(risk_names), figsize=(18, 5), squeeze=False)
    for axis, risk_name in zip(axes[0], risk_names):
        for score_name in score_names:
            subset = [
                row
                for row in curve_rows
                if row["score"] == score_name and row["risk"] == risk_name
            ]
            axis.plot(
                [float(row["coverage"]) for row in subset],
                [float(row["prefix_mean_risk"]) for row in subset],
                label=score_name,
                linewidth=1.4,
            )
        axis.set_title(risk_name)
        axis.set_xlabel("retained coverage")
        axis.set_ylabel("prefix mean risk")
        axis.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure_path = workspace / "risk_coverage.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary_path = workspace / "h1_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_count": len(rows),
                "base_failure_prevalence": float(np.mean(failures)),
                "risk_source": risk_contract["source"],
                "provenance": {
                    "stage3_config": str(args.config),
                    "stage3_config_sha256": sha256(args.config),
                    "uncertainty_scoring_manifest": str(scoring_manifest_path),
                    "uncertainty_scoring_manifest_sha256": sha256(
                        scoring_manifest_path
                    ),
                    "stage2_metrics": str(risk_contract["metrics_csv"]),
                    "stage2_metrics_sha256": sha256(
                        Path(risk_contract["metrics_csv"])
                    ),
                },
                "results": nested,
                "low_entropy_high_fa_ipd_failure_example": select_example(rows),
                "illustrative_example_rule": config["evaluation"][
                    "illustrative_example"
                ],
                "gateway_status": "PENDING_SCIENTIFIC_ASSESSMENT",
                "gateway_rule": config["evaluation"]["gateway"],
                "artifacts": {
                    "comparison": str(comparison_path),
                    "risk_coverage": str(curve_path),
                    "stratified_analysis": str(strata_path),
                    "figure": str(figure_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote Stage 3 H1 evaluation: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
