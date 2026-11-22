#!/usr/bin/env python3
"""Read saved CV artifacts; no fitting, new CV, bootstrap or volume reads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


def diagnose(root: Path) -> dict:
    sys.path.insert(0, str(root / "src"))
    from exchange_ml.e1_statistics import aurc_batch

    base = root / "outputs/partition_v3_development_v1"
    manifest = json.loads((base / "partition_manifest.json").read_text())
    assert manifest["status"] == "complete" and not manifest["summary"]["smoke_only"]
    assert manifest["signature"] == hashlib.sha256(json.dumps(manifest["provenance"], sort_keys=True).encode()).hexdigest()
    for name, digest in {**manifest["output_sha256"], **manifest["cache_sha256"]}.items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["provenance"]["code_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    paths = [base / name for name in ("oof_scores.csv", "fold_models.json", "tuning.json",
                                     "comparison.csv", "paired_differences.csv", "ct_review_budget.csv",
                                     "ct_review_differences.csv", "partition_manifest.json")]
    paths += [root / "src/exchange_ml" / name for name in ("partition_cv.py", "e1_statistics.py")]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (base / "oof_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name, value in list(row.items()):
            if name not in ("case_id", "bone", "scan_type", "all_members_empty"):
                row[name] = float(value)
    saved = json.loads((base / "fold_models.json").read_text())
    heads = ("ridge_p1", "ridge_p1_a", "ridge_p1_g_a")
    p1 = np.array([r["partition_v1"] for r in rows])
    graph = np.array([r["graph_degree"] for r in rows])
    risk = np.array([r["member_mean_merge_split"] for r in rows])
    target = np.log1p(risk)
    folds = np.array([int(r["outer_fold"]) for r in rows])
    pred = np.array([[r[h] for h in heads] for r in rows])

    def aurc(score: np.ndarray, outcome: np.ndarray) -> float:
        return float(aurc_batch(score[None], outcome[None, :, None])[0, 0])

    def identify(i: int) -> dict:
        return {k: rows[i][k] for k in ("case_id", "bone", "outer_fold", "partition_v1",
                                       "graph_degree", "presence", "member_mean_merge_split")}

    fold_models = []
    intercept_null = np.zeros(len(rows))
    for record in saved:
        model, fold, head = record["model"], record["outer_fold"], record["head"]
        selected = folds == fold
        train = ~selected
        assert set(model["train_case_ids"]) == {rows[i]["case_id"] for i in np.flatnonzero(train)}
        assert set(record["test_case_ids"]) == {rows[i]["case_id"] for i in np.flatnonzero(selected)}
        features = np.array([[r[n] for n in model["features"]] for r in rows])
        z = (features - model["mean"]) / model["scale"]
        reproduced = z @ model["coefficient"] + model["intercept"]
        max_error = float(np.max(np.abs(reproduced[selected] - pred[selected, heads.index(head)])))
        assert max_error <= 1e-15
        # Diagonal of the saved model's ridge smoother: a predictor-only
        # influence diagnostic. No response-dependent model is fitted here.
        inverse = np.linalg.inv(z[train].T @ z[train] + model["alpha"] * np.eye(z.shape[1]))
        leverage = 1 / int(train.sum()) + np.einsum("ij,jk,ik->i", z[train], inverse, z[train])
        train_indices = np.flatnonzero(train)
        top = np.argsort(leverage)[-3:][::-1]
        result = {"outer_fold": fold, "head": head, "alpha": model["alpha"],
                  "intercept": model["intercept"], "feature_mean": model["mean"],
                  "feature_scale": model["scale"], "coefficient": model["coefficient"],
                  "n_train_units": int(train.sum()), "n_test_units": int(selected.sum()),
                  "max_saved_prediction_error": max_error,
                  "same_fold_p1_aurc": aurc(p1[selected], risk[selected]),
                  "same_fold_head_aurc": aurc(pred[selected, heads.index(head)], risk[selected]),
                  "test_mean_log1p_risk": float(target[selected].mean()),
                  "test_mean_prediction": float(reproduced[selected].mean()),
                  "top_train_ridge_smoother_diagonal": [
                      {**identify(int(train_indices[i])), "diagonal": float(leverage[i])} for i in top]}
        fold_models.append(result)
        if head == "ridge_p1":
            result["raw_p1_affine_slope"] = model["coefficient"][0] / model["scale"][0]
            result["predicted_logrisk_at_p1_zero"] = model["intercept"] - model["mean"][0] * result["raw_p1_affine_slope"]
            intercept_null[selected] = model["intercept"]
    triangle = np.triu(np.ones((len(rows), len(rows)), dtype=bool), 1)
    p1_difference = p1[:, None] - p1
    eligible = triangle & (p1_difference != 0)
    inversions = []
    mse = {}
    zero_positive = (p1[:, None] == 0) & (p1[None, :] > 0) & (risk[:, None] == 0) & (risk[None, :] > 0)
    for index, head in enumerate(heads):
        difference = pred[:, index, None] - pred[:, index]
        for scope, mask in (("within_fold", folds[:, None] == folds),
                            ("between_folds", folds[:, None] != folds)):
            pairs = eligible & mask
            inversions.append({"head": head, "scope": scope,
                               "n_non_tied_p1_pairs": int(pairs.sum()),
                               "n_p1_order_inversions": int((pairs & (p1_difference * difference < 0)).sum())})
        mse[head] = float(np.mean((pred[:, index] - target) ** 2))
    mse["outer_train_intercept_only"] = float(np.mean((intercept_null - target) ** 2))

    def read_table(name: str) -> list[dict]:
        with (base / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    examples = [{k: row[k] for k in ("case_id", "bone", "partition_v1", "graph_degree", "presence",
                                   "member_mean_merge_split", "member_mean_one_minus_f1",
                                   "member_mean_missing_volume", "stage2_missing_volume")}
                for row in rows if row["case_id"] == "292" and row["bone"] == "femur"]
    cases = sorted({r["case_id"] for r in rows})
    structural = {
        "all_members_empty_bone_flags": sum(r["all_members_empty"] == "True" for r in rows),
        "complete_omission_positive_cts": {
            source: sum(any(r[f"{source}_missing_volume"] == 1. for r in rows if r["case_id"] == cid)
                        for cid in cases) for source in ("member_mean", "stage2")},
        "coverage_blindness_illustration_not_selected_benchmark": examples,
        "existing_comparison_rows": [r for r in read_table("comparison.csv")
                                     if r["risk_source"] == "member_mean"
                                     and r["score"] in ("partition_v1", "presence", "graph_degree", "fa_v2")],
        "existing_graph_minus_p1_rows": [r for r in read_table("paired_differences.csv")
                                        if r["score"] == "graph_degree" and r["baseline"] == "partition_v1"],
        "existing_ct_policy_rows": [r for r in read_table("ct_review_budget.csv")
                                    if r["stratum"] == "pooled" and r["estimand"] == "case_macro"
                                    and r["risk_source"] == "member_mean"
                                    and r["policy"] in ("partition_v1", "presence", "graph_degree", "guard_ridge_p1_g_a")],
    }
    return {"role": "read_only_descriptive_diagnosis_no_refit_no_new_validation",
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "recorded_output_hashes_verified": len(manifest["output_sha256"]),
            "recorded_cache_hashes_verified": len(manifest["cache_sha256"]),
            "source_sha256": hashes, "n_cases": len({r["case_id"] for r in rows}), "n_units": len(rows),
            "p1_graph_pearson": float(np.corrcoef(p1, graph)[0, 1]),
            "p1_graph_spearman": float(spearmanr(p1, graph).statistic),
            "p1_distribution": {"min": float(p1.min()), "median": float(np.median(p1)), "max": float(p1.max())},
            "fold_models": fold_models, "p1_order_inversions": inversions, "oof_logrisk_mse": mse,
            "zero_support": {"p1_zero_units": int((p1 == 0).sum()),
                             "p1_zero_and_risk_zero": int(((p1 == 0) & (risk == 0)).sum()),
                             "p1_zero_and_risk_positive": int(((p1 == 0) & (risk > 0)).sum()),
                             "p1_positive_and_risk_zero": int(((p1 > 0) & (risk == 0)).sum()),
                             "p1_and_graph_zero_masks_equal": bool(np.array_equal(p1 == 0, graph == 0)),
                             "zero_vs_positive_risk_pairs": int(zero_positive.sum()),
                             "inverted_zero_vs_positive_pairs": {
                                 h: int((zero_positive & (pred[:, j, None] > pred[:, j])).sum())
                                 for j, h in enumerate(heads)}},
            "largest_p1_units": [identify(int(i)) for i in np.argsort(p1)[-5:][::-1]],
            "structural_and_coverage_readout": structural,
            "interpretation_limits": [
                "Same-fold AURCs are diagnostics, not a replacement primary endpoint.",
                "Pooled OOF compares scores from different fitted models; deployed final model has one common scale.",
                "Within-fold preservation exonerates the one-feature monotone transform only; final candidate loses within four of five folds.",
                "Alpha was selected for prediction MSE, not ranking/AURC.",
                "No source or completed experiment artifact was modified."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/sdb/shared/dang.cpm/Exchange-ML"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/partition_v3_ridge_diagnosis.json"))
    args = parser.parse_args()
    result = diagnose(args.root.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Read-only ridge diagnosis PASS: {result['n_cases']} CTs, {len(result['fold_models'])} saved models; {args.output}")
