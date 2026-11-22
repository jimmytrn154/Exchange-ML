from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.partition_cv import (FINAL_HEAD, HEAD_FEATURES, fit_model, grouped_folds,
                                      nested_cv, predict_model, validate_rows)


def synthetic_rows(n_cases: int = 8) -> list[dict]:
    """Predictor/label fixture: three pelvic bones versus one femur per CT."""
    rows = []
    for i in range(n_cases):
        scan = "pelvic" if i < n_cases // 2 else "femur"
        bones = ("sacrum", "left_hip", "right_hip") if scan == "pelvic" else ("femur",)
        for j, bone in enumerate(bones):
            p1, presence, graph = (i + j + 1) / (n_cases + 3), (i % 3) / 3, (i + 2 * j) / 5
            row = {"case_id": f"case_{i:03d}", "scan_type": scan, "bone": bone,
                   **{s: (i + j + 1) / (n_cases + 3) for s in SCORE_NAMES},
                   "partition_v1": p1, "presence": presence, "graph_degree": graph,
                   "all_members_empty": False}
            for source, factor in (("member_mean", 1.), ("stage2", .7)):
                row.update({source + "_merge_split": (2 * p1 + .3 * graph) * factor,
                            source + "_one_minus_f1": p1 * factor,
                            source + "_missing_volume": presence * factor})
            rows.append(row)
    return rows


class PartitionCVTests(unittest.TestCase):
    def test_case_groups_strata_and_determinism(self):
        rows = synthetic_rows(20)
        folds = grouped_folds(rows, 5, 42)
        self.assertEqual(folds, grouped_folds(rows[::-1], 5, 42))
        for scan in ("pelvic", "femur"):
            cases = {r["case_id"] for r in rows if r["scan_type"] == scan}
            self.assertEqual([sum(folds[c] == f for c in cases) for f in range(5)], [2] * 5)
        cv = nested_cv(synthetic_rows(4), {"outer_folds": 2, "inner_folds": 2, "alphas": [.1, 1]})
        for case in folds:
            self.assertIsInstance(folds[case], int)
        for case in {r["case_id"] for r in cv["oof_scores"]}:
            self.assertEqual(len({r["outer_fold"] for r in cv["oof_scores"] if r["case_id"] == case}), 1)
        for record in cv["fold_models"]:
            self.assertTrue(set(record["test_case_ids"]).isdisjoint(record["model"]["train_case_ids"]))

    def test_ridge_matches_closed_form_and_json_predicts_without_labels(self):
        rows = synthetic_rows()
        alpha = 2.5
        model = fit_model(rows, FINAL_HEAD, alpha)
        features = HEAD_FEATURES[FINAL_HEAD]
        x = np.array([[r[f] for f in features] for r in rows])
        z = (x - x.mean(0)) / x.std(0)
        y = np.log1p([r["member_mean_merge_split"] for r in rows])
        design = np.column_stack((np.ones(len(rows)), z))
        penalty = np.diag([0., alpha, alpha, alpha])
        direct = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        np.testing.assert_allclose([model["intercept"], *model["coefficient"]], direct, atol=1e-12)
        features_only = [{f: r[f] for f in features} for r in rows]
        np.testing.assert_allclose(predict_model(json.loads(json.dumps(model)), features_only), design @ direct)
        self.assertEqual(predict_model(model, []).shape, (0,))
        bad = copy.deepcopy(model)
        bad["features"] = ["n_GT", "graph_degree", "presence"]
        with self.assertRaises(ValueError):
            predict_model(bad, rows)

    def test_heldout_features_and_labels_cannot_change_fold_fit_or_tuning(self):
        rows = synthetic_rows()
        config = {"outer_folds": 2, "inner_folds": 2, "alphas": [.1, 1., 10.], "seed": 27}
        initial = nested_cv(rows, config)
        test_cases = {r["case_id"] for r in initial["folds"] if r["outer_fold"] == 0}
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["case_id"] in test_cases:
                row["member_mean_merge_split"] += 1e6
                row["graph_degree"] += 1e5
                row["partition_v1"] = 1 - row["partition_v1"]
                row["presence"] = 1 - row["presence"]
                row["n_GT"] = 999999  # Forbidden fields cannot enter the model.
        other = nested_cv(changed, config)
        self.assertEqual([r for r in initial["fold_models"] if r["outer_fold"] == 0],
                         [r for r in other["fold_models"] if r["outer_fold"] == 0])
        self.assertEqual([r for r in initial["tuning"] if r["scope"] == "outer_0"],
                         [r for r in other["tuning"] if r["scope"] == "outer_0"])
        self.assertEqual(initial["folds"], other["folds"])
        # Changing only outer-test labels leaves those OOF predictions intact.
        for row in changed:
            original = next(r for r in rows if (r["case_id"], r["bone"]) == (row["case_id"], row["bone"]))
            for feature in ("partition_v1", "graph_degree", "presence"):
                row[feature] = original[feature]
        labels_only = nested_cv(changed, config)
        for old, new in zip(initial["oof_scores"], labels_only["oof_scores"]):
            if old["case_id"] in test_cases:
                for head in HEAD_FEATURES:
                    self.assertEqual(old[head], new[head])

    def test_constant_feature_target_and_log_label_range(self):
        rows = synthetic_rows(4)
        for row in rows:
            row.update(partition_v1=.2, graph_degree=0., presence=0., member_mean_merge_split=3.)
        model = fit_model(rows, FINAL_HEAD, 1.)
        np.testing.assert_allclose(model["coefficient"], 0)
        np.testing.assert_allclose(model["scale"], 1)
        np.testing.assert_allclose(predict_model(model, rows), np.log1p(3.))
        cv = nested_cv(rows, {"outer_folds": 2, "inner_folds": 2, "alphas": [.1, 1.]})
        self.assertEqual(cv["summary"]["final_head"], FINAL_HEAD)
        self.assertTrue(all(m["alpha"] == .1 for m in cv["models"].values()))
        self.assertEqual(cv, nested_cv(rows[::-1], {"outer_folds": 2, "inner_folds": 2, "alphas": [.1, 1.]}))
        rows[0]["member_mean_merge_split"] = -1
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_invalid_inputs_and_tiny_cohort_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "insufficient_cases"):
            nested_cv(synthetic_rows(2), {})
        rows = synthetic_rows()
        with self.assertRaises(ValueError):
            validate_rows([*rows, rows[0]])
        with self.assertRaises(ValueError):
            nested_cv(rows, {"alphas": [0.]})
        for name, value in (("partition_v1", 2.), ("presence", -.1), ("graph_degree", float("nan"))):
            broken = copy.deepcopy(rows)
            broken[0][name] = value
            with self.assertRaises(ValueError):
                fit_model(broken, FINAL_HEAD, 1.)


if __name__ == "__main__":
    unittest.main()
