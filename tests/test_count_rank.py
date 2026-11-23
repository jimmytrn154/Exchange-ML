from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from exchange_ml.count_rank import (FEATURES, FINAL_HEAD, TARGET, fit_model, nested_cv,
                                         pairwise_differences, pairwise_loss_gradient,
                                         predict_model, validate_rows)


def synthetic_rows(n_cases: int = 12) -> list[dict]:
    rng = np.random.default_rng(71)
    rows = []
    for i in range(n_cases):
        scan = "pelvic" if i < n_cases // 2 else "femur"
        bones = ("sacrum", "left_hip", "right_hip") if scan == "pelvic" else ("femur",)
        for bone in bones:
            p1, graph = float(rng.uniform(0, .1)), float(rng.uniform(0, 2))
            rows.append({"case_id": f"synthetic_{i:03}", "scan_type": scan, "bone": bone,
                         "partition_v1": p1, "predicted_count_mean": graph,
                         TARGET: float(3 * p1 + .4 * graph)})
    return rows


class CountRankTests(unittest.TestCase):
    def test_pair_construction_ties_and_hand_loss(self):
        x = np.array([[0., 1.], [1., 0.], [3., 2.]])
        d = pairwise_differences(x, np.array([0., 0., 2.]))
        np.testing.assert_array_equal(d, [[3., 1.], [2., 2.]])
        loss, gradient = pairwise_loss_gradient(np.zeros(2), d, .1)
        self.assertAlmostEqual(loss, np.log(2))
        np.testing.assert_allclose(gradient, -.5 * d.mean(0))
        self.assertEqual(pairwise_differences(x, np.zeros(3)).shape, (0, 2))
        # High raw M+S is oriented toward a larger ranking score.
        self.assertLess(pairwise_loss_gradient(np.ones(2), d, .001)[0], loss)

    def test_gradient_and_extreme_margins_are_stable(self):
        d = np.array([[2., -3.], [.5, 4.], [0., 1.]])
        beta = np.array([.3, -.2])
        _, analytic = pairwise_loss_gradient(beta, d, .01)
        delta = np.eye(2) * 1e-6
        numeric = np.array([(pairwise_loss_gradient(beta + change, d, .01)[0]
                             - pairwise_loss_gradient(beta - change, d, .01)[0]) / 2e-6
                            for change in delta])
        np.testing.assert_allclose(analytic, numeric, atol=1e-9)
        for extreme in (np.array([1e5, -1e5]), np.array([-1e5, 1e5])):
            loss, gradient = pairwise_loss_gradient(extreme, d, .01)
            self.assertTrue(np.isfinite(loss) and np.isfinite(gradient).all())

    def test_optimizer_stationarity_serialization_and_monotonic_labels(self):
        rows = synthetic_rows()
        model = fit_model(rows, .01)
        self.assertEqual(model["optimizer"]["status"], "converged")
        self.assertLessEqual(model["optimizer"]["gradient_infinity_norm"], 1e-7)
        self.assertEqual(model["intercept"], 0.)
        self.assertEqual(model["features"], list(FEATURES))
        features_only = [{name: row[name] for name in FEATURES} for row in rows]
        np.testing.assert_array_equal(predict_model(json.loads(json.dumps(model)), features_only),
                                      predict_model(model, rows))
        self.assertEqual(predict_model(model, []).shape, (0,))
        changed = [dict(row, **{TARGET: 100 * row[TARGET] + 2}) for row in rows]
        self.assertEqual(fit_model(changed, .01)["coefficient"], model["coefficient"])
        self.assertEqual(fit_model(rows[::-1], .01), model)
        bad = copy.deepcopy(model); bad["features"] = ["n_GT", "predicted_count_mean"]
        with self.assertRaises(ValueError):
            predict_model(bad, rows)
        bad = copy.deepcopy(model); bad["intercept"] = 1.
        with self.assertRaises(ValueError):
            predict_model(bad, rows)

    def test_no_pairs_constant_features_and_exact_alpha_ties(self):
        rows = synthetic_rows(8)
        for row in rows:
            row[TARGET] = 3.
        model = fit_model(rows, .1)
        self.assertEqual(model["optimizer"]["status"], "no_risk_pairs")
        self.assertEqual(model["n_training_pairs"], 0)
        np.testing.assert_array_equal(predict_model(model, rows), 0.)
        cv = nested_cv(rows, {"outer_folds": 2, "inner_folds": 2, "alphas": [.01, .1]})
        self.assertEqual(cv["final_model"]["alpha"], .1)
        self.assertTrue(all(r["model"]["alpha"] == .1 for r in cv["fold_models"]))
        for row in rows:
            row.update(partition_v1=.1, predicted_count_mean=0.)
            row[TARGET] = float(int(row["case_id"][-3:]))
        model = fit_model(rows, .1)
        self.assertEqual(model["optimizer"]["status"], "converged")
        np.testing.assert_array_equal(model["coefficient"], 0.)
        np.testing.assert_array_equal(model["scale"], 1.)

    def test_nested_grouping_train_only_scaling_and_heldout_sentinels(self):
        rows = synthetic_rows(12)
        config = {"outer_folds": 3, "inner_folds": 2, "alphas": [.01, .1], "seed": 73}
        original = nested_cv(rows, config)
        test_cases = {r["case_id"] for r in original["folds"] if r["outer_fold"] == 0}
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["case_id"] in test_cases:
                row[TARGET] += 1e6
                row["partition_v1"] = 1 - row["partition_v1"]
                row["predicted_count_mean"] += 1e5
                row["presence"] = float("nan")  # Never a fitting feature.
        other = nested_cv(changed, config)
        self.assertEqual([r for r in original["fold_models"] if r["outer_fold"] == 0],
                         [r for r in other["fold_models"] if r["outer_fold"] == 0])
        self.assertEqual([r for r in original["tuning"] if r["scope"] == "outer_0"],
                         [r for r in other["tuning"] if r["scope"] == "outer_0"])
        for row in changed:
            if row["case_id"] in test_cases:
                reference = next(r for r in rows if (r["case_id"], r["bone"]) == (row["case_id"], row["bone"]))
                for feature in FEATURES:
                    row[feature] = reference[feature]
        labels_only = nested_cv(changed, config)
        for before, after in zip(original["oof_scores"], labels_only["oof_scores"]):
            if before["case_id"] in test_cases:
                self.assertEqual(before[FINAL_HEAD], after[FINAL_HEAD])
        for record in original["fold_models"]:
            self.assertFalse(set(record["test_case_ids"]) & set(record["model"]["train_case_ids"]))
            training = [r for r in rows if r["case_id"] in record["model"]["train_case_ids"]]
            np.testing.assert_allclose(record["model"]["mean"], np.mean([[r[f] for f in FEATURES] for r in training], 0))
        self.assertEqual(original, nested_cv(rows[::-1], config))
        for record in original["tuning"]:
            self.assertAlmostEqual(record["mean_inner_fold_aurc"], np.mean([r["aurc"] for r in record["inner_fold_results"]]))
            self.assertEqual(record["selection_target"], "unweighted_mean_inner_fold_member_mean_merge_split_AURC")

    def test_rejects_bad_anatomy_features_and_insufficient_cv(self):
        rows = synthetic_rows()
        for name, value in (("partition_v1", 1.1), ("predicted_count_mean", -.1), (TARGET, float("nan"))):
            bad = copy.deepcopy(rows); bad[0][name] = value
            with self.assertRaises(ValueError):
                fit_model(bad, .1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows([*rows, rows[0]])
        with self.assertRaisesRegex(ValueError, "complete expected anatomy"):
            validate_rows(rows[1:])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            nested_cv(synthetic_rows(2), {})
        with self.assertRaisesRegex(ValueError, "insufficient inner-fold"):
            nested_cv(synthetic_rows(4), {"outer_folds": 2, "inner_folds": 2})
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_model(rows, 0.)
        with self.assertRaisesRegex(RuntimeError, "failed convergence"):
            fit_model(rows, .001, {"max_iter": 1})


class CountNumericalAndLeakageTests(unittest.TestCase):
    def test_abnormal_solver_status_requires_independent_stationarity(self):
        from scipy.optimize import OptimizeResult
        from unittest.mock import patch
        import exchange_ml.count_rank as rank
        rows = synthetic_rows()
        good = rank.fit_model(rows, .1)
        result = OptimizeResult(success=False, x=np.array(good['coefficient']),
                                jac=np.zeros(2), nit=1, message='ABNORMAL: near stationary')
        with patch.object(rank, 'minimize', return_value=result):
            verified = rank.fit_model(rows, .1)
        self.assertFalse(verified['optimizer']['native_solver_success'])
        self.assertEqual(verified['optimizer']['status'], 'stationary_verified_despite_solver_status')
        self.assertLessEqual(verified['optimizer']['gradient_infinity_norm'], 1e-7)
        # A false/stale solver jacobian cannot hide a nonstationary coefficient.
        result.x = np.array([100., -100.])
        with patch.object(rank, 'minimize', return_value=result), self.assertRaisesRegex(RuntimeError, 'failed convergence'):
            rank.fit_model(rows, .1)

    def test_gt_complexity_and_old_control_never_enter_predictor(self):
        rows = synthetic_rows()
        before = fit_model(rows, .1)
        changed = [dict(row, n_GT=10**8, pairwise_p1_g=float('nan'), graph_degree=float('nan')) for row in rows]
        self.assertEqual(before, fit_model(changed, .1))


if __name__ == "__main__":
    unittest.main()
