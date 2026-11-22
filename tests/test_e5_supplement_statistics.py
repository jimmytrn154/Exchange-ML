from __future__ import annotations

from itertools import permutations
import unittest
from unittest.mock import patch

import numpy as np

from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.e1_statistics import expected_curve
from exchange_ml.e4_e5_analysis import review_budget
from exchange_ml.e5_supplement_statistics import budget_analysis, retained_risk_batch


def synthetic_rows() -> list[dict]:
    rows = []
    for case, scan, bones in (("001", "pelvic", ("sacrum", "left_hip", "right_hip")),
                              ("002", "pelvic", ("sacrum", "left_hip", "right_hip")),
                              ("003", "femur", ("femur",)), ("004", "femur", ("femur",))):
        for bone in bones:
            i = len(rows)
            y = i / 8
            r = {"case_id": case, "scan_type": scan, "bone": bone,
                 **{s: float(i % 3) for s in SCORE_NAMES}, "presence": float(7 - i),
                 "fa_v2": float(i // 2)}
            for source, factor in (("member_mean", 1.), ("stage2", .5)):
                r.update({source + "_merge_split": y * 2 * factor,
                          source + "_one_minus_f1": y * factor,
                          source + "_missing_volume": y * factor})
            rows.append(r)
    return rows


class SupplementStatisticsTests(unittest.TestCase):
    def test_batched_cutoffs_match_exhaustive_tie_orders(self):
        x = np.array([0, 0, 1, 1.])
        y = np.array([0, 2, 4, 6.])
        valid = [p for p in permutations(range(4)) if all(x[p[i]] <= x[p[i + 1]] for i in range(3))]
        observed = retained_risk_batch(x[None], y[None, :, None], (1, 2, 3, 4))[0, :, 0]
        expected = np.array([np.mean([y[list(p[:k])].mean() for p in valid]) for k in range(1, 5)])
        np.testing.assert_allclose(observed, expected)
        np.testing.assert_allclose(observed, expected_curve(x, y))
        self.assertTrue(np.isnan(retained_risk_batch(x[None], y[None, :, None], (0,))[0, 0, 0]))

    def test_bootstrap_preserves_case_multiplicity(self):
        rows = []
        for case, value in (("001", 0.), ("002", .5), ("003", 1.)):
            for bone in ("sacrum", "left_hip", "right_hip"):
                rows.append({"case_id": case, "scan_type": "pelvic", "bone": bone,
                             **{s: 0. for s in SCORE_NAMES},
                             **{f"{source}_{risk}": value for source in ("member_mean", "stage2")
                                for risk in ("merge_split", "one_minus_f1", "missing_volume")}})
        draws = np.array([[0, 1, 2, 0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8, 6, 7, 8]])
        with patch("exchange_ml.e5_supplement_statistics.cluster_draws", return_value=draws) as mocked:
            points, _, _ = budget_analysis(rows, resamples=2, seed=1)
            self.assertEqual(mocked.call_count, 2)  # one shared draw matrix per nonempty stratum
        row = next(r for r in points if r["stratum"] == "pooled")
        low, high = np.quantile([1 / 6, 5 / 6], [.025, .975])
        self.assertAlmostEqual(row["ci_low"], low)
        self.assertAlmostEqual(row["ci_high"], high)
        self.assertAlmostEqual(row["retained_mean_risk"], .5)

    def test_seed_and_batch_size_do_not_change_results(self):
        rows = synthetic_rows()
        a = budget_analysis(rows, resamples=32, seed=42, batch_size=1)
        b = budget_analysis(rows, resamples=32, seed=42, batch_size=17)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 468)
        self.assertEqual(len(a[1]), 864)
        self.assertEqual(len(a[2]), 1248)  # 13 policies x six endpoints x 16 stratum-units

    def test_pairing_self_contrasts_and_scaled_endpoints(self):
        points, paired, _ = budget_analysis(synthetic_rows(), resamples=32, seed=19)
        self_contrasts = [r for r in paired if r["score"] == r["baseline"]]
        self.assertTrue(self_contrasts)
        for r in self_contrasts:
            self.assertEqual(r["delta_retained_risk"], 0.)
            self.assertEqual(r["ci_low"], 0.)
            self.assertEqual(r["ci_high"], 0.)
        lookup = {(r["stratum"], r["score"], r["requested_coverage"], r["risk_source"], r["risk"]): r for r in points}
        for r in points:
            if r["risk_source"] == "member_mean":
                other = lookup[r["stratum"], r["score"], r["requested_coverage"], "stage2", r["risk"]]
                for field in ("retained_mean_risk", "ci_low", "ci_high"):
                    self.assertAlmostEqual(other[field], .5 * r[field])

    def test_original_merge_split_budget_and_CT_cost_are_preserved(self):
        rows = synthetic_rows()
        original = review_budget(rows, {"retained_fractions": [.8, .9]})
        points, _, curves = budget_analysis(rows, resamples=8, seed=1)
        lookup = {(r["stratum"], r["score"], r["requested_coverage"], r["risk_source"]): r
                  for r in points if r["risk"] == "merge_split"}
        for old in original:
            new = lookup[old["stratum"], old["score"], old["requested_coverage"], old["risk_source"]]
            for field in ("retained_mean_risk", "expected_reviewed_cases", "retained_units", "reviewed_units",
                          "coverage", "delta_vs_random", "delta_vs_presence"):
                self.assertAlmostEqual(old[field], new[field])
        for row in points:
            if row["score"] == "random":
                self.assertAlmostEqual(row["retained_mean_risk"], row["random_mean_risk"])
        self.assertTrue(all(0 <= r["prefix_mean_risk"] <= 1 for r in curves if r["risk"] != "merge_split"))

    def test_single_case_and_zero_retained_units_have_no_claimed_CI(self):
        rows = [r for r in synthetic_rows() if r["case_id"] == "003"]
        points, contrasts, _ = budget_analysis(rows, resamples=32, seed=2)
        self.assertTrue(all(r["retained_mean_risk"] is None and r["ci_low"] is None for r in points))
        self.assertTrue(all(r["ci_status"] == "zero_retained_units" for r in points))
        self.assertTrue(all(r["delta_retained_risk"] is None for r in contrasts))
        rows = [r for r in synthetic_rows() if r["case_id"] == "001"]
        points, _, _ = budget_analysis(rows, resamples=32, seed=2)
        self.assertTrue(all(r["retained_mean_risk"] is not None and r["ci_low"] is None for r in points))
        self.assertTrue(all(r["ci_status"] == "insufficient_cases" and r["bootstrap_resamples"] == 0 for r in points))

    def test_invalid_inputs_are_rejected_without_silent_clipping(self):
        rows = synthetic_rows()
        rows[0]["member_mean_one_minus_f1"] = 1.1
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            budget_analysis(rows, resamples=32, seed=2)
        rows = synthetic_rows()
        rows[0]["stage2_missing_volume"] = None
        with self.assertRaisesRegex(ValueError, "silently dropped"):
            budget_analysis(rows, resamples=32, seed=2)
        rows = synthetic_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            budget_analysis(rows + rows[:1], resamples=32, seed=2)
        with self.assertRaises(ValueError):
            retained_risk_batch(np.zeros((1, 2)), np.zeros((1, 2, 1)), (3,))


if __name__ == "__main__":
    unittest.main()
