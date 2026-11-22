from __future__ import annotations

from itertools import combinations
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from exchange_ml.e1_statistics import aurc_batch, cluster_draws
from exchange_ml.partition_cv import HEAD_FEATURES, nested_cv
from exchange_ml.partition_reporting import (ALL_SCORES, CI_ROLE, _ct_tables, report,
                                            retained_ct_expectation)
from test_partition_cv import synthetic_rows


class PartitionReportingTests(unittest.TestCase):
    @staticmethod
    def omission_rows() -> list[dict]:
        rows = [r for r in synthetic_rows(8) if r["scan_type"] == "femur"]
        for index, row in enumerate(rows):
            row.update({head: row["partition_v1"] for head in HEAD_FEATURES})
            row["graph_degree"] = [0., 1., 3., 2.][index]
            row["member_mean_missing_volume"] = float(index in (0, 2))
            row["stage2_missing_volume"] = 0.
        return rows

    def test_ct_ties_match_exhaustive_case_and_bone_macro(self):
        scores = np.array([0., 0., 0., 0.])
        counts = np.array([3, 3, 1, 1])
        sums = np.array([[0., 3.], [3., 6.], [5., 0.], [7., 2.]])
        for retained in (1, 2, 3, 4):
            options = [list(c) for c in combinations(range(4), retained)]
            expected_case = np.mean([(sums[c] / counts[c, None]).mean(0) for c in options], axis=0)
            expected_bone = np.mean([sums[c].sum(0) / counts[c].sum() for c in options], axis=0)
            observed_case, observed_bone = retained_ct_expectation(scores, sums, counts, retained)
            np.testing.assert_allclose(observed_case, expected_case)
            np.testing.assert_allclose(observed_bone, expected_bone)
        # A ratio of expected sums is wrong when a CT tie spans 3/1-bone cases.
        _, bone = retained_ct_expectation(scores, sums, counts, 2)
        self.assertFalse(np.allclose(bone, sums.sum(0) / counts.sum()))
        # A non-tied prefix is always included, only the cutoff tie is sampled.
        scores[0] = -1
        options = [[0, *c] for c in combinations((1, 2, 3), 2)]
        case, bone = retained_ct_expectation(scores, sums, counts, 3)
        np.testing.assert_allclose(case, np.mean([(sums[c] / counts[c, None]).mean(0) for c in options], 0))
        np.testing.assert_allclose(bone, np.mean([sums[c].sum(0) / counts[c].sum() for c in options], 0))

    def test_guard_excludes_cts_and_reports_infeasible_budget(self):
        x, sums, sizes = np.arange(4.), np.arange(4.)[:, None], np.ones(4, int)
        case, bone = retained_ct_expectation(x, sums, sizes, 3, np.array([False, True, True, True]))
        np.testing.assert_allclose(case, [2.])
        np.testing.assert_allclose(bone, [2.])
        self.assertTrue(np.isnan(retained_ct_expectation(x, sums, sizes, 3, np.array([False, False, True, True]))[0]).all())
        self.assertTrue(np.isnan(retained_ct_expectation(x, sums, sizes, 0)[0]).all())
        rows = synthetic_rows(4)
        for row in rows:
            for head in HEAD_FEATURES:
                row[head] = row["partition_v1"]
            row["all_members_empty"] = row["case_id"] in ("case_000", "case_001")
        result = report(rows, {"bootstrap_resamples": 4, "ct_review_fractions": [.1]})
        guarded = [r for r in result["ct_review_budget"] if r["stratum"] == "pooled" and r["policy"].startswith("guard_")]
        self.assertTrue(guarded)
        self.assertTrue(all(r["status"] == "guard_budget_infeasible" and r["retained_mean_risk"] is None for r in guarded))
        for row in result["ct_review_budget"]:
            if row["stratum"] == "pooled":
                self.assertEqual(row["reviewed_cts"], 1)
        json.dumps(result, allow_nan=False)

    def test_oof_report_matches_existing_aurc_and_paired_case_bootstrap(self):
        rows = nested_cv(synthetic_rows(), {"outer_folds": 2, "inner_folds": 2, "alphas": [.1, 1.]})["oof_scores"]
        config = {"bootstrap_resamples": 12, "bootstrap_seed": 13, "ct_review_fractions": [.1],
                  "include_omission_guard": False}
        result = report(rows, config)
        self.assertEqual(len(result["summary"]["scores"]), 16)
        self.assertEqual(result["summary"]["ci_role"], CI_ROLE)
        self.assertFalse(result["summary"]["human_correction_assumed"])
        y = np.array([r["member_mean_merge_split"] for r in rows])
        p1 = np.array([r["partition_v1"] for r in rows])
        head = np.array([r["ridge_p1_g_a"] for r in rows])
        selected = next(r for r in result["comparison"] if r["stratum"] == "pooled" and r["score"] == "partition_v1" and r["risk_source"] == "member_mean" and r["risk"] == "merge_split")
        self.assertAlmostEqual(selected["aurc"], aurc_batch(p1[None], y[None, :, None])[0, 0])
        draws = cluster_draws(rows, 12, 13)
        diffs = aurc_batch(head[draws], y[draws][..., None])[..., 0] - aurc_batch(p1[draws], y[draws][..., None])[..., 0]
        contrast = next(r for r in result["paired_differences"] if r["contrast_role"] == "extension_main")
        np.testing.assert_allclose([contrast["ci_low"], contrast["ci_high"]], np.quantile(diffs, [.025, .975]))
        self.assertEqual(result, report(rows, config))
        self.assertTrue(all(r["reviewed_cts"] == 1 for r in result["ct_review_budget"]))
        self.assertEqual({r["score"] for r in result["comparison"]}, set(ALL_SCORES))
        json.dumps(result, allow_nan=False)

    def test_constant_risk_and_single_case_are_uninformative(self):
        rows = synthetic_rows(4)
        for row in rows:
            for name in list(row):
                if name.startswith(("member_mean_", "stage2_")):
                    row[name] = 0.
            row.update({head: 0. for head in HEAD_FEATURES})
        result = report(rows, {"bootstrap_resamples": 2, "ct_review_fractions": [.1], "include_omission_guard": False})
        self.assertTrue(all(r["status"] == "constant_risk_uninformative" and r["rho"] is None for r in result["comparison"]))
        self.assertTrue(all(r["status"] == "no_positive_cts" and r["complete_omission_sensitivity"] is None
                            for r in result["ct_omission_sensitivity"]))
        single = report([rows[-1]], {"bootstrap_resamples": 2, "ct_review_fractions": [.1]})
        self.assertTrue(all(r["status"] == "insufficient_cases" and r["ci_low"] is None for r in single["comparison"]))
        self.assertTrue(all(r["status"] == "zero_retained_cases" for r in single["ct_review_budget"]))
        json.dumps(single, allow_nan=False)
        broken = copy.deepcopy(rows)
        broken[0]["stage2_missing_volume"] = 2.
        with self.assertRaises(ValueError):
            report(broken, {})

    def test_omission_ties_any_bone_label_and_guard_at_same_ct_cost(self):
        rows = synthetic_rows(4)
        for row in rows:
            row.update({name: .2 for name in ALL_SCORES})
            positive = row["case_id"] == "case_000" and row["bone"] == "sacrum"
            row["member_mean_missing_volume"] = float(positive)
            row["stage2_missing_volume"] = 0.
            row["all_members_empty"] = positive
        result = report(rows, {"bootstrap_resamples": 4, "ct_review_fractions": [.25]})
        points = [r for r in result["ct_omission_sensitivity"] if r["stratum"] == "pooled"]
        for policy, expected in (("random", .25), ("partition_v1", .25), ("guard_only", 1.),
                                 ("guard_ridge_p1_g_a", 1.)):
            point = next(r for r in points if r["policy"] == policy and r["risk_source"] == "member_mean")
            self.assertAlmostEqual(point["complete_omission_sensitivity"], expected)
            self.assertAlmostEqual(point["expected_reviewed_positive_cts"], expected)
            self.assertEqual(point["positive_cts"], 1)  # CT counted once despite its three bones.
            self.assertEqual(point["reviewed_cts"], 1)
        absent = [r for r in points if r["risk_source"] == "stage2"]
        self.assertTrue(all(r["status"] == "no_positive_cts" and r["ci_low"] is None for r in absent))
        contrast = next(r for r in result["ct_omission_differences"]
                        if r["stratum"] == "pooled" and r["policy"] == "guard_only"
                        and r["baseline"] == "random" and r["risk_source"] == "member_mean")
        self.assertAlmostEqual(contrast["delta_complete_omission_sensitivity"], .75)
        json.dumps(result, allow_nan=False)

    def test_omission_paired_bootstrap_preserves_duplicates_and_empty_draws(self):
        rows = self.omission_rows()
        draws = np.array([[0, 0, 2, 3], [0, 1, 1, 2]])
        with patch("exchange_ml.partition_reporting.cluster_draws", return_value=draws):
            _, _, points, contrasts = _ct_tables(rows, 2, 42, (.25,), False)
        point = next(r for r in points if r["stratum"] == "pooled" and r["policy"] == "graph_degree"
                     and r["risk_source"] == "member_mean")
        self.assertAlmostEqual(point["complete_omission_sensitivity"], .5)
        # Each draw reviews one positive; denominators include duplicated CTs.
        np.testing.assert_allclose([point["ci_low"], point["ci_high"]], np.quantile([1 / 3, 1 / 2], [.025, .975]))
        contrast = next(r for r in contrasts if r["stratum"] == "pooled" and r["policy"] == "graph_degree"
                        and r["baseline"] == "random" and r["risk_source"] == "member_mean")
        np.testing.assert_allclose([contrast["ci_low"], contrast["ci_high"]],
                                   np.quantile([1 / 3 - .25, 1 / 2 - .25], [.025, .975]))
        # An undefined zero-positive draw cannot be dropped to create a CI.
        draws[1] = [1, 1, 3, 3]
        with patch("exchange_ml.partition_reporting.cluster_draws", return_value=draws):
            _, _, points, contrasts = _ct_tables(rows, 2, 42, (.25,), False)
        point = next(r for r in points if r["stratum"] == "pooled" and r["policy"] == "graph_degree"
                     and r["risk_source"] == "member_mean")
        self.assertEqual(point["bootstrap_defined_fraction"], .5)
        self.assertEqual(point["ci_status"], "no_positive_or_infeasible_bootstrap_draws")
        self.assertIsNone(point["ci_low"])
        contrast = next(r for r in contrasts if r["stratum"] == "pooled" and r["policy"] == "graph_degree"
                        and r["baseline"] == "random" and r["risk_source"] == "member_mean")
        self.assertIsNone(contrast["ci_low"])

    def test_observed_infeasible_guard_has_no_ci_even_if_all_bootstraps_feasible(self):
        rows = self.omission_rows()
        for index, row in enumerate(rows):
            row["all_members_empty"] = index < 2
            row["member_mean_missing_volume"] = 1.
        # Observed: two flags exceed one reviewed CT. Resample: zero flags.
        with patch("exchange_ml.partition_reporting.cluster_draws", return_value=np.array([[2, 2, 3, 3]])):
            points, contrasts, omissions, omission_deltas = _ct_tables(rows, 1, 42, (.25,), True)
        for table in (points, contrasts, omissions, omission_deltas):
            selected = [r for r in table if r["policy"].startswith("guard_")]
            self.assertTrue(selected)
            self.assertTrue(all(r["ci_low"] is None and r["ci_high"] is None for r in selected))
        self.assertTrue(all(r["status"] == "guard_budget_infeasible" for r in omissions
                            if r["policy"].startswith("guard_")))
        # At an all-reviewed budget sensitivity is one, even though retained risk is undefined.
        _, _, omissions, _ = _ct_tables(rows[:1], 1, 42, (.25,), True)
        self.assertTrue(all(r["complete_omission_sensitivity"] == 1. for r in omissions
                            if r["risk_source"] == "member_mean"))


if __name__ == "__main__":
    unittest.main()
