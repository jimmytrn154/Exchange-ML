from __future__ import annotations

from itertools import permutations
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from exchange_ml.e1_scoring import pair_components as e1_pair_components, score_unit, SCORE_NAMES
from exchange_ml.e1_statistics import aurc_batch, expected_curve
from exchange_ml.instances import _anatomy_match
from exchange_ml.e6_evaluation import (
    RISKS, STRUCTURAL_RISKS, evaluate_predictions, model_risk, pair_components,
    report_tables, score_case,
)


def volume(values):
    return np.asarray(values, dtype=np.uint32).reshape(1, 1, -1)


class E6ScoreTests(unittest.TestCase):
    def test_generic_id_permutation_including_more_than_fifty_instances(self):
        first = np.arange(1, 121, dtype=np.uint32)
        second = 100000 + np.arange(120, 0, -1, dtype=np.uint32)
        row = pair_components(first, second)
        self.assertEqual(row["presence"], 0)
        self.assertAlmostEqual(row["fa_v2"], 0)
        self.assertAlmostEqual(row["hungarian"], 0)
        self.assertEqual(row["k_i"], 120)
        self.assertEqual(row["k_j"], 120)

    def test_frozen_e1_components_match_single_anatomy_contract(self):
        rng = np.random.default_rng(3)
        for _ in range(12):
            first, second = rng.integers(0, 6, (2, 100), dtype=np.uint32)
            observed, frozen = pair_components(first, second), e1_pair_components(first, second)
            self.assertEqual(observed.keys(), frozen.keys())
            for key in frozen:
                self.assertAlmostEqual(observed[key], frozen[key], msg=key)

    def test_sparse_hungarian_matches_dense_reference_and_no_overlap(self):
        rng = np.random.default_rng(21)
        for _ in range(20):
            first = rng.integers(0, 9, 40)
            second = rng.integers(0, 5, 40)
            lids, rids = np.unique(first[first > 0]), np.unique(second[second > 0])
            iou = np.array([[np.count_nonzero((first == i) & (second == j)) /
                             np.count_nonzero((first == i) | (second == j))
                             for j in rids] for i in lids])
            rows, cols = linear_sum_assignment(-iou)
            expected = 1 - iou[rows, cols].sum() / max(len(lids), len(rids))
            self.assertAlmostEqual(pair_components(first, second)["hungarian"], expected)
        self.assertEqual(pair_components(np.array([1, 0]), np.array([0, 300]))["hungarian"], 1)

    def test_sparse_many_instances_do_not_require_dense_joint_matrix(self):
        # A dense joint would require 400 million entries; sparse has 20k edges.
        first = np.arange(1, 20001, dtype=np.uint32)
        row = pair_components(first, first + 30000)
        self.assertAlmostEqual(row["hungarian"], 0)
        self.assertAlmostEqual(row["fa_v2"], 0)

    def test_exact_merge_and_empty_conventions(self):
        separate, merged = np.repeat([1, 500], 32), np.ones(64, dtype=np.uint32)
        for first, second in ((separate, merged), (merged, separate)):
            pair = pair_components(first, second)
            self.assertAlmostEqual(pair["q_log_n"], 1 / 6)
            self.assertAlmostEqual(pair["q_joint"], 1)
            self.assertEqual(pair["presence"], 0)
            self.assertAlmostEqual(pair["hungarian"], .75)
        empty = np.zeros((1, 2, 4), dtype=np.uint32)
        full = np.ones_like(empty)
        self.assertEqual(pair_components(empty, full)["fa_v2"], 1)
        scores, pairs = score_case([empty] * 3, [np.full(empty.shape, .5)] * 3)
        self.assertTrue(all(value == 0 for value in scores.values()))
        self.assertEqual(len(pairs), 3)

    def test_probabilities_and_scores_are_gt_free(self):
        maps = [volume([0, 1, 1]), volume([0, 20, 21]), volume([0, 5000, 5000])]
        probabilities = [np.zeros((1, 1, 3)), np.ones((1, 1, 3)), np.zeros((1, 1, 3))]
        scores, _ = score_case(maps, probabilities)
        self.assertEqual(set(scores), set(SCORE_NAMES))
        self.assertAlmostEqual(scores["count_variance"], np.var([1, 2, 1]))
        p = 1 / 3
        self.assertAlmostEqual(scores["mean_entropy"], -(p * np.log(p) + (1 - p) * np.log1p(-p)))
        # GT cannot enter this API; whole-anatomy all-members-empty failures
        # retain zero disagreement by design and are caught only by evaluation.
        for invalid in (np.full((1, 1, 3), np.nan), np.full((1, 1, 3), 1.1),
                        np.ones((1, 1, 3), dtype=int), np.ones((1, 2, 3))):
            with self.assertRaises(ValueError):
                score_case(maps, [invalid] * 3)
        with self.assertRaises(ValueError):
            score_case(maps[:2], probabilities[:2])
        for labels in (np.array([[-1]]), np.ones((1, 1), float), np.ones((1, 1), bool)):
            with self.assertRaises(ValueError):
                pair_components(labels, labels)

    def test_all_twelve_scores_equal_frozen_port_without_anatomy_constraint(self):
        rng = np.random.default_rng(55)
        for _ in range(8):
            maps = [rng.integers(0, 9, (3, 4, 5), dtype=np.uint32) for _ in range(3)]
            probabilities = [rng.random((3, 4, 5), dtype=np.float32) for _ in range(3)]
            observed, _ = score_case(maps, probabilities)
            frozen, _ = score_unit(maps, probabilities)
            for score in SCORE_NAMES:
                self.assertAlmostEqual(observed[score], frozen[score], msg=score)
        with self.assertRaises(ValueError):
            score_case([np.ones(3, dtype=int)] * 3, [np.ones(3)] * 3)


class E6RiskTests(unittest.TestCase):
    def test_e1_matching_preserved_without_id_range_assumptions(self):
        rng = np.random.default_rng(19)
        for _ in range(20):
            pred = rng.integers(0, 5, (3, 4, 5), dtype=np.uint32)
            gt = rng.integers(0, 7, (3, 4, 5), dtype=np.uint32)
            expected = _anatomy_match(gt, pred, .1)
            actual = model_risk(pred * 1000, gt * 900, (1, 1, 1))
            for key in ("gt_instance_count", "pred_instance_count", "instance_precision",
                        "instance_recall", "instance_f1", "merge_errors", "split_errors"):
                self.assertAlmostEqual(actual[key], expected[key], msg=key)

    def test_merge_split_and_volume_blindspots_remain_separate(self):
        gt = volume([1, 1, 1, 1, 90, 90, 90, 90])
        merged = volume([400] * 8)
        row = model_risk(merged, gt, (2, 3, 4))
        self.assertEqual(row["merge_errors"], 1)
        self.assertEqual(row["merge_split"], .5)
        self.assertEqual(row["instance_f1"], 1)
        self.assertEqual(row["missing_volume"], 0)
        self.assertEqual(row["gt_volume_mm3"], 8 * 24)
        row = model_risk(volume([1, 1, 2, 2, 90, 90, 90, 90]), gt, (1, 1, 1))
        self.assertEqual(row["split_errors"], 1)
        self.assertEqual(row["merge_errors"], 0)
        partial = volume([1, 1, 1, 0, 90, 90, 90, 0])
        row = model_risk(partial, gt, (1, 1, 1))
        self.assertEqual(row["instance_f1"], 1)
        self.assertEqual(row["missing_volume"], .25)

    def test_negative_cases_have_na_structural_endpoints_and_false_positives(self):
        empty = volume([0, 0, 0, 0])
        for pred, expected_f1, expected_dice, expected_fp in ((empty, 1, 1, 0),
                (volume([9, 0, 600, 600]), 0, 0, 2)):
            row = model_risk(pred, empty, (1, 1, 1))
            for risk in STRUCTURAL_RISKS:
                self.assertIsNone(row[risk])
            self.assertEqual(row["instance_f1"], expected_f1)
            self.assertEqual(row["binary_dice"], expected_dice)
            self.assertEqual(row["false_positive_instances"], expected_fp)
        row = model_risk(empty, volume([1, 1, 0, 0]), (1, 1, 1))
        self.assertEqual(row["merge_split"], 0)
        self.assertEqual(row["missing"], 1)
        self.assertEqual(row["missed_gt_instances"], 1)
        self.assertEqual(row["missing_volume"], 1)

    def test_ignore_mask_only_affects_evaluation_and_preserves_ids(self):
        gt = volume([500, 500, 0, 900])
        pred = volume([6, 6, 7, 0])
        ignored = np.array([False, False, True, True]).reshape(gt.shape)
        row = model_risk(pred, gt, (1, 1, 1), ignore_mask=ignored)
        self.assertEqual(row["n_GT"], 1)
        self.assertEqual(row["n_pred"], 1)
        self.assertEqual(row["instance_f1"], 1)
        self.assertEqual(row["n_ignored_voxels"], 2)
        for mask in (ignored.astype(int), np.ones(gt.shape, bool), np.zeros((2, 2), bool)):
            with self.assertRaises(ValueError):
                model_risk(pred, gt, (1, 1, 1), ignore_mask=mask)
        with self.assertRaises(ValueError):
            model_risk(pred, gt, (1, 0, 1))

    def test_aggregate_mean_is_per_model_not_consensus_prediction(self):
        gt = volume([1, 1, 0, 20, 20])
        maps = [gt, volume([1, 1, 0, 1, 1]), np.zeros_like(gt)]
        rows = evaluate_predictions("RibFrac421", maps, gt, (1, 1, 1))
        self.assertEqual(len(rows["model_risks"]), 3)
        self.assertAlmostEqual(rows["risks"][0]["member_mean_merge_split"], 1 / 6)
        self.assertAlmostEqual(rows["risks"][0]["member_mean_missing"], 1 / 3)
        self.assertEqual(len(rows["matches"]), 6)


class E6StatisticsTests(unittest.TestCase):
    @staticmethod
    def tables(n=5):
        scores, risks = [], []
        for i in range(n):
            scores.append({"case_id": f"case{i}", **{s: .1 * ((i + j) % n) for j, s in enumerate(SCORE_NAMES)}})
            scores[-1]["mean_entropy"] = min(scores[-1]["mean_entropy"], .6)
            scores[-1]["p95_entropy"] = min(scores[-1]["p95_entropy"], .6)
            risks.append({"case_id": f"case{i}", "n_GT": 1,
                          **{"member_mean_" + r: i / n for r in RISKS}})
        return scores, risks

    def test_paired_bootstrap_is_deterministic_and_matches_explicit_draws(self):
        scores, risks = self.tables()
        report = report_tables(scores, risks, bootstrap_resamples=32, bootstrap_seed=71)
        self.assertEqual(report, report_tables(scores, risks, bootstrap_resamples=32, bootstrap_seed=71))
        first_risk = RISKS[0]
        y = np.array([r["member_mean_" + first_risk] for r in risks])
        draws = np.random.default_rng(71).integers(5, size=(32, 5))
        f = np.array([r["fa_v2"] for r in scores])
        p = np.array([r["presence"] for r in scores])
        expected = aurc_batch(f[draws], y[draws, None])[:, 0] - aurc_batch(p[draws], y[draws, None])[:, 0]
        row = next(r for r in report["paired_differences"] if r["risk"] == first_risk and r["baseline"] == "presence")
        np.testing.assert_allclose([row["ci_low"], row["ci_high"]], np.quantile(expected, [.025, .975]))
        self.assertEqual(len(report["comparison"]), len(RISKS) * 12)
        self.assertEqual(len(report["paired_differences"]), len(RISKS) * 11)
        self.assertNotIn("meets_rule", report["summary"])

    def test_tie_curve_matches_enumerated_orderings(self):
        scores, risks = self.tables(4)
        for i, row in enumerate(scores):
            row["presence"] = [0., 1., 1., 1.][i]
        result = report_tables(scores, risks, bootstrap_resamples=8)
        curve = [r["retained_risk"] for r in result["risk_coverage"]
                 if r["risk"] == "merge_split" and r["score"] == "presence"]
        y = np.array([r["member_mean_merge_split"] for r in risks])
        expected = np.mean([np.cumsum(y[[0, *order]]) / np.arange(1, 5)
                            for order in permutations((1, 2, 3))], axis=0)
        np.testing.assert_allclose(curve, expected)

    def test_negative_cases_excluded_only_where_undefined_and_constant_uninformative(self):
        scores, risks = self.tables(3)
        risks[0]["n_GT"] = 0
        for risk in STRUCTURAL_RISKS:
            risks[0]["member_mean_" + risk] = None
        for row in risks[1:]:
            row["member_mean_merge_split"] = 0
        report = report_tables(scores, risks, bootstrap_resamples=8)
        endpoint = report["summary"]["endpoints"]["merge_split"]
        self.assertEqual(endpoint["status"], "constant_risk_uninformative")
        self.assertEqual(endpoint["n_cases"], 2)
        self.assertEqual(endpoint["excluded_undefined_cases"], 1)
        self.assertEqual(report["summary"]["endpoints"]["one_minus_f1"]["n_cases"], 3)
        row = next(r for r in report["comparison"] if r["risk"] == "merge_split")
        self.assertEqual(row["aurc"], 0)
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(row["spearman"])

    def test_one_case_and_all_negatives_are_na_without_crash(self):
        scores, risks = self.tables(1)
        risks[0]["n_GT"] = 0
        for risk in STRUCTURAL_RISKS:
            risks[0]["member_mean_" + risk] = None
        report = report_tables(scores, risks, bootstrap_resamples=2)
        self.assertTrue(all(r["aurc"] is None for r in report["comparison"]))
        self.assertEqual(report["summary"]["n_negative_gt_cases"], 1)

        scores, risks = self.tables(3)
        for row in risks:
            row["n_GT"] = 0
            for risk in STRUCTURAL_RISKS:
                row["member_mean_" + risk] = None
        report = report_tables(scores, risks, bootstrap_resamples=2)
        self.assertEqual(report["summary"]["n_negative_gt_cases"], 3)
        for risk in STRUCTURAL_RISKS:
            self.assertEqual(report["summary"]["endpoints"][risk]["n_cases"], 0)
        self.assertEqual(report["summary"]["endpoints"]["one_minus_dice"]["n_cases"], 3)

    def test_membership_invalid_ranges_and_eligibility_are_rejected(self):
        for kind in ("duplicate", "missing", "invalid_score", "invalid_risk", "eligibility"):
            scores, risks = self.tables()
            if kind == "duplicate":
                scores.append(scores[0])
            elif kind == "missing":
                risks.pop()
            elif kind == "invalid_score":
                scores[0]["mean_entropy"] = 1
            elif kind == "invalid_risk":
                risks[0]["member_mean_missing_volume"] = 1.1
            else:
                risks[0]["n_GT"] = 0
            with self.assertRaises(ValueError, msg=kind):
                report_tables(scores, risks, bootstrap_resamples=2)


if __name__ == "__main__":
    unittest.main()
