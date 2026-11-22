from __future__ import annotations

from itertools import permutations
import unittest

import numpy as np
from scipy.stats import rankdata

from exchange_ml.e1_data import bone_risk, clean_prediction
from exchange_ml.e1_scoring import information_from_counts, pair_components, score_unit
from exchange_ml.e1_statistics import aurc_batch, cluster_draws, correlation, expected_curve
from exchange_ml.instances import instance_metrics, prune_small_components
from exchange_ml.stage3_h1 import case_uncertainty_scores


class E1ScoringTests(unittest.TestCase):
    def test_exact_merge_phantom_and_reverse(self):
        separate = np.repeat([1, 2], 32).astype(np.uint8)
        merged = np.ones(64, dtype=np.uint8)
        for left, right in ((separate, merged), (merged, separate)):
            pair = pair_components(left, right)
            self.assertAlmostEqual(pair["vi"], np.log(2))
            self.assertAlmostEqual(pair["q_log_n"], 1 / 6)
            self.assertAlmostEqual(pair["q_joint"], 1)
            self.assertEqual(pair["presence"], 0)
            self.assertAlmostEqual(pair["fa_v2"], 1)

    def test_replication_isolates_voxel_denominator(self):
        counts = np.array([[16, 0], [16, 0]])
        original = information_from_counts(counts)
        for r in (2, 4, 8, 16):
            result = information_from_counts(r * counts)
            for key in ("vi", "joint_entropy", "q_joint"):
                self.assertAlmostEqual(result[key], original[key])
            self.assertAlmostEqual(result["q_log_n"], original["vi"] / np.log(32 * r))

    def test_permutation_and_zero_entropy(self):
        first = np.array([1, 1, 2, 2, 0])
        second = np.array([7, 7, 3, 3, 0])
        pair = pair_components(first, second)
        self.assertAlmostEqual(pair["fa_v2"], 0)
        for n in (0, 1, 20):
            pair = pair_components(np.ones(n, int), np.ones(n, int))
            self.assertEqual(pair["q_joint"], 0)
            self.assertEqual(pair["q_log_n"], 0)

    def test_empty_pairs_and_empty_unit_are_retained(self):
        empty = np.zeros(8, dtype=int)
        full = np.ones(8, dtype=int)
        self.assertEqual(pair_components(empty, full)["fa_v2"], 1)
        self.assertEqual(pair_components(full, empty)["fa_v2"], 1)
        scores, pairs = score_unit([empty] * 3, [np.full(8, .5)] * 3)
        self.assertTrue(all(value == 0 for value in scores.values()))
        self.assertEqual(len(pairs), 3)

    def test_mean_probability_entropy_and_id_cardinality(self):
        maps = [np.array([1, 1]), np.array([2, 3]), np.array([1, 1])]
        probabilities = [np.zeros(2), np.ones(2), np.zeros(2)]
        scores, _ = score_unit(maps, probabilities)
        entropy = -(1 / 3 * np.log(1 / 3) + 2 / 3 * np.log(2 / 3))
        self.assertAlmostEqual(scores["mean_entropy"], entropy)
        self.assertAlmostEqual(scores["count_variance"], np.var([1, 2, 1]))
        with self.assertRaises(ValueError):
            score_unit(maps, [np.full(2, np.nan)] * 3)

    def test_v1_scores_match_existing_whole_case_path(self):
        # Multiple anatomical groups: Hungarian must forbid cross-bone matches.
        maps = [np.array([0, 1, 2, 51, 52, 151]),
                np.array([1, 1, 2, 52, 0, 152]),
                np.array([0, 3, 3, 0, 51, 151])]
        probabilities = [np.linspace(.1, .9, 6)] * 3
        current, _ = score_unit(maps, probabilities)
        legacy = case_uncertainty_scores(maps, probabilities)
        mapping = {"mean_entropy": "mean_entropy", "p95_entropy": "p95_entropy",
                   "pairwise_dice": "pairwise_semantic_dice_disagreement",
                   "presence": "foreground_jaccard_disagreement", "union_nvi": "union_foreground_nvi",
                   "count_variance": "fragment_count_variance", "hungarian": "hungarian_structural_disagreement",
                   "fa_v1": "fa_ipd"}
        for new, old in mapping.items():
            self.assertAlmostEqual(current[new], legacy[old], msg=new)

    def test_cleanup_crop_preserves_existing_connectivity_and_threshold(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 5, (8, 9, 10), dtype=np.uint8)
        labels[1:5, 2:6, 3:7] = 51
        for spacing in ((1., 1., 1.), (.7, .8, 2.)):
            for threshold in (0., 3., 10.):
                legacy = prune_small_components(labels, spacing, min_volume_mm3=threshold, connectivity=26)
                np.testing.assert_array_equal(clean_prediction(labels, spacing, threshold, 26), legacy)

    def test_risk_reuses_matching_without_pruning_gt(self):
        gt = np.zeros((10, 10, 10), dtype=np.uint8)
        gt[1:3, 1:3, 1:3] = 1
        gt[5:9, 5:9, 5:9] = 2
        pred = gt.copy()
        pred[gt == 2] = 1
        reference = instance_metrics(pred, gt, (1, 1, 1), connectivity=26,
                                     prediction_component_prune_min_volume_mm3=0)
        row = bone_risk(pred, gt, (1, 1, 1), "sacrum", .1)
        self.assertEqual(row["n_GT"], 2)
        self.assertEqual(row["min_gt_fragment_volume_mm3"], 8)
        self.assertEqual(row["merge_split"], reference["normalized_merge_split_error"])
        self.assertAlmostEqual(row["one_minus_f1"], 1 - reference["instance_f1"])
        self.assertEqual(row["missing_volume"], 0)


class E1StatisticsTests(unittest.TestCase):
    def test_tie_expectation_matches_enumerated_rankings(self):
        scores = np.array([0., 1., 1., 1.])
        risks = np.array([2., 0., 1., 7.])
        curves = [np.cumsum(risks[[0, *order]]) / np.arange(1, 5)
                  for order in permutations((1, 2, 3))]
        expected = np.mean(curves, axis=0)
        np.testing.assert_allclose(expected_curve(scores, risks), expected)
        area = np.trapz(expected, dx=1 / 4)
        self.assertAlmostEqual(aurc_batch(scores[None], risks[None, :, None])[0, 0], area)

    def test_batch_matches_curves_for_ties_and_unequal_risk_scales(self):
        rng = np.random.default_rng(15)
        scores = rng.integers(0, 3, (9, 8))
        risks = rng.random((9, 8, 3)) * 4
        area = aurc_batch(scores, risks)
        for b in range(9):
            for k in range(3):
                self.assertAlmostEqual(area[b, k], np.trapz(expected_curve(scores[b], risks[b, :, k]), dx=1 / 8))

    def test_constant_risk_random_reference_and_small_n(self):
        scores = np.array([[2., 1., 1., 0.]])
        self.assertAlmostEqual(aurc_batch(scores, np.full((1, 4, 1), 3.))[0, 0], 3 * .75)
        self.assertTrue(np.isnan(aurc_batch(np.zeros((1, 1)), np.zeros((1, 1, 1)))[0, 0]))

    def test_cluster_bootstrap_preserves_bones_strata_and_duplicate_cases(self):
        rows = [{"case_id": case, "scan_type": scan, "bone": bone}
                for case, scan, bones in (("p1", "pelvic", range(3)), ("p2", "pelvic", range(3)),
                                          ("f1", "femur", range(1)), ("f2", "femur", range(1)))
                for bone in bones]
        draws = cluster_draws(rows, 100, 20261119)
        np.testing.assert_array_equal(draws, cluster_draws(rows, 100, 20261119))
        duplicate = False
        for take in draws:
            self.assertEqual(sum(rows[i]["scan_type"] == "pelvic" for i in take), 6)
            for case in ("p1", "p2"):
                counts = [sum(rows[i]["case_id"] == case and rows[i]["bone"] == bone for i in take) for bone in range(3)]
                self.assertEqual(len(set(counts)), 1)
                duplicate |= counts[0] == 2
        self.assertTrue(duplicate)

    def test_partial_spearman_uses_ranks_and_reports_undefined(self):
        rng = np.random.default_rng(4)
        x, y, c = rng.random(15), rng.random(15), rng.random((15, 3))
        design = np.column_stack((np.ones(15), rankdata(c, axis=0)))
        residuals = [r - design @ np.linalg.lstsq(design, r, rcond=None)[0] for r in (rankdata(x), rankdata(y))]
        self.assertAlmostEqual(correlation(x, y, c), np.corrcoef(*residuals)[0, 1])
        self.assertIsNone(correlation(x, np.ones(15)))
        self.assertIsNone(correlation(x, y, np.ones((15, 3))))


if __name__ == "__main__":
    unittest.main()
