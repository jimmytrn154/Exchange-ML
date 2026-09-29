from __future__ import annotations

import unittest

import numpy as np

from exchange_ml.stage3_h1 import (
    aurc,
    binary_predictive_entropy,
    case_uncertainty_scores,
    hungarian_instance_disagreement,
    residualized_spearman,
    risk_coverage_curve,
    top_k_failure_summary,
)


class Stage3H1Tests(unittest.TestCase):
    def test_binary_entropy_extremes_and_half(self) -> None:
        roi = np.ones((2, 2, 2), dtype=bool)
        certain = binary_predictive_entropy(
            [np.zeros(roi.shape), np.zeros(roi.shape), np.zeros(roi.shape)], roi
        )
        self.assertEqual(certain["mean_entropy"], 0.0)
        half = binary_predictive_entropy(
            [np.zeros(roi.shape), np.ones(roi.shape)], roi
        )
        self.assertAlmostEqual(half["mean_entropy"], np.log(2.0))
        self.assertAlmostEqual(half["p95_entropy"], np.log(2.0))

    def test_case_scores_detect_missing_fragment(self) -> None:
        first = np.zeros((8, 8, 8), dtype=np.int16)
        first[1:3, 1:3, 1:3] = 1
        first[5:7, 5:7, 5:7] = 2
        missing = first.copy()
        missing[missing == 2] = 0
        probabilities = [(partition > 0).astype(np.float32) for partition in (first, missing, first)]
        scores = case_uncertainty_scores(
            [first, missing, first], probabilities
        )
        self.assertGreater(scores["fa_ipd"], 0.0)
        self.assertGreater(scores["foreground_jaccard_disagreement"], 0.0)
        self.assertGreater(scores["fragment_count_variance"], 0.0)
        self.assertGreater(scores["pairwise_semantic_dice_disagreement"], 0.0)

    def test_hungarian_is_permutation_invariant_and_penalizes_unmatched(self) -> None:
        first = np.zeros((6, 6, 6), dtype=np.int16)
        first[:3] = 1
        first[3:] = 2
        relabeled = np.where(first == 1, 2, np.where(first == 2, 1, 0))
        self.assertAlmostEqual(hungarian_instance_disagreement(first, relabeled), 0.0)
        missing = np.where(first == 2, 0, first)
        self.assertGreater(hungarian_instance_disagreement(first, missing), 0.0)

    def test_risk_coverage_and_aurc_use_low_uncertainty_prefix(self) -> None:
        uncertainty = [0.3, 0.1, 0.2]
        risk = [1.0, 0.0, 0.5]
        case_ids = ["003", "001", "002"]
        coverage, prefix, order = risk_coverage_curve(uncertainty, risk, case_ids)
        self.assertEqual(order, ["001", "002", "003"])
        np.testing.assert_allclose(coverage, [1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(prefix, [0.0, 0.25, 0.5])
        self.assertAlmostEqual(aurc(uncertainty, risk, case_ids), 0.25)

    def test_value_quartiles_do_not_split_ties(self) -> None:
        from scripts.evaluate_stage3_h1 import value_quartiles

        values = [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
        bins = value_quartiles(values)
        self.assertEqual({bins[index] for index in (0, 1, 2)}, {bins[0]})
        self.assertEqual({bins[index] for index in (5, 6, 7)}, {bins[5]})

    def test_example_selection_requires_failure_and_uses_ascending_tie_break(self) -> None:
        from scripts.evaluate_stage3_h1 import select_example

        rows = [
            {"case_id": "002", "mean_entropy": 0.1, "fa_ipd": 0.9, "failure_event": True},
            {"case_id": "001", "mean_entropy": 0.1, "fa_ipd": 0.9, "failure_event": True},
            {"case_id": "003", "mean_entropy": 0.0, "fa_ipd": 1.0, "failure_event": False},
        ]
        self.assertEqual(select_example(rows), "001")
        self.assertIsNone(
            select_example([{**row, "failure_event": False} for row in rows])
        )

    def test_top_k_is_deterministic_and_residualization_is_finite(self) -> None:
        result = top_k_failure_summary(
            [0.9, 0.9, 0.1],
            [True, False, False],
            [1.0, 0.5, 0.0],
            ["002", "001", "003"],
            1 / 3,
        )
        self.assertEqual(result["selected_case_ids"], ["001"])
        rho = residualized_spearman(
            [0.1, 0.2, 0.8, 0.9],
            [0.2, 0.1, 0.7, 1.0],
            np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]], dtype=float),
        )
        self.assertTrue(np.isfinite(rho))


if __name__ == "__main__":
    unittest.main()
