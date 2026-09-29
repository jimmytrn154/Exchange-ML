from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage as ndi

from exchange_ml.instances import connectivity_structure, instance_metrics
from exchange_ml.metrics import (
    binary_dice,
    foreground_aware_partition_components,
    foreground_aware_partition_disagreement,
    foreground_jaccard_disagreement,
    ipd,
    normalized_variation_of_information,
    surface_distances,
)
from exchange_ml.perturbations import (
    matched_boundary_perturbation,
    merge_instances,
    relabel_instances,
    remove_instance,
    split_instance,
)


def synthetic_instances() -> np.ndarray:
    labels = np.zeros((16, 16, 16), dtype=np.int16)
    labels[2:7, 2:7, 2:7] = 1
    labels[9:14, 9:14, 9:14] = 2
    return labels


class Stage1MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt = synthetic_instances()
        self.kwargs = {
            "connectivity": 26,
            "prediction_component_prune_min_volume_mm3": 0.0,
            "iou_match_threshold": 0.1,
        }

    def test_gt_vs_gt_is_perfect(self) -> None:
        result = instance_metrics(self.gt, self.gt, (1.0, 1.0, 1.0), **self.kwargs)
        self.assertEqual(result["binary_dice"], 1.0)
        self.assertEqual(result["instance_f1"], 1.0)
        self.assertEqual(result["merge_errors"], 0)
        self.assertEqual(result["split_errors"], 0)

    def test_partition_metrics_are_permutation_invariant(self) -> None:
        relabeled = relabel_instances(self.gt, {1: 17, 2: 23})
        self.assertAlmostEqual(
            normalized_variation_of_information(self.gt, relabeled), 0.0
        )
        self.assertAlmostEqual(
            foreground_aware_partition_disagreement(self.gt, relabeled),
            0.0,
        )
        self.assertAlmostEqual(ipd([self.gt, relabeled, self.gt]), 0.0)

    def test_merge_and_split_change_partition_without_changing_foreground(self) -> None:
        merged = merge_instances(self.gt, 1, 2)
        split = split_instance(self.gt, 1, new_id=3, axis=0)
        self.assertEqual(binary_dice(merged > 0, self.gt > 0), 1.0)
        self.assertEqual(binary_dice(split > 0, self.gt > 0), 1.0)
        for changed in (merged, split):
            union_nvi = normalized_variation_of_information(self.gt, changed)
            self.assertGreater(union_nvi, 0.0)
            self.assertAlmostEqual(
                foreground_aware_partition_disagreement(self.gt, changed),
                union_nvi,
            )
        self.assertEqual(
            instance_metrics(merged, self.gt, (1, 1, 1), **self.kwargs)["merge_errors"],
            1,
        )
        self.assertEqual(
            instance_metrics(split, self.gt, (1, 1, 1), **self.kwargs)["split_errors"],
            1,
        )

    def test_missing_fragment_reduces_semantic_and_instance_scores(self) -> None:
        missing = remove_instance(self.gt, 2)
        result = instance_metrics(missing, self.gt, (1, 1, 1), **self.kwargs)
        self.assertLess(result["binary_dice"], 1.0)
        self.assertLess(result["instance_f1"], 1.0)
        self.assertEqual(normalized_variation_of_information(self.gt, missing), 0.0)
        self.assertAlmostEqual(
            foreground_aware_partition_disagreement(self.gt, missing),
            0.5,
        )

    def test_foreground_presence_edge_cases_and_symmetry(self) -> None:
        empty = np.zeros_like(self.gt)
        self.assertEqual(foreground_aware_partition_disagreement(empty, empty), 0.0)
        self.assertEqual(foreground_aware_partition_disagreement(empty, self.gt), 1.0)
        self.assertEqual(foreground_aware_partition_disagreement(self.gt, empty), 1.0)

        missing = remove_instance(self.gt, 2)
        forward = foreground_aware_partition_disagreement(self.gt, missing)
        reverse = foreground_aware_partition_disagreement(missing, self.gt)
        self.assertAlmostEqual(forward, reverse)

    def test_fa_ipd_component_breakdown_is_consistent(self) -> None:
        missing = remove_instance(self.gt, 2)
        components = foreground_aware_partition_components(self.gt, missing)
        self.assertEqual(components["union_voxels"], 250)
        self.assertEqual(components["intersection_voxels"], 125)
        self.assertAlmostEqual(components["presence_disagreement"], 0.5)
        self.assertEqual(components["partition_disagreement"], 0.0)
        self.assertEqual(components["weighted_partition_disagreement"], 0.0)
        self.assertAlmostEqual(
            foreground_jaccard_disagreement(self.gt, missing),
            components["presence_disagreement"],
        )
        self.assertAlmostEqual(
            foreground_aware_partition_disagreement(self.gt, missing),
            components["score"],
        )

    def test_pairwise_domains_are_aggregated_for_ensemble_ipd(self) -> None:
        missing_first = remove_instance(self.gt, 1)
        missing_second = remove_instance(self.gt, 2)
        expected = np.mean(
            [
                foreground_aware_partition_disagreement(self.gt, missing_first),
                foreground_aware_partition_disagreement(self.gt, missing_second),
                foreground_aware_partition_disagreement(missing_first, missing_second),
            ]
        )
        self.assertAlmostEqual(ipd([self.gt, missing_first, missing_second]), expected)

    def test_roi_restricts_presence_and_partition_domains(self) -> None:
        roi = np.zeros_like(self.gt, dtype=bool)
        roi[2:7, 2:7, 2:7] = True
        missing_outside_roi = remove_instance(self.gt, 2)
        self.assertEqual(
            foreground_aware_partition_disagreement(
                self.gt, missing_outside_roi, roi=roi
            ),
            0.0,
        )
        with self.assertRaises(ValueError):
            foreground_aware_partition_disagreement(
                self.gt, missing_outside_roi, roi=np.ones((2, 2))
            )

    def test_score_is_not_a_triangle_inequality_metric(self) -> None:
        first = np.asarray([0, 0, 0, 1, 1])
        bridge = np.asarray([0, 1, 1, 2, 2])
        third = np.asarray([0, 1, 1, 2, 3])
        direct = foreground_aware_partition_disagreement(first, third)
        via_bridge = foreground_aware_partition_disagreement(
            first, bridge
        ) + foreground_aware_partition_disagreement(bridge, third)
        self.assertAlmostEqual(direct, 1.0)
        self.assertAlmostEqual(via_bridge, 0.75)
        self.assertGreater(direct, via_bridge)

    def test_boundary_perturbation_targets_semantic_dice(self) -> None:
        perturbed, metadata = matched_boundary_perturbation(
            self.gt, 1, target_binary_dice=0.9, connectivity=26, max_iterations=3
        )
        self.assertNotEqual(metadata["achieved_binary_dice"], 1.0)
        self.assertLess(abs(float(metadata["achieved_binary_dice"]) - 0.9), 0.01)
        self.assertGreater(
            foreground_aware_partition_disagreement(self.gt, perturbed), 0.0
        )

    def test_surface_distances_identity(self) -> None:
        mask = self.gt == 1
        distances = surface_distances(mask, mask, (1.0, 1.0, 1.0))
        self.assertEqual(distances, {"assd_mm": 0.0, "hd95_mm": 0.0})

    def test_prediction_cleanup_never_removes_small_gt_instances(self) -> None:
        reference = np.zeros((12, 12, 12), dtype=np.int16)
        reference[1:11, 1:11, 1:11] = 1
        reference[0, 0, 0] = 2
        prediction = reference.copy()
        result = instance_metrics(
            prediction,
            reference,
            (1.0, 1.0, 1.0),
            connectivity=26,
            prediction_component_prune_min_volume_mm3=500.0,
            iou_match_threshold=0.1,
        )
        self.assertEqual(result["retained_gt_instance_count"], 2)
        self.assertEqual(result["per_anatomy"]["sacrum"]["gt_instance_count"], 2)
        self.assertEqual(result["per_anatomy"]["sacrum"]["pred_instance_count"], 1)
        self.assertLess(result["instance_recall"], 1.0)

    def test_out_of_contract_instance_ids_are_rejected(self) -> None:
        invalid_gt = self.gt.copy()
        invalid_gt[0, 0, 0] = 201
        with self.assertRaisesRegex(ValueError, "GT instance IDs"):
            instance_metrics(invalid_gt, invalid_gt, (1, 1, 1), **self.kwargs)

        invalid_prediction = self.gt.copy()
        invalid_prediction[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "prediction instance IDs"):
            instance_metrics(invalid_prediction, self.gt, (1, 1, 1), **self.kwargs)

    def test_connectivity_conventions(self) -> None:
        face = np.zeros((4, 4, 4), dtype=bool)
        face[1, 1, 1] = face[2, 1, 1] = True
        edge = np.zeros_like(face)
        edge[1, 1, 1] = edge[2, 2, 1] = True
        corner = np.zeros_like(face)
        corner[1, 1, 1] = corner[2, 2, 2] = True
        self.assertEqual(ndi.label(face, connectivity_structure(6))[1], 1)
        self.assertEqual(ndi.label(edge, connectivity_structure(6))[1], 2)
        self.assertEqual(ndi.label(edge, connectivity_structure(18))[1], 1)
        self.assertEqual(ndi.label(corner, connectivity_structure(18))[1], 2)
        self.assertEqual(ndi.label(corner, connectivity_structure(26))[1], 1)


if __name__ == "__main__":
    unittest.main()
