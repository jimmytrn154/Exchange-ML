from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

from exchange_ml.e1_scoring import pair_components
from exchange_ml.e2_perturbations import (
    METRICS, components, contingency, extract_bone, merge_pairs, phantoms,
    replication, responses, split_largest,
)
from exchange_ml.perturbations import matched_boundary_perturbation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e2 import selection  # noqa: E402


def fixture() -> np.ndarray:
    labels = np.zeros((7, 7, 22), dtype=np.uint8)
    for i in range(6):
        labels[2:4, 2:4, 1 + 3 * i:3 + 3 * i] = i + 1
    return labels


class E2Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/e2.json").read_text())

    def test_phantom_exact_assertions(self):
        checks, table = phantoms(1e-10)
        self.assertEqual(len(checks), 32)
        self.assertTrue(all(r["passed"] for r in checks))
        self.assertEqual(table.sum(), 64)
        self.assertAlmostEqual(components(table)["q_log_n"], 1 / 6)
        self.assertEqual(components(table)["q_joint"], 1)

    def test_contingency_matches_e1_and_ignores_zero_padding(self):
        left = np.array([0, 51, 52, 52, 0, 53])
        right = np.array([51, 51, 53, 0, 0, 53])
        first = components(contingency(left, right, "left_hip"))
        old = pair_components(left, right)
        for key in (*METRICS, "n_shared_voxels", "n_union_voxels"):
            self.assertAlmostEqual(first[key], old[key])
        padded = components(contingency(np.pad(left, 100), np.pad(right, 100), "left_hip"))
        self.assertEqual(first, padded)
        with self.assertRaises(ValueError):
            contingency(left, right, "sacrum")

    def test_replication_scales_mismatches_as_well_as_shared_counts(self):
        joint = np.array([[0, 3, 1], [2, 6, 0], [0, 2, 4]], dtype=np.int64)
        base = components(joint)
        rows = replication(joint, [1, 2, 4, 8, 16], 1e-10)
        for row in rows:
            self.assertEqual(row["N"], base["n_shared_voxels"] * row["r"])
            for actual, expected in (("VI", "vi"), ("H", "joint_entropy"), ("A", "presence"), ("w", "overlap_weight"), ("Q2", "q_joint")):
                self.assertAlmostEqual(row[actual], base[expected])
            self.assertAlmostEqual(row["Q1"], base["vi"] / np.log(row["N"]))
        self.assertLess(rows[-1]["Q1"], rows[0]["Q1"])

    def test_replication_degenerate_zero_and_single_voxel_domains(self):
        for joint in (np.zeros((2, 2), int), np.array([[0, 3], [0, 0]]), np.array([[0, 0], [0, 1]])):
            for row in replication(joint, [1, 2, 16], 1e-10):
                self.assertFalse(row["positive_vi"])
                self.assertEqual(row["Q1"], 0)
                self.assertEqual(row["Q2"], 0)

    def test_selected_cases_are_balanced_unique_and_order_independent(self):
        metadata = [{"case_id": f"{offset + i:03}", "scan_type": scan}
                    for scan, offset in (("pelvic", 0), ("femur", 200)) for i in range(20)]
        chosen = selection(metadata, 20261118, 10)
        self.assertEqual(chosen, selection(list(reversed(metadata)), 20261118, 10))
        self.assertEqual(len({r["case_id"] for r in chosen}), 20)
        self.assertEqual(sum(r["scan_type"] == "pelvic" for r in chosen), 10)
        self.assertNotEqual(chosen, selection(metadata, 1, 10))

    def test_merge_greedy_pairs_are_disjoint_and_tie_broken_by_ids(self):
        pairs = merge_pairs(fixture(), (1, 1, 1))
        self.assertEqual(pairs, [(1, 2), (3, 4), (5, 6)])
        self.assertEqual(len({x for pair in pairs for x in pair}), 6)

    def test_split_physical_axis_balanced_chunks_and_anatomy_ids(self):
        source = np.zeros((5, 7, 5), dtype=np.uint8)
        source[1:3, 1:5, 1:3] = 51
        source[4, 6, 4] = 52
        original = source.copy()
        changed, metadata = split_largest(source, (3, 1, 1), "left_hip", 4)
        self.assertEqual(metadata["split_axis_zyx"], 0)
        self.assertEqual(metadata["new_ids"], [51, 53, 54, 55])
        sizes = [int((changed == i).sum()) for i in metadata["new_ids"]]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        np.testing.assert_array_equal(changed > 0, source > 0)
        np.testing.assert_array_equal(source, original)

    def test_split_axis_ties_prefer_z(self):
        source = np.ones((2, 2, 2), dtype=np.uint8)
        changed, meta = split_largest(source, (1, 1, 1), "sacrum", 2)
        self.assertEqual(meta["split_axis_zyx"], 0)
        np.testing.assert_array_equal(changed[0], 1)
        np.testing.assert_array_equal(changed[1], 2)

    def test_no_unused_ids_is_explicitly_ineligible(self):
        source = np.arange(1, 51, dtype=np.uint8).reshape(2, 5, 5)
        source[0, :2, :2] = 1
        # Use all 50 IDs with enough voxels in the largest instance.
        source = np.concatenate((np.arange(1, 51), np.repeat(1, 10))).astype(np.uint8).reshape(3, 4, 5)
        changed, metadata = split_largest(source, (1, 1, 1), "sacrum", 4)
        self.assertIsNone(changed)
        self.assertIn("unused_anatomy_ids", metadata["reason"])

    def test_responses_independent_from_original_with_deduplicated_missing(self):
        source = fixture()
        original = source.copy()
        rows = responses(source, (1, 1, 1), "sacrum", self.config)
        np.testing.assert_array_equal(source, original)
        missing = [r for r in rows if r["operation"] == "missing"]
        self.assertEqual(len(missing), 2)  # equal sizes: small and large both choose ID 1
        self.assertEqual({r["affected_ids"] for r in missing}, {"[1]", "[4]"})
        for row in rows:
            if row["operation"] in ("identity", "permutation", "merge", "split"):
                self.assertAlmostEqual(row["presence"], 0)
            if row["operation"] in ("missing", "boundary"):
                self.assertAlmostEqual(row["vi"], 0)
            self.assertEqual(row["gt_fragment_count"], 6)
        for row in missing:
            self.assertAlmostEqual(row["changed_volume_fraction"], 1 / 6)
        self.assertEqual(len([r for r in rows if r["operation"] == "merge" and r["eligible"]]), 3)

    def test_small_fragments_are_not_pruned_and_targets_not_faked(self):
        source = np.zeros((5, 5, 5), dtype=np.uint8)
        source[2, 2, 2] = 1
        rows = responses(source, (1, 1, 1), "sacrum", self.config)
        self.assertTrue(all(r["gt_volume_mm3"] == 1 for r in rows))
        self.assertEqual(sum(r["operation"] == "missing" for r in rows), 1)
        self.assertTrue(all(not r["eligible"] for r in rows if r["operation"] == "merge"))
        for row in rows:
            if row["operation"] == "boundary":
                self.assertFalse(row["target_reached"])
                self.assertEqual(row["reason"], "target_not_reached")
        self.assertTrue(all("entropy" not in r for r in rows))

    def test_boundary_crop_only_removes_zero_padding(self):
        source = np.zeros((35, 35, 35), dtype=np.uint8)
        source[15:18, 15:18, 15:18] = 1
        source[5:7, 5:7, 5:7] = 2
        cropped = extract_bone(source, "sacrum", 8)
        full_changed, _ = matched_boundary_perturbation(source, 1, target_binary_dice=.95, connectivity=26)
        crop_changed, _ = matched_boundary_perturbation(cropped, 1, target_binary_dice=.95, connectivity=26)
        full = components(contingency(source, full_changed, "sacrum"))
        actual = components(contingency(cropped, crop_changed, "sacrum"))
        self.assertEqual(full, actual)


if __name__ == "__main__":
    unittest.main()
