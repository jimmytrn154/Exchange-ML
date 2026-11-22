from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exchange_ml.partition_graph import aggregate_graph, extract_case, graph_pair, phantom_checks


def maps(directory: Path, scan_type: str = "pelvic") -> list[Path]:
    """Native integer maps: one retained instance versus a balanced split."""
    label = 1 if scan_type == "pelvic" else 151
    first = np.full((2, 3, 4), label, np.uint8)
    split = first.copy()
    split[:, :, 2:] += 1
    paths = []
    for index, array in enumerate((first, split, first)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((10., 10., 10.))  # Each voxel is exactly the 1000 mm3 cutoff.
        path = directory / f"{scan_type}-{index}.mha"
        sitk.WriteImage(image, str(path))
        paths.append(path)
    return paths


class PartitionGraphTests(unittest.TestCase):
    def test_mechanism_phantoms_and_unequal_split(self):
        phantoms = phantom_checks()
        self.assertEqual(len(phantoms), 20)
        self.assertTrue(all(row["passed"] for row in phantoms))
        tiny = next(row for row in phantoms if row["check"] == "tiny_split_unpruned")
        self.assertGreater(tiny["value"], 0)
        self.assertLess(tiny["value"], .01)
        self.assertEqual(tiny["partition_v2"], 1.)  # Known joint-entropy saturation is preserved.
        balanced = graph_pair(np.ones(10, int), np.repeat([1, 2], 5))
        unequal = graph_pair(np.ones(10, int), np.r_[np.ones(9, int), 2])
        self.assertEqual(balanced["graph_degree"], .5)
        self.assertLess(unequal["graph_degree"], balanced["graph_degree"])

    def test_sparse_implementation_matches_dense_contingencies(self):
        rng = np.random.default_rng(13)
        for _ in range(30):
            a, b = rng.integers(0, 5, (2, 80))
            shared = (a > 0) & (b > 0)
            ids_a, ids_b = np.unique(a[shared]), np.unique(b[shared])
            table = np.array([[np.count_nonzero((a == i) & (b == j)) for j in ids_b] for i in ids_a])
            forward = np.mean(table.sum(1) ** 2 / (table ** 2).sum(1) - 1)
            reverse = np.mean(table.sum(0) ** 2 / (table ** 2).sum(0) - 1)
            observed = graph_pair(a, b)
            self.assertAlmostEqual(observed["graph_degree"], (forward + reverse) / 2)
            self.assertEqual(observed["graph_shared_voxels"], int(shared.sum()))
            renamed = graph_pair(a * 10000019, b * 10000079)
            self.assertEqual(observed, renamed)

    def test_background_unmatched_instances_and_empty_support(self):
        result = graph_pair(np.array([1, 1, 2, 0]), np.array([1, 1, 0, 3]))
        self.assertEqual(result["graph_degree"], 0)
        self.assertEqual(result["graph_unmatched_instances_first"], 1)
        self.assertEqual(result["graph_unmatched_instances_second"], 1)
        features, _ = aggregate_graph([np.zeros(0, int)] * 3)
        self.assertTrue(features["all_members_empty"])
        self.assertEqual(features["graph_no_shared_fraction"], 1)
        features, _ = aggregate_graph([np.ones(3, int), np.zeros(3, int), np.zeros(3, int)])
        self.assertFalse(features["all_members_empty"])
        self.assertEqual(features["graph_no_shared_fraction"], 1)

    def test_rejects_invalid_partitions(self):
        for bad in (np.array([1.]), np.array([-1]), np.array([True])):
            with self.assertRaises(ValueError):
                graph_pair(bad, bad)
        with self.assertRaisesRegex(ValueError, "shapes"):
            graph_pair(np.ones(2, int), np.ones(3, int))
        with self.assertRaises(ValueError):
            aggregate_graph([np.ones(2, int)] * 2)

    def test_prediction_only_native_volume_and_cleanup(self):
        with TemporaryDirectory() as directory:
            paths = maps(Path(directory))
            config = {"connectivity": 26, "prune_mm3": 1000.}
            result = extract_case("004", "pelvic", paths, config)
            self.assertEqual(len(result["features"]), 3)
            self.assertEqual(len(result["pairs"]), 9)
            sacrum = result["features"][0]
            self.assertAlmostEqual(sacrum["graph_degree"], 1 / 3)
            self.assertFalse(sacrum["all_members_empty"])
            self.assertTrue(result["features"][1]["all_members_empty"])
            # Physical pruning is unchanged: at 1 mm spacing every component is too small.
            for path in paths:
                image = sitk.ReadImage(str(path)); image.SetSpacing((1., 1., 1.))
                sitk.WriteImage(image, str(path))
            self.assertTrue(all(r["all_members_empty"] for r in extract_case("004", "pelvic", paths, config)["features"]))
            image = sitk.ReadImage(str(paths[-1])); image.SetOrigin((1., 0., 0.))
            sitk.WriteImage(image, str(paths[-1]))
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                extract_case("004", "pelvic", paths, config)


if __name__ == "__main__":
    unittest.main()
