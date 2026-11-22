from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import SimpleITK as sitk

from exchange_ml.e6_data import (binary_target, case_number, geometry, inventory,
                               read_case, read_label_codes, read_volume,
                               require_geometry, validate_labels)


def write_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["public_id", "label_id", "label_code"])
        writer.writerows(rows)


def fixture(root: Path) -> None:
    for folder in ("Part1", "Part2", "ribfrac-val-images", "ribfrac-val-labels"):
        (root / folder).mkdir()
    for cid, folder, metadata in (("RibFrac1", "Part1", "ribfrac-train-info-1.csv"),
                                  ("RibFrac301", "Part2", "ribfrac-train-info-2.csv"),
                                  ("RibFrac421", "ribfrac-val-images", "ribfrac-val-info.csv")):
        label_folder = "ribfrac-val-labels" if cid == "RibFrac421" else folder
        label = np.zeros((2, 3, 4), np.uint16)
        label[0, 1, 1] = 301
        for kind, directory, array in (("image", folder, label.astype(np.int16)),
                                        ("label", label_folder, label)):
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((.7, .8, 1.2))
            image.SetOrigin((10., 20., 30.))
            image.SetDirection((-1., 0., 0., 0., -1., 0., 0., 0., 1.))
            sitk.WriteImage(image, str(root / directory / f"{cid}-{kind}.nii.gz"))
        write_csv(root / metadata, [(cid, 0, 0), (cid, 301, -1)])


class E6DataTests(unittest.TestCase):
    def test_inventory_pairs_metadata_and_excludes_test(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            (root / "ribfrac-test-images").mkdir()
            (root / "ribfrac-test-images/RibFrac501-image.nii.gz").write_bytes(b"sealed")
            observed = inventory(root, strict=False)
            self.assertEqual(list(observed), ["RibFrac1", "RibFrac301", "RibFrac421"])
            self.assertEqual(observed["RibFrac421"]["split"], "validation")
            self.assertEqual(observed["RibFrac421"]["label_codes"], {0: 0, 301: -1})
            with self.assertRaisesRegex(ValueError, "official"):
                inventory(root)
            (root / "ribfrac-val-labels/RibFrac421-label.nii.gz").unlink()
            with self.assertRaisesRegex(ValueError, "membership differs"):
                inventory(root, strict=False)

    def test_public_ids_and_split_leakage_rejected(self):
        for case in ("RibFrac0", "RibFrac01", "RibFrac661", "421", "RibFrac-1"):
            with self.subTest(case=case), self.assertRaises(ValueError):
                case_number(case)
        self.assertEqual(case_number("RibFrac421"), 421)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            for kind in ("image", "label"):
                source = root / "Part1" / f"RibFrac1-{kind}.nii.gz"
                source.rename(root / "Part1" / f"RibFrac421-{kind}.nii.gz")
            write_csv(root / "ribfrac-train-info-1.csv", [("RibFrac421", 0, 0), ("RibFrac421", 301, -1)])
            with self.assertRaisesRegex(ValueError, "split leakage"):
                inventory(root, strict=False)

    def test_untyped_fractures_are_positive_and_ids_do_not_encode_anatomy(self):
        labels = np.array([[[0, 1, 72, 301]]], dtype=np.uint16)
        codes = {0: 0, 1: -1, 72: 2, 301: 4}
        self.assertIs(validate_labels(labels, codes), labels)
        np.testing.assert_array_equal(binary_target(labels, codes), [[[0, 1, 1, 1]]])
        for invalid_codes in ({0: 0, 1: 1}, {0: 0, 1: 1, 72: 2, 301: 5}, {1: 1, 72: 2, 301: 4}):
            with self.assertRaises(ValueError):
                validate_labels(labels, invalid_codes)
        for invalid in (labels.astype(float), np.array([[[-1]]]), np.zeros((2, 2), int)):
            with self.assertRaises(ValueError):
                validate_labels(invalid, {0: 0})
        self.assertEqual(binary_target(np.zeros((2, 2, 2), np.uint8), {0: 0}).sum(), 0)

    def test_csv_duplicates_unknown_codes_and_background_errors_rejected(self):
        rows = [("RibFrac421", 0, 0), ("RibFrac421", 1, -1)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "info.csv"
            write_csv(path, rows)
            self.assertEqual(read_label_codes(path), {"RibFrac421": {0: 0, 1: -1}})
            for invalid_rows in (rows + [rows[-1]], [("RibFrac421", 1, -1)],
                                 [("RibFrac421", 0, -1)], [("RibFrac421", 0, 0), ("RibFrac421", 1, 5)]):
                write_csv(path, invalid_rows)
                with self.assertRaises(ValueError):
                    read_label_codes(path)

    def test_one_case_round_trip_preserves_orientation_and_large_ids(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            record = inventory(root, strict=False)["RibFrac421"]
            image, labels, observed = read_case(record)
            self.assertEqual(image.shape, (2, 3, 4))
            self.assertEqual(labels.dtype, np.uint32)
            self.assertEqual(labels.max(), 301)
            self.assertEqual(observed["direction"], (-1., 0., 0., 0., -1., 0., 0., 0., 1.))
            self.assertIsNone(read_case(record, load_image=False)[0])
            changed = dict(observed, spacing=(1., 1., 1.))
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                read_volume(record["label"], changed)
            changed = dict(observed, direction=(1., 0., 0., 0., 1., 0., 0., 0., 1.))
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                require_geometry(observed, changed)

    def test_probabilities_validated_and_scalar_grid_required(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probability.mha"
            for value in (float("nan"), -.1, 1.1):
                sitk.WriteImage(sitk.GetImageFromArray(np.full((2, 2, 2), value, np.float32)), str(path))
                with self.assertRaisesRegex(ValueError, "probabilities must"):
                    read_volume(path, probability=True)
            sitk.WriteImage(sitk.GetImageFromArray(np.full((2, 2, 2), .5, np.float32)), str(path))
            result, _ = read_volume(path, probability=True)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.mean(), .5)
            with self.assertRaisesRegex(ValueError, "scalar 3D"):
                geometry(sitk.Image([2, 2], sitk.sitkUInt8))


if __name__ == "__main__":
    unittest.main()
