from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from exchange_ml.stage2_backend import (
    bbox_from_mask,
    bone_lut_normalize,
    contiguous_anatomy_instances,
    select_stage2_case_ids,
    semantic_anatomy_target,
    validate_grouped_samples,
)


class Stage2BackendTests(unittest.TestCase):
    def test_anatomy_mapping_and_instance_relabel(self) -> None:
        labels = np.asarray([0, 1, 7, 51, 105, 151, 200], dtype=np.int16)
        expected = np.asarray([0, 1, 1, 2, 3, 4, 4], dtype=np.uint8)
        np.testing.assert_array_equal(semantic_anatomy_target(labels), expected)
        local, mapping = contiguous_anatomy_instances(labels, "Femur")
        self.assertEqual(mapping, {151: 1, 200: 2})
        np.testing.assert_array_equal(
            local, np.asarray([0, 0, 0, 0, 0, 1, 2], dtype=np.uint8)
        )

    def test_bbox_is_clipped_and_padded(self) -> None:
        mask = np.zeros((8, 9, 10), dtype=bool)
        mask[0:2, 4:6, 8:10] = True
        bbox = bbox_from_mask(mask, 2)
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertEqual(
            [(value.start, value.stop) for value in bbox],
            [(0, 4), (2, 8), (6, 10)],
        )

    def test_lut_is_bounded_and_monotonic(self) -> None:
        values = np.asarray(
            [-2000, -1000, -200, 150, 700, 1500, 3000], dtype=np.float32
        )
        mapped = bone_lut_normalize(values)
        self.assertTrue(np.all(np.diff(mapped) >= 0))
        self.assertGreaterEqual(float(mapped.min()), 0.0)
        self.assertLessEqual(float(mapped.max()), 1.0)

    def test_frozen_split_has_no_overlap(self) -> None:
        split = json.loads(Path("splits/pengwin_stage0_split.json").read_text())
        selected = select_stage2_case_ids(split, 20260920)
        self.assertEqual(
            {key: len(value) for key, value in selected.items()},
            {"train": 40, "validation": 40, "test": 50},
        )
        validate_grouped_samples(
            [f"PENGWIN_{case_id}_Sacrum" for case_id in selected["train"]],
            [f"PENGWIN_{case_id}_Sacrum" for case_id in selected["validation"]],
        )

    def test_group_leakage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_grouped_samples(
                ["PENGWIN_002_Sacrum"], ["PENGWIN_002_LeftHip"]
            )


if __name__ == "__main__":
    unittest.main()
