from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from exchange_ml.e1_data import MEMBERS
from exchange_ml.e5_supplement_figures import annotation_target, plot_budget_figures, plot_qualitative_supplement


class E5SupplementFigureTests(unittest.TestCase):
    def example(self):
        truth = np.zeros((8, 12), np.uint8)
        truth[2:6, 3:9] = 1
        empty = np.zeros_like(truth)
        panels = [{"ct": np.zeros_like(truth, dtype=float), "labels": [truth, empty, empty, empty],
                   "original": [truth, empty, empty, empty], "spacing_yx": (1., .8), "z": z} for z in (2, 4)]
        selected = {"case_id": "187", "bone": "left_hip", "category": "shared_error_low_disagreement", "status": "selected"}
        example = {"selection": selected, "panels": panels}
        candidates = [{**selected, "selected": True, "fa_v2": 0., "partition_v2": 0., "presence": 0., "mean_entropy": 0.}]
        risks = [{"case_id": "187", "bone": "left_hip", "model_id": m, "one_minus_f1": 1.,
                  "merge_split": 0., "missing_volume": 1.} for m in MEMBERS]
        return example, selected, candidates, risks

    def test_omission_panel_arrows_and_export_preserve_input_arrays(self):
        example, selected, candidates, risks = self.example()
        before = deepcopy(example)
        target = annotation_target(example["panels"][0])
        self.assertIsNotNone(target)
        self.assertEqual(example["panels"][0]["labels"][0][target], 1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plot_qualitative_supplement(root, [example], [selected], candidates, risks)
            self.assertGreater((root / "figures/qualitative.pdf").stat().st_size, 1000)
            self.assertTrue((root / "figures/qualitative_shared_error_low_disagreement.png").exists())
            arrows = (root / "figures/qualitative_annotations.csv").read_text()
            self.assertIn("display_only_not_clinical_adjudication", arrows)
        for panel, original in zip(example["panels"], before["panels"]):
            np.testing.assert_array_equal(panel["ct"], original["ct"])
            for a, b in zip(panel["labels"], original["labels"]):
                np.testing.assert_array_equal(a, b)

    def test_no_examples_and_frozen_identity_mismatch(self):
        example, selected, candidates, risks = self.example()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plot_qualitative_supplement(root, [], [selected], candidates, risks)
            self.assertTrue((root / "figures/qualitative.png").exists())
            changed = {**selected, "case_id": "999"}
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                plot_qualitative_supplement(root, [example], [changed], candidates, risks)

    def test_budget_plot_handles_null_cis_and_random_ct_cost(self):
        points, curves = [], []
        for source in ("member_mean", "stage2"):
            for stratum in ("pooled", "pelvic", "femur"):
                for policy in ("fa_v2", "presence", "mean_entropy", "partition_v1", "partition_v2", "random"):
                    for retained in (.8, .9):
                        points.append({"split": "heldout", "risk": "merge_split", "stratum": stratum,
                                       "risk_source": source, "score": policy, "requested_coverage": retained,
                                       "coverage": retained, "retained_mean_risk": .4,
                                       "expected_reviewed_cases": 3.5, "ci_low": None, "ci_high": None})
                    curves.extend({"split": "heldout", "risk": "merge_split", "stratum": stratum,
                                   "risk_source": source, "score": policy, "coverage": i / 10,
                                   "prefix_mean_risk": .4} for i in range(1, 11))
        before = deepcopy(points)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plot_budget_figures(root, points, [], curves)
            self.assertTrue((root / "figures/risk_coverage_heldout_merge_split.pdf").exists())
            self.assertTrue((root / "figures/fixed_budget_heldout_merge_split.png").exists())
        self.assertEqual(points, before)


if __name__ == "__main__":
    unittest.main()
