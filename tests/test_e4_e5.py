from __future__ import annotations

from itertools import combinations, permutations
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from exchange_ml.e1_data import MEMBERS, bone_risk
from exchange_ml.e1_statistics import expected_curve
from exchange_ml.e4_e5_analysis import (CATEGORIES, enrichment, expected_reviewed_cases, normalization_scope,
    qualitative_candidates, review_budget, runtime_summary, top_weights)
from exchange_ml.e4_e5_data import fragment_coverage, scope_scores
from exchange_ml.e5_visualization import display_alignment, plot_qualitative, prepare_example
from exchange_ml.instances import _anatomy_match
from run_e4_e5 import check_config, dataset_table_rows, read_csv, read_json, signature, write_csv


def config() -> dict:
    return read_json(ROOT / "configs/e4_e5.json")


class E4E5Tests(unittest.TestCase):
    def test_dataset_split_and_numeric_split_risk_survive_export(self):
        rows = [{"case_id": "004", "scan_type": "pelvic", "split": 0., "missing_volume": .25},
                {"case_id": "001", "scan_type": "pelvic", "split": 1., "missing_volume": .3}]
        fixed = [dataset_table_rows([rows[0]], "validation", rename_split_risk=True)[0],
                 dataset_table_rows([rows[1]], "heldout", rename_split_risk=True)[0]]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "blindspots.csv"
            write_csv(path, fixed)
            actual = read_csv(path)
        self.assertEqual([r["split"] for r in actual], ["validation", "heldout"])
        self.assertEqual([r["split_risk"] for r in actual], [0., 1.])
        self.assertEqual([r["missing_volume"] for r in actual], [.25, .3])
        self.assertNotIn("split_risk", rows[0])
        with self.assertRaisesRegex(ValueError, "reserved"):
            dataset_table_rows(rows, "validation")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            dataset_table_rows([{**rows[0], "split_risk": 1}], "heldout", rename_split_risk=True)
        with self.assertRaisesRegex(ValueError, "finite"):
            dataset_table_rows([{"split": "validation"}], "heldout", rename_split_risk=True)

    def test_cutoff_weights_and_distinct_cases_match_exhaustive_sampling(self):
        scores = np.array([3, 2, 2, 2, 1.])
        cases = ["a", "a", "b", "c", "d"]
        for count in range(6):
            orderings = sorted(permutations(range(5)), key=lambda p: tuple(-scores[i] for i in p))
            valid = [p for p in orderings if all(scores[p[i]] >= scores[p[i + 1]] for i in range(4))]
            expected = np.mean([len({cases[i] for i in p[:count]}) for p in valid])
            self.assertAlmostEqual(expected_reviewed_cases(scores, cases, count), expected)
            self.assertAlmostEqual(top_weights(scores, count).sum(), count)
        np.testing.assert_allclose(top_weights(np.ones(4), 1.5), np.full(4, .375))
        with self.assertRaises(ValueError):
            top_weights(scores, 6)

    def test_review_ties_keep_exact_budget_and_curve(self):
        rows = []
        for i, risk in enumerate([0, 1, 2, 3, 4.]):
            rows.append({"case_id": str(i // 3), "scan_type": "pelvic", **{s: 1. for s in (
                "mean_entropy", "p95_entropy", "pairwise_dice", "presence", "union_nvi", "count_variance",
                "hungarian", "fa_v1", "partition_v1", "partition_v2", "fa_v2", "q_joint_unweighted")},
                "member_mean_merge_split": risk, "stage2_merge_split": risk})
        for r in review_budget(rows, config()):
            self.assertEqual(r["retained_units"], int(np.floor(r["requested_coverage"] * 5)))
            self.assertAlmostEqual(r["retained_mean_risk"], 2)
            self.assertAlmostEqual(r["delta_vs_random"], 0)
            self.assertAlmostEqual(r["delta_vs_presence"], 0)
        x, y = np.array([0, 0, 1, 1, 1.]), np.array([0, 2, 1, 3, 4.])
        for retained in range(1, 6):
            keep = 1 - top_weights(x, 5 - retained)
            self.assertAlmostEqual(keep @ y / retained, expected_curve(x, y)[retained - 1])

    def test_fragment_matching_matches_frozen_evaluator_and_union_coverage(self):
        gt = np.array([1, 1, 1, 1, 2, 2, 0, 0], dtype=np.uint8).reshape(1, 2, 4)
        pred = np.array([1, 1, 2, 0, 2, 2, 3, 0], dtype=np.uint8).reshape(gt.shape)
        fragments, risk = fragment_coverage(pred, gt, "sacrum", (1, 2, 3), .1)
        original = _anatomy_match(gt, pred, .1)
        self.assertEqual([(r["gt_id"], r["matched_pred_id"], r["matched_iou"]) for r in fragments],
                         [(r["gt_id"], r["pred_id"], r["iou"]) for r in original["matches"]])
        self.assertEqual(risk["split_errors"], original["split_errors"])
        self.assertEqual(risk["merge_errors"], original["merge_errors"])
        self.assertAlmostEqual(fragments[0]["gt_fraction_covered"], .75)
        self.assertEqual(fragments[0]["gt_volume_mm3"], 24)
        expected = bone_risk(pred, gt, (1, 2, 3), "sacrum", .1)
        for k, v in risk.items():
            self.assertAlmostEqual(v, expected[k])

    def test_perfect_f1_can_hide_volume_loss_and_wrong_anatomy_is_not_coverage(self):
        gt = np.ones((2, 2, 5), dtype=np.uint8)
        pred = gt.copy()
        pred[:, :, 2:] = 151
        fragments, risk = fragment_coverage(pred, gt, "sacrum", (1, 1, 1), .1)
        self.assertEqual(risk["one_minus_f1"], 0)
        self.assertAlmostEqual(risk["missing_volume"], .6)
        self.assertAlmostEqual(fragments[0]["gt_fraction_covered"], .4)

    def test_random_fragment_matching_agrees_with_reference_including_empty(self):
        rng = np.random.default_rng(17)
        for _ in range(15):
            gt = rng.integers(0, 5, size=(2, 3, 4), dtype=np.uint8)
            pred = rng.integers(0, 5, size=gt.shape, dtype=np.uint8)
            fragments, risk = fragment_coverage(pred, gt, "sacrum", (.8, .9, 1), .1)
            reference = _anatomy_match(gt, pred, .1)
            self.assertAlmostEqual(risk["one_minus_f1"], 1 - reference["instance_f1"])
            self.assertEqual([r["matched_pred_id"] for r in fragments], [r["pred_id"] for r in reference["matches"]])
        empty = np.zeros((1, 1, 1), np.uint8)
        rows, risk = fragment_coverage(empty, empty, "sacrum", (1, 1, 1), .1)
        self.assertFalse(rows)
        self.assertIsNone(risk["missing_volume"])

    def test_scope_uses_predicted_anatomy_and_background_is_not_outside(self):
        a = np.array([1, 1, 151, 0, 0], dtype=np.uint8)
        b = np.array([1, 0, 151, 151, 151], dtype=np.uint8)
        result, pairs = scope_scores([a, b, b], "pelvic")
        outside = [r for r in pairs if r["scope"] == "outside_annotation"]
        self.assertAlmostEqual(outside[0]["fraction_outside_scope"], 2 / 3)
        self.assertIsNone(outside[2]["fraction_outside_scope"])
        self.assertEqual(result["n_defined_outside_scope_pairs"], 2)
        self.assertNotEqual(result["all_log_n"], result["annotated_log_n"])
        for p in [r for r in pairs if r["scope"] == "annotated"]:
            self.assertLessEqual(p["n_union_voxels"], 2)
        same = scope_scores([a, a, a], "pelvic")[0]
        self.assertIsNone(same["fraction_outside_scope"])

    def test_case_2x2_only_changes_declared_factor_and_single_case_ci_is_null(self):
        rows = []
        for i in range(4):
            r = {"case_id": str(i), "scan_type": "pelvic" if i < 3 else "femur",
                 "all_log_n": float(i), "all_joint": float(i),
                 "annotated_log_n": float(3 - i), "annotated_joint": float(3 - i),
                 "member_mean_merge_split": float(i), "stage2_merge_split": float(i)}
            r.update({k + "_partition_share": .2 for k in ("all_log_n", "all_joint", "annotated_log_n", "annotated_joint")})
            rows.append(r)
        estimates, contrasts = normalization_scope(rows, 32, 4)
        self.assertEqual(len(estimates), 24)
        for row in contrasts:
            if row["stratum"] == "femur":
                self.assertIsNone(row["ci_low"])
            elif row["contrast"].startswith("normalization"):
                self.assertEqual(row["delta_aurc"], 0)
                self.assertEqual(row["ci_low"], 0)
                self.assertEqual(row["ci_high"], 0)

    def test_constant_risk_enrichment_is_undefined(self):
        from exchange_ml.e1_scoring import SCORE_NAMES
        rows = [{"case_id": str(i), "scan_type": "femur", **{s: i for s in SCORE_NAMES},
                 "member_mean_merge_split": 1., "stage2_merge_split": 0.} for i in range(4)]
        result = enrichment(rows, config())
        self.assertTrue(all(r["enrichment"] is None for r in result))
        self.assertEqual({r["status"] for r in result}, {"constant_risk", "zero_prevalence"})

    def test_no_eligible_example_is_explicit_and_cutoffs_are_not_relaxed(self):
        rows = [{"case_id": str(i), "bone": "femur", "scan_type": "femur",
                 **{s: 0 for s in ("partition_v2", "presence", "mean_entropy", "fa_v2")},
                 **{"member_mean_" + k: 0 for k in ("merge_split", "missing", "missing_volume", "one_minus_f1")}}
                for i in range(4)]
        models = [{"case_id": r["case_id"], "bone": "femur", "model_id": m, "one_minus_f1": 0}
                  for r in rows for m in MEMBERS]
        candidates, selected = qualitative_candidates(rows, models, config())
        self.assertEqual(candidates, [])
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(r["status"] == "no_eligible_case" for r in selected))
        with TemporaryDirectory() as directory:
            plot_qualitative(Path(directory) / "qualitative", [], selected, [], models)
            self.assertTrue((Path(directory) / "qualitative.pdf").is_file())

    def test_qualitative_selection_deterministic_tie_break(self):
        rows = []
        for i in range(8):
            rows.append({"case_id": f"{7-i:03}", "bone": "femur", "scan_type": "femur",
                         "partition_v2": float(i), "presence": float(7-i), "mean_entropy": float(7-i),
                         "fa_v2": float(i), "member_mean_merge_split": 1., "member_mean_missing": 1.,
                         "member_mean_missing_volume": .3, "member_mean_one_minus_f1": .3})
        models = [{"case_id": r["case_id"], "bone": "femur", "model_id": m, "one_minus_f1": .3}
                  for r in rows for m in MEMBERS]
        candidates, selected = qualitative_candidates(rows, models, config())
        self.assertEqual(selected, qualitative_candidates(rows[::-1], models[::-1], config())[1])
        self.assertTrue(all(r["status"] == "selected" for r in selected))
        self.assertEqual(len([r for r in candidates if r["selected"]]), 3)
        self.assertEqual(next(r for r in selected if r["category"] == CATEGORIES[1])["case_id"], "005")

    def test_visualization_preserves_split_ids_and_writes_geometry(self):
        gt = np.zeros((4, 6, 8), dtype=np.uint8)
        gt[:, 1:5, 1:7] = 1
        pred = gt.copy()
        pred[:, :, 4:] *= 2
        display, mapping = display_alignment([pred, gt, gt], gt)
        self.assertEqual(len(np.unique(display[1][display[1] > 0])), 2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for model, labels in [("gt", gt), ("stage2", gt), *zip(MEMBERS, (pred, gt, gt))]:
                p = root / (model + ".mha")
                image = sitk.GetImageFromArray(labels)
                image.SetSpacing((.8, 1, 2))
                sitk.WriteImage(image, str(p))
                paths[model] = p
            image = sitk.Cast(image, sitk.sitkInt16)
            paths["image"] = root / "image.mha"
            sitk.WriteImage(image, str(paths["image"]))
            settings = {**config(), "prune_mm3": 0, "connectivity": 26, "iou_threshold": .1}
            selection = {"case_id": "001", "bone": "sacrum", "category": CATEGORIES[0], "status": "selected"}
            example = prepare_example(selection, paths, settings)
            self.assertEqual(len(example["panels"]), 2)
            self.assertTrue(all(r["center_z_mm"] == r["z_index"] * 2 for r in example["coordinates"]))
            candidates = [{**selection, "selected": True, "fa_v2": .5, "partition_v2": .4, "presence": .1, "mean_entropy": .1}]
            risks = [{"case_id": "001", "bone": "sacrum", "model_id": m, "one_minus_f1": 0, "merge_split": 0} for m in MEMBERS]
            plot_qualitative(root / "qualitative", [example], [selection], candidates, risks)
            self.assertGreater((root / "qualitative.pdf").stat().st_size, 1000)

    def test_missing_runtime_is_not_fabricated_and_ids_survive_csv_roundtrip(self):
        rows = runtime_summary([], [], [])
        missing = [r for r in rows if r["status"] == "not_recorded_separately_in_E3"]
        self.assertTrue(missing)
        self.assertTrue(all(r["median"] is None and r["n"] == 0 for r in missing))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            write_csv(path, [{"case_id": "004", "value": .25}])
            self.assertEqual(read_csv(path), [{"case_id": "004", "value": .25}])
        settings = config()
        check_config(settings)
        settings["qualitative"]["high_percentile"] = .7
        with self.assertRaises(ValueError):
            check_config(settings)


if __name__ == "__main__":
    unittest.main()
