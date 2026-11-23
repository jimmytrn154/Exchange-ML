"""Separate reporting never treats repeated models as independent patients."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ribfrac_cohorts_report", ROOT / "scripts/report_ribfrac_cohorts.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def example_tables(count: int = 3):
    risks, models = [], []
    for i in range(count):
        n_gt = 2 if i < count * 3 // 4 else 0
        structural = {name: 0.25 if n_gt else None for name in report.STRUCTURAL}
        common = {"case_id": f"RibFrac{421+i}", "n_GT": n_gt,
                  **{"member_mean_" + k: v for k, v in structural.items()},
                  "member_mean_false_positive_instances": 1,
                  "member_mean_one_minus_f1": 0.25 if n_gt else 1,
                  "member_mean_one_minus_dice": 0.1 if n_gt else 1}
        risks.append(common)
        for member in report.MEMBERS:
            models.append({"case_id": common["case_id"], "model_id": member,
                           "n_GT": n_gt, "n_pred": 3 if n_gt else 1,
                           "false_positive_instances": 1,
                           "one_minus_f1": common["member_mean_one_minus_f1"],
                           "one_minus_dice": common["member_mean_one_minus_dice"], **structural})
    return risks, models


class CohortTests(unittest.TestCase):
    def test_negative_cts_are_counted_once_and_structural_risk_is_undefined(self):
        risks, models = example_tables()
        rows, summary = report.cohort_tables(risks, models)
        self.assertEqual(summary["n_cases"], 3)
        self.assertEqual(summary["n_negative_cases"], 1)
        negative = next(r for r in rows if r["cohort"] == "negative" and r["model"] == "member_mean")
        self.assertEqual(negative["n_cases"], 1)
        self.assertEqual(negative["sum_fp_instances"], 1)
        self.assertIsNone(negative["mean_merge_split"])
        self.assertEqual(negative["mean_instance_f1"], 0)

    def test_member_mean_fp_probability_is_not_a_consensus_prediction(self):
        risks, models = example_tables()
        negatives = [row for row in models if not row["n_GT"]]
        for row, count in zip(negatives, (0, 0, 3)):
            row.update(n_pred=count, false_positive_instances=count)
        rows, _ = report.cohort_tables(risks, models)
        negative = next(r for r in rows if r["cohort"] == "negative" and r["model"] == "member_mean")
        self.assertAlmostEqual(negative["fraction_CT_with_any_FP"], 1/3)
        self.assertEqual(negative["mean_fp_instances_per_CT"], 1)

    def test_missing_models_duplicate_rows_and_false_positive_drift_are_rejected(self):
        risks, models = example_tables()
        for bad in (models[:-1], models + [models[0]]):
            with self.assertRaises(ValueError):
                report.cohort_tables(risks, bad)
        risks[0]["member_mean_false_positive_instances"] = 100
        with self.assertRaisesRegex(ValueError, "member-mean"):
            report.cohort_tables(risks, models)

    def test_empty_subgroup_has_explicit_undefined_results(self):
        risks, models = example_tables()
        risks = [r for r in risks if r["n_GT"]]
        ids = {r["case_id"] for r in risks}
        models = [r for r in models if r["case_id"] in ids]
        rows, _ = report.cohort_tables(risks, models)
        self.assertTrue(all(r["status"] == "undefined_no_cases" for r in rows if r["cohort"] == "negative"))

    def test_all_active_or_frozen_roots_are_rejected(self):
        source = Path("/tmp/Exchange-ML-E6-fast-v1").resolve()
        for dest in (source, source / "reports", source.parent,
                     ROOT / "__docs__", ROOT / "outputs/stage3_h1_v2/new", ROOT.parent / "dataset"):
            with self.assertRaises(ValueError):
                report.validate_output(dest.resolve(), source)

    def test_completed_pipeline_is_idempotent_and_detects_corrupted_csv(self):
        with TemporaryDirectory() as folder:
            base = Path(folder)
            source, output = base / "source", base / "reports"
            source.mkdir()
            risks, models = example_tables(80)
            for name, rows in (("risks.csv", risks), ("model_risks.csv", models)):
                with (source / name).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
            summary = {"smoke_only": False, "n_positive_gt_cases": 60, "n_negative_gt_cases": 20}
            (source / "summary.json").write_text(json.dumps(summary))
            provenance = {"role": "synthetic_unit_test_only"}
            state = {"status": "complete", "provenance": provenance, "signature": report.signature(provenance),
                     "summary": summary, "output_sha256": {name: report.sha(source / name)
                         for name in ("risks.csv", "model_risks.csv", "summary.json")}}
            (source / "e6_manifest.json").write_text(json.dumps(state))
            with patch.object(sys, "argv", ["report", "--source-root", str(source), "--output-root", str(output)]):
                report.main()
                before = (output / "manifest.json").read_bytes()
                report.main()
                self.assertEqual((output / "manifest.json").read_bytes(), before)
                with (source / "risks.csv").open("a") as handle:
                    handle.write("corrupt\n")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    report.main()

    def test_incomplete_e6_is_not_reported_as_finished(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "not complete yet"):
                report.completed_sources(Path(folder))


if __name__ == "__main__":
    unittest.main()
