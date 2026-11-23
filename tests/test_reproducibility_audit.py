from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_reproducibility import case_id, exposure_audit, inspect_metadata, output_path


class ReproducibilityAuditTests(unittest.TestCase):
    def test_outcome_rows_are_exposure_but_gt_stats_are_not(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out").mkdir()
            (root / "out/risks.csv").write_text(
                "case_id,member_mean_merge_split,member_mean_one_minus_f1\n005,0,0.2\n006,,\n007,1,1\n")
            (root / "out/gt.csv").write_text("case_id,n_GT,scan_type\n006,12,pelvic\n")
            report = exposure_audit(root, {"005", "006"}, ("out",))
            self.assertEqual(report["known_recorded_outcome_case_ids"], ["005"])
            self.assertEqual(report["case_status"]["006"], "no_record_found_exposure_unknown")
            self.assertFalse(report["confirmation_ready"])
            self.assertFalse(report["exposure_audit_completed"])

    def test_manifest_case_lists_are_membership_not_model_outcomes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "membership_manifest.json"
            p.write_text(json.dumps({"case_ids": ["005", "006"], "status": "complete"}))
            self.assertEqual(inspect_metadata(p, {"005", "006"}), [])
            p.write_text(json.dumps({"rows": [{"case_id": "005", "prediction": "005.mha",
                                               "status": "COMPLETE"}]}))
            hits = inspect_metadata(p, {"005"})
            self.assertEqual(hits[0]["kind"], "prediction_record_not_proof_outcomes_viewed")
            p.write_text(json.dumps({"rows": [{"case_id": "005", "instance_f1": 0}]}))
            self.assertEqual(inspect_metadata(p, {"005"})[0]["kind"], "recorded_model_outcomes")

    def test_absence_and_unreadable_history_never_authorize_confirmation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out").mkdir()
            (root / "out/broken_manifest.json").write_text("bad json")
            report = exposure_audit(root, {"005"}, ("out", "missing-history"))
            self.assertFalse(report["confirmation_ready"])
            self.assertTrue(report["unrecorded_history_unknown"])
            self.assertEqual(report["missing_roots"], ["missing-history"])
            self.assertEqual(len(report["skipped_metadata"]), 1)
            self.assertEqual(report["case_status"]["005"], "no_record_found_exposure_unknown")

    def test_prediction_presence_does_not_decode_volume_or_follow_directory_symlinks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "out/predictions").mkdir(parents=True)
            # Invalid image bytes intentionally prove no image reader is used.
            (root / "out/predictions/005.mha").write_bytes(b"not an image")
            (root / "external").mkdir()
            (root / "external/risks.csv").write_text("case_id,instance_f1\n006,0\n")
            (root / "out/external-link").symlink_to(root / "external", target_is_directory=True)
            report = exposure_audit(root, {"005", "006"}, ("out",))
            self.assertEqual(report["prediction_evidence_case_ids"], ["005"])
            self.assertEqual(report["known_recorded_outcome_case_ids"], [])
            self.assertEqual(report["case_status"]["006"], "no_record_found_exposure_unknown")

    def test_output_guard_rejects_sources_campaigns_and_parent_namespace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "outputs/research_followup_v1/preflight"
            self.assertEqual(output_path(root, allowed), allowed.resolve())
            for path in (root, root / "__docs__", root / "outputs/stage3_h1_v2",
                         root / "outputs/research_followup_v1", root / "outputs/external_ribfrac"):
                with self.assertRaises(ValueError):
                    output_path(root, path)
            allowed.parent.mkdir(parents=True)
            allowed.symlink_to(root / "__docs__", target_is_directory=True)
            with self.assertRaises(ValueError):
                output_path(root, allowed)

    def test_case_identifiers_preserve_zero_padding_without_parsing_other_fields(self):
        self.assertEqual(case_id(5), "005")
        self.assertEqual(case_id("005"), "005")
        for value in (None, True, "RibFrac5", "foo", "1000"):
            self.assertIsNone(case_id(value))


if __name__ == "__main__":
    unittest.main()
