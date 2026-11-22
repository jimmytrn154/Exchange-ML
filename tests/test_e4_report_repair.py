"""Reporting-only correction preserves experiment evidence and rejects drift."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_e4_e5 import read_csv, read_json, save_json, sha256, signature, write_csv
from repair_e4_split_column import REVISION, TABLE, repair_report


class E4ReportRepairTests(unittest.TestCase):
    def fixture(self, root: Path, *, empty: bool = False) -> tuple[dict, Path, dict]:
        config_path = root / "config.json"
        save_json(config_path, {"blindspot_f1_tolerance": 1e-12, "blindspot_missing_volume": .1})
        archive = root / "report_corrections" / REVISION
        archive.mkdir(parents=True)
        runner = archive / "run_e4_e5.before.py"
        runner.write_bytes(b"# original experiment runner\n")
        sources, hashes, legacy = {}, {}, []
        for split, cid, risk in (("validation", "004", 0.), ("heldout", "001", 1.)):
            rows = [{"case_id": cid, "scan_type": "pelvic", "bone": "sacrum", "model_id": "member_01",
                     "one_minus_f1": .5 if empty else 0., "missing_volume": .25, "split": risk}]
            path = root / (split + "_source.csv")
            write_csv(path, rows)
            sources[split] = path
            hashes[split + "/model_risks"] = sha256(path)
            if not empty:
                legacy.append({"split": split, **rows[0], "stratum": "pelvic"})
        write_csv(root / TABLE, legacy, ("split", "case_id", "missing_volume"))
        cache = root / "case_cache" / "fixture.json"
        save_json(cache, {"result": "untouched"})
        manifest = {"status": "complete", "provenance": {
            "config_sha256": sha256(config_path), "upstream_tables": hashes,
            "code_sha256": {"scripts/run_e4_e5.py": sha256(runner)}},
            "summary": {"n_blindspot_model_units": len(legacy)},
            "input_sha256": {"fixture": "untouched"}, "output_sha256": {TABLE: sha256(root / TABLE)},
            "cache_sha256": {"case_cache/fixture.json": sha256(cache)}}
        manifest["signature"] = signature(manifest["provenance"])
        save_json(root / "e4_e5_manifest.json", manifest)
        return sources, config_path, manifest

    def test_repair_preserves_original_values_provenance_caches_and_is_idempotent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources, config_path, original = self.fixture(root)
            before = (root / "e4_e5_manifest.json").read_bytes()
            cache_bytes = (root / "case_cache/fixture.json").read_bytes()
            self.assertEqual(repair_report(root, sources, config_path, check_only=True)["status"], "correction_needed")
            self.assertEqual(before, (root / "e4_e5_manifest.json").read_bytes())
            result = repair_report(root, sources, config_path)
            self.assertEqual(result["status"], "corrected")
            rows = read_csv(root / TABLE)
            self.assertEqual([r["split"] for r in rows], ["validation", "heldout"])
            self.assertEqual([r["split_risk"] for r in rows], [0., 1.])
            self.assertEqual([r["case_id"] for r in rows], ["004", "001"])
            self.assertEqual([r["missing_volume"] for r in rows], [.25, .25])
            manifest = read_json(root / "e4_e5_manifest.json")
            for key in ("signature", "provenance", "summary", "cache_sha256", "input_sha256"):
                self.assertEqual(manifest[key], original[key])
            self.assertEqual(cache_bytes, (root / "case_cache/fixture.json").read_bytes())
            self.assertEqual(before, (root / "report_corrections" / REVISION / "e4_e5_manifest.before.json").read_bytes())
            snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(repair_report(root, sources, config_path)["status"], "already_corrected")
            self.assertEqual(snapshot, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_empty_table_includes_split_risk_schema(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources, config_path, _ = self.fixture(root, empty=True)
            repair_report(root, sources, config_path)
            self.assertEqual(read_csv(root / TABLE), [])
            self.assertIn("split_risk", (root / TABLE).read_text().splitlines()[0].split(","))

    def test_interrupted_correction_finishes_from_staged_receipt_and_csv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources, config_path, _ = self.fixture(root)
            original_manifest = (root / "e4_e5_manifest.json").read_bytes()
            repair_report(root, sources, config_path)
            revised_manifest = (root / "e4_e5_manifest.json").read_bytes()
            # Simulate a stop after the receipt/CSV were written, before manifest commit.
            (root / "e4_e5_manifest.json").write_bytes(original_manifest)
            self.assertEqual(repair_report(root, sources, config_path)["status"], "corrected")
            self.assertEqual((root / "e4_e5_manifest.json").read_bytes(), revised_manifest)

    def test_source_output_and_parent_drift_are_rejected_before_mutation(self):
        for target in ("source", "output", "parent"):
            with self.subTest(target=target), TemporaryDirectory() as directory:
                root = Path(directory)
                sources, config_path, original = self.fixture(root)
                manifest_path = root / "e4_e5_manifest.json"
                if target == "source":
                    sources["validation"].write_text("corrupted\n")
                elif target == "output":
                    (root / TABLE).write_text("corrupted\n")
                else:
                    altered = deepcopy(original)
                    altered["provenance"]["config_sha256"] = "invalid"
                    save_json(manifest_path, altered)
                before = manifest_path.read_bytes()
                with self.assertRaises(ValueError):
                    repair_report(root, sources, config_path)
                self.assertEqual(before, manifest_path.read_bytes())
                self.assertFalse((root / "report_corrections" / REVISION / "correction.json").exists())


if __name__ == "__main__":
    unittest.main()
