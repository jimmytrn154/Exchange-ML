from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_partition_v4 as runner


def config_for_fixture(folder: Path) -> dict:
    return {"version": 1, "candidate": "pairwise_p1_g", "features": ["partition_v1", "graph_degree"],
            "target": "member_mean_merge_split", "parent_root": "parent_v3", "output_root": "outputs/full",
            "smoke_output_root": "outputs/smoke", "confirmation_ready": False,
            "cv": {"seed": 20261122, "outer_folds": 5, "inner_folds": 3, "alphas": [.001, .01, .1, 1.],
                   "max_iter": 1000, "tolerance": 1e-9},
            "reporting": {"bootstrap_resamples": 10000, "bootstrap_seed": 20261122,
                          "primary_estimand": "mean_outer_fold_aurc"}}


def source_fixture(folder: Path) -> dict:
    """Small-file fixture with all 40 CTs but no volume/checkpoint files."""
    (folder / "configs").mkdir()
    (folder / "parent_e1/validation").mkdir(parents=True)
    parent = folder / "parent_v3"; parent.mkdir()
    (parent / ".partition.lock").touch()
    cases, rows = [], []
    for i in range(40):
        scan = "pelvic" if i < 20 else "femur"
        cid = f"case_{i:03}"
        cases.append({"case_id": cid, "scan_type": scan})
        for bone in (("sacrum", "left_hip", "right_hip") if scan == "pelvic" else ("femur",)):
            rows.append({"case_id": cid, "bone": bone, "scan_type": scan,
                         **{name: .05 for name in runner.SCORE_NAMES}, "graph_degree": .1,
                         **{name: .2 for name in runner.RISKS}, "all_members_empty": False})
    folds = runner.grouped_folds(rows, 5, 20261122)
    rows = [dict(row, outer_fold=folds[row["case_id"]], n_GT=999) for row in rows]
    runner.write_csv(folder / "configs/cases.csv", cases)
    runner.save_json(folder / "configs/split.json", {"split": {"validation": [c["case_id"] for c in cases]}})
    runner.save_json(folder / "configs/e1.json", {"split": "configs/split.json", "scan_metadata": "configs/cases.csv"})
    runner.write_csv(folder / "parent_e1/validation/scores.csv", rows)
    runner.write_csv(folder / "parent_e1/validation/risks.csv", rows)
    e1prov = {"config_sha256": runner.sha256(folder / "configs/e1.json"),
              "split_sha256": runner.sha256(folder / "configs/split.json"),
              "scan_metadata_sha256": runner.sha256(folder / "configs/cases.csv"), "code_sha256": {}}
    tables = {f"validation/{name}.csv": runner.sha256(folder / f"parent_e1/validation/{name}.csv") for name in ("scores", "risks")}
    runner.save_json(folder / "parent_e1/e1_manifest.json", {"status": "complete", "smoke_only": False,
                     "provenance": e1prov, "signature": runner.signature(e1prov),
                     "case_ids": [c["case_id"] for c in cases], "output_sha256": tables})
    runner.write_csv(parent / "oof_scores.csv", rows)
    runner.write_csv(parent / "graph_features.csv", rows)
    v3config = {"parent_root": "parent_e1", "parent_config": "configs/e1.json", "cv": {"outer_folds": 5, "seed": 20261122}}
    runner.save_json(folder / "configs/partition_v3.json", v3config)
    prov = {"smoke_only": False, "config": v3config,
            "config_sha256": runner.sha256(folder / "configs/partition_v3.json"),
            "parent_manifest_sha256": runner.sha256(folder / "parent_e1/e1_manifest.json"),
            "source_tables": tables, "code_sha256": {}}
    runner.save_json(parent / "partition_manifest.json", {"status": "complete", "summary": {"smoke_only": False},
                     "provenance": prov, "signature": runner.signature(prov), "cache_sha256": {},
                     "output_sha256": {n: runner.sha256(parent / n) for n in ("oof_scores.csv", "graph_features.csv")}})
    config = config_for_fixture(folder)
    runner.save_json(folder / "configs/partition_v4.json", config)
    return config


def fake_cv(rows: list[dict], config: dict) -> dict:
    assignment = runner.grouped_folds(rows, config["cv"]["outer_folds"], config["cv"]["seed"])
    return {"oof_scores": [dict(row, outer_fold=assignment[row["case_id"]], pairwise_p1_g=row["partition_v1"]) for row in rows],
            "folds": [{"case_id": cid, "outer_fold": fold} for cid, fold in assignment.items()],
            "tuning": [{"alpha": .1}], "fold_models": [
                {"outer_fold": fold, "test_case_ids": [cid for cid, f in assignment.items() if f == fold],
                 "model": {"train_case_ids": [cid for cid, f in assignment.items() if f != fold]}}
                for fold in sorted(set(assignment.values()))],
            "final_model": {"head": "pairwise_p1_g", "features": ["partition_v1", "graph_degree"],
                            "train_case_ids": sorted(assignment)},
            "summary": {"n_cases": len(assignment), "n_units": len(rows)}}


class StructuralRankRunnerTests(unittest.TestCase):
    def test_source_audit_projection_and_corruption_detection(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory); config = source_fixture(folder)
            with patch.object(runner, "ROOT", folder):
                inputs = runner.source_inputs(config)
                self.assertEqual(len(inputs["cases"]), 40)
                self.assertEqual(len(inputs["rows"]), 80)
                self.assertFalse(any("n_GT" in row or "outer_fold" in row for row in inputs["rows"]))
                path = folder / "parent_v3/graph_features.csv"
                path.write_text(path.read_text() + "corruption")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    runner.source_inputs(config)

    def test_source_rejects_wrong_membership_even_with_rehashed_parent(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory); config = source_fixture(folder)
            path = folder / "parent_v3/graph_features.csv"
            rows = runner.read_csv(path); rows[0]["case_id"] = "foreign"
            runner.write_csv(path, rows)
            manifest = folder / "parent_v3/partition_manifest.json"
            saved = runner.read_json(manifest); saved["output_sha256"][path.name] = runner.sha256(path)
            runner.save_json(manifest, saved)
            with patch.object(runner, "ROOT", folder), self.assertRaisesRegex(ValueError, "exact 80"):
                runner.source_inputs(config)

    def test_resume_checkpoint_then_completed_run_is_read_only(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory); config = source_fixture(folder)
            config_path = folder / "configs/partition_v4.json"
            for name in runner.CODE:
                path = folder / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fixture-code")
            model_module = ModuleType("exchange_ml.structural_rank"); model_module.nested_cv = fake_cv
            model_module.predict_model = lambda model, rows: runner.np.array([r["partition_v1"] for r in rows])
            report_module = ModuleType("exchange_ml.structural_rank_reporting")
            report_module.report = lambda rows, settings: {"comparison": [{"score": "pairwise_p1_g", "aurc": .2}], "summary": {"synthetic": True}}
            with patch.object(runner, "ROOT", folder), patch.dict(sys.modules, {
                    "exchange_ml.structural_rank": model_module, "exchange_ml.structural_rank_reporting": report_module}):
                output = folder / "outputs/smoke"; output.parent.mkdir()
                with patch.object(report_module, "report", side_effect=RuntimeError("interrupted reporting")):
                    with self.assertRaisesRegex(RuntimeError, "interrupted reporting"):
                        runner.run(config, config_path, output, smoke=True)
                state = runner.read_json(output / "rank_manifest.json")
                self.assertEqual(state["stage"], "cv_complete")
                checkpoint = output / "cv_checkpoint.json"
                saved_checkpoint = checkpoint.read_bytes(); checkpoint.unlink()
                with patch.object(model_module, "nested_cv", side_effect=AssertionError("deleted recorded checkpoint must not refit")):
                    with self.assertRaises(FileNotFoundError):
                        runner.run(config, config_path, output, smoke=True)
                checkpoint.write_bytes(saved_checkpoint)
                with patch.object(model_module, "nested_cv", side_effect=AssertionError("checkpoint should prevent refit")):
                    runner.run(config, config_path, output, smoke=True)
                    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
                    runner.run(config, config_path, output, smoke=True)
                    self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()})
                self.assertTrue(runner.read_json(output / "final_model.json")["smoke_only"])
                self.assertEqual(len(runner.read_csv(output / "oof_scores.csv")), 40)
                self.assertTrue(all(r["case_id"].startswith("synthetic_") for r in runner.read_csv(output / "oof_scores.csv")))
                changed = dict(config, role="changed")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    runner.run(changed, config_path, output, smoke=True)
                (output / "comparison.csv").write_text("corruption")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    runner.run(config, config_path, output, smoke=True)

    def test_checkpoint_schema_rejects_rehashed_input_drift(self):
        rows = runner.synthetic_rows(); config = config_for_fixture(ROOT)
        cv = fake_cv(rows, config)
        module = ModuleType("exchange_ml.structural_rank")
        module.predict_model = lambda model, rows: runner.np.array([r["partition_v1"] for r in rows])
        with patch.dict(sys.modules, {"exchange_ml.structural_rank": module}):
            runner.validate_cv_payload(cv, rows, config)
            cv["oof_scores"][0]["partition_v1"] += .1
            with self.assertRaisesRegex(ValueError, "inputs or fold identity"):
                runner.validate_cv_payload(cv, rows, config)

    def test_unrecognized_existing_outputs_are_preserved(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory); config = source_fixture(folder)
            output = folder / "out"; output.mkdir(); (output / "evidence.csv").write_text("preserve")
            model = ModuleType("exchange_ml.structural_rank"); model.nested_cv = fake_cv
            reporting = ModuleType("exchange_ml.structural_rank_reporting"); reporting.report = None
            with patch.object(runner, "ROOT", folder), patch.dict(sys.modules, {
                    "exchange_ml.structural_rank": model, "exchange_ml.structural_rank_reporting": reporting}), patch.object(runner, "provenance", return_value={}):
                with self.assertRaisesRegex(ValueError, "nonempty output"):
                    runner.run(config, folder / "configs/partition_v4.json", output, smoke=True)
            self.assertEqual((output / "evidence.csv").read_text(), "preserve")

    def test_output_protection_and_confirmation_gate(self):
        config = config_for_fixture(ROOT)
        for path in (ROOT, ROOT / "outputs", ROOT / "outputs/stage3_h1_v2/sub", ROOT / "src/sub",
                     ROOT.parent / "dataset", Path("/tmp/Exchange-ML-E6/sub"), ROOT / "parent_v3/sub",
                     ROOT / "outputs/smoke/sub"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.output_path(config, path, False)
        self.assertEqual(runner.output_path(config, Path("/tmp/v4-fixture"), False), Path("/tmp/v4-fixture"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            runner.save_json(path, dict(config, confirmation_ready=True))
            with self.assertRaisesRegex(ValueError, "cannot authorize"):
                runner.load_config(path)

    def test_launcher_cpu_dependency_and_submission_failures(self):
        for mode, fail_at, expected in (("full", 0, 2), ("smoke", 0, 1), ("full", 1, 1), ("invalid", 0, 0)):
            with self.subTest(mode=mode, fail_at=fail_at), TemporaryDirectory() as directory:
                folder = Path(directory); capture = folder / "calls.json"
                fake = folder / "sbatch"
                fake.write_text(f"#!{sys.executable}\nimport json,os,pathlib,sys\np=pathlib.Path(os.environ['RANK_TEST_CAPTURE'])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a))\nif len(a)==int(os.environ['RANK_TEST_FAIL']):sys.exit(1)\nprint(str(100+len(a))+';test_cluster')\n")
                fake.chmod(0o755)
                env = dict(os.environ, PATH=str(folder) + os.pathsep + os.environ["PATH"],
                           RANK_TEST_CAPTURE=str(capture), RANK_TEST_FAIL=str(fail_at))
                result = subprocess.run(["bash", str(ROOT / "scripts/submit_partition_v4.sh"), mode], env=env, text=True, capture_output=True)
                calls = json.loads(capture.read_text()) if capture.exists() else []
                self.assertEqual(len(calls), expected)
                self.assertEqual(result.returncode == 0, fail_at == 0 and mode != "invalid", result.stderr)
                if len(calls) == 2:
                    self.assertEqual(calls, [["--parsable", "scripts/partition_v4.sbatch", "smoke"],
                                             ["--parsable", "--dependency=afterok:101", "scripts/partition_v4.sbatch", "full"]])
        directives = [line for line in (ROOT / "scripts/partition_v4.sbatch").read_text().splitlines() if line.startswith("#SBATCH")]
        self.assertFalse(any("--gres" in line or "--gpus" in line for line in directives))


if __name__ == "__main__":
    unittest.main()
