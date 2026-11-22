from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import run_partition_v3 as runner
from test_partition_graph import maps
from exchange_ml.e1_data import MEMBERS
from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.partition_graph import extract_case


class PartitionRunnerTests(unittest.TestCase):
    def fixture(self, folder: Path):
        config = runner.load_config(ROOT / "configs/partition_v3.json")
        parent = folder / "parent"; parent.mkdir()
        (parent / ".e1.lock").touch()
        manifest = parent / "e1_manifest.json"; manifest.write_text("{}")
        config["parent_root"] = str(parent)
        config_path = folder / "config.json"; runner.save_json(config_path, config)
        cases, rows, sources = [], [], {}
        for cid, scan in (("004", "pelvic"), ("254", "femur")):
            cases.append({"case_id": cid, "scan_type": scan})
            paths = maps(folder, scan)
            sources[cid] = {member: {"path": str(path), "sha256": runner.sha256(path)}
                            for member, path in zip(MEMBERS, paths)}
            for features in extract_case(cid, scan, paths, config)["features"]:
                row = {**{n: .1 for n in SCORE_NAMES}, **features}
                for source in ("member_mean", "stage2"):
                    row.update({source + "_" + risk: .2 for risk in
                                ("merge_split", "one_minus_f1", "missing_volume")})
                rows.append(row)
        inputs = {"cases": cases, "rows": rows, "sources": sources,
                  "parent_manifest_sha256": runner.sha256(manifest), "source_tables": {}}
        return config, config_path, inputs

    def test_cache_resume_source_integrity_and_baseline_parity(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            config, _, inputs = self.fixture(folder)
            case = inputs["cases"][0]; sources = inputs["sources"]["004"]
            output = folder / "output"
            payload, digest = runner.process_cached_case(case, sources, inputs["rows"], config, output, "test")
            with patch.object(runner, "extract_case", side_effect=AssertionError("cache should be reused")):
                reloaded, reloaded_digest = runner.process_cached_case(case, sources, inputs["rows"], config, output, "test")
                self.assertEqual(runner.signature(reloaded), runner.signature(payload))
                self.assertEqual(reloaded_digest, digest)
            bad_rows = [dict(row) for row in inputs["rows"]]; bad_rows[0]["presence"] = .99
            with self.assertRaisesRegex(ValueError, "E1 feature mismatch"):
                runner.process_cached_case(case, sources, bad_rows, config, output, "test")
            with self.assertRaisesRegex(ValueError, "identity/content drift"):
                runner.process_cached_case(case, sources, inputs["rows"], config, output, "changed")
            path = Path(sources["member_01"]["path"])
            original = path.read_bytes(); path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner.process_cached_case(case, sources, inputs["rows"], config, output, "test")

    def test_end_to_end_smoke_resume_then_read_only_completion(self):
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            config, config_path, inputs = self.fixture(folder)
            output = folder / "output"
            with patch.object(runner, "source_inputs", return_value=inputs), patch.object(runner, "candidate_inventory", return_value={"confirmation_ready": False}):
                with patch("exchange_ml.partition_cv.nested_cv", side_effect=RuntimeError("simulated interruption after features")):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        runner.run(config, config_path, output, smoke=True)
                state = runner.read_json(output / "partition_manifest.json")
                self.assertEqual(state["status"], "running")
                self.assertEqual(len(state["cache_sha256"]), 2)
                with patch.object(runner, "extract_case", side_effect=AssertionError("resume must use caches")):
                    runner.run(config, config_path, output, smoke=True)
                state = runner.read_json(output / "partition_manifest.json")
                self.assertEqual(state["status"], "complete")
                self.assertEqual(state["summary"]["real_feature_units"], 4)
                self.assertFalse(state["summary"]["fresh_confirmation_ready"])
                self.assertEqual(len(runner.read_csv(output / "oof_scores.csv")), 40)
                before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
                runner.run(config, config_path, output, smoke=True)
                self.assertEqual(before, {p: p.read_bytes() for p in output.rglob("*") if p.is_file()})
                changed = dict(config, role="changed")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    runner.run(changed, config_path, output, smoke=True)
                with (output / "oof_scores.csv").open("a") as handle:
                    handle.write("corruption")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    runner.run(config, config_path, output, smoke=True)

    def test_protected_roots_smoke_separation_and_confirmation_gate(self):
        config = runner.load_config(ROOT / "configs/partition_v3.json")
        for path in (ROOT, ROOT / "outputs", ROOT / config["parent_root"] / "new", ROOT.parent / "dataset",
                     Path("/tmp/Exchange-ML-E6"), Path("/tmp/Exchange-ML-E6/sub"), ROOT / "outputs/external_ribfrac/sub",
                     ROOT / "src/new_outputs"):
            with self.assertRaises(ValueError):
                runner.output_path(config, path, False)
        with self.assertRaises(ValueError):
            runner.output_path(config, ROOT / config["output_root"], True)
        self.assertEqual(runner.output_path(config, Path("/tmp/partition-test"), True), Path("/tmp/partition-test"))
        with TemporaryDirectory() as directory:
            config["confirmation"]["ready"] = True
            path = Path(directory) / "config.json"; runner.save_json(path, config)
            with self.assertRaisesRegex(ValueError, "cannot authorize"):
                runner.load_config(path)

    def test_unrecognized_outputs_are_never_overwritten(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            config, config_path, inputs = self.fixture(folder)
            output = folder / "output"; output.mkdir()
            saved = output / "oof_scores.csv"; saved.write_text("existing evidence")
            with patch.object(runner, "source_inputs", return_value=inputs), patch.object(runner, "candidate_inventory", return_value={}):
                with self.assertRaisesRegex(ValueError, "nonempty output root"):
                    runner.run(config, config_path, output, smoke=True)
            self.assertEqual(saved.read_text(), "existing evidence")

    def test_launcher_dependencies_failure_and_cpu_resources(self):
        for mode, fail_at, expected in (("full", 0, 2), ("smoke", 0, 1), ("full", 1, 1), ("invalid", 0, 0)):
            with self.subTest(mode=mode, fail_at=fail_at), TemporaryDirectory() as directory:
                folder = Path(directory); capture = folder / "calls.json"
                fake = folder / "sbatch"
                fake.write_text(f"#!{sys.executable}\nimport json,os,pathlib,sys\np=pathlib.Path(os.environ['PARTITION_TEST_CAPTURE'])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a))\nif len(a)==int(os.environ['PARTITION_TEST_FAIL_AT']):sys.exit(1)\nprint(str(100+len(a))+';test_cluster')\n")
                fake.chmod(0o755)
                env = dict(os.environ, PATH=str(folder) + os.pathsep + os.environ["PATH"],
                           PARTITION_TEST_CAPTURE=str(capture), PARTITION_TEST_FAIL_AT=str(fail_at))
                result = subprocess.run(["bash", str(ROOT / "scripts/submit_partition_v3.sh"), mode], env=env, text=True, capture_output=True)
                calls = json.loads(capture.read_text()) if capture.exists() else []
                self.assertEqual(len(calls), expected)
                self.assertEqual(result.returncode == 0, mode != "invalid" and fail_at == 0, result.stderr)
                if len(calls) == 2:
                    self.assertEqual(calls[0], ["--parsable", "scripts/partition_v3.sbatch", "smoke"])
                    self.assertEqual(calls[1], ["--parsable", "--dependency=afterok:101", "scripts/partition_v3.sbatch", "full"])
        directives = [line for line in (ROOT / "scripts/partition_v3.sbatch").read_text().splitlines() if line.startswith("#SBATCH")]
        self.assertFalse(any("--gres" in line or "--gpus" in line for line in directives))


if __name__ == "__main__":
    unittest.main()
