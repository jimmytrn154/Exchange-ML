"""Continuation scheduling and rejection of incomplete/corrupt E6 artifacts."""
from __future__ import annotations

from contextlib import ExitStack
import csv
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
import check_ribfrac_fast as checker


class SubmissionTests(unittest.TestCase):
    def launch(self, *, active: str = "", fail_at: int = 0):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            capture = folder / "calls.json"
            (folder / "squeue").write_text("#!/bin/sh\nprintf '%s' \"$FINISH_ACTIVE\"\n")
            (folder / "sbatch").write_text(
                f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                "p=pathlib.Path(os.environ['FINISH_CAPTURE'])\n"
                "a=json.loads(p.read_text()) if p.exists() else []\n"
                "a.append(sys.argv[1:]); p.write_text(json.dumps(a))\n"
                "if len(a)==int(os.environ['FINISH_FAIL']):sys.exit(1)\n"
                "print(str(800+len(a))+';cluster')\n")
            for name in ("squeue", "sbatch"):
                (folder / name).chmod(0o755)
            env = dict(os.environ, PATH=str(folder) + os.pathsep + os.environ["PATH"],
                       FINISH_CAPTURE=str(capture), FINISH_ACTIVE=active, FINISH_FAIL=str(fail_at))
            result = subprocess.run(["bash", str(ROOT / "scripts/finish_ribfrac_fast.sh")],
                                    env=env, capture_output=True, text=True)
            return result, json.loads(capture.read_text()) if capture.exists() else []

    def test_two_gpu_inference_then_cpu_evaluation_and_audit(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([c[-1] for c in calls], ["preflight", "infer", "evaluate", "audit"])
        for i, call in enumerate(calls):
            self.assertEqual("--gres=gpu:1" in call, i == 1)
            self.assertEqual("--array=0-2%2" in call, i == 1)
            if i:
                self.assertIn(f"--dependency=afterok:{800+i}", call)
                self.assertIn("--kill-on-invalid-dep=yes", call)
        self.assertNotIn("cluster", result.stdout)

    def test_existing_writer_blocks_all_submissions(self):
        result, calls = self.launch(active="ribfrac-fast-infer\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_submission_failure_stops_downstream_jobs(self):
        result, calls = self.launch(fail_at=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 2)


class AuditTests(unittest.TestCase):
    def fixture(self, folder: Path, stack: ExitStack) -> Path:
        r = checker.runner
        identity = {"test_campaign": "frozen"}
        members = {f"member_0{i}": 32000 + i for i in (1, 2, 3)}
        config = {"output_root": str(folder), "dataset_root": str(folder),
                  "members": members, "backend": {"epochs": 1000}, "storage": {"reserve_gib": 0}}
        image, label, metadata = [folder / name for name in ("image", "label", "metadata")]
        for path in (image, label, metadata):
            path.write_text(path.name)
        cases = {f"RibFrac{i}": {"split": "validation", "image": image, "label": label,
                                 "metadata": metadata} for i in range(421, 501)}
        stack.enter_context(patch.object(r, "load_config", return_value=config))
        stack.enter_context(patch.object(r, "output_path", return_value=folder))
        stack.enter_context(patch.object(r, "identity", return_value=identity))
        stack.enter_context(patch.object(r, "inventory", return_value=cases))
        stack.enter_context(patch.object(checker, "header_voxels", return_value=100))
        training = {m: {"final_epoch": 1000, "checkpoint": {"sha256": m}} for m in members}
        stack.enter_context(patch.object(r, "verify_training", side_effect=lambda c, p, m: training[m]))
        r.save_json(folder / "handoff.json", {"signature": r.signature(identity)})
        inference_hashes = {}
        for member in members:
            model = folder / "models" / member
            model.mkdir(parents=True)
            (model / "training_manifest.json").write_text("{}")
            records = {}
            for cid in cases:
                saved = {"source_image": {"sha256": r.sha256(image)}}
                for kind, dirname in (("prediction", "predictions"), ("probability", "probabilities")):
                    path = model / dirname / f"{cid}.mha"
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(kind)
                    saved[kind] = {"sha256": r.sha256(path)}
                records[cid] = saved
            provenance = {**identity, "member": member,
                "training_manifest_sha256": r.sha256(model / "training_manifest.json"),
                "checkpoint_sha256": member, "case_ids": sorted(cases)}
            path = model / "inference_manifest.json"
            r.save_json(path, {"status": "complete", "signature": r.signature(provenance),
                               "provenance": provenance, "records": records})
            inference_hashes[member] = r.sha256(path)
        records = {}
        for cid in cases:
            cache, scores = folder / "cache" / f"{cid}.json", folder / "cache" / f"{cid}.scores.json"
            r.save_json(cache, {}); r.save_json(scores, {})
            records[cid] = {"gt": {"sha256": r.sha256(label)}, "cache_sha256": r.sha256(cache),
                            "score_cache_sha256": r.sha256(scores)}
        with (folder / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id", *checker.SCORE_NAMES])
            writer.writeheader()
            writer.writerows({"case_id": cid, **dict.fromkeys(checker.SCORE_NAMES, 0)} for cid in cases)
        summary = {"smoke_only": False}
        r.save_json(folder / "summary.json", summary)
        provenance = {**identity, "inference_manifest_sha256": inference_hashes,
                      "external_metadata_sha256": {str(metadata): r.sha256(metadata)}}
        r.save_json(folder / "e6_manifest.json", {"status": "complete", "signature": r.signature(provenance),
            "provenance": provenance, "records": records, "summary": summary,
            "output_sha256": {name: r.sha256(folder / name) for name in ("scores.csv", "summary.json")}})
        return folder

    def test_complete_240_record_audit(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            self.fixture(Path(directory), stack)
            self.assertEqual(checker.check(audit=True)["inference_records"], 240)

    def test_corrupt_prediction_rejected(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            folder = self.fixture(Path(directory), stack)
            (folder / "models/member_01/predictions/RibFrac421.mha").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                checker.check(audit=True)

    def test_incomplete_member_rejected(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            folder = self.fixture(Path(directory), stack)
            path = folder / "models/member_01/inference_manifest.json"
            state = checker.runner.read_json(path)
            state["records"].pop("RibFrac421")
            checker.runner.save_json(path, state)
            with self.assertRaisesRegex(ValueError, "80-case inference"):
                checker.check(audit=True)


if __name__ == "__main__":
    unittest.main()
