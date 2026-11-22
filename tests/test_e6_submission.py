"""Exercise the real E6 launcher with a local fake scheduler; never submit jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class E6SubmissionTests(unittest.TestCase):
    def launch(self, mode: str, *, fail_at: int = 0) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            capture = folder / "submissions.json"
            executable = folder / "sbatch"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, sys\n"
                "p = pathlib.Path(os.environ['E6_SBATCH_CAPTURE'])\n"
                "calls = json.loads(p.read_text()) if p.exists() else []\n"
                "calls.append(sys.argv[1:])\n"
                "p.write_text(json.dumps(calls))\n"
                "if len(calls) == int(os.environ.get('E6_FAKE_FAIL_AT', '0')):\n"
                "    sys.exit(1)\n"
                "print(str(1000 + len(calls)) + ';test_cluster')\n"
            )
            executable.chmod(0o755)
            env = dict(os.environ, PATH=str(folder) + os.pathsep + os.environ["PATH"],
                       E6_SBATCH_CAPTURE=str(capture), E6_FAKE_FAIL_AT=str(fail_at),
                       E6_OUTPUT_ROOT=str(folder / "full"), E6_SMOKE_ROOT=str(folder / "smoke"))
            result = subprocess.run(["bash", str(ROOT / "scripts/submit_e6.sh"), mode],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            calls = json.loads(capture.read_text()) if capture.exists() else []
            return result, calls

    def test_full_dependencies_and_two_concurrent_independent_gpu_tasks(self):
        result, calls = self.launch("full")
        self.assertEqual(result.returncode, 0, result.stderr)
        stages = ["smoke", "prepare", "preprocess", "train", "infer", "evaluate"]
        self.assertEqual(len(calls), len(stages))
        for index, (stage, call) in enumerate(zip(stages, calls)):
            self.assertIn("--parsable", call)
            self.assertEqual(call[-1], f"scripts/e6_{stage}.sbatch")
            if index:
                self.assertIn(f"--dependency=afterok:{1000 + index}", call)
            self.assertEqual("--array=0-2%2" in call, stage in ("train", "infer"))
        self.assertNotIn("test_cluster", result.stdout.split("E6 jobs:")[-1])

    def test_smoke_submits_only_one_synthetic_job(self):
        result, calls = self.launch("smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [["--parsable", "scripts/e6_smoke.sbatch"]])

    def test_submission_failure_does_not_launch_downstream_jobs(self):
        result, calls = self.launch("full", fail_at=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 3)
        self.assertFalse(any("scripts/e6_train.sbatch" in call for call in calls))

    def test_invalid_mode_does_not_contact_scheduler(self):
        result, calls = self.launch("invalid")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_cpu_stages_do_not_request_gpus(self):
        for stage in ("prepare", "preprocess", "evaluate"):
            source = (ROOT / f"scripts/e6_{stage}.sbatch").read_text()
            directives = [line for line in source.splitlines() if line.startswith("#SBATCH")]
            self.assertFalse(any("--gres" in line or "--gpus" in line for line in directives))
        for stage in ("train", "infer", "smoke"):
            source = (ROOT / f"scripts/e6_{stage}.sbatch").read_text()
            self.assertIn("#SBATCH --gres=gpu:1", source)
            self.assertNotIn("CUDA_VISIBLE_DEVICES=", source)


if __name__ == "__main__":
    unittest.main()
