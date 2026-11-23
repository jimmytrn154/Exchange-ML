"""Exercise the real follow-up launcher with a fake scheduler, never Slurm."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FollowupSubmissionTests(unittest.TestCase):
    def launch(self, mode: str, fail_at: int = 0, invalid_at: int = 0):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            capture = folder / "calls.json"
            executable = folder / "sbatch"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json,os,pathlib,sys\n"
                "p=pathlib.Path(os.environ['FOLLOWUP_CAPTURE'])\n"
                "calls=json.loads(p.read_text()) if p.exists() else []\n"
                "calls.append(sys.argv[1:]);p.write_text(json.dumps(calls))\n"
                "n=len(calls)\n"
                "if n==int(os.environ['FOLLOWUP_FAIL_AT']):sys.exit(1)\n"
                "print('invalid' if n==int(os.environ['FOLLOWUP_INVALID_AT']) else str(1000+n)+';cluster')\n"
            )
            executable.chmod(0o755)
            env = dict(os.environ, PATH=str(folder) + os.pathsep + os.environ["PATH"],
                       FOLLOWUP_CAPTURE=str(capture), FOLLOWUP_FAIL_AT=str(fail_at),
                       FOLLOWUP_INVALID_AT=str(invalid_at))
            result = subprocess.run(["bash", str(ROOT / "scripts/submit_research_followup.sh"), mode],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
            return result, json.loads(capture.read_text()) if capture.exists() else []

    def test_full_runs_preflight_then_independent_smokes_and_dependent_fulls(self):
        result, calls = self.launch("full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0], ["--parsable", "scripts/research_followup.sbatch"])
        for index in (1, 2):
            self.assertIn("--dependency=afterok:1001", calls[index])
            self.assertEqual(calls[index][-1], "smoke")
        self.assertIn("--dependency=afterok:1002", calls[3])
        self.assertIn("--dependency=afterok:1003", calls[4])
        self.assertEqual(calls[3][-2:], ["scripts/count_rank.sbatch", "full"])
        self.assertEqual(calls[4][-2:], ["scripts/evaluator_sensitivity.sbatch", "full"])
        self.assertFalse(any("e6" in arg or "ribfrac" in arg or "--gres" in arg
                             for call in calls for arg in call))

    def test_check_and_smoke_do_not_submit_full(self):
        for mode, count in (("check", 1), ("smoke", 3)):
            result, calls = self.launch(mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), count)
            self.assertFalse(any(call[-1] == "full" for call in calls))

    def test_submission_failure_or_malformed_id_stops_downstream(self):
        for failure, invalid in ((2, 0), (0, 2)):
            result, calls = self.launch("full", failure, invalid)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(calls), 2)

    def test_invalid_mode_never_calls_scheduler(self):
        result, calls = self.launch("bad")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(calls)

    def test_cpu_preflight_has_no_gpu_request_or_cancellation(self):
        source = (ROOT / "scripts/research_followup.sbatch").read_text()
        directives = [line for line in source.splitlines() if line.startswith("#SBATCH")]
        self.assertFalse(any("--gres" in line or "--gpus" in line for line in directives))
        self.assertNotIn("scancel", source)


if __name__ == "__main__":
    unittest.main()
