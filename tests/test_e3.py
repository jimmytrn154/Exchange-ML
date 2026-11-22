from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e3_reporting import primary_result, top_weights
from run_e3 import file_state, verify_record, write_checked


class E3Tests(unittest.TestCase):
    def test_volume_write_round_trips_and_corrupt_resume_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = sitk.Image([2, 3, 4], sitk.sitkInt16)
            reference.SetSpacing([.8, .9, 1.1])
            source = root / "image.mha"
            sitk.WriteImage(reference, str(source))
            label = np.zeros((4, 3, 2), dtype=np.uint16)
            label[1, 1, 1] = 151
            prediction, probability = root / "prediction.mha", root / "probability.mha"
            record = {"image_state": file_state(source), "outputs": {
                "prediction": write_checked(prediction, label, reference, probability=False),
                "probability": write_checked(probability, label.astype(np.float32) / 200, reference, probability=True),
            }}
            files = {"prediction": prediction, "probability": probability}
            verify_record(record, source, files)
            data = probability.read_bytes()
            probability.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_record(record, source, files)
            with self.assertRaisesRegex(ValueError, "unexpected whole-volume empty"):
                write_checked(prediction, np.zeros_like(label), reference, probability=False)

    def test_enrichment_cutoff_shares_ties(self):
        np.testing.assert_allclose(top_weights(np.array([3, 2, 2, 1]), 2), [1, .5, .5, 0])
        np.testing.assert_allclose(top_weights(np.array([1, 1, 1]), .75), [.25, .25, .25])

    def test_primary_requires_both_cis_below_zero_and_effect_floor(self):
        protocol = {"primary": {"baselines": ["presence", "mean_entropy"], "risk": "merge_split", "max_delta_aurc": -.02}}
        rows = [{"stratum": "pooled", "risk_source": source, "risk": "merge_split", "baseline": baseline,
                 "delta_aurc": -.02, "ci_low": -.05, "ci_high": -.001}
                for source in ("member_mean", "stage2") for baseline in ("presence", "mean_entropy")]
        self.assertTrue(primary_result(rows, protocol, False)["robust_across_risk_sources"])
        rows[-1]["ci_high"] = 0.
        self.assertFalse(primary_result(rows, protocol, False)["robust_across_risk_sources"])
        rows[0]["delta_aurc"] = -.0199
        self.assertFalse(primary_result(rows, protocol, False)["member_mean"]["meets_rule"])
        self.assertIsNone(primary_result(rows, protocol, True)["member_mean"]["meets_rule"])


if __name__ == "__main__":
    unittest.main()
