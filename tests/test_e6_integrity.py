"""Tiny file-level E6 integration checks; no models or real cohort is run."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_e6
from exchange_ml.e6_backend import fingerprint, save_json, sha256, signature, write_volume


class EvaluationFixture:
    """Three tiny cases and signed inference records; training alone is mocked."""

    def __init__(self, directory: Path):
        self.root = directory / "outputs"
        self.root.mkdir()
        self.source = directory / "dataset"
        self.source.mkdir()
        self.config = deepcopy(run_e6.read_json(run_e6.ROOT / "configs/e6.json"))
        self.config["dataset_root"] = str(self.source)
        self.config["evaluation"]["bootstrap_resamples"] = 32
        self.metadata = self.source / "ribfrac-val-info.csv"
        with self.metadata.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["public_id", "label_id", "label_code"])
            for case_id, codes in (("RibFrac421", {0: 0}), ("RibFrac422", {0: 0, 1: -1, 2: 1}),
                                   ("RibFrac423", {0: 0, 1: 4})):
                writer.writerows((case_id, i, c) for i, c in codes.items())
        self.records = {}
        self.gts = {}
        for i in range(3):
            cid = f"RibFrac{421+i}"
            gt = np.zeros((4, 5, 6), np.uint32)
            if i:
                gt[1, 1, 1:3] = 1
            if i == 1:
                gt[2, 3, 3:5] = 2
            image = sitk.GetImageFromArray(np.arange(gt.size, dtype=np.float32).reshape(gt.shape))
            image.SetSpacing((.7, .8, 1.5))
            image.SetOrigin((2., 3., 4.))
            image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
            image_path = self.source / f"{cid}-image.mha"
            label_path = self.source / f"{cid}-label.mha"
            write_volume(image_path, sitk.GetArrayFromImage(image), image)
            write_volume(label_path, gt, image)
            self.records[cid] = {"case_id": cid, "split": "validation", "image": image_path,
                "label": label_path, "metadata": self.metadata,
                "label_codes": ({0: 0} if i == 0 else {0: 0, 1: -1, 2: 1} if i == 1 else {0: 0, 1: 4})}
            self.gts[cid] = gt
        self.training = {}
        for index, member in enumerate(self.config["members"]):
            folder = self.root / "models" / member
            folder.mkdir(parents=True)
            checkpoint_hash = f"{index+1:064x}"
            training = {"checkpoint": {"sha256": checkpoint_hash}}
            self.training[member] = training
            save_json(folder / "training_manifest.json", training)
            provenance = {**run_e6.identity(self.config, self.root), "member": member,
                "training_manifest_sha256": sha256(folder / "training_manifest.json"),
                "checkpoint_sha256": checkpoint_hash, "case_ids": sorted(self.records)}
            inference = {"status": "complete", "signature": signature(provenance),
                         "provenance": provenance, "records": {}}
            for cid, record in self.records.items():
                pred = self.gts[cid].copy()
                if index == 1:
                    pred[pred > 0] = 501  # generic merge; IDs carry no anatomy
                if index == 2:
                    pred[:] = 0
                    if cid == "RibFrac421":
                        pred[1, 2, 3] = 1001  # false positive in negative case
                probability = np.where(pred > 0, .8 + .05 * index, .1).astype(np.float32)
                image = sitk.ReadImage(str(record["image"]))
                pred_fp = write_volume(folder / "predictions" / f"{cid}.mha", pred, image)
                prob_fp = write_volume(folder / "probabilities" / f"{cid}.mha", probability, image)
                inference["records"][cid] = {"source_image": fingerprint(record["image"]),
                                              "prediction": pred_fp, "probability": prob_fp}
            save_json(folder / "inference_manifest.json", inference)

    def patched(self):
        stack = ExitStack()
        stack.enter_context(patch.object(run_e6, "inventory", return_value=self.records))
        stack.enter_context(patch.object(run_e6, "verify_training",
                                        side_effect=lambda config, root, member: self.training[member]))
        return stack

    def run(self):
        with self.patched():
            run_e6.evaluate(self.config, self.root)

    def bytes(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def assert_score_persisted(self, case_id: str):
        # The persisted record must contain the actual score, not just an empty
        # file or the run's identity manifest. Permit the runner's chosen cache
        # layout while verifying semantic contents before GT voxel reads.
        for path in self.root.rglob("*.json"):
            if case_id not in path.name and "score" not in str(path.relative_to(self.root)):
                continue
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                row = data.get("scores", data.get("score", data))
                if isinstance(row, list):
                    candidates = row
                else:
                    candidates = [row]
                if any(isinstance(r, dict) and "fa_v2" in r and "presence" in r
                       and r.get("case_id", case_id) == case_id for r in candidates):
                    return
        raise AssertionError(f"No semantic GT-free score record persisted before reading {case_id} GT")


class E6EvaluationIntegrityTests(unittest.TestCase):
    def test_real_volume_evaluation_persists_scores_before_gt_and_is_byte_idempotent(self):
        with TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            actual_read = sitk.ReadImage
            labels = {str(r["label"]): cid for cid, r in fixture.records.items()}
            observed = []

            def read_with_guard(path, *args, **kwargs):
                if str(path) in labels:
                    cid = labels[str(path)]
                    fixture.assert_score_persisted(cid)
                    observed.append(cid)
                return actual_read(path, *args, **kwargs)

            with fixture.patched(), patch.object(run_e6.sitk, "ReadImage", side_effect=read_with_guard):
                run_e6.evaluate(fixture.config, fixture.root)
            self.assertEqual(set(observed), set(fixture.records))
            summary = json.loads((fixture.root / "summary.json").read_text())
            self.assertEqual(summary["n_cases"], 3)
            self.assertEqual(summary["n_positive_gt_cases"], 2)
            self.assertEqual(summary["n_negative_gt_cases"], 1)
            self.assertEqual(summary["bootstrap_resamples"], 32)
            with (fixture.root / "model_risks.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            negative = [r for r in rows if r["case_id"] == "RibFrac421"]
            self.assertTrue(all(r["merge_split"] == "" for r in negative))
            before = fixture.bytes()
            fixture.run()
            self.assertEqual(before, fixture.bytes())

    def test_completed_evaluation_refuses_source_gt_and_metadata_drift(self):
        for kind in ("gt", "metadata"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory))
                fixture.run()
                if kind == "gt":
                    path = fixture.records["RibFrac423"]["label"]
                    image = sitk.ReadImage(str(path))
                    changed = sitk.GetArrayFromImage(image)
                    changed[0, 0, 0] = 1
                    write_volume(path, changed, image)
                else:
                    fixture.metadata.write_text(fixture.metadata.read_text() + "\n")
                before = fixture.bytes()
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.run()
                self.assertEqual(before, fixture.bytes())

    def test_completed_evaluation_refuses_prediction_probability_and_source_image_drift(self):
        for kind in ("prediction", "probability", "image"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                fixture = EvaluationFixture(Path(directory))
                fixture.run()
                if kind == "image":
                    path = fixture.records["RibFrac421"]["image"]
                else:
                    folder = "predictions" if kind == "prediction" else "probabilities"
                    path = fixture.root / "models/member_01" / folder / "RibFrac421.mha"
                image = sitk.ReadImage(str(path))
                changed = sitk.GetArrayFromImage(image)
                changed[0, 0, 0] = .2 if kind == "probability" else 1
                write_volume(path, changed, image)
                before = fixture.bytes()
                with self.assertRaises((ValueError, RuntimeError)):
                    fixture.run()
                self.assertEqual(before, fixture.bytes())

    def test_all_outputs_with_same_wrong_geometry_are_rejected_against_source_ct(self):
        with TemporaryDirectory() as directory:
            fixture = EvaluationFixture(Path(directory))
            # Shift prediction/probability and GT together; pairwise geometry
            # checks alone would pass, but source CT geometry remains original.
            cid = "RibFrac422"
            paths = [fixture.records[cid]["label"]]
            paths += [fixture.root / "models" / member / folder / f"{cid}.mha"
                      for member in fixture.config["members"] for folder in ("predictions", "probabilities")]
            for path in paths:
                image = sitk.ReadImage(str(path))
                image.SetOrigin((102., 3., 4.))
                write_volume(path, sitk.GetArrayFromImage(image), image)
            for member in fixture.config["members"]:
                path = fixture.root / "models" / member / "inference_manifest.json"
                inference = json.loads(path.read_text())
                for name, folder in (("prediction", "predictions"), ("probability", "probabilities")):
                    inference["records"][cid][name] = fingerprint(path.parent / folder / f"{cid}.mha")
                save_json(path, inference)
            with self.assertRaisesRegex(ValueError, "[Gg]eometry"):
                fixture.run()


if __name__ == "__main__":
    unittest.main()
