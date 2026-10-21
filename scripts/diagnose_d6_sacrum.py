#!/usr/bin/env python3
"""D6 diagnosis: where does the retrained Stage 2 lose the sacrum?

Reruns the exact validation-inference pipeline on a few cases and records, per
anatomy, the voxel count after each Stage B step: model foreground, affinity
decode, resampling to the original grid, and painting into the label map.
Writes only under ``--out``; real predictions and manifests are not touched.

User-typed (single GPU, a few minutes per case). Ground truth is read only to
report coverage after the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from exchange_ml.stage2_backend import discover_case_directories  # noqa: E402
from predict_stage2_backend import load_configured_backend  # noqa: E402
from stage2_backend import check, load_config  # noqa: E402

ABBC = ("bg", "border", "boundary", "core")
ANATOMY_TAG = re.compile(r"^\[(Sacrum|LeftHip|RightHip|Femur)\] (.*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_backend.json"))
    parser.add_argument("--cases", nargs="+", default=["030", "047", "060", "004"])
    parser.add_argument("--out", type=Path, default=Path("outputs/stage3_h1/diagnostics/d6"))
    return parser.parse_args()


def instance_summary(labels: np.ndarray) -> dict:
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    return {"voxels": int(counts.sum()), "instances": int(ids.size),
            "largest": sorted((int(c) for c in counts), reverse=True)[:8]}


class Recorder:
    """Hooks the backend's own functions; the pipeline logic is not changed."""

    def __init__(self, backend):
        self.backend = backend
        self.rows: dict[str, dict] = {}
        self.pending_bbox = None
        self.pending_resample = None
        self._log = backend.log
        self._bbox = backend.bbox_from_mask
        self._resample = backend.resample_label_map_to_original

    def install(self) -> None:
        backend = self.backend

        def bbox_from_mask(*args, **kwargs):
            box = self._bbox(*args, **kwargs)
            self.pending_bbox = box
            return box

        def resample_label_map_to_original(label_pp, *args, **kwargs):
            out = self._resample(label_pp, *args, **kwargs)
            self.pending_resample = (np.asarray(label_pp).copy(), np.asarray(out).copy())
            return out

        def log(msg: str) -> None:
            self._log(msg)
            match = ANATOMY_TAG.match(msg)
            if not match:
                return
            anatomy, text = match.groups()
            row = self.rows.setdefault(anatomy, {})
            if text.startswith("bbox crop=") and self.pending_bbox is not None:
                row["bbox"] = [[s.start, s.stop] for s in self.pending_bbox]
            elif text.startswith("painted ") and self.pending_resample is not None:
                decoded_pp, decoded_crop = self.pending_resample
                numbers = re.findall(r"\d+", text)
                row["after_decode_pp"] = instance_summary(decoded_pp)
                row["after_resample_crop"] = instance_summary(decoded_crop)
                row["pp_to_original_scale"] = float(decoded_crop.size) / float(decoded_pp.size)
                row["painted_voxels"] = int(numbers[0])
                row["painted_fragments"] = int(numbers[1])
                row["_arrays"] = (decoded_pp, decoded_crop)
                self.pending_resample = None
            elif "emit zero" in text or "no fragments" in text:
                row["note"] = text

        backend.bbox_from_mask = bbox_from_mask
        backend.resample_label_map_to_original = resample_label_map_to_original
        backend.log = log


def model_foreground(probs_path: Path) -> dict:
    probs = np.load(probs_path).astype(np.float32)
    abbc = probs[:4]
    argmax = abbc.argmax(0)
    return {
        "fg_prob_gt_0.5_voxels_pp": int(((1.0 - abbc[0]) > 0.5).sum()),
        "argmax_class_voxels_pp": {name: int((argmax == i).sum()) for i, name in enumerate(ABBC)},
        "mean_short_affinity_in_fg": float(probs[4:7][:, (1.0 - abbc[0]) > 0.5].mean())
        if ((1.0 - abbc[0]) > 0.5).any() else None,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    check(config)
    backend = load_configured_backend(config)
    recorder = Recorder(backend)
    recorder.install()
    cases = discover_case_directories(Path(config["paths"]["dataset_root"]))
    report = {}
    for case_id in args.cases:
        case_dir = args.out / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PENGWIN_DUMP_PROBS"] = str(case_dir.resolve())
        recorder.rows = {}
        source = cases[case_id] / "image.mha"
        reference = sitk.ReadImage(str(source))
        prediction = backend.run_per_anatomy(source, reference, prerouted_bone_masks=None, anatomies=None)
        backend.write_label_map(prediction, reference, case_dir / "prediction.mha")
        rows = recorder.rows
        for anatomy, row in rows.items():
            arrays = row.pop("_arrays", None)
            probs_path = case_dir / f"probs13_{anatomy}.npy"
            if probs_path.is_file():
                fg = model_foreground(probs_path)
                fg["fg_prob_gt_0.5_voxels_original_est"] = int(
                    fg["fg_prob_gt_0.5_voxels_pp"] * row.get("pp_to_original_scale", float("nan"))
                )
                row["model_foreground"] = fg
                if anatomy != "Sacrum":
                    probs_path.unlink()  # keep only the Sacrum dump (~200 MB each)
            if arrays is not None and anatomy == "Sacrum":
                np.savez_compressed(case_dir / "sacrum_decoded.npz",
                                    decoded_pp=arrays[0].astype(np.int32),
                                    decoded_crop=arrays[1].astype(np.int32))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(cases[case_id] / "label.mha")))
        sacrum_gt = (gt >= 1) & (gt <= 50)
        if sacrum_gt.any():
            pred = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "prediction.mha")))
            rows.setdefault("Sacrum", {})["final_vs_gt"] = {
                "gt_sacrum_voxels": int(sacrum_gt.sum()),
                "final_sacrum_label_voxels": int(((pred >= 1) & (pred <= 50)).sum()),
                "gt_sacrum_covered_by_any_label": float(((pred > 0) & sacrum_gt).sum() / sacrum_gt.sum()),
            }
        report[case_id] = rows
        (args.out / "d6_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        sac = rows.get("Sacrum", {})
        print(f"[d6] {case_id} Sacrum: model_fg_est={sac.get('model_foreground', {}).get('fg_prob_gt_0.5_voxels_original_est')}"
              f" decode_pp={sac.get('after_decode_pp', {}).get('voxels')}"
              f" resampled={sac.get('after_resample_crop', {}).get('voxels')}"
              f" painted={sac.get('painted_voxels')} gt={sac.get('final_vs_gt', {}).get('gt_sacrum_voxels')}",
              flush=True)
    print(f"[d6] report: {args.out / 'd6_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
