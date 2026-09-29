#!/usr/bin/env python3
"""Build split-safe nnU-Net raw inputs for the frozen Stage 2 backend.

The complete 80-case conversion is a user-typed full preprocessing command.
``--case-limit`` exists only for a representative smoke check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from exchange_ml.stage2_backend import (  # noqa: E402
    ANATOMY_LABELS,
    ANATOMY_RANGES,
    bbox_from_mask,
    bone_lut_normalize,
    contiguous_anatomy_instances,
    discover_case_directories,
    select_stage2_case_ids,
    semantic_anatomy_target,
    validate_grouped_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_backend.json"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def canonicalize(image: sitk.Image) -> sitk.Image:
    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    return image if orientation == "LPS" else sitk.DICOMOrient(image, "LPS")


def assert_matching_geometry(image: sitk.Image, label: sitk.Image, case_id: str) -> None:
    if image.GetSize() != label.GetSize():
        raise RuntimeError(f"{case_id}: size mismatch")
    for name, left, right in (
        ("spacing", image.GetSpacing(), label.GetSpacing()),
        ("origin", image.GetOrigin(), label.GetOrigin()),
        ("direction", image.GetDirection(), label.GetDirection()),
    ):
        if not np.allclose(left, right, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"{case_id}: {name} mismatch")


def image_from_array(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def crop_image(image: sitk.Image, bbox_zyx: tuple[slice, slice, slice]) -> sitk.Image:
    z_slice, y_slice, x_slice = bbox_zyx
    index_xyz = [int(x_slice.start), int(y_slice.start), int(z_slice.start)]
    size_xyz = [
        int(x_slice.stop - x_slice.start),
        int(y_slice.stop - y_slice.start),
        int(z_slice.stop - z_slice.start),
    ]
    return sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)


def write_image(image: sitk.Image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def dataset_jsons(max_instances: int) -> tuple[dict[str, Any], dict[str, Any]]:
    anatomy = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, **ANATOMY_LABELS},
        "numTraining": 0,
        "file_ending": ".mha",
        "description": "PENGWIN Stage 2 five-class anatomy localization; frozen split only.",
    }
    fragments = {
        "channel_names": {"0": "nonorm"},
        "labels": {
            "background": 0,
            **{f"fragment_{index:02d}": index for index in range(1, max_instances + 1)},
        },
        "numTraining": 0,
        "file_ending": ".mha",
        "description": (
            "PENGWIN Stage 2 per-anatomy CT-LUT crops with contiguous instance labels; "
            "ABBC and affinity targets are generated inside the pinned V308 trainer."
        ),
    }
    return anatomy, fragments


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    split = load_json(Path(config["split"]["manifest"]))
    selected = select_stage2_case_ids(split, int(config["split"]["trajectory_seed"]))
    cases = discover_case_directories(Path(config["paths"]["dataset_root"]))
    output_root = args.output_root or Path(config["paths"]["workspace_root"]) / "nnUNet_raw"
    anatomy_cfg = config["anatomy_stage"]
    fragment_cfg = config["fragment_stage"]
    anatomy_root = output_root / anatomy_cfg["dataset_name"]
    fragment_root = output_root / fragment_cfg["dataset_name"]
    anatomy_json, fragment_json = dataset_jsons(
        max_instances=int(fragment_cfg["maximum_local_instance_id"])
    )
    rows: list[dict[str, Any]] = []
    split_samples = {
        "anatomy": {"train": [], "validation": []},
        "fragment": {"train": [], "validation": []},
    }

    for role in args.roles:
        case_ids = selected[role]
        if args.case_limit is not None:
            if args.case_limit <= 0:
                raise ValueError("--case-limit must be positive")
            case_ids = case_ids[: args.case_limit]
        for case_id in case_ids:
            case_dir = cases[case_id]
            image = canonicalize(sitk.ReadImage(str(case_dir / "image.mha")))
            label = canonicalize(sitk.ReadImage(str(case_dir / "label.mha")))
            assert_matching_geometry(image, label, case_id)
            image_array = np.clip(
                sitk.GetArrayFromImage(image).astype(np.float32), -1000.0, 2000.0
            )
            instances = sitk.GetArrayFromImage(label).astype(np.int16)

            anatomy_id = f"PENGWIN_{case_id}"
            anatomy_image = image_from_array(image_array, image)
            anatomy_target = image_from_array(semantic_anatomy_target(instances), label)
            write_image(
                anatomy_image,
                anatomy_root / "imagesTr" / f"{anatomy_id}_0000.mha",
                args.overwrite,
            )
            write_image(
                anatomy_target,
                anatomy_root / "labelsTr" / f"{anatomy_id}.mha",
                args.overwrite,
            )
            split_samples["anatomy"][role].append(anatomy_id)

            lut_image = image_from_array(bone_lut_normalize(image_array), image)
            for anatomy, (lower, upper) in ANATOMY_RANGES.items():
                anatomy_mask = (instances >= lower) & (instances <= upper)
                bbox = bbox_from_mask(anatomy_mask, int(fragment_cfg["roi_pad_voxels"]))
                if bbox is None:
                    continue
                local_instances, mapping = contiguous_anatomy_instances(instances, anatomy)
                sample_id = f"PENGWIN_{case_id}_{anatomy}"
                cropped_image = crop_image(lut_image, bbox)
                local_image = image_from_array(local_instances, label)
                cropped_label = crop_image(local_image, bbox)
                write_image(
                    cropped_image,
                    fragment_root / "imagesTr" / f"{sample_id}_0000.mha",
                    args.overwrite,
                )
                write_image(
                    cropped_label,
                    fragment_root / "labelsTr" / f"{sample_id}.mha",
                    args.overwrite,
                )
                split_samples["fragment"][role].append(sample_id)
                rows.append(
                    {
                        "role": role,
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "anatomy": anatomy,
                        "source_instance_to_local": {
                            str(key): value for key, value in mapping.items()
                        },
                        "bbox_zyx": [[int(value.start), int(value.stop)] for value in bbox],
                    }
                )

    validate_grouped_samples(
        split_samples["fragment"]["train"],
        split_samples["fragment"]["validation"],
    )
    anatomy_json["numTraining"] = sum(
        len(values) for values in split_samples["anatomy"].values()
    )
    fragment_json["numTraining"] = sum(
        len(values) for values in split_samples["fragment"].values()
    )
    for root, payload in ((anatomy_root, anatomy_json), (fragment_root, fragment_json)):
        root.mkdir(parents=True, exist_ok=True)
        (root / "dataset.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "config": str(args.config),
        "trajectory_seed": int(config["split"]["trajectory_seed"]),
        "case_limit": args.case_limit,
        "roles": list(args.roles),
        "split_samples": split_samples,
        "fragment_rows": rows,
        "test_case_ids": selected["test"],
        "test_data_read": False,
    }
    manifest_path = output_root.parent / "stage2_preparation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage 2 preparation complete: {manifest_path}")
    print(f"anatomy samples={anatomy_json['numTraining']}")
    print(f"fragment samples={fragment_json['numTraining']}")
    print("test data read=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
