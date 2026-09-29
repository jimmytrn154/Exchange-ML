"""Audit PENGWIN instance connectivity and physical-volume reference thresholds.

The full 340-case command is intentionally user-owned under ``__docs__/rule.md``.
Use ``--max-cases 1`` only for a representative implementation smoke test.
The thresholds in this descriptive audit do not filter or rewrite GT labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from exchange_ml.instances import connectivity_structure


CONNECTIVITIES = (6, 18, 26)


def discover_labels(dataset_root: Path) -> list[Path]:
    labels = sorted(dataset_root.glob("PENGWIN26_task1_2_train_part*/*/label.mha"))
    if not labels:
        raise FileNotFoundError(
            f"no PENGWIN label.mha files found under {dataset_root}"
        )
    case_ids = [path.parent.name for path in labels]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs found across PENGWIN training parts")
    return labels


def audit_label_array(
    labels: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    component_prune_min_volume_mm3: float,
    gt_fragment_min_volume_mm3: float,
) -> dict[str, Any]:
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError(f"expected a 3D label array, got {values.shape}")
    label_ids = sorted(int(value) for value in np.unique(values) if int(value) > 0)
    invalid_ids = [label_id for label_id in label_ids if label_id > 200]
    if invalid_ids:
        raise ValueError(f"labels outside the documented 1..200 range: {invalid_ids}")

    voxel_volume_mm3 = float(np.prod(spacing_zyx))
    if voxel_volume_mm3 <= 0:
        raise ValueError(f"invalid spacing {spacing_zyx}")
    component_threshold_voxels = max(
        1,
        int(np.ceil(component_prune_min_volume_mm3 / voxel_volume_mm3)),
    )
    objects = ndi.find_objects(values, max_label=max(label_ids, default=0))
    per_connectivity: dict[str, dict[str, Any]] = {
        str(connectivity): {
            "component_count": 0,
            "disconnected_instance_count": 0,
            "small_component_count": 0,
            "small_component_voxels": 0,
            "component_count_by_id": {},
        }
        for connectivity in CONNECTIVITIES
    }
    fragment_volumes_mm3: dict[str, float] = {}

    for label_id in label_ids:
        label_slice = objects[label_id - 1]
        if label_slice is None:
            raise RuntimeError(f"label {label_id} has no bounding box")
        instance_mask = values[label_slice] == label_id
        fragment_volumes_mm3[str(label_id)] = float(
            instance_mask.sum() * voxel_volume_mm3
        )
        for connectivity in CONNECTIVITIES:
            components, count = ndi.label(
                instance_mask,
                structure=connectivity_structure(connectivity),
            )
            sizes = np.bincount(components.ravel())[1:]
            small = sizes[sizes < component_threshold_voxels]
            result = per_connectivity[str(connectivity)]
            result["component_count"] += int(count)
            result["disconnected_instance_count"] += int(count > 1)
            result["small_component_count"] += int(len(small))
            result["small_component_voxels"] += int(small.sum())
            result["component_count_by_id"][str(label_id)] = int(count)

    foreground_voxels = int(np.count_nonzero(values))
    for result in per_connectivity.values():
        result["small_component_fraction_of_foreground"] = float(
            result["small_component_voxels"] / max(foreground_voxels, 1)
        )

    return {
        "shape_zyx": list(values.shape),
        "spacing_zyx_mm": [float(value) for value in spacing_zyx],
        "foreground_voxels": foreground_voxels,
        "label_ids": label_ids,
        "fragment_count": len(label_ids),
        "fragment_volumes_mm3": fragment_volumes_mm3,
        "gt_ids_below_min_volume": [
            int(label_id)
            for label_id, volume in fragment_volumes_mm3.items()
            if volume < gt_fragment_min_volume_mm3
        ],
        "connectivity": per_connectivity,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"case_count": len(records), "connectivity": {}}
    for connectivity in CONNECTIVITIES:
        key = str(connectivity)
        rows = [record["connectivity"][key] for record in records]
        summary["connectivity"][key] = {
            "component_count": int(sum(row["component_count"] for row in rows)),
            "disconnected_instance_count": int(
                sum(row["disconnected_instance_count"] for row in rows)
            ),
            "small_component_count": int(
                sum(row["small_component_count"] for row in rows)
            ),
            "small_component_voxels": int(
                sum(row["small_component_voxels"] for row in rows)
            ),
            "cases_with_disconnected_instances": int(
                sum(row["disconnected_instance_count"] > 0 for row in rows)
            ),
            "cases_with_small_components": int(
                sum(row["small_component_count"] > 0 for row in rows)
            ),
        }
    summary["gt_ids_below_min_volume"] = int(
        sum(len(record["gt_ids_below_min_volume"]) for record in records)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--component-prune-min-volume-mm3",
        type=float,
        default=1000.0,
        help="Report components below this threshold; do not alter GT.",
    )
    parser.add_argument(
        "--gt-fragment-min-volume-mm3",
        type=float,
        default=500.0,
        help="Report complete GT IDs below this threshold; do not filter them.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke-test limit only. Omit for the required full 340-case audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.component_prune_min_volume_mm3 < 0 or args.gt_fragment_min_volume_mm3 < 0:
        raise ValueError("physical-volume thresholds must be non-negative")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive")

    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "SimpleITK is required for the Stage 1 dataset audit"
        ) from error

    label_paths = discover_labels(args.dataset_root)
    selected = (
        label_paths[: args.max_cases] if args.max_cases is not None else label_paths
    )
    records: list[dict[str, Any]] = []
    for label_path in selected:
        image = sitk.ReadImage(str(label_path))
        labels = sitk.GetArrayFromImage(image)
        record = audit_label_array(
            labels,
            tuple(reversed(tuple(float(value) for value in image.GetSpacing()))),
            component_prune_min_volume_mm3=args.component_prune_min_volume_mm3,
            gt_fragment_min_volume_mm3=args.gt_fragment_min_volume_mm3,
        )
        record["case_id"] = label_path.parent.name
        record["label_path"] = str(label_path)
        records.append(record)

    payload = {
        "dataset_root": str(args.dataset_root.resolve()),
        "discovered_case_count": len(label_paths),
        "analyzed_case_count": len(records),
        "complete_dataset": len(records) == len(label_paths) == 340,
        "connectivities": list(CONNECTIVITIES),
        "component_prune_min_volume_mm3": args.component_prune_min_volume_mm3,
        "gt_fragment_min_volume_mm3": args.gt_fragment_min_volume_mm3,
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "discovered_case_count",
                    "analyzed_case_count",
                    "complete_dataset",
                    "summary",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
