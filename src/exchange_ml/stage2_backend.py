"""Split-safe data helpers for the frozen PENGWIN Stage 2 backend."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


ANATOMY_RANGES: dict[str, tuple[int, int]] = {
    "Sacrum": (1, 50),
    "LeftHip": (51, 100),
    "RightHip": (101, 150),
    "Femur": (151, 200),
}
ANATOMY_LABELS = {"Sacrum": 1, "LeftHip": 2, "RightHip": 3, "Femur": 4}
BONE_LUT_X = np.asarray([-1000.0, -200.0, 150.0, 700.0, 1500.0, 2000.0], dtype=np.float32)
BONE_LUT_Y = np.asarray([0.0, 0.05, 0.35, 0.70, 0.92, 1.0], dtype=np.float32)


def semantic_anatomy_target(instance_map: np.ndarray) -> np.ndarray:
    """Map subject-local instance IDs to the five-class anatomy target."""
    instances = np.asarray(instance_map)
    invalid = (instances < 0) | (instances > 200)
    if invalid.any():
        raise ValueError(f"instance labels outside 0..200: {np.unique(instances[invalid]).tolist()}")
    target = np.zeros(instances.shape, dtype=np.uint8)
    for anatomy, (lower, upper) in ANATOMY_RANGES.items():
        target[(instances >= lower) & (instances <= upper)] = ANATOMY_LABELS[anatomy]
    return target


def bone_lut_normalize(image: np.ndarray) -> np.ndarray:
    """Apply the frozen ABBC piecewise-linear CT mapping to ``[0, 1]``."""
    values = np.asarray(image, dtype=np.float32)
    return np.interp(
        np.clip(values, BONE_LUT_X[0], BONE_LUT_X[-1]), BONE_LUT_X, BONE_LUT_Y
    ).astype(np.float32)


def bbox_from_mask(mask: np.ndarray, pad_voxels: int) -> tuple[slice, slice, slice] | None:
    """Return a clipped ``(z, y, x)`` bounding box around a non-empty mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"expected a 3D mask, got {binary.shape}")
    coordinates = np.argwhere(binary)
    if coordinates.size == 0:
        return None
    pad = max(int(pad_voxels), 0)
    lower = np.maximum(coordinates.min(axis=0) - pad, 0)
    upper = np.minimum(coordinates.max(axis=0) + pad + 1, binary.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))  # type: ignore[return-value]


def contiguous_anatomy_instances(
    instance_map: np.ndarray, anatomy: str
) -> tuple[np.ndarray, dict[int, int]]:
    """Extract one anatomy and relabel its instances to contiguous ``1..K``."""
    if anatomy not in ANATOMY_RANGES:
        raise KeyError(f"unknown anatomy: {anatomy}")
    instances = np.asarray(instance_map)
    lower, upper = ANATOMY_RANGES[anatomy]
    source_ids = sorted(int(v) for v in np.unique(instances) if lower <= int(v) <= upper)
    mapping = {source_id: local_id for local_id, source_id in enumerate(source_ids, 1)}
    output = np.zeros(instances.shape, dtype=np.uint8)
    for source_id, local_id in mapping.items():
        output[instances == source_id] = local_id
    return output, mapping


def select_stage2_case_ids(split: dict[str, Any], trajectory_seed: int) -> dict[str, list[str]]:
    """Resolve the exact initial-train and static-validation IDs."""
    trajectories = {int(row["seed"]): row["initial_labeled_case_ids"] for row in split["trajectories"]}
    if trajectory_seed not in trajectories:
        raise KeyError(f"trajectory seed {trajectory_seed} is absent")
    training = [str(case_id).zfill(3) for case_id in trajectories[trajectory_seed]]
    validation = [str(case_id).zfill(3) for case_id in split["split"]["validation"]]
    test = [str(case_id).zfill(3) for case_id in split["split"]["test"]]
    if set(training) & set(validation) or set(training) & set(test) or set(validation) & set(test):
        raise ValueError("Stage 2 train/validation/test case IDs overlap")
    if (len(training), len(validation), len(test)) != (40, 40, 50):
        raise ValueError(
            f"unexpected Stage 2 sizes: train={len(training)}, val={len(validation)}, test={len(test)}"
        )
    return {"train": training, "validation": validation, "test": test}


def discover_case_directories(dataset_root: Path) -> dict[str, Path]:
    """Index the 340 source case directories and reject incomplete inventory."""
    cases: dict[str, Path] = {}
    for part in sorted(dataset_root.glob("PENGWIN26_task1_2_train_part*")):
        for path in sorted(part.iterdir()):
            if path.is_dir():
                cases[path.name.zfill(3)] = path
    incomplete = [case_id for case_id, path in cases.items() if not (path / "image.mha").is_file() or not (path / "label.mha").is_file()]
    if len(cases) != 340 or incomplete:
        raise RuntimeError(f"invalid PENGWIN inventory: cases={len(cases)}, incomplete={incomplete}")
    return cases


def source_case_from_sample(sample_id: str) -> str:
    return sample_id.removeprefix("PENGWIN_").split("_", 1)[0].zfill(3)


def validate_grouped_samples(train_samples: Iterable[str], validation_samples: Iterable[str]) -> None:
    """Reject source-case leakage across train and validation ROI samples."""
    train_cases = {source_case_from_sample(value) for value in train_samples}
    validation_cases = {source_case_from_sample(value) for value in validation_samples}
    overlap = sorted(train_cases & validation_cases)
    if overlap:
        raise ValueError(f"source-case leakage across Stage 2 split: {overlap}")
