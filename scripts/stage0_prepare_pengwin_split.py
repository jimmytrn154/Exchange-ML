#!/usr/bin/env python3
"""Audit all public PENGWIN labels and create frozen split manifests.

This is a user-typed, dataset-wide command. It reads all 340 label volumes and
must not be run as an agent smoke check.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PELVIC_IDS = tuple(range(1, 121)) + tuple(range(151, 201))
FEMUR_IDS = tuple(range(251, 421))
BONE_RANGES = {
    "sacrum": (1, 50),
    "left_hip": (51, 100),
    "right_hip": (101, 150),
    "femur": (151, 200),
}
CLICK_STRATEGIES = (
    "boundary_internal_margin",
    "center_of_mass",
    "euclidean_distance_transform",
    "uniformly_sampled",
)
SPLIT_TARGETS = {
    "test": {"pelvic": 25, "femur": 25},
    "validation": {"pelvic": 20, "femur": 20},
    "pool": {"pelvic": 125, "femur": 125},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/PENGWIN26_task1_2"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--trajectory-seeds",
        type=int,
        nargs="+",
        default=(20260920, 20260921, 20260922),
    )
    parser.add_argument(
        "--search-trials",
        type=int,
        default=20000,
        help="Randomized candidates evaluated after the label audit.",
    )
    return parser.parse_args()


def load_simpleitk() -> Any:
    try:
        import SimpleITK as sitk  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SimpleITK is required in the user-run data environment. "
            "It was absent from the Stage 0 inspection environment."
        ) from exc
    return sitk


def subject_anatomy(subject_id: int) -> str:
    if subject_id in PELVIC_IDS:
        return "pelvic"
    if subject_id in FEMUR_IDS:
        return "femur"
    raise ValueError(f"unexpected public training subject ID: {subject_id:03d}")


def discover_cases(dataset_root: Path) -> list[Path]:
    cases: list[Path] = []
    for part in sorted(dataset_root.glob("PENGWIN26_task1_2_train_part*")):
        cases.extend(path for path in sorted(part.iterdir()) if path.is_dir())
    expected = {f"{value:03d}" for value in PELVIC_IDS + FEMUR_IDS}
    observed = {path.name for path in cases}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"case inventory mismatch; missing={missing}, extra={extra}")
    if len(cases) != 340:
        raise RuntimeError(f"expected 340 cases, found {len(cases)}")
    return sorted(cases, key=lambda path: int(path.name))


def image_information(sitk: Any, path: Path) -> dict[str, Any]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    pixel_id_value = int(reader.GetPixelID())
    pixel_id_name = (
        sitk.GetPixelIDValueAsString(pixel_id_value)
        if hasattr(sitk, "GetPixelIDValueAsString")
        else str(pixel_id_value)
    )
    return {
        "size_xyz": [int(value) for value in reader.GetSize()],
        "spacing_xyz_mm": [float(value) for value in reader.GetSpacing()],
        "origin_xyz_mm": [float(value) for value in reader.GetOrigin()],
        "direction": [float(value) for value in reader.GetDirection()],
        "pixel_id": pixel_id_name,
        "pixel_id_value": pixel_id_value,
    }


def close_sequence(
    left: Iterable[float], right: Iterable[float], tolerance: float = 1e-6
) -> bool:
    left_values = list(left)
    right_values = list(right)
    return len(left_values) == len(right_values) and all(
        math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
        for a, b in zip(left_values, right_values)
    )


def verify_geometry(
    image_info: dict[str, Any], label_info: dict[str, Any], case_id: str
) -> None:
    if image_info["size_xyz"] != label_info["size_xyz"]:
        raise RuntimeError(f"{case_id}: image/label size mismatch")
    for key in ("spacing_xyz_mm", "origin_xyz_mm", "direction"):
        if not close_sequence(image_info[key], label_info[key]):
            raise RuntimeError(f"{case_id}: image/label {key} mismatch")


def bone_name(label_id: int) -> str:
    for name, (lower, upper) in BONE_RANGES.items():
        if lower <= label_id <= upper:
            return name
    raise ValueError(f"label outside documented PENGWIN range: {label_id}")


def click_count(dataset_root: Path, strategy: str, case_id: str) -> int:
    path = (
        dataset_root
        / "PENGWIN26_task2_train_clicks"
        / strategy
        / case_id
        / "peripelvic-fragment-clicks.json"
    )
    if not path.is_file():
        raise RuntimeError(f"missing click file: {path}")
    points = json.loads(path.read_text(encoding="utf-8")).get("points")
    if not isinstance(points, list):
        raise RuntimeError(f"invalid click points in {path}")
    return len(points)


def audit_case(sitk: Any, dataset_root: Path, case_dir: Path) -> dict[str, Any]:
    case_id = case_dir.name
    anatomy = subject_anatomy(int(case_id))
    image_path = case_dir / "image.mha"
    label_path = case_dir / "label.mha"
    if not image_path.is_file() or not label_path.is_file():
        raise RuntimeError(f"{case_id}: missing image.mha or label.mha")

    image_info = image_information(sitk, image_path)
    label_info = image_information(sitk, label_path)
    verify_geometry(image_info, label_info, case_id)

    label_image = sitk.ReadImage(str(label_path))
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(label_image)
    labels = sorted(int(value) for value in statistics.GetLabels() if int(value) != 0)
    if not labels:
        raise RuntimeError(f"{case_id}: no non-background labels")
    if any(value < 1 or value > 200 for value in labels):
        raise RuntimeError(f"{case_id}: label outside 1..200: {labels}")
    if anatomy == "pelvic" and any(value > 150 for value in labels):
        raise RuntimeError(f"{case_id}: pelvic case contains femur-range label")
    if anatomy == "femur" and any(value < 151 for value in labels):
        raise RuntimeError(f"{case_id}: femur case contains pelvic-range label")

    click_counts = {
        strategy: click_count(dataset_root, strategy, case_id)
        for strategy in CLICK_STRATEGIES
    }
    if any(count != len(labels) for count in click_counts.values()):
        raise RuntimeError(
            f"{case_id}: fragment/click count mismatch; "
            f"labels={len(labels)}, clicks={click_counts}"
        )

    voxel_volume_mm3 = math.prod(float(v) for v in label_info["spacing_xyz_mm"])
    foreground_voxels = sum(int(statistics.GetNumberOfPixels(v)) for v in labels)
    return {
        "case_id": case_id,
        "anatomy": anatomy,
        "fragment_count": len(labels),
        "label_ids": labels,
        "bone_groups": sorted({bone_name(value) for value in labels}),
        "foreground_voxels": foreground_voxels,
        "foreground_volume_mm3": foreground_voxels * voxel_volume_mm3,
        "click_counts": click_counts,
        "image": image_info,
        "label": label_info,
    }


def quantile_cutpoints(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        float(ordered[math.ceil(q * len(ordered) / 4) - 1]) for q in (1, 2, 3)
    ]


def attach_strata(records: list[dict[str, Any]]) -> None:
    for anatomy in ("pelvic", "femur"):
        group = [record for record in records if record["anatomy"] == anatomy]
        fragment_cuts = quantile_cutpoints(
            [float(record["fragment_count"]) for record in group]
        )
        volume_cuts = quantile_cutpoints(
            [float(record["foreground_volume_mm3"]) for record in group]
        )
        for record in group:
            record["strata"] = {
                "fragment_count_quartile": bisect.bisect_left(
                    fragment_cuts, float(record["fragment_count"])
                ),
                "foreground_volume_quartile": bisect.bisect_left(
                    volume_cuts, float(record["foreground_volume_mm3"])
                ),
                "bone_signature": "+".join(record["bone_groups"]),
            }


def categorical_distance(
    selected: list[dict[str, Any]], population: list[dict[str, Any]], key: str
) -> float:
    selected_counts = Counter(str(record["strata"][key]) for record in selected)
    population_counts = Counter(str(record["strata"][key]) for record in population)
    return sum(
        abs(
            selected_counts[category] / len(selected)
            - population_counts[category] / len(population)
        )
        for category in population_counts
    )


def subset_score(selected: list[dict[str, Any]], population: list[dict[str, Any]]) -> float:
    return sum(
        categorical_distance(selected, population, key)
        for key in (
            "fragment_count_quartile",
            "foreground_volume_quartile",
            "bone_signature",
        )
    )


def best_partition(
    records: list[dict[str, Any]],
    first_size: int,
    second_size: int,
    seed: int,
    trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...], list[dict[str, Any]]] | None = None
    for _ in range(trials):
        candidate = records.copy()
        rng.shuffle(candidate)
        first = candidate[:first_size]
        second = candidate[first_size : first_size + second_size]
        score = subset_score(first, records)
        if second:
            score += subset_score(second, records)
        tie_break = tuple(record["case_id"] for record in candidate)
        if best is None or (score, tie_break) < (best[0], best[1]):
            best = (score, tie_break, candidate)
    if best is None:
        raise RuntimeError("no split candidate generated")
    ordered = best[2]
    return (
        ordered[:first_size],
        ordered[first_size : first_size + second_size],
        ordered[first_size + second_size :],
    )


def select_initial_set(
    pool: list[dict[str, Any]], seed: int, trials: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for offset, anatomy in enumerate(("pelvic", "femur")):
        population = [record for record in pool if record["anatomy"] == anatomy]
        first, _, _ = best_partition(
            population,
            first_size=20,
            second_size=0,
            seed=seed + offset,
            trials=trials,
        )
        selected.extend(first)
    return sorted(selected, key=lambda record: int(record["case_id"]))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    anatomy_counts = Counter(record["anatomy"] for record in records)
    fragment_counts = [int(record["fragment_count"]) for record in records]
    volumes = [float(record["foreground_volume_mm3"]) for record in records]
    return {
        "n_cases": len(records),
        "anatomy_counts": dict(sorted(anatomy_counts.items())),
        "fragment_count": {
            "min": min(fragment_counts),
            "max": max(fragment_counts),
            "mean": sum(fragment_counts) / len(fragment_counts),
        },
        "foreground_volume_mm3": {
            "min": min(volumes),
            "max": max(volumes),
            "mean": sum(volumes) / len(volumes),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.search_trials < 1:
        raise SystemExit("--search-trials must be positive")
    if len(args.trajectory_seeds) < 3 or len(set(args.trajectory_seeds)) != len(
        args.trajectory_seeds
    ):
        raise SystemExit("provide at least three distinct --trajectory-seeds")

    sitk = load_simpleitk()
    dataset_root = args.dataset_root.resolve()
    records = [
        audit_case(sitk, dataset_root, case)
        for case in discover_cases(dataset_root)
    ]
    attach_strata(records)

    split_records: dict[str, list[dict[str, Any]]] = {
        "test": [],
        "validation": [],
        "pool": [],
    }
    for offset, anatomy in enumerate(("pelvic", "femur")):
        population = [record for record in records if record["anatomy"] == anatomy]
        test, validation, pool = best_partition(
            population,
            first_size=SPLIT_TARGETS["test"][anatomy],
            second_size=SPLIT_TARGETS["validation"][anatomy],
            seed=args.seed + offset,
            trials=args.search_trials,
        )
        split_records["test"].extend(test)
        split_records["validation"].extend(validation)
        split_records["pool"].extend(pool)

    for name, expected_size in (("test", 50), ("validation", 40), ("pool", 250)):
        if len(split_records[name]) != expected_size:
            raise RuntimeError(f"{name}: expected {expected_size} cases")
    all_assigned = [
        record["case_id"]
        for name in ("test", "validation", "pool")
        for record in split_records[name]
    ]
    if len(all_assigned) != len(set(all_assigned)) or set(all_assigned) != {
        record["case_id"] for record in records
    }:
        raise RuntimeError("split assignments are not disjoint and exhaustive")

    trajectories = []
    for seed in args.trajectory_seeds:
        initial = select_initial_set(split_records["pool"], seed, args.search_trials)
        trajectories.append(
            {
                "seed": seed,
                "initial_labeled_case_ids": [record["case_id"] for record in initial],
                "initial_labeled_size": 40,
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": "PENGWIN 2026 Task 1 public labeled training set",
        "split_seed": args.seed,
        "search_trials": args.search_trials,
        "ground_truth_usage": (
            "GT-derived strata define only the static split and initial sets; "
            "they must never be exposed to acquisition."
        ),
        "split": {
            name: sorted(record["case_id"] for record in selected)
            for name, selected in split_records.items()
        },
        "split_summary": {
            name: summarize(selected) for name, selected in split_records.items()
        },
        "trajectories": trajectories,
        "budget_sequence": [40, 60, 80, 100, 120, 140],
        "query_rounds": 5,
        "query_batch_size": 20,
    }
    manifest_hash_source = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["content_sha256_without_this_field"] = hashlib.sha256(
        manifest_hash_source
    ).hexdigest()
    audit = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "n_cases": len(records),
        "summary": summarize(records),
        "records": records,
        "warning": "Contains GT-derived metadata; never load this file in acquisition code.",
    }

    write_json(args.output_dir / "pengwin_stage0_split.json", manifest)
    write_json(args.output_dir / "pengwin_stage0_gt_audit.json", audit)
    print(f"Wrote {args.output_dir / 'pengwin_stage0_split.json'}")
    print(f"Wrote {args.output_dir / 'pengwin_stage0_gt_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
