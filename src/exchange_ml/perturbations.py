"""Controlled partition perturbations for H4 construct validation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .instances import connectivity_structure
from .metrics import binary_dice


def relabel_instances(instance_map: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    source = np.asarray(instance_map)
    output = source.copy()
    for old_id, new_id in mapping.items():
        output[source == int(old_id)] = int(new_id)
    return output


def merge_instances(
    instance_map: np.ndarray, first_id: int, second_id: int
) -> np.ndarray:
    source = np.asarray(instance_map)
    if not np.any(source == first_id) or not np.any(source == second_id):
        raise ValueError("both merge IDs must be present")
    output = source.copy()
    output[source == int(second_id)] = int(first_id)
    return output


def _next_id_in_anatomy(instance_map: np.ndarray, instance_id: int) -> int:
    bounds = ((1, 50), (51, 100), (101, 150), (151, 200))
    lower, upper = next(
        ((lo, hi) for lo, hi in bounds if lo <= instance_id <= hi),
        (1, 200),
    )
    used = {int(value) for value in np.unique(instance_map)}
    for candidate in range(lower, upper + 1):
        if candidate not in used:
            return candidate
    raise ValueError(f"no unused label ID in anatomy range {lower}..{upper}")


def split_instance(
    instance_map: np.ndarray,
    instance_id: int,
    *,
    new_id: int | None = None,
    axis: int = 0,
) -> np.ndarray:
    source = np.asarray(instance_map)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    coordinates = np.argwhere(source == int(instance_id))
    if len(coordinates) < 2:
        raise ValueError("split instance must contain at least two voxels")
    order = np.lexsort(
        tuple(coordinates[:, dim] for dim in reversed(range(3)))
        + (coordinates[:, axis],)
    )
    selected = coordinates[order[len(order) // 2 :]]
    split_id = (
        _next_id_in_anatomy(source, instance_id) if new_id is None else int(new_id)
    )
    if split_id == instance_id or np.any(source == split_id):
        raise ValueError("new split ID must be unused and differ from the source ID")
    output = source.copy()
    output[tuple(selected.T)] = split_id
    return output


def remove_instance(instance_map: np.ndarray, instance_id: int) -> np.ndarray:
    source = np.asarray(instance_map)
    if not np.any(source == int(instance_id)):
        raise ValueError(f"instance {instance_id} is absent")
    output = source.copy()
    output[source == int(instance_id)] = 0
    return output


def matched_boundary_perturbation(
    instance_map: np.ndarray,
    instance_id: int,
    *,
    target_binary_dice: float,
    connectivity: int,
    max_iterations: int = 8,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Choose a deterministic partial erosion/dilation near a requested Dice.

    Complete morphology iterations are too coarse for small synthetic objects.
    Voxels are therefore ordered by successive boundary shells, then a prefix
    is selected to match the requested whole-foreground Dice as closely as the
    finite voxel grid permits.
    """
    source = np.asarray(instance_map)
    original_instance = source == int(instance_id)
    if not original_instance.any():
        raise ValueError(f"instance {instance_id} is absent")
    if not 0 < target_binary_dice <= 1:
        raise ValueError("target_binary_dice must be in (0, 1]")
    structure = connectivity_structure(connectivity)
    foreground = source > 0
    candidates: list[tuple[float, np.ndarray, str, int, float]] = []
    for operation in ("erosion", "dilation"):
        current = original_instance.copy()
        ordered_coordinates: list[np.ndarray] = []
        for iterations in range(1, max_iterations + 1):
            if operation == "erosion":
                updated = ndi.binary_erosion(current, structure=structure)
                shell = np.logical_and(current, np.logical_not(updated))
            else:
                updated = ndi.binary_dilation(current, structure=structure)
                updated &= np.logical_or(source == 0, original_instance)
                shell = np.logical_and(updated, np.logical_not(current))
            shell_coordinates = np.argwhere(shell)
            if shell_coordinates.size:
                ordered_coordinates.extend(shell_coordinates)
            current = updated
            if not current.any() or not shell_coordinates.size:
                break

        if not ordered_coordinates:
            continue
        available = np.asarray(ordered_coordinates, dtype=np.int64)
        foreground_size = int(foreground.sum())
        if operation == "erosion":
            target_count = (
                2.0
                * foreground_size
                * (1.0 - target_binary_dice)
                / (2.0 - target_binary_dice)
            )
            maximum = min(len(available), int(original_instance.sum()) - 1)
        else:
            target_count = (
                2.0 * foreground_size * (1.0 - target_binary_dice) / target_binary_dice
            )
            maximum = len(available)
        if maximum < 1:
            continue

        candidate_counts = {
            max(1, min(maximum, int(np.floor(target_count)))),
            max(1, min(maximum, int(np.ceil(target_count)))),
        }
        for changed_count in sorted(candidate_counts):
            candidate = source.copy()
            selected = available[:changed_count]
            if operation == "erosion":
                candidate[tuple(selected.T)] = 0
            else:
                candidate[tuple(selected.T)] = int(instance_id)
            score = binary_dice(candidate > 0, foreground)
            candidates.append(
                (
                    abs(score - target_binary_dice),
                    candidate,
                    operation,
                    changed_count,
                    score,
                )
            )
    if not candidates:
        raise ValueError("no non-empty boundary perturbation could be generated")
    _, output, operation, changed_voxels, achieved = min(
        candidates, key=lambda row: (row[0], row[2], row[3])
    )
    return output, {
        "operation": operation,
        "changed_voxels": changed_voxels,
        "max_shell_iterations": int(max_iterations),
        "target_binary_dice": float(target_binary_dice),
        "achieved_binary_dice": float(achieved),
    }
