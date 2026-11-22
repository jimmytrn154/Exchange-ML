"""Prediction-only object-macro overlap features, separate from frozen E1–E6.

For each shared instance, the inverse Simpson effective number of counterparts
is sum(n_ij)^2 / sum(n_ij^2). Macro-average (d-1), then average directions and
the three member pairs. This is an unbounded descriptor, not a probability or
a new mathematical metric. No-shared-support pairs have value zero and a flag.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from .e1_data import MEMBERS, clean_prediction, mask_bone, read_volume
from .e1_scoring import pair_components


def _labels(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_) or (value < 0).any():
        raise ValueError("nonnegative integer instance IDs required")
    return value


def graph_pair(first: np.ndarray, second: np.ndarray) -> dict:
    left, right = _labels(first), _labels(second)
    if left.shape != right.shape:
        raise ValueError("partition shapes differ")
    shared = (left > 0) & (right > 0)
    union = (left > 0) | (right > 0)
    nl, nr = (int(np.count_nonzero(np.unique(x))) for x in (left, right))
    forward = reverse = 0.
    sl = sr = 0
    if shared.any():
        _, li = np.unique(left[shared], return_inverse=True)
        _, ri = np.unique(right[shared], return_inverse=True)
        sl, sr = int(li.max()) + 1, int(ri.max()) + 1
        codes, counts = np.unique(li.astype(np.int64) * sr + ri, return_counts=True)
        rows, columns = codes // sr, codes % sr
        counts = counts.astype(np.float64)

        def directional(indices: np.ndarray, size: int) -> float:
            totals = np.bincount(indices, weights=counts, minlength=size)
            squares = np.bincount(indices, weights=counts**2, minlength=size)
            return float(np.maximum(totals**2 / squares - 1, 0).mean())

        forward, reverse = directional(rows, sl), directional(columns, sr)
    return {"graph_degree": (forward + reverse) / 2,
            "graph_forward": forward, "graph_reverse": reverse,
            "graph_shared_support": bool(shared.any()),
            "graph_shared_voxels": int(shared.sum()), "graph_union_voxels": int(union.sum()),
            "graph_instances_first": nl, "graph_instances_second": nr,
            "graph_shared_instances_first": sl, "graph_shared_instances_second": sr,
            "graph_unmatched_instances_first": nl - sl, "graph_unmatched_instances_second": nr - sr}


def aggregate_graph(partitions: Sequence[np.ndarray]) -> tuple[dict, list[dict]]:
    if len(partitions) != 3:
        raise ValueError("exactly three members required")
    maps = [_labels(p) for p in partitions]
    pairs = [graph_pair(maps[i], maps[j]) for i, j in combinations(range(3), 2)]
    return {"graph_degree": float(np.mean([p["graph_degree"] for p in pairs])),
            "graph_no_shared_fraction": float(np.mean([not p["graph_shared_support"] for p in pairs])),
            "all_members_empty": bool(all(not np.any(p) for p in maps))}, pairs


def extract_case(case_id: str, scan_type: str, prediction_paths: Sequence[Path],
                 cleanup_config: dict) -> dict:
    """Read three native-grid maps [Z,Y,X]; never read GT or probabilities."""
    if scan_type not in ("pelvic", "femur") or len(prediction_paths) != 3:
        raise ValueError("declared scan type and three prediction paths required")
    if cleanup_config["connectivity"] != 26 or cleanup_config["prune_mm3"] != 1000:
        raise ValueError("preserve E1 cleanup: 26-connectivity and 1000 mm3")
    maps, expected = [], None
    for path in prediction_paths:
        array, expected = read_volume(Path(path), expected)
        spacing = tuple(expected["spacing"][::-1])
        maps.append(clean_prediction(array, spacing, 1000., 26))
    features, pair_rows = [], []
    bones = ("sacrum", "left_hip", "right_hip") if scan_type == "pelvic" else ("femur",)
    for bone in bones:
        roi = np.logical_or.reduce([mask_bone(m, bone) for m in maps])
        parts = [np.where(mask_bone(m, bone)[roi], m[roi], 0) for m in maps]
        result, pairs = aggregate_graph(parts)
        old_pairs = [pair_components(a, b) for a, b in combinations(parts, 2)]
        mean = lambda name: float(np.mean([p[name] for p in old_pairs]))
        common = {"case_id": case_id, "scan_type": scan_type, "bone": bone}
        features.append({**common, **result, "presence": mean("presence"),
            "partition_v1": mean("weighted_q_log_n"), "partition_v2": mean("weighted_q_joint"),
            "component_vi": mean("vi"),
            "predicted_count_mean": float(np.mean([np.count_nonzero(np.unique(p)) for p in parts])),
            "predicted_volume_mm3": float(np.mean([np.count_nonzero(p) for p in parts]) * np.prod(spacing))})
        pair_rows.extend({**common, "member_i": i, "member_j": j, **p}
                         for (i, j), p in zip(combinations(MEMBERS, 2), pairs))
    return {"features": features, "pairs": pair_rows, "geometry": expected}


def phantom_checks() -> list[dict]:
    """Deterministic mechanism assertions; not clinical performance evidence."""
    rows = []

    def check(name: str, a: np.ndarray, b: np.ndarray, expected: float) -> None:
        result = graph_pair(a, b)
        old = pair_components(a, b)
        shared = (a > 0) & (b > 0)
        macro_entropies = []
        for parent, counterpart in ((a, b), (b, a)):
            entropies = []
            for label in np.unique(parent[shared]):
                _, counts = np.unique(counterpart[shared & (parent == label)], return_counts=True)
                probabilities = counts / counts.sum()
                entropies.append(float(-np.sum(probabilities * np.log(probabilities))))
            macro_entropies.append(float(np.mean(entropies)) if entropies else 0.)
        assert np.isclose(result["graph_degree"], expected)
        rows.append({"check": name, "value": result["graph_degree"], "expected": expected,
                     "presence": old["presence"], "vi": old["vi"],
                     "partition_v1": old["weighted_q_log_n"], "partition_v2": old["weighted_q_joint"],
                     "macro_conditional_entropy": float(np.mean(macro_entropies)),
                     "hungarian": old["hungarian"],
                     "count_variance": float(np.var([old["k_i"], old["k_j"]])),
                     "shared_support": result["graph_shared_support"], "passed": True})

    base = np.ones(12, np.uint8)
    for k in (2, 3, 4):
        other = np.repeat(np.arange(1, k + 1), 12 // k)
        check(f"balanced_split_{k}", base, other, (k - 1) / 2)
        check(f"balanced_merge_{k}", other, base, (k - 1) / 2)
    small = np.r_[np.ones(990, int), np.full(5, 2), np.full(5, 3)]
    large = np.r_[np.ones(495, int), np.full(495, 3), np.full(10, 2)]
    reference = np.r_[np.ones(990, int), np.full(10, 2)]
    check("size_swap_unpruned_small", reference, small, .25)
    check("size_swap_unpruned_large", reference, large, .25)
    for name, a, b in (("identity", base, base), ("permutation", reference, reference * 91),
                       ("both_empty", base * 0, base * 0), ("one_empty", base, base * 0),
                       ("single_shared_voxel", np.array([1]), np.array([900]))):
        check(name, a, b, 0.)
    original = graph_pair(reference, small)["graph_degree"]
    assert np.isclose(original, graph_pair(small, reference)["graph_degree"])
    assert np.isclose(original, graph_pair(np.tile(reference, 3), np.tile(small, 3))["graph_degree"])
    check("symmetry", small, reference, original)
    check("count_replication", np.tile(reference, 3), np.tile(small, 3), original)
    equal = np.r_[np.ones(50, int), np.full(50, 2)]
    jitter = equal.copy(); jitter[49], jitter[50] = 2, 1
    check("boundary_label_jitter", equal, jitter, 50**2 / (49**2 + 1) - 1)
    check("foreground_boundary_only", np.ones(100, int), np.r_[0, np.ones(98, int), 0], 0.)
    check("missing_entire_instance", equal, np.r_[np.ones(50, int), np.zeros(50, int)], 0.)
    check("tiny_split_unpruned", np.ones(1000, int), np.r_[np.ones(999, int), 2],
          (1000**2 / (999**2 + 1) - 1) / 2)
    check("spurious_nonoverlapping_instance", np.r_[np.ones(99, int), 0], np.r_[np.ones(99, int), 2], 0.)
    return rows
