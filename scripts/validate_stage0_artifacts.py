#!/usr/bin/env python3
"""Lightweight validation of frozen Stage 0 JSON artifacts.

This reads JSON manifests only. It does not read medical volumes and is safe as
an agent-typed smoke check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/study_protocol.json"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object in {path}")
    return payload


def validate(protocol_path: Path) -> None:
    protocol = load_json(protocol_path)
    pengwin = protocol["pengwin"]
    split_path = Path(pengwin["split_manifest"])
    audit_path = Path(pengwin["gt_audit"])
    manifest = load_json(split_path)
    audit = load_json(audit_path)

    embedded_hash = manifest.pop("content_sha256_without_this_field")
    computed_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert embedded_hash == computed_hash, "split manifest embedded hash mismatch"
    assert embedded_hash == pengwin["split_manifest_embedded_sha256"], (
        "protocol references a different split manifest"
    )

    split = manifest["split"]
    expected_sizes = pengwin["split_sizes"]
    assert {name: len(ids) for name, ids in split.items()} == expected_sizes
    all_ids = split["test"] + split["validation"] + split["pool"]
    assert len(all_ids) == len(set(all_ids)) == pengwin["public_labeled_cases"]

    assert audit["n_cases"] == len(audit["records"]) == len(all_ids)
    by_id = {record["case_id"]: record for record in audit["records"]}
    assert set(all_ids) == set(by_id)
    assert "never load" in audit["warning"].lower()

    expected_quotas = pengwin["anatomy_quota_per_split"]
    for split_name, ids in split.items():
        observed = {
            anatomy: sum(by_id[case_id]["anatomy"] == anatomy for case_id in ids)
            for anatomy in ("pelvic", "femur")
        }
        assert observed == expected_quotas[split_name]

    pool = set(split["pool"])
    expected_seeds = pengwin["trajectory_seeds"]
    assert [row["seed"] for row in manifest["trajectories"]] == expected_seeds
    for trajectory in manifest["trajectories"]:
        ids = trajectory["initial_labeled_case_ids"]
        assert len(ids) == len(set(ids)) == pengwin["initial_labeled_cases_per_trajectory"]
        assert set(ids) <= pool
        assert sum(by_id[case_id]["anatomy"] == "pelvic" for case_id in ids) == 20
        assert sum(by_id[case_id]["anatomy"] == "femur" for case_id in ids) == 20

    assert manifest["budget_sequence"] == pengwin["budget_sequence"]
    assert manifest["query_rounds"] == pengwin["query_rounds"]
    assert manifest["query_batch_size"] == pengwin["query_batch_size"]
    assert all(
        all(count == record["fragment_count"] for count in record["click_counts"].values())
        for record in audit["records"]
    )

    margin = protocol["metric_contract"]["h2"][
        "semantic_noninferiority_margin_absolute_dice"
    ]
    assert margin == 0.01
    assert protocol["data_leakage_guards"]["gt_allowed_for_acquisition"] is False
    assert str(audit_path) in protocol["data_leakage_guards"]["acquisition_code_must_not_load"]

    print("Stage 0 artifact validation: PASS")
    print(f"Cases: {len(all_ids)}")
    print(f"Split sizes: {expected_sizes}")
    print(f"Trajectory seeds: {expected_seeds}")
    print(f"Split hash: {embedded_hash}")


def main() -> int:
    args = parse_args()
    validate(args.protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
