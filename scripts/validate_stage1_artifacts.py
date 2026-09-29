#!/usr/bin/env python3
"""Validate the completed Stage 1 PENGWIN connectivity-audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONNECTIVITIES = (6, 18, 26)
EXPECTED_CASE_COUNT = 340
EXPECTED_FRAGMENT_COUNT = 2427


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("splits/pengwin_stage1_connectivity_audit.json"),
    )
    return parser.parse_args()


def _sum(records: list[dict[str, Any]], connectivity: int, key: str) -> int:
    return sum(
        int(record["connectivity"][str(connectivity)][key]) for record in records
    )


def validate(audit_path: Path) -> dict[str, Any]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    records = payload["records"]
    assert payload["discovered_case_count"] == EXPECTED_CASE_COUNT
    assert payload["analyzed_case_count"] == EXPECTED_CASE_COUNT
    assert payload["complete_dataset"] is True
    assert payload["connectivities"] == list(CONNECTIVITIES)
    assert payload["component_prune_min_volume_mm3"] == 1000.0
    assert payload["gt_fragment_min_volume_mm3"] == 500.0
    assert len(records) == EXPECTED_CASE_COUNT

    case_ids = [record["case_id"] for record in records]
    assert len(set(case_ids)) == EXPECTED_CASE_COUNT
    assert all(Path(record["label_path"]).is_file() for record in records)

    total_fragments = sum(int(record["fragment_count"]) for record in records)
    assert total_fragments == EXPECTED_FRAGMENT_COUNT
    assert payload["summary"]["case_count"] == EXPECTED_CASE_COUNT

    below_500 = 0
    below_1000 = 0
    total_gt_volume_mm3 = 0.0
    below_1000_volume_mm3 = 0.0
    for record in records:
        label_ids = [int(value) for value in record["label_ids"]]
        volumes = {
            int(key): float(value)
            for key, value in record["fragment_volumes_mm3"].items()
        }
        assert record["fragment_count"] == len(label_ids) == len(volumes)
        assert sorted(label_ids) == sorted(volumes)
        assert all(1 <= label_id <= 200 for label_id in label_ids)
        expected_below_500 = sorted(
            label_id for label_id, volume in volumes.items() if volume < 500.0
        )
        assert sorted(record["gt_ids_below_min_volume"]) == expected_below_500
        below_500 += len(expected_below_500)
        below_1000 += sum(volume < 1000.0 for volume in volumes.values())
        total_gt_volume_mm3 += sum(volumes.values())
        below_1000_volume_mm3 += sum(
            volume for volume in volumes.values() if volume < 1000.0
        )

        counts = []
        for connectivity in CONNECTIVITIES:
            row = record["connectivity"][str(connectivity)]
            per_id = {
                int(key): int(value)
                for key, value in row["component_count_by_id"].items()
            }
            assert sorted(per_id) == sorted(label_ids)
            assert row["component_count"] == sum(per_id.values())
            assert row["disconnected_instance_count"] == sum(
                count > 1 for count in per_id.values()
            )
            counts.append(int(row["component_count"]))
        assert counts[0] >= counts[1] >= counts[2] >= len(label_ids)

    assert payload["summary"]["gt_ids_below_min_volume"] == below_500
    for connectivity in CONNECTIVITIES:
        expected = {
            "component_count": _sum(records, connectivity, "component_count"),
            "disconnected_instance_count": _sum(
                records, connectivity, "disconnected_instance_count"
            ),
            "small_component_count": _sum(
                records, connectivity, "small_component_count"
            ),
            "small_component_voxels": _sum(
                records, connectivity, "small_component_voxels"
            ),
            "cases_with_disconnected_instances": sum(
                record["connectivity"][str(connectivity)]["disconnected_instance_count"]
                > 0
                for record in records
            ),
            "cases_with_small_components": sum(
                record["connectivity"][str(connectivity)]["small_component_count"] > 0
                for record in records
            ),
        }
        assert payload["summary"]["connectivity"][str(connectivity)] == expected

    digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    return {
        "status": "PASS",
        "audit": str(audit_path),
        "sha256": digest,
        "case_count": len(records),
        "gt_instance_count": total_fragments,
        "gt_instances_below_500_mm3": below_500,
        "gt_instances_below_1000_mm3": below_1000,
        "gt_volume_fraction_below_1000_mm3": (
            below_1000_volume_mm3 / total_gt_volume_mm3
        ),
        "frozen_policy": {
            "preserve_all_gt_instance_ids": True,
            "prediction_connectivity": 26,
            "prediction_component_prune_min_volume_mm3": 1000.0,
            "prediction_morphology": "none beyond per-ID component pruning",
        },
    }


def main() -> int:
    args = parse_args()
    print(json.dumps(validate(args.audit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
