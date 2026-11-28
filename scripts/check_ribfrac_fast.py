#!/usr/bin/env python3
"""Read-only checks around the frozen E6 inference/evaluation entry points.

No training, scoring, bootstrap, checkpoint selection, or output modification.
The original launcher is part of the training identity and stays unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import run_ribfrac_fast as runner
from exchange_ml.e6_backend import header_voxels
from exchange_ml.e1_scoring import SCORE_NAMES


def check(audit: bool = False) -> dict:
    config = runner.load_config(runner.ROOT / "configs/ribfrac_fast.json")
    root = runner.output_path(config, Path(config["output_root"]))
    identity = runner.identity(config, root)
    if runner.read_json(root / "handoff.json")["signature"] != runner.signature(identity):
        raise ValueError("Handoff identity differs from the completed campaign")
    cases = {cid: row for cid, row in runner.inventory(Path(config["dataset_root"])).items()
             if row["split"] == "validation"}
    expected_cases = {f"RibFrac{i}" for i in range(421, 501)}
    if set(cases) != expected_cases:
        raise ValueError("Exactly the 80 official external cases are required")
    checkpoint_hashes, inference_hashes = [], {}
    missing_voxels = 0
    inference_records = 0
    for member in config["members"]:
        training = runner.verify_training(config, root, member)
        if training["final_epoch"] != config["backend"]["epochs"]:
            raise ValueError(f"Fixed final epoch not reached: {member}")
        checkpoint_hashes.append(training["checkpoint"]["sha256"])
        folder = root / "models" / member
        path = folder / "inference_manifest.json"
        records = {}
        if path.exists():
            state = runner.stage_state(path, {**identity, "member": member,
                "training_manifest_sha256": runner.sha256(folder / "training_manifest.json"),
                "checkpoint_sha256": training["checkpoint"]["sha256"], "case_ids": sorted(cases)})
            records = state["records"]
            if not set(records) <= expected_cases or state["status"] not in ("running", "complete"):
                raise ValueError(f"Invalid inference status or case membership: {member}")
            if (audit or state["status"] == "complete") and (
                    state["status"] != "complete" or set(records) != expected_cases):
                raise ValueError(f"Complete 80-case inference required: {member}")
            if audit:
                for cid, saved in records.items():
                    runner.require_hash(cases[cid]["image"], saved["source_image"]["sha256"])
                    runner.require_hash(folder / "predictions" / f"{cid}.mha", saved["prediction"]["sha256"])
                    runner.require_hash(folder / "probabilities" / f"{cid}.mha", saved["probability"]["sha256"])
                inference_hashes[member] = runner.sha256(path)
        elif audit:
            raise ValueError(f"Missing inference manifest: {member}")
        inference_records += len(records)
        if not audit:
            missing_voxels += sum(header_voxels(cases[cid]["image"]) for cid in expected_cases - records.keys())
        print(f"{member}: final epoch {training['final_epoch']}; checkpoint hash PASS; "
              f"inference {len(records)}/80", flush=True)
    if len(set(checkpoint_hashes)) != 3:
        raise ValueError("Three distinct independently seeded checkpoints required")
    if not audit:
        # Each missing model/case writes uint32 instances + float32 probability.
        required = 8 * missing_voxels + int(config["storage"]["reserve_gib"] * 2**30)
        available = shutil.disk_usage(root).free
        runner.require_disk({"sufficient": available >= required,
                             "estimated_required_bytes": required, "available_bytes": available})
        return {"status": "ready_for_inference", "external_cases": 80,
                "inference_records": inference_records, "required_records": 240,
                "estimated_remaining_bytes_including_reserve": required, "available_bytes": available}

    path = root / "e6_manifest.json"
    state = runner.stage_state(path, {**identity, "inference_manifest_sha256": inference_hashes,
        "external_metadata_sha256": {str(p): runner.sha256(p)
                                     for p in sorted({row["metadata"] for row in cases.values()})}})
    if state["status"] != "complete" or set(state["records"]) != expected_cases:
        raise ValueError("Complete 80-case external evaluation required")
    runner.verify_outputs(root, state["output_sha256"])
    if runner.read_json(root / "summary.json") != state["summary"] or state["summary"]["smoke_only"]:
        raise ValueError("Full evaluation summary differs from manifest")
    for cid, saved in state["records"].items():
        runner.require_hash(cases[cid]["label"], saved["gt"]["sha256"])
        runner.require_hash(root / "cache" / f"{cid}.json", saved["cache_sha256"])
        runner.require_hash(root / "cache" / f"{cid}.scores.json", saved["score_cache_sha256"])
    with (root / "scores.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"case_id", *SCORE_NAMES}:
            raise ValueError("Expected all 12 fixed disagreement scores")
        rows = list(reader)
    if len(rows) != 80 or {row["case_id"] for row in rows} != expected_cases:
        raise ValueError("Score table lacks the exact 80-case membership")
    return {"status": "E6_audit_pass", "trained_members": 3, "external_cases": 80,
            "inference_records": inference_records, "scores": len(SCORE_NAMES),
            "verified_output_hashes": len(state["output_sha256"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "audit"))
    args = parser.parse_args()
    print(json.dumps(check(audit=args.stage == "audit"), indent=2))
