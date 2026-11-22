#!/usr/bin/env python3
"""Repair the E4 blindspot CSV schema without recomputing the experiment.

The original execution identity, metrics, caches and figures are preserved.
A hash-bound receipt and exact before-files document the reporting correction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path

from run_e4_e5 import (ROOT, dataset_table_rows, locked, read_csv, read_json,
                       require_hash, save_json, sha256, signature)
import csv
import json

REVISION = "split_column_v1"
TABLE = "tables/f1_volume_blindspots.csv"
RUNNER = "scripts/run_e4_e5.py"
SCOPE = "report-only schema repair; no inference, scoring, bootstrap or selection rerun"


def _csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["split", "stratum", "case_id", "scan_type", "bone", "model_id",
                                       "one_minus_f1", "missing_volume", "split_risk"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _archive(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"archive differs from original: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)


def repair_report(output: Path, sources: dict[str, Path], config_path: Path, *,
                  check_only: bool = False, original_runner: Path | None = None) -> dict:
    """Validate/rebuild only the small CSV, and preserve compute provenance.

    A receipt is staged before replacing the CSV. An interrupted correction can
    finish from either the old or corrected CSV; other states are rejected.
    """
    archive = output / "report_corrections" / REVISION
    manifest_path = output / "e4_e5_manifest.json"
    old_manifest_path = archive / "e4_e5_manifest.before.json"
    old_table_path = archive / "f1_volume_blindspots.before.csv"
    old_runner_path = archive / "run_e4_e5.before.py"
    receipt_path = archive / "correction.json"
    with locked(output / ".e4_e5.lock"):
        current_bytes = manifest_path.read_bytes()
        original_bytes = old_manifest_path.read_bytes() if old_manifest_path.exists() else current_bytes
        original = json.loads(original_bytes)
        parent_hash = hashlib.sha256(original_bytes).hexdigest()
        if original["status"] != "complete" or signature(original["provenance"]) != original["signature"]:
            raise ValueError("a complete, internally consistent original manifest is required")
        require_hash(config_path, original["provenance"]["config_sha256"])
        config = read_json(config_path)
        corrected, legacy = [], []
        source_hashes = {}
        for split in ("validation", "heldout"):
            source = sources[split]
            expected = original["provenance"]["upstream_tables"][split + "/model_risks"]
            require_hash(source, expected)
            source_hashes[split] = {"path": str(source.resolve()), "sha256": expected}
            rows = [r for r in read_csv(source) if r["one_minus_f1"] <= config["blindspot_f1_tolerance"]
                    and r["missing_volume"] > config["blindspot_missing_volume"]]
            legacy.extend({"split": split, **r, "stratum": r["scan_type"]} for r in rows)
            corrected.extend(dataset_table_rows(rows, split, rename_split_risk=True))
        if len(corrected) != original["summary"]["n_blindspot_model_units"]:
            raise ValueError("blindspot count differs from the original summary")
        original_table = old_table_path if old_table_path.exists() else output / TABLE
        require_hash(original_table, original["output_sha256"][TABLE])
        if read_csv(original_table) != legacy:
            raise ValueError("original table is not the expected legacy serialization of the source risks")
        payload = _csv_bytes(corrected)
        new_hash = hashlib.sha256(payload).hexdigest()
        prior_receipt = read_json(receipt_path) if receipt_path.exists() else None
        for name, digest in original["output_sha256"].items():
            if name == TABLE and prior_receipt and sha256(output / name) == new_hash:
                continue  # interrupted correction after the CSV replacement
            require_hash(output / name, digest)
        for name, digest in original["cache_sha256"].items():
            require_hash(output / name, digest)
        runner_bytes = (old_runner_path if old_runner_path.exists() else original_runner or ROOT / RUNNER).read_bytes()
        if hashlib.sha256(runner_bytes).hexdigest() != original["provenance"]["code_sha256"][RUNNER]:
            raise ValueError("the original runner must be archived before patching its code")
        receipt = {
            "id": REVISION, "scope": SCOPE,
            "created_at_utc": prior_receipt["created_at_utc"] if prior_receipt else datetime.now(timezone.utc).isoformat(),
            "parent_manifest_sha256": parent_hash, "original_compute_signature": original["signature"],
            "source_tables": source_hashes, "row_count": len(corrected),
            "old_table_sha256": original["output_sha256"][TABLE], "new_table_sha256": new_hash,
            "archives": {"e4_e5_manifest.before.json": parent_hash,
                         "f1_volume_blindspots.before.csv": original["output_sha256"][TABLE],
                         "run_e4_e5.before.py": original["provenance"]["code_sha256"][RUNNER]},
            "correction_code_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                                        RUNNER: sha256(ROOT / RUNNER)},
            "renamed_field": "numeric split -> split_risk; split now denotes validation/heldout",
        }
        if prior_receipt is not None and prior_receipt != receipt:
            raise ValueError("existing correction receipt differs; retain it and investigate")
        # save_json's canonical bytes, used to construct the final manifest before mutation.
        receipt_bytes = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
        receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
        if prior_receipt is not None:
            require_hash(receipt_path, receipt_hash)
        revised = deepcopy(original)
        receipt_name = str(receipt_path.relative_to(output))
        revised["output_sha256"][TABLE] = new_hash
        revised["output_sha256"][receipt_name] = receipt_hash
        revised["report_corrections"] = [{"id": REVISION, "scope": SCOPE,
                                          "receipt": receipt_name, "receipt_sha256": receipt_hash,
                                          "parent_manifest_sha256": parent_hash}]
        current = json.loads(current_bytes)
        if current != original and current != revised:
            raise ValueError("active manifest is neither the original nor this reporting revision")
        if current == revised:
            require_hash(output / TABLE, new_hash)
            require_hash(receipt_path, receipt_hash)
            for name, digest in receipt["archives"].items():
                require_hash(archive / name, digest)
            return {"status": "already_corrected", "rows": len(corrected), "receipt": receipt_name}
        if check_only:
            return {"status": "correction_needed", "rows": len(corrected), "receipt": receipt_name}
        archive.mkdir(parents=True, exist_ok=True)
        _archive(old_manifest_path, original_bytes)
        _archive(old_table_path, original_table.read_bytes())
        _archive(old_runner_path, runner_bytes)
        if prior_receipt is None:
            save_json(receipt_path, receipt)
        temporary = (output / TABLE).with_suffix(".csv.tmp")
        temporary.write_bytes(payload)
        temporary.replace(output / TABLE)
        save_json(manifest_path, revised)
        require_hash(output / TABLE, new_hash)
        require_hash(receipt_path, receipt_hash)
        return {"status": "corrected", "rows": len(corrected), "receipt": receipt_name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/stage3_h1_v2/e4_e5")
    parser.add_argument("--check", action="store_true", help="verify correction state without modifying reports")
    args = parser.parse_args()
    config_path = ROOT / "configs/e4_e5.json"
    config = read_json(config_path)
    base = ROOT / read_json(ROOT / config["e1_config"])["output_root"]
    heldout = ROOT / read_json(ROOT / config["e3_config"])["output_root"]
    result = repair_report(args.output_root.resolve(), {"validation": base / "validation/model_risks.csv",
                           "heldout": heldout / "model_risks.csv"}, config_path, check_only=args.check)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
