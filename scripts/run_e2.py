#!/usr/bin/env python3
"""CPU-only E2. Agent checks: --check, --smoke. Full run is user-typed."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exchange_ml.e1_data import MEMBERS, clean_prediction, mask_bone, read_volume  # noqa: E402
from exchange_ml.e2_perturbations import (  # noqa: E402
    METRICS, components, contingency, extract_bone, phantoms, replication, responses,
)
from exchange_ml.e2_reporting import report  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selection(metadata: list[dict], seed: int, per_type: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    selected = []
    for scan in ("pelvic", "femur"):
        candidates = sorted(r["case_id"] for r in metadata if r["scan_type"] == scan)
        chosen = sorted(rng.choice(candidates, per_type, replace=False).tolist())
        selected.extend({"case_id": cid, "scan_type": scan} for cid in chosen)
    return selected


def inputs(config_path: Path, config: dict) -> tuple[list[dict], dict, dict, dict, dict]:
    if (config["selection_seed"], config["cases_per_scan_type"], config["replication_factors"],
        config["boundary_targets"], config["boundary_dice_tolerance"], config["numeric_tolerance"]) != (
            20261118, 10, [1, 2, 4, 8, 16], [.99, .95, .90], .005, 1e-10):
        raise ValueError("E2 config differs from declared exps.md protocol")
    if config["boundary_max_iterations"] != 8:
        raise ValueError("boundary helper shell limit must remain explicit at 8")
    e1_root = ROOT / config["e1_root"]
    manifest_path = e1_root / "e1_manifest.json"
    e1 = json.loads(manifest_path.read_text())
    if e1["status"] != "complete" or e1["smoke_only"] or len(e1["case_ids"]) != 40:
        raise ValueError("E2c requires completed full E1")
    preserved = {str(manifest_path): sha256(manifest_path), str(e1_root / "summary.json"): sha256(e1_root / "summary.json")}
    for name, digest in e1["output_sha256"].items():
        path = e1_root / name
        if sha256(path) != digest:
            raise ValueError(f"E1 output hash mismatch: {path}")
        preserved[str(path)] = digest
    for name, digest in e1["provenance"]["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"E1 implementation changed since its run: {name}")
    with (ROOT / config["scan_metadata"]).open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    ids = set(json.loads((ROOT / config["split"]).read_text())["split"]["validation"])
    if len(metadata) != 40 or {r["case_id"] for r in metadata} != ids or set(e1["case_ids"]) != ids:
        raise ValueError("validation case identity mismatch")
    if any(sum(r["scan_type"] == scan for r in metadata) != 20 for scan in ("pelvic", "femur")):
        raise ValueError("validation metadata must have 20 pelvic and 20 femur cases")
    selected = selection(metadata, config["selection_seed"], config["cases_per_scan_type"])
    with (e1_root / "validation/pair_components.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    pair_keys = [(r["case_id"], r["bone"], r["member_i"], r["member_j"]) for r in pairs]
    if len(pairs) != 240 or len(set(pair_keys)) != 240:
        raise ValueError("E1 must have 240 unique bone/member-pair rows")
    paths = {}
    by_id = {r["case_id"]: r for r in metadata}
    for cid in sorted(ids):
        row = by_id[cid]
        bones = ("sacrum", "left_hip", "right_hip") if row["scan_type"] == "pelvic" else ("femur",)
        expected = {(cid, b, i, j) for b in bones for i, j in combinations(MEMBERS, 2)}
        if {k for k in pair_keys if k[0] == cid} != expected:
            raise ValueError(f"{cid}: missing or incorrect E1 pair identities")
        paths[cid] = {key: Path(e1["sources"][cid]["states"][key]["path"])
                      for key in ("gt", *(m + "_prediction" for m in MEMBERS))}
        for key, path in paths[cid].items():
            observed = path.stat()
            expected_state = e1["sources"][cid]["states"][key]
            if observed.st_size != expected_state["bytes"] or observed.st_mtime_ns != expected_state["mtime_ns"]:
                raise ValueError(f"{cid}/{key}: input metadata drift since E1")
            with path.open("rb") as handle:
                handle.read(1)
    names = ["e2_perturbations.py", "e2_reporting.py", "e1_data.py", "e1_scoring.py", "instances.py", "metrics.py", "perturbations.py", "__init__.py"]
    files = [Path(__file__), *(ROOT / "src/exchange_ml" / n for n in names)]
    provenance = {"config_sha256": sha256(config_path), "e1_manifest_sha256": preserved[str(manifest_path)],
                  "metadata_sha256": sha256(ROOT / config["scan_metadata"]), "split_sha256": sha256(ROOT / config["split"]),
                  "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
                  "software": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "SimpleITK", "matplotlib")},
                  "python": sys.version, "full_20_case_selection": selected}
    return sorted(metadata, key=lambda r: r["case_id"]), {"selected": selected, "paths": paths, "pairs": pairs, "e1": e1}, provenance, preserved, config


def sparse_table(joint: np.ndarray, identity: dict) -> dict:
    positions = np.argwhere(joint > 0)
    return {**identity, "shape": list(joint.shape), "label_index": "0=background, 1..50=bone-local IDs",
            "counts": [[int(i), int(j), int(joint[i, j])] for i, j in positions]}


def case_result(case: dict, paths: dict, expected_hashes: dict, pair_rows: list[dict], selected: bool, config: dict) -> tuple[dict, dict]:
    used = {k: p for k, p in paths.items() if selected or k != "gt"}
    hashes = {key: sha256(p) for key, p in used.items()}
    if any(hashes[k] != expected_hashes[k] for k in hashes):
        raise ValueError(f"{case['case_id']}: source volume bytes differ from E1")
    bones = ("sacrum", "left_hip", "right_hip") if case["scan_type"] == "pelvic" else ("femur",)
    response_rows = []
    if selected:
        gt, observed = read_volume(paths["gt"], None)
        spacing = tuple(observed["spacing"][::-1])
        for bone in bones:
            cropped = extract_bone(gt, bone, config["boundary_max_iterations"])
            response_rows.extend({**case, **r} for r in responses(cropped, spacing, bone, config))
        del gt, cropped
    maps, expected = {}, None
    for member in MEMBERS:
        array, observed = read_volume(paths[member + "_prediction"], expected)
        expected = observed
        maps[member] = clean_prediction(array, tuple(observed["spacing"][::-1]), 1000., 26)
    replicated, tables = [], []
    for bone in bones:
        roi = np.logical_or.reduce([mask_bone(m, bone) for m in maps.values()])
        vectors = {member: np.where(mask_bone(m, bone)[roi], m[roi], 0) for member, m in maps.items()}
        for first, second in combinations(MEMBERS, 2):
            joint = contingency(vectors[first], vectors[second], bone)
            observed = components(joint)
            old = next(r for r in pair_rows if r["bone"] == bone and r["member_i"] == first and r["member_j"] == second)
            for key in ("n_shared_voxels", "n_union_voxels", *METRICS):
                if not np.isclose(observed[key], float(old[key]), atol=config["numeric_tolerance"], rtol=0):
                    raise ArithmeticError(f"{case['case_id']}/{bone}/{first}/{second}: E1 {key} mismatch")
            identity = {"source": "E1", **case, "bone": bone, "member_i": first, "member_j": second}
            tables.append(sparse_table(joint, identity))
            replicated.extend({**identity, **r} for r in replication(joint, config["replication_factors"], config["numeric_tolerance"]))
    return {"responses": response_rows, "replication": replicated, "contingencies": tables}, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e2.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="one selected pelvic and one selected femur; separate output")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    with (ROOT / config["e1_root"] / ".e1.lock").open("r") as e1_lock:
        fcntl.flock(e1_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        metadata, source, provenance, preserved, config = inputs(config_path, config)
        phantom_rows, phantom_joint = phantoms(config["numeric_tolerance"])
        print("Preflight OK: E1 hashes verified, 20 GT cases selected, 240 E1 pairs available; 32 phantom assertions pass", flush=True)
        if args.check:
            return
        selected = source["selected"]
        if args.smoke:
            selected = [next(r for r in selected if r["scan_type"] == scan) for scan in ("pelvic", "femur")]
            metadata = selected
        selected_ids = {r["case_id"] for r in selected}
        output = ROOT / ("outputs/smoke_e2" if args.smoke else config["output_root"])
        e1_root = (ROOT / config["e1_root"]).resolve()
        if (not output.resolve().is_relative_to(ROOT / "outputs") or
            (output.resolve().is_relative_to(e1_root) and output.resolve() != e1_root / "perturbations") or
            output.resolve().is_relative_to(ROOT / "outputs/stage3_h1")):
            raise ValueError("E2 requires a dedicated output directory outside frozen inputs")
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".e2.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            signature = hashlib.sha256(json.dumps({"provenance": provenance, "smoke_only": args.smoke}, sort_keys=True).encode()).hexdigest()
            manifest_path = output / "e2_manifest.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
                raise ValueError("existing E2 identity differs; preserve it and choose a new output root")
            manifest = {"status": "running", "signature": signature, "smoke_only": args.smoke,
                        "provenance": provenance, "response_case_ids": sorted(selected_ids),
                        "replication_case_ids": [r["case_id"] for r in metadata],
                        "sources": {}, "case_seconds": {}, "started_at_unix": time.time(),
                        "gt_pruned": False, "entropy_from_hard_labels": False,
                        "e1_outputs_modified": False, "source_datasets_modified": False}
            save(manifest_path, manifest)
            all_responses, all_tables = [], []
            phantom_id = {"source": "phantom", "case_id": "", "scan_type": "phantom", "bone": "sacrum", "member_i": "original", "member_j": "merged"}
            all_replication = [{**phantom_id, **r} for r in replication(phantom_joint, config["replication_factors"], config["numeric_tolerance"])]
            all_tables.append(sparse_table(phantom_joint, phantom_id))
            try:
                for index, case in enumerate(metadata, 1):
                    cid = case["case_id"]
                    paths = source["paths"][cid]
                    used = {k: p for k, p in paths.items() if cid in selected_ids or k != "gt"}
                    states = {k: {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for k, p in used.items()}
                    cache = output / "case_cache" / (cid + ".json")
                    if cache.exists():
                        record = json.loads(cache.read_text())
                        if record["signature"] != signature or record["input_states"] != states:
                            raise ValueError(f"{cid}: cache/input identity changed")
                        print(f"[{index}/{len(metadata)}] {cid}: valid cached case", flush=True)
                    else:
                        print(f"[{index}/{len(metadata)}] {cid}: GT responses={cid in selected_ids}, E1 count replication", flush=True)
                        start = time.monotonic()
                        result, hashes = case_result(case, paths, source["e1"]["sources"][cid]["sha256"],
                                                     [r for r in source["pairs"] if r["case_id"] == cid], cid in selected_ids, config)
                        for key, path in used.items():
                            if path.stat().st_size != states[key]["bytes"] or path.stat().st_mtime_ns != states[key]["mtime_ns"]:
                                raise ValueError(f"{cid}/{key}: input changed during processing")
                        record = {"signature": signature, "input_states": states, "input_sha256": hashes,
                                  "result": result, "seconds": time.monotonic() - start}
                        save(cache, record)
                    all_responses.extend(record["result"]["responses"])
                    all_replication.extend(record["result"]["replication"])
                    all_tables.extend(record["result"]["contingencies"])
                    manifest["sources"][cid] = {"states": states, "sha256": record["input_sha256"]}
                    manifest["case_seconds"][cid] = record["seconds"]
                    save(manifest_path, manifest)
                    print(f"  complete: {record['seconds']:.1f}s", flush=True)
                expected_units = sum(3 if r["scan_type"] == "pelvic" else 1 for r in selected)
                units = {(r["case_id"], r["bone"]) for r in all_responses}
                if len(units) != expected_units:
                    raise ValueError("response bone count mismatch")
                expected_pairs = sum(9 if r["scan_type"] == "pelvic" else 3 for r in metadata)
                if len(all_tables) != expected_pairs + 1:
                    raise ValueError("replication pair count mismatch")
                summary = report(output, all_responses, all_replication, phantom_rows, selected)
                with (output / "contingencies.jsonl").open("w") as handle:
                    for table in all_tables:
                        handle.write(json.dumps(table) + "\n")
                for name, digest in preserved.items():
                    if sha256(Path(name)) != digest:
                        raise ValueError(f"E1 artifact changed during E2: {name}")
                save(output / "summary.json", summary)
                manifest.update({"status": "complete", "summary": summary, "completed_at_unix": time.time(),
                                 "output_sha256": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                                                   if p.is_file() and p.suffix in {".csv", ".png", ".pdf", ".jsonl"}}})
                save(manifest_path, manifest)
                print(f"E2 {'SMOKE' if args.smoke else 'FULL'} COMPLETE: {output}", flush=True)
            except BaseException as error:
                manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                save(manifest_path, manifest)
                raise


if __name__ == "__main__":
    main()
