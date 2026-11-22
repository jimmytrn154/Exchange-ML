#!/usr/bin/env python3
"""E1 validation decomposition and risk comparison, CPU only.

Agent-typed: --check and --smoke (one pelvic + one femur).
User-typed: default full 40-case run. No training or inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exchange_ml.e1_data import MEMBERS, process_case  # noqa: E402
from exchange_ml.e1_reporting import report  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_inputs(config_path: Path, config: dict) -> tuple[list[dict], dict, str]:
    split_path = ROOT / config["split"]
    split = json.loads(split_path.read_text())
    ids = set(split["split"]["validation"])
    with (ROOT / config["scan_metadata"]).open(newline="") as handle:
        cases = sorted(csv.DictReader(handle), key=lambda r: r["case_id"])
    if len(cases) != 40 or {r["case_id"] for r in cases} != ids:
        raise ValueError("E1 must contain precisely the 40 frozen validation cases")
    if [r["scan_type"] for r in cases].count("pelvic") != 20 or [r["scan_type"] for r in cases].count("femur") != 20:
        raise ValueError("expected 20 pelvic and 20 femur cases")
    if config["connectivity"] != 26 or config["prune_mm3"] != 1000 or config["iou_threshold"] != .1:
        raise ValueError("E1 cleanup/matching differs from the frozen backend")
    if config["bootstrap_resamples"] != 10000:
        raise ValueError("full E1 requires the declared 10000 bootstrap resamples")
    dataset = ROOT / config["dataset_root"]
    directories = {}
    for path in dataset.glob("PENGWIN26_task1_2_train_part*/*"):
        if path.is_dir():
            if path.name in directories:
                raise ValueError(f"duplicate dataset case {path.name}")
            directories[path.name] = path
    members_root = ROOT / config["member_root"]
    provenance = {"config_sha256": sha256(config_path), "split_sha256": sha256(split_path),
                  "scan_metadata_sha256": sha256(ROOT / config["scan_metadata"]), "members": {}}
    expected_seeds = (31001, 31002, 31003)
    manifests = {}
    for member, seed in zip(MEMBERS, expected_seeds):
        path = members_root / member / "validation_inference_manifest.json"
        manifest = json.loads(path.read_text())
        rows = manifest["rows"]
        if manifest["member_id"] != member or manifest["model_seed"] != seed:
            raise ValueError(f"{member}: identity/seed mismatch")
        if len(rows) != 40 or {r["case_id"] for r in rows} != ids:
            raise ValueError(f"{member}: validation case mismatch")
        if any(r["status"] not in {"COMPLETE", "VALID_RESUME"} for r in rows):
            raise ValueError(f"{member}: incomplete inference")
        manifests[member] = {r["case_id"]: r for r in rows}
        provenance["members"][member] = {"manifest_sha256": sha256(path),
                                         **{key: manifest[key] for key in ("anatomy_checkpoint_sha256", "fragment_checkpoint_sha256", "probability_semantics")}}
    stage2_manifest = ROOT / config["stage2_root"] / "validation_inference_manifest.json"
    stage2_rows = json.loads(stage2_manifest.read_text())["rows"]
    if len(stage2_rows) != 40 or {r["case_id"] for r in stage2_rows} != ids:
        raise ValueError("Stage 2 validation case mismatch")
    if any(r["status"] not in {"COMPLETE", "VALID_RESUME"} for r in stage2_rows):
        raise ValueError("Stage 2 incomplete inference")
    provenance["stage2_manifest_sha256"] = sha256(stage2_manifest)
    sources = {}
    for case in cases:
        cid = case["case_id"]
        paths = {"gt": directories[cid] / "label.mha",
                 "stage2": ROOT / config["stage2_root"] / "validation_predictions" / (cid + ".mha")}
        for member in MEMBERS:
            for key, folder, manifest_key in (("prediction", "validation_predictions", "prediction"),
                                               ("probability", "validation_probabilities", "foreground_probability")):
                path = members_root / member / folder / (cid + ".mha")
                if path.resolve() != (ROOT / manifests[member][cid][manifest_key]).resolve():
                    raise ValueError(f"{member}/{cid}: inference path drift")
                paths[member + "_" + key] = path
        for path in paths.values():
            with path.open("rb") as handle:
                handle.read(1)
        sources[cid] = paths
    code = [Path(__file__), *sorted((ROOT / "src/exchange_ml").glob("*.py"))]
    provenance["code_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in code}
    provenance["software"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "SimpleITK", "matplotlib")}
    provenance["python"] = sys.version
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    return cases, {"paths": sources, "provenance": provenance}, signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e1.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only inventory/dependency/provenance check")
    mode.add_argument("--smoke", action="store_true", help="two cases, 32 bootstrap resamples; separate smoke output")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    cases, inputs, signature = load_inputs(config_path, config)
    print("Preflight OK: 40 cases, 120 member maps, 120 probabilities, 40 Stage 2 maps, 40 GT labels", flush=True)
    if args.check:
        return
    if args.smoke:
        cases = [next(r for r in cases if r["scan_type"] == scan) for scan in ("pelvic", "femur")]
    output = ROOT / ("outputs/smoke_e1" if args.smoke else config["output_root"])
    if output.resolve() == (ROOT / "outputs/stage3_h1").resolve() or not output.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("E1 output must be separate from frozen v1 and inside outputs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".e1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "e1_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
            raise ValueError("existing E1 run has different code/config/provenance; retain it and use a new output_root")
        manifest = {"status": "running", "signature": signature, "smoke_only": args.smoke,
                    "case_ids": [r["case_id"] for r in cases], "provenance": inputs["provenance"],
                    "scoring_gt_voxels_loaded": False, "evaluation_gt_loaded": True,
                    "v1_outputs_modified": False, "bootstrap_resamples": 32 if args.smoke else config["bootstrap_resamples"],
                    "bootstrap_seed": config["bootstrap_seed"], "started_at_unix": time.time(),
                    "sources": {}, "case_seconds": {}}
        save_json(manifest_path, manifest)
        tables = {name: [] for name in ("scores", "pair_components", "model_risks", "risks", "case_scores", "case_risks", "case_model_risks")}
        try:
            for index, case in enumerate(cases, 1):
                cid = case["case_id"]
                paths = inputs["paths"][cid]
                states = {k: {"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for k, p in paths.items()}
                cache = output / "case_cache" / (cid + ".json")
                started = time.monotonic()
                if cache.exists():
                    record = json.loads(cache.read_text())
                    if record["signature"] != signature or record["input_states"] != states:
                        raise ValueError(f"{cid}: cache identity/source metadata changed; retain output and investigate")
                    print(f"[{index}/{len(cases)}] {cid}: valid cached case", flush=True)
                else:
                    print(f"[{index}/{len(cases)}] {cid} ({case['scan_type']}): score -> evaluate", flush=True)
                    # Full content digests recorded independently of the scoring arrays.
                    digests = {k: sha256(p) for k, p in paths.items()}
                    result = process_case(cid, case["scan_type"], paths, config)
                    for k, p in paths.items():
                        if p.stat().st_size != states[k]["bytes"] or p.stat().st_mtime_ns != states[k]["mtime_ns"]:
                            raise ValueError(f"source changed while processing: {p}")
                    record = {"signature": signature, "input_states": states, "input_sha256": digests,
                              "result": result, "seconds": time.monotonic() - started}
                    save_json(cache, record)
                for name in tables:
                    tables[name].extend(record["result"][name])
                manifest["sources"][cid] = {"states": states, "sha256": record["input_sha256"]}
                manifest["case_seconds"][cid] = record["seconds"]
                save_json(manifest_path, manifest)
                print(f"  complete: {record['seconds']:.1f}s, {len(record['result']['scores'])} bone units", flush=True)
            expected_units = sum(3 if c["scan_type"] == "pelvic" else 1 for c in cases)
            if (len(tables["scores"]), len(tables["pair_components"]), len(tables["model_risks"])) != (expected_units, 3 * expected_units, 4 * expected_units):
                raise ValueError("output row count mismatch")
            print("Writing E1a/E1b/E1c tables and case-cluster paired bootstrap...", flush=True)
            summary = report(output, tables, manifest["bootstrap_resamples"], config["bootstrap_seed"])
            save_json(output / "summary.json", summary)
            manifest.update({"status": "complete", "summary": summary, "completed_at_unix": time.time(),
                             "output_sha256": {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file() and p.suffix in {".csv", ".png", ".pdf"}}})
            save_json(manifest_path, manifest)
            print(f"E1 {'SMOKE' if args.smoke else 'VALIDATION'} COMPLETE: {output}", flush=True)
        except BaseException as error:
            manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            save_json(manifest_path, manifest)
            raise


if __name__ == "__main__":
    main()
