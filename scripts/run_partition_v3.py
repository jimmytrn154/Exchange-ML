#!/usr/bin/env python3
"""CPU-only graph features and nested grouped development; full run is user-run.

Only the forty E1 validation CTs enter development. No E3 held-out scores,
RibFrac results, GT volumes or probability volumes are read by this runner.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exchange_ml.e1_data import MEMBERS
from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.partition_graph import extract_case, phantom_checks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refuse undefined/empty report schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def lock(path: Path, *, shared: bool = False):
    with path.open("r" if shared else "a") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another writer owns {path}") from error
        yield


def load_config(path: Path) -> dict:
    config = read_json(path)
    if (config["version"], config["candidate"], config["features"], config["target"],
        config["connectivity"], config["prune_mm3"]) != (
            1, "ridge_p1_g_a", ["partition_v1", "graph_degree", "presence"],
            "log1p_member_mean_merge_split", 26, 1000.):
        raise ValueError("configuration differs from fixed graph-development protocol")
    if config["confirmation"]["ready"] or config["confirmation"]["exposure_audit_completed"]:
        raise ValueError("this development runner cannot authorize fresh confirmation")
    if config["reporting"]["bootstrap_resamples"] != 10000:
        raise ValueError("full development requires 10000 bootstrap draws")
    return config


def output_path(config: dict, override: Path | None, smoke: bool) -> Path:
    path = (override or ROOT / config["smoke_output_root" if smoke else "output_root"]).resolve()
    parent = (ROOT / config["parent_root"]).resolve()
    protected = [parent, ROOT / "outputs/stage3_h1", ROOT / "outputs/stage2_backend",
                 ROOT / "outputs/archive", ROOT / "outputs/local_copy_v1", ROOT.parent / "dataset"]
    protected.extend(ROOT / name for name in ("src", "scripts", "configs", "splits", "tests", "__docs__", "paper", "environment"))
    if path == ROOT / "outputs" or path == ROOT or path in ROOT.parents:
        raise ValueError("dedicated development output root required")
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError("output overlaps frozen artifacts or data")
    if any(part in ("external_ribfrac", "Exchange-ML-E6") for part in path.parts):
        raise ValueError("keep active E6 outputs separate")
    full = (ROOT / config["output_root"]).resolve()
    smoke_root = (ROOT / config["smoke_output_root"]).resolve()
    other = full if smoke else smoke_root
    if (full == smoke_root or full.is_relative_to(smoke_root) or smoke_root.is_relative_to(full)
            or path == other or path.is_relative_to(other) or other.is_relative_to(path)):
        raise ValueError("smoke and full roots must be separate")
    return path


def source_inputs(config: dict) -> dict:
    base = ROOT / config["parent_root"]
    manifest_path = base / "e1_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["status"] != "complete" or manifest["smoke_only"] or manifest["signature"] != signature(manifest["provenance"]):
        raise ValueError("complete signed full E1 parent required")
    require_hash(ROOT / config["parent_config"], manifest["provenance"]["config_sha256"])
    for name, digest in manifest["provenance"]["code_sha256"].items():
        require_hash(ROOT / name, digest)
    e1 = read_json(ROOT / config["parent_config"])
    require_hash(ROOT / e1["split"], manifest["provenance"]["split_sha256"])
    require_hash(ROOT / e1["scan_metadata"], manifest["provenance"]["scan_metadata_sha256"])
    split = read_json(ROOT / e1["split"])
    cases = read_csv(ROOT / e1["scan_metadata"])
    ids = [r["case_id"] for r in cases]
    if (len(ids) != 40 or len(set(ids)) != 40 or set(ids) != set(split["split"]["validation"])
        or set(ids) != set(manifest["case_ids"]) or set(ids) != set(manifest["sources"])):
        raise ValueError("exact frozen 40 validation cases required")
    if [r["scan_type"] for r in cases].count("pelvic") != 20 or [r["scan_type"] for r in cases].count("femur") != 20:
        raise ValueError("expected 20 pelvic and 20 femur validation CTs")
    scores, risks, table_hashes = {}, {}, {}
    for name, target in (("scores", scores), ("risks", risks)):
        relative = f"validation/{name}.csv"
        require_hash(base / relative, manifest["output_sha256"][relative])
        table_hashes[relative] = manifest["output_sha256"][relative]
        for row in read_csv(base / relative):
            key = (row["case_id"], row["bone"])
            if key in target:
                raise ValueError("duplicate source case/bone")
            target[key] = row
    expected = {(r["case_id"], bone) for r in cases for bone in
                (("sacrum", "left_hip", "right_hip") if r["scan_type"] == "pelvic" else ("femur",))}
    if scores.keys() != expected or risks.keys() != expected:
        raise ValueError("source scores/risks need exact 80 bone units")
    sources = {}
    for case in cases:
        cid = case["case_id"]
        sources[cid] = {}
        for member in MEMBERS:
            name = member + "_prediction"
            record = manifest["sources"][cid]
            path = Path(record["states"][name]["path"])
            if not path.is_file() or path.stat().st_size != record["states"][name]["bytes"]:
                raise ValueError(f"missing or resized source prediction: {path}")
            sources[cid][member] = {"path": str(path), "sha256": record["sha256"][name]}
    rows = []
    scan_types = {r["case_id"]: r["scan_type"] for r in cases}
    for key in sorted(expected):
        row = {k: scores[key][k] for k in ("case_id", "scan_type", "bone")}
        if row["scan_type"] != risks[key]["scan_type"] or row["scan_type"] != scan_types[key[0]]:
            raise ValueError("source score/risk scan types differ")
        row.update({k: float(scores[key][k]) for k in (*SCORE_NAMES, "component_vi")})
        row.update({k: float(v) for k, v in risks[key].items() if k.startswith(("member_mean_", "stage2_"))})
        if not np.isfinite([v for v in row.values() if isinstance(v, float)]).all():
            raise ValueError("nonfinite source score/risk")
        rows.append(row)
    return {"cases": sorted(cases, key=lambda r: r["case_id"]), "rows": rows, "sources": sources,
            "parent_manifest_sha256": sha256(manifest_path), "source_tables": table_hashes}


def provenance(config: dict, config_path: Path, inputs: dict, smoke: bool, output: Path) -> dict:
    files = [Path(__file__), *sorted((ROOT / "src/exchange_ml").glob("partition_*.py")),
             ROOT / "scripts/partition_v3.sbatch", ROOT / "scripts/submit_partition_v3.sh"]
    return {"config": config, "config_sha256": sha256(config_path),
            "parent_manifest_sha256": inputs["parent_manifest_sha256"], "source_tables": inputs["source_tables"],
            "confirmation_membership": candidate_inventory(config),
            "sources": inputs["sources"], "smoke_only": smoke, "output_root": str(output),
            "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
            "software": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "SimpleITK")},
            "python": sys.version}


def verify_case_sources(sources: dict) -> None:
    for record in sources.values():
        require_hash(Path(record["path"]), record["sha256"])


def process_cached_case(case: dict, sources: dict, baseline_rows: list[dict], config: dict,
                        output: Path, run_signature: str) -> tuple[dict, str]:
    cid = case["case_id"]
    cache = output / "cases" / f"{cid}.json"
    verify_case_sources(sources)
    identity = signature({"run": run_signature, "case": case, "sources": sources})
    if cache.exists():
        saved = read_json(cache)
        if saved["identity"] != identity or saved["payload_sha256"] != signature(saved["payload"]):
            raise ValueError(f"case cache identity/content drift: {cid}")
        payload = saved["payload"]
    else:
        paths = [Path(sources[m]["path"]) for m in MEMBERS]
        started = time.perf_counter()
        payload = extract_case(cid, case["scan_type"], paths, config)
        payload["feature_extraction_seconds"] = time.perf_counter() - started
        verify_case_sources(sources)
    expected_bones = {"sacrum", "left_hip", "right_hip"} if case["scan_type"] == "pelvic" else {"femur"}
    if (len(payload["features"]) != len(expected_bones)
            or {r["bone"] for r in payload["features"]} != expected_bones
            or len(payload["pairs"]) != 3 * len(expected_bones)):
        raise ValueError(f"case feature schema mismatch: {cid}")
    old = {(r["case_id"], r["bone"]): r for r in baseline_rows}
    for row in payload["features"]:
        if row["case_id"] != cid or row["scan_type"] != case["scan_type"]:
            raise ValueError(f"case feature identity mismatch: {cid}")
        source = old[row["case_id"], row["bone"]]
        for name in ("presence", "partition_v1", "partition_v2", "component_vi"):
            if not np.isclose(row[name], source[name], rtol=1e-10, atol=1e-10):
                raise ValueError(f"E1 feature mismatch: {cid}/{row['bone']}/{name}")
    if not cache.exists():
        save_json(cache, {"identity": identity, "payload_sha256": signature(payload), "payload": payload})
    return payload, sha256(cache)


def synthetic_rows() -> list[dict]:
    """Twenty tiny fake CTs for testing CV/reporting, never external evidence."""
    rng = np.random.default_rng(6122)
    rows = []
    for index in range(20):
        scan = "pelvic" if index < 10 else "femur"
        for bone in (("sacrum", "left_hip", "right_hip") if scan == "pelvic" else ("femur",)):
            p, a, g = rng.uniform(0, .15), rng.uniform(0, 1), rng.uniform(0, 2)
            row = {"case_id": f"synthetic_{index:03}", "scan_type": scan, "bone": bone,
                   **{n: float(rng.uniform(0, .5)) for n in SCORE_NAMES},
                   "partition_v1": float(p), "presence": float(a), "graph_degree": float(g),
                   "all_members_empty": bool(index == 0), "graph_no_shared_fraction": 0.}
            for source in ("member_mean", "stage2"):
                row.update({source + "_" + r: float(rng.uniform(0, 1)) for r in
                            ("missing_volume", "one_minus_f1", "missing", "merge", "split", "one_minus_dice")})
                row[source + "_merge_split"] = float(g + .2 * a + rng.uniform(0, .3))
            rows.append(row)
    return rows


def candidate_inventory(config: dict) -> dict:
    """Set arithmetic only. No volume, risk or inference result is opened."""
    settings = config["confirmation"]
    split_path, e3_path = ROOT / settings["split"], ROOT / settings["e3_protocol"]
    split, e3 = read_json(split_path), read_json(e3_path)
    from exchange_ml.stage2_backend import select_stage2_case_ids
    initial = set(select_stage2_case_ids(split, settings["trajectory_seed"])["train"])
    pool, held = set(split["split"]["pool"]), set(e3["case_ids"])
    remaining = sorted(pool - initial - held)
    if len(remaining) != 110 or set(remaining) & set(split["split"]["validation"] + split["split"]["test"]):
        raise ValueError("candidate case membership differs from inspected split")
    return {"n_candidate_cases": len(remaining), "case_ids": remaining,
            "split_sha256": sha256(split_path), "e3_protocol_sha256": sha256(e3_path),
            "confirmation_ready": False, "exposure_audit_completed": False,
            "role": "membership_only_not_proof_of_unseen_outcomes"}


def run(config: dict, config_path: Path, output: Path, *, smoke: bool) -> None:
    from exchange_ml.partition_cv import nested_cv
    from exchange_ml.partition_reporting import report
    parent = ROOT / config["parent_root"]
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(lock(parent / ".e1.lock", shared=True))
        stack.enter_context(lock(output / ".partition.lock"))
        inputs = source_inputs(config)
        prov = provenance(config, config_path, inputs, smoke, output)
        run_signature = signature(prov)
        path = output / "partition_manifest.json"
        if path.exists():
            state = read_json(path)
            if state["signature"] != run_signature or state["provenance"] != prov:
                raise ValueError("run identity changed: preserve outputs and choose a new output root")
        else:
            if any(p.name != ".partition.lock" for p in output.iterdir()):
                raise ValueError("nonempty output root without a partition manifest; preserve it and choose a new root")
            state = {"status": "running", "signature": run_signature, "provenance": prov,
                     "started_at_unix": time.time(), "cache_sha256": {}}
        cases = inputs["cases"]
        if smoke:
            cases = [next(c for c in cases if c["case_id"] == cid) for cid in ("004", "254")]
        if state["status"] == "complete":
            for case in cases:
                verify_case_sources(inputs["sources"][case["case_id"]])
            for name, digest in {**state["output_sha256"], **state["cache_sha256"]}.items():
                require_hash(output / name, digest)
            print("PARTITION DEVELOPMENT ALREADY COMPLETE" if not smoke else "PARTITION SMOKE ALREADY COMPLETE")
            return
        save_json(path, state)
        write_csv(output / "phantom_checks.csv", phantom_checks())
        features, pairs, runtime = [], [], []
        for index, case in enumerate(cases, 1):
            cid = case["case_id"]
            relative = f"cases/{cid}.json"
            if relative in state["cache_sha256"]:
                require_hash(output / relative, state["cache_sha256"][relative])
            payload, digest = process_cached_case(case, inputs["sources"][cid], inputs["rows"], config, output, run_signature)
            features.extend(payload["features"])
            pairs.extend(payload["pairs"])
            runtime.append({"case_id": cid, "scan_type": case["scan_type"],
                            "feature_extraction_seconds": payload["feature_extraction_seconds"],
                            "scope": "three_map_read_cleanup_graph_and_E1_component_checks_excludes_SHA256"})
            state["cache_sha256"][relative] = digest
            save_json(path, state)
            print(f"Graph features {index}/{len(cases)}: {cid}", flush=True)
        write_csv(output / "graph_features.csv", features)
        write_csv(output / "graph_pairs.csv", pairs)
        write_csv(output / "runtime.csv", runtime)
        baseline = {(r["case_id"], r["bone"]): r for r in inputs["rows"]}
        rows = [{**baseline[r["case_id"], r["bone"]], **r} for r in features]
        effective = json.loads(json.dumps(config))
        if smoke:
            rows = synthetic_rows()
            effective["cv"].update(outer_folds=2, inner_folds=2)
            effective["reporting"]["bootstrap_resamples"] = 32
        cv = nested_cv(rows, effective)
        tables = report(cv["oof_scores"], effective)
        write_csv(output / "oof_scores.csv", cv["oof_scores"])
        write_csv(output / "folds.csv", cv["folds"])
        save_json(output / "tuning.json", cv["tuning"])
        save_json(output / "fold_models.json", cv["fold_models"])
        save_json(output / "models.json", cv["models"])
        save_json(output / "final_model.json", cv["final_model"])
        for name, table in tables.items():
            if name != "summary":
                write_csv(output / f"{name}.csv", table)
        save_json(output / "confirmation_candidates.json", prov["confirmation_membership"])
        summary = {"role": "synthetic_CV_reporting_smoke_only" if smoke else "development_nested_OOF_not_confirmation",
                   "smoke_only": smoke, "real_feature_cases": len(cases), "real_feature_units": len(features),
                   "cv": cv["summary"], "reporting": tables["summary"],
                   "primary_candidate": "ridge_p1_g_a", "primary_control": "partition_v1",
                   "fresh_confirmation_ready": False, "scoring_gt_voxels_loaded": False,
                   "risk_labels": "frozen E1 CSV only", "CI_scope": "conditional_on_fixed_OOF_predictions_not_training_procedure"}
        save_json(output / "summary.json", summary)
        require_hash(parent / "e1_manifest.json", inputs["parent_manifest_sha256"])
        names = [p for p in output.iterdir() if p.is_file() and p.name not in (path.name, ".partition.lock") and not p.name.endswith(".tmp")]
        state.update(status="complete", completed_at_unix=time.time(), summary=summary,
                     output_sha256={p.name: sha256(p) for p in sorted(names)})
        save_json(path, state)
    print("PARTITION CPU SMOKE COMPLETE" if smoke else "PARTITION DEVELOPMENT COMPLETE; NOT FRESH CONFIRMATION")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/partition_v3.json")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = output_path(config, args.output_root, args.mode == "smoke")
    if args.mode == "check":
        inputs = source_inputs(config)
        inventory = candidate_inventory(config)
        inventory.pop("case_ids")
        print(json.dumps({"status": "preflight_pass", "n_cases": len(inputs["cases"]),
                          "n_units": len(inputs["rows"]), "output": str(output),
                          "confirmation": inventory}, indent=2))
    else:
        run(config, args.config, output, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
