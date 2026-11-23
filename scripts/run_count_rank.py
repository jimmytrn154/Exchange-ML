#!/usr/bin/env python3
"""CPU-only P1+count ablation from audited frozen v3/v4 tables; full is user-run."""
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
from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.partition_cv import grouped_folds

RISKS = tuple(f"{source}_{risk}" for source in ("member_mean", "stage2")
              for risk in ("merge_split", "one_minus_f1", "missing_volume"))
CODE = ("scripts/run_count_rank.py", "scripts/count_rank.sbatch", "scripts/submit_count_rank.sh",
        "src/exchange_ml/count_rank.py", "src/exchange_ml/count_rank_reporting.py",
        "scripts/run_partition_v4.py", "src/exchange_ml/structural_rank.py",
        "src/exchange_ml/structural_rank_reporting.py", "src/exchange_ml/partition_cv.py", "src/exchange_ml/e1_statistics.py",
        "src/exchange_ml/e1_scoring.py", "src/exchange_ml/metrics.py", "src/exchange_ml/__init__.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"nonempty report table required: {path}")
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
        config["reporting"]["primary_estimand"]) != (
            1, "pairwise_p1_count", ["partition_v1", "predicted_count_mean"], "member_mean_merge_split",
            "mean_outer_fold_aurc"):
        raise ValueError("configuration differs from fixed P1+count ranking protocol")
    if config["confirmation_ready"]:
        raise ValueError("development runner cannot authorize confirmation")
    if config["reporting"]["bootstrap_resamples"] != 10000:
        raise ValueError("full development requires 10000 bootstrap draws")
    return config


def output_path(config: dict, override: Path | None, smoke: bool) -> Path:
    output = (override or ROOT / config["smoke_output_root" if smoke else "output_root"]).resolve()
    protected = [(ROOT / config["parent_root"]).resolve(), (ROOT / config["v4_root"]).resolve(), ROOT.parent / "dataset"]
    protected.extend(ROOT / name for name in ("src", "scripts", "configs", "splits", "tests", "__docs__", "paper", "environment"))
    protected.extend(ROOT / "outputs" / name for name in (
        "stage3_h1_v2", "stage3_h1", "stage2_backend", "archive", "local_copy_v1", "smoke_partition_v3", "partition_v4_development_v1", "smoke_partition_v4"))
    protected = [p.resolve() for p in protected]
    if output == ROOT or output == ROOT / "outputs" or output in ROOT.parents:
        raise ValueError("dedicated output root required")
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("output overlaps protected source/data/artifacts")
    if any(p == "external_ribfrac" or p.startswith("Exchange-ML-E6") for p in output.parts):
        raise ValueError("keep active E6 outputs separate")
    full = (ROOT / config["output_root"]).resolve()
    smoke_root = (ROOT / config["smoke_output_root"]).resolve()
    other = full if smoke else smoke_root
    if (full == smoke_root or full.is_relative_to(smoke_root) or smoke_root.is_relative_to(full)
            or output == other or output.is_relative_to(other) or other.is_relative_to(output)):
        raise ValueError("smoke and full output roots must be separate")
    return output


def indexed(rows: list[dict], description: str) -> dict:
    result = {}
    for row in rows:
        key = row["case_id"], row["bone"]
        if key in result:
            raise ValueError(f"duplicate {description} case/bone")
        result[key] = row
    return result


def source_inputs(config: dict) -> dict:
    """Audit small recorded artifacts only. Never open images or checkpoints."""
    base = ROOT / config["parent_root"]
    manifest_path = base / "partition_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest["status"] != "complete" or manifest["provenance"]["smoke_only"]
            or manifest["summary"]["smoke_only"] or manifest["signature"] != signature(manifest["provenance"])):
        raise ValueError("complete full v3 parent with valid signature required")
    for name, digest in {**manifest["output_sha256"], **manifest["cache_sha256"]}.items():
        require_hash(base / name, digest)
    for name, digest in manifest["provenance"]["code_sha256"].items():
        require_hash(ROOT / name, digest)
    require_hash(ROOT / "configs/partition_v3.json", manifest["provenance"]["config_sha256"])
    v3 = manifest["provenance"]["config"]
    if read_json(ROOT / "configs/partition_v3.json") != v3:
        raise ValueError("parent embedded config differs from hash-bound config")
    e1_root = ROOT / v3["parent_root"]
    e1_path = e1_root / "e1_manifest.json"
    require_hash(e1_path, manifest["provenance"]["parent_manifest_sha256"])
    e1 = read_json(e1_path)
    if e1["status"] != "complete" or e1["smoke_only"] or e1["signature"] != signature(e1["provenance"]):
        raise ValueError("complete E1 validation parent required")
    for name, digest in e1["provenance"]["code_sha256"].items():
        require_hash(ROOT / name, digest)
    require_hash(ROOT / v3["parent_config"], e1["provenance"]["config_sha256"])
    e1_config = read_json(ROOT / v3["parent_config"])
    for key, filename in (("split_sha256", "split"), ("scan_metadata_sha256", "scan_metadata")):
        require_hash(ROOT / e1_config[filename], e1["provenance"][key])
    split = read_json(ROOT / e1_config["split"])
    cases = read_csv(ROOT / e1_config["scan_metadata"])
    ids = [c["case_id"] for c in cases]
    if (len(ids) != 40 or len(set(ids)) != 40 or set(ids) != set(split["split"]["validation"])
            or set(ids) != set(e1["case_ids"])):
        raise ValueError("exact frozen 40 validation CTs required")
    if [c["scan_type"] for c in cases].count("pelvic") != 20 or [c["scan_type"] for c in cases].count("femur") != 20:
        raise ValueError("20 pelvic and 20 femur CTs required")
    expected = {(c["case_id"], bone) for c in cases for bone in
                (("sacrum", "left_hip", "right_hip") if c["scan_type"] == "pelvic" else ("femur",))}
    source_hashes = {}
    for name in ("scores", "risks"):
        relative = f"validation/{name}.csv"
        digest = e1["output_sha256"][relative]
        if manifest["provenance"]["source_tables"][relative] != digest:
            raise ValueError("v3/E1 source table identity differs")
        require_hash(e1_root / relative, digest)
        source_hashes[str((e1_root / relative).relative_to(ROOT))] = digest
    baseline = indexed(read_csv(e1_root / "validation/scores.csv"), "E1 scores")
    risks = indexed(read_csv(e1_root / "validation/risks.csv"), "E1 risks")
    features = indexed(read_csv(base / "graph_features.csv"), "v3 graph features")
    oof = indexed(read_csv(base / "oof_scores.csv"), "v3 OOF")
    if any(set(table) != expected for table in (baseline, risks, features, oof)):
        raise ValueError("exact 80 validation case/bone units required")
    scans = {c["case_id"]: c["scan_type"] for c in cases}
    parent_assignment = grouped_folds(list(oof.values()), v3["cv"]["outer_folds"], v3["cv"]["seed"])
    rows = []
    for key in sorted(expected):
        source = oof[key]
        if any(table[key]["scan_type"] != scans[key[0]] for table in (baseline, risks, features, oof)):
            raise ValueError("source scan type mismatch")
        if int(source["outer_fold"]) != parent_assignment[key[0]]:
            raise ValueError("parent OOF fold membership drift")
        row = {name: source[name] for name in ("case_id", "scan_type", "bone")}
        for name in (*SCORE_NAMES, "graph_degree", "predicted_count_mean", *RISKS):
            value = float(source[name])
            reference = risks[key][name] if name in RISKS else features[key][name] if name in ("graph_degree", "predicted_count_mean") else baseline[key][name]
            if not np.isfinite(value) or value != float(reference):
                raise ValueError(f"frozen feature/risk mismatch: {key}/{name}")
            row[name] = value
        flag = source["all_members_empty"]
        if flag not in ("True", "False") or flag != features[key]["all_members_empty"]:
            raise ValueError("source omission flag mismatch")
        row["all_members_empty"] = flag == "True"
        n_gt = int(risks[key]["n_GT"])
        if n_gt < 1:
            raise ValueError("positive GT-count diagnostic metadata required")
        row["n_GT"] = n_gt  # Diagnostic-only: never part of FEATURES.
        rows.append(row)
    v4_root = ROOT / config["v4_root"]
    v4_path = v4_root / "rank_manifest.json"
    v4 = read_json(v4_path)
    if (v4["status"] != "complete" or v4["provenance"]["smoke_only"]
            or v4["signature"] != signature(v4["provenance"])):
        raise ValueError("completed real v4 control required")
    v4_config = read_json(ROOT / "configs/partition_v4.json")
    require_hash(ROOT / "configs/partition_v4.json", v4["provenance"]["config_sha256"])
    if v4_config != v4["provenance"]["config"] or config["cv"] != v4_config["cv"]:
        raise ValueError("count ablation must retain exact v4 CV settings")
    if v4["provenance"]["parent_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("v4 control does not share the same v3 parent")
    for name, digest in v4["provenance"]["code_sha256"].items():
        require_hash(ROOT / name, digest)
    for name, digest in v4["output_sha256"].items():
        require_hash(v4_root / name, digest)
    control = indexed(read_csv(v4_root / "oof_scores.csv"), "v4 OOF control")
    if set(control) != expected:
        raise ValueError("v4 control membership differs")
    for row in rows:
        key = row["case_id"], row["bone"]
        saved = control[key]
        if int(saved["outer_fold"]) != parent_assignment[key[0]]:
            raise ValueError("v4 control folds differ")
        for name in (*SCORE_NAMES, "graph_degree", *RISKS):
            if float(saved[name]) != row[name]:
                raise ValueError("v4 raw control feature/risk differs")
        row["pairwise_p1_g"] = float(saved["pairwise_p1_g"])
        if not np.isfinite(row["pairwise_p1_g"]):
            raise ValueError("v4 OOF control must be finite")
    return {"v4_manifest_sha256": sha256(v4_path), "rows": rows, "cases": cases, "parent_manifest_sha256": sha256(manifest_path),
            "source_table_sha256": source_hashes, "parent_output_hashes_verified": len(manifest["output_sha256"]),
            "parent_cache_hashes_verified": len(manifest["cache_sha256"])}


def provenance(config: dict, config_path: Path, inputs: dict, smoke: bool, output: Path) -> dict:
    return {"config": config, "config_sha256": sha256(config_path),
            "parent_manifest_sha256": inputs["parent_manifest_sha256"],
            "v4_manifest_sha256": inputs["v4_manifest_sha256"],
            "source_table_sha256": inputs["source_table_sha256"], "smoke_only": smoke,
            "output_root": str(output), "code_sha256": {p: sha256(ROOT / p) for p in CODE},
            "software": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}, "python": sys.version}


def synthetic_rows() -> list[dict]:
    """Twenty artificial CTs; never mixed with the real development rows."""
    rng = np.random.default_rng(4122)
    rows = []
    for index in range(20):
        scan = "pelvic" if index < 10 else "femur"
        for bone in (("sacrum", "left_hip", "right_hip") if scan == "pelvic" else ("femur",)):
            p1, graph = float(rng.uniform(0, .1)), float(rng.uniform(0, 1))
            row = {"case_id": f"synthetic_{index:03}", "scan_type": scan, "bone": bone,
                   **{name: float(rng.uniform(0, .5)) for name in SCORE_NAMES},
                   "partition_v1": p1, "graph_degree": graph, "predicted_count_mean": float(rng.integers(1, 12)),
                   "pairwise_p1_g": graph - .2, "n_GT": int(rng.integers(1, 12)), "all_members_empty": False}
            for source in ("member_mean", "stage2"):
                row[source + "_merge_split"] = float(np.round(2 * p1 + .15 * row["predicted_count_mean"] + graph, 1))
                row[source + "_one_minus_f1"] = float(rng.uniform(0, 1))
                row[source + "_missing_volume"] = float(rng.uniform(0, 1))
            rows.append(row)
    return rows


def validate_cv_payload(cv: dict, rows: list[dict], config: dict) -> None:
    """Verify an orphan/reused checkpoint against inputs, groups and models."""
    from exchange_ml.count_rank import predict_model
    original = indexed(rows, "CV input")
    oof = indexed(cv["oof_scores"], "CV checkpoint")
    if original.keys() != oof.keys():
        raise ValueError("CV checkpoint case/bone membership drift")
    cases = {r["case_id"] for r in rows}
    assignment = grouped_folds(rows, min(config["cv"]["outer_folds"], len(cases)), config["cv"]["seed"])
    for key, row in original.items():
        saved = oof[key]
        if any(saved.get(name) != value for name, value in row.items()) or saved["outer_fold"] != assignment[key[0]]:
            raise ValueError("CV checkpoint inputs or fold identity drift")
        if not np.isfinite(saved["pairwise_p1_count"]):
            raise ValueError("CV checkpoint has nonfinite predictions")
    recorded_folds = cv["folds"]
    if len(recorded_folds) != len(cases) or {r["case_id"]: r["outer_fold"] for r in recorded_folds} != assignment:
        raise ValueError("CV checkpoint fold manifest drift")
    records = cv["fold_models"]
    if len(records) != len(set(assignment.values())) or {r["outer_fold"] for r in records} != set(assignment.values()):
        raise ValueError("CV checkpoint fold models incomplete")
    for record in records:
        fold, model = record["outer_fold"], record["model"]
        test = [r for r in cv["oof_scores"] if assignment[r["case_id"]] == fold]
        expected_test = {r["case_id"] for r in test}
        if (set(model["train_case_ids"]) != cases - expected_test
                or set(record["test_case_ids"]) != expected_test):
            raise ValueError("CV checkpoint training/test group drift")
        prediction = predict_model(model, test)
        if not np.allclose(prediction, [r["pairwise_p1_count"] for r in test], rtol=1e-12, atol=1e-12):
            raise ValueError("CV checkpoint predictions differ from saved model")
    if set(cv["final_model"]["train_case_ids"]) != cases:
        raise ValueError("CV checkpoint final model training membership drift")
    predict_model(cv["final_model"], [])  # Validate serialized predictor schema without labels.


def run(config: dict, config_path: Path, output: Path, *, smoke: bool) -> None:
    from exchange_ml.count_rank import nested_cv
    from exchange_ml.count_rank_reporting import report
    parent = ROOT / config["parent_root"]
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(lock(parent / ".partition.lock", shared=True))
        stack.enter_context(lock(ROOT / config["v4_root"] / ".rank.lock", shared=True))
        stack.enter_context(lock(output / ".rank.lock"))
        print("Auditing frozen v3 tables and recorded hashes", flush=True)
        inputs = source_inputs(config)
        prov = provenance(config, config_path, inputs, smoke, output)
        identity = signature(prov)
        path = output / "rank_manifest.json"
        if path.exists():
            state = read_json(path)
            if state["signature"] != identity or state["provenance"] != prov:
                raise ValueError("run identity changed; preserve outputs and choose a new output root")
        else:
            if any(p.name != ".rank.lock" for p in output.iterdir()):
                raise ValueError("nonempty output root without rank manifest")
            state = {"status": "running", "signature": identity, "provenance": prov, "started_at_unix": time.time()}
        if state["status"] == "complete":
            for name, digest in state["output_sha256"].items():
                require_hash(output / name, digest)
            print("COUNT ABLATION ALREADY COMPLETE (verified, no refit)")
            return
        save_json(path, state)
        effective = json.loads(json.dumps(config))
        rows = synthetic_rows() if smoke else inputs["rows"]
        if smoke:
            effective["reporting"]["bootstrap_resamples"] = 32
        checkpoint = output / "cv_checkpoint.json"
        if "cv_checkpoint_sha256" in state:
            require_hash(checkpoint, state["cv_checkpoint_sha256"])
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved["signature"] != identity or saved["payload_sha256"] != signature(saved["payload"]):
                raise ValueError("CV checkpoint identity/content drift")
            cv = saved["payload"]
            validate_cv_payload(cv, rows, effective)
            print("Reusing verified nested-CV checkpoint", flush=True)
        else:
            print(f"Nested grouped ranking CV: {len({r['case_id'] for r in rows})} CTs, {len(rows)} bones; synthetic={smoke}", flush=True)
            cv = nested_cv(rows, effective)
            validate_cv_payload(cv, rows, effective)
            save_json(checkpoint, {"signature": identity, "payload_sha256": signature(cv), "payload": cv})
        state["cv_checkpoint_sha256"] = sha256(checkpoint)
        state["stage"] = "cv_complete"
        save_json(path, state)
        print(f"Fold-local and pooled diagnostics; bootstrap draws={effective['reporting']['bootstrap_resamples']}", flush=True)
        tables = report(cv["oof_scores"], effective)
        write_csv(output / "oof_scores.csv", cv["oof_scores"])
        write_csv(output / "folds.csv", cv["folds"])
        save_json(output / "tuning.json", cv["tuning"])
        save_json(output / "fold_models.json", cv["fold_models"])
        final_model = {**cv["final_model"], "smoke_only": smoke,
                       "analysis_role": "synthetic_smoke" if smoke else "development_not_confirmed",
                       "execution_signature": identity}
        save_json(output / "final_model.json", final_model)
        for name, table in tables.items():
            if name != "summary":
                write_csv(output / f"{name}.csv", table)
        summary = {"role": "synthetic_CV_reporting_smoke_only" if smoke else "development_after_count_diagnosis_not_confirmation",
                   "smoke_only": smoke, "source_cases": len(inputs["cases"]), "source_units": len(inputs["rows"]),
                   "cv": cv["summary"], "reporting": tables["summary"],
                   "candidate": "pairwise_p1_count", "primary_control": "partition_v1",
                   "primary_estimand": "mean_outer_fold_aurc", "confirmation_ready": False,
                   "volumes_or_checkpoints_read": False, "CT_policy_tested": False,
                   "CI_scope": "conditional_fixed_outer_folds_and_OOF_scores_not_training_uncertainty"}
        save_json(output / "summary.json", summary)
        require_hash(parent / "partition_manifest.json", inputs["parent_manifest_sha256"])
        require_hash(ROOT / config["v4_root"] / "rank_manifest.json", inputs["v4_manifest_sha256"])
        files = [p for p in output.iterdir() if p.is_file() and p.name not in (path.name, ".rank.lock") and not p.name.endswith(".tmp")]
        state.update(status="complete", stage="reporting_complete", summary=summary, completed_at_unix=time.time(),
                     output_sha256={p.name: sha256(p) for p in sorted(files)})
        save_json(path, state)
    print("COUNT ABLATION CPU SMOKE COMPLETE" if smoke else "COUNT ABLATION DEVELOPMENT COMPLETE; NOT CONFIRMATION")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "full"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/count_rank.json")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = output_path(config, args.output_root, args.mode == "smoke")
    if args.mode == "check":
        with lock(ROOT / config["parent_root"] / ".partition.lock", shared=True):
            inputs = source_inputs(config)
        print(json.dumps({"status": "preflight_pass", "n_cases": len(inputs["cases"]), "n_units": len(inputs["rows"]),
                          "parent_output_hashes_verified": inputs["parent_output_hashes_verified"],
                          "parent_cache_hashes_verified": inputs["parent_cache_hashes_verified"],
                          "output_root": str(output), "confirmation_ready": False}, indent=2))
    else:
        run(config, args.config, output, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
