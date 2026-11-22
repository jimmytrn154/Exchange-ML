#!/usr/bin/env python3
"""E4/E5 CPU analysis of frozen E1/E3 artifacts; no training or inference.

Agent smoke: --check or --smoke. User full run: scripts/e4_e5.sbatch.
Outputs live in a separate directory; E1/E2/E3 manifests stay untouched.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exchange_ml.e1_data import MEMBERS
from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.e1_statistics import aurc_batch
from exchange_ml.e4_e5_analysis import (STRATA, enrichment, normalization_scope, qualitative_candidates,
                                       review_budget, risk_source_table, runtime_summary)
from exchange_ml.e4_e5_data import annotated_bones, process_case
from exchange_ml.e5_visualization import plot_qualitative, prepare_example
from run_e1 import save_json, sha256
from run_e3 import load_protocol

MODELS = (*MEMBERS, "stage2")
TABLES = ("scores", "risks", "model_risks", "case_risks", "case_model_risks", "comparison", "paired_differences")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_csv(path: Path) -> list[dict]:
    """Preserve zero-padded case IDs; parse numeric cells without rounding."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key == "case_id":
                continue
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def write_csv(path: Path, rows: list[dict], empty_fields: tuple[str, ...] = ()) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    if not fields:
        raise ValueError(f"missing empty-table schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@contextmanager
def locked(path: Path, *, shared: bool = False):
    # A shared lock requires an existing upstream file and never creates one.
    with path.open("r" if shared else "a") as handle:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield


def state(path: Path) -> dict:
    info = path.stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def check_config(config: dict) -> None:
    expected = {"bootstrap_seed": 20261119, "bootstrap_resamples": 10000,
                "review_fractions": [.1, .2], "retained_fractions": [.8, .9], "severity_fraction": .25,
                "blindspot_f1_tolerance": 1e-12, "blindspot_missing_volume": .1}
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"{key} differs from the declared E4/E5 protocol")
    if config["qualitative"] != {"high_percentile": .75, "low_percentile": .5,
        "shared_error_min_mean_one_minus_f1": .2, "shared_error_max_fa_percentile": .25,
        "crop_margin_voxels": 8, "slices_per_example": 2}:
        raise ValueError("qualitative settings differ from the declared rules")


def validate_rows(data: dict, cases: list[dict]) -> None:
    expected = {(r["case_id"], b) for r in cases for b in annotated_bones(r["scan_type"])}
    metadata = {r["case_id"]: r["scan_type"] for r in cases}
    for name in ("scores", "risks"):
        rows = data[name]
        keys = [(r["case_id"], r["bone"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"{name}: missing/duplicate bone units")
    keys = [(r["case_id"], r["bone"], r["model_id"]) for r in data["model_risks"]]
    if len(keys) != len(set(keys)) or set(keys) != {(cid, b, m) for cid, b in expected for m in MODELS}:
        raise ValueError("model_risks: incomplete model/bone grid")
    for name in ("case_risks", "case_model_risks"):
        keys = [(r["case_id"], r.get("model_id")) for r in data[name]]
        expected_keys = ({(cid, m) for cid in metadata for m in MODELS} if name == "case_model_risks"
                         else {(cid, None) for cid in metadata})
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError(f"{name}: incomplete/duplicate case rows")
    for name in TABLES[:5]:
        if any(r["scan_type"] != metadata[r["case_id"]] for r in data[name]):
            raise ValueError(f"{name}: scan type mismatch")
    by_model = {(r["case_id"], r["bone"], r["model_id"]): r for r in data["model_risks"]}
    cases_by_model = {(r["case_id"], r["model_id"]): r for r in data["case_model_risks"]}
    for case in cases:
        cid = case["case_id"]
        for model in MODELS:
            rows = [by_model[cid, b, model] for b in annotated_bones(case["scan_type"])]
            n = sum(r["n_GT"] for r in rows)
            risk = sum(r["merge_errors"] + r["split_errors"] for r in rows) / max(n, 1)
            if not np.isclose(risk, cases_by_model[cid, model]["merge_split"], atol=1e-12, rtol=1e-12):
                raise ValueError("case-level risk is not the fixed total-error/total-GT risk")
        aggregate = next(r for r in data["case_risks"] if r["case_id"] == cid)
        for source, models in (("member_mean", MEMBERS), ("stage2", ("stage2",))):
            value = np.mean([cases_by_model[cid, m]["merge_split"] for m in models])
            if not np.isclose(value, aggregate[source + "_merge_split"], atol=1e-12, rtol=1e-12):
                raise ValueError("case-level risk aggregation drift")


def load_inputs(config_path: Path, config: dict) -> tuple[dict, dict, dict, dict]:
    """Read-only preflight: verify small artifacts, identities and volume sizes."""
    check_config(config)
    e1_path, e3_path = ROOT / config["e1_config"], ROOT / config["e3_config"]
    e1_config = read_json(e1_path)
    protocol, heldout, directories = load_protocol(e3_path)
    base = ROOT / e1_config["output_root"]
    roots = {"validation": base / "validation", "heldout": ROOT / protocol["output_root"]}
    manifests = {"validation": base / "e1_manifest.json", "heldout": roots["heldout"] / "e3_manifest.json"}
    e1, e3 = (read_json(manifests[name]) for name in ("validation", "heldout"))
    if e1["status"] != "complete" or e1["smoke_only"] or e3["status"] != "complete" or e3["provenance"]["smoke"]:
        raise ValueError("E4/E5 require complete non-smoke E1 and E3")
    require_hash(manifests["validation"], protocol["e1_manifest_sha256"])
    require_hash(e1_path, e1["provenance"]["config_sha256"])
    require_hash(e3_path, e3["provenance"]["protocol_sha256"])
    if signature(e3["provenance"]) != e3["signature"]:
        raise ValueError("E3 provenance signature mismatch")
    for filename, digest in e3["provenance"]["code"].items():
        require_hash(ROOT / filename, digest)
    for name in ("connectivity", "prune_mm3", "iou_threshold"):
        if e1_config[name] != protocol[name]:
            raise ValueError(f"E1/E3 {name} differ")
    validation = read_csv(ROOT / e1_config["scan_metadata"])
    require_hash(ROOT / e1_config["scan_metadata"], e1["provenance"]["scan_metadata_sha256"])
    cases = {"validation": validation, "heldout": heldout}
    if set(e1["case_ids"]) != {r["case_id"] for r in validation} or e3["case_ids"] != [r["case_id"] for r in heldout]:
        raise ValueError("case manifest membership mismatch")
    if set(e1["case_ids"]) & set(e3["case_ids"]):
        raise ValueError("validation overlaps held-out")
    provenance = {"config_sha256": sha256(config_path), "upstream_manifests": {},
                  "upstream_tables": {}, "gt_caches": {}, "software": {
                      name: importlib.metadata.version(name) for name in ("numpy", "scipy", "SimpleITK", "matplotlib")}}
    datasets, specs = {}, {}
    inference = {}
    for model, digest in e3["provenance"]["models"].items():
        p = roots["heldout"] / "models" / model / "inference_manifest.json"
        record = read_json(p)
        identity = record["identity"]
        if (signature(record) != digest or record["status"] != "complete" or record.get("error")
            or set(record["records"]) != set(e3["case_ids"]) or identity["case_ids"] != e3["case_ids"]
            or identity["smoke"] or identity["model"] != model
            or identity["protocol_sha256"] != sha256(e3_path)):
            raise ValueError(f"{model}: E3 inference manifest mismatch")
        inference[model] = record
    if set(inference) != set(MODELS):
        raise ValueError("four E3 inference manifests required")
    for split in cases:
        provenance["upstream_manifests"][split] = sha256(manifests[split])
        manifest = e1 if split == "validation" else e3
        datasets[split] = {}
        for name in TABLES + (("inference_runtime", "evaluation_runtime") if split == "heldout" else ()):
            p = roots[split] / (name + ".csv")
            key = str(p.relative_to(base if split == "validation" else roots[split]))
            require_hash(p, manifest["output_sha256"][key])
            provenance["upstream_tables"][split + "/" + name] = sha256(p)
            datasets[split][name] = read_csv(p)
        validate_rows(datasets[split], cases[split])
        for case in cases[split]:
            cid = case["case_id"]
            if split == "validation":
                original = e1["sources"][cid]
                aliases = {"gt": "gt", "stage2": "stage2", **{m: m + "_prediction" for m in MEMBERS}}
                paths = {k: Path(original["states"][v]["path"]) for k, v in aliases.items()}
                hashes = {k: original["sha256"][v] for k, v in aliases.items()}
                sizes = {k: original["states"][v]["bytes"] for k, v in aliases.items()}
            else:
                cache_path = roots[split] / "case_cache" / (cid + ".json")
                cache = read_json(cache_path)
                if cache["signature"] != e3["signature"]:
                    raise ValueError(f"E3 GT cache signature mismatch: {cid}")
                provenance["gt_caches"][cid] = sha256(cache_path)
                paths = {"gt": directories[cid] / "label.mha"}
                hashes, sizes = {"gt": cache["gt"]["sha256"]}, {"gt": cache["gt"]["bytes"]}
                for model in MODELS:
                    saved = inference[model]["records"][cid]
                    paths[model] = roots[split] / "models" / model / "predictions" / (cid + ".mha")
                    hashes[model] = saved["outputs"]["prediction"]["sha256"]
                    sizes[model] = saved["outputs"]["prediction"]["bytes"]
                paths["image"] = directories[cid] / "image.mha"
                image_state = state(paths["image"])
                if any(inference[m]["records"][cid]["image_state"] != image_state for m in MODELS):
                    raise ValueError(f"E3 source image metadata changed: {cid}")
            for key, size in sizes.items():
                if paths[key].stat().st_size != size:
                    raise ValueError(f"input volume size changed: {paths[key]}")
            specs[split, cid] = {"paths": paths, "hashes": hashes}
    code = [Path(__file__), ROOT / "scripts/run_e1.py", ROOT / "scripts/run_e3.py",
            *(ROOT / "src/exchange_ml" / name for name in ("e4_e5_analysis.py", "e4_e5_data.py", "e5_visualization.py",
              "e1_data.py", "e1_scoring.py", "e1_statistics.py", "instances.py", "metrics.py"))]
    provenance["code_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in code}
    return cases, datasets, specs, {"identity": provenance, "protocol": protocol, "base": base}


def joined_rows(data: dict, ids: set[str]) -> list[dict]:
    risks = {(r["case_id"], r["bone"]): r for r in data["risks"]}
    return [{**r, **risks[r["case_id"], r["bone"]]} for r in data["scores"] if r["case_id"] in ids]


def smoke_comparison(rows: list[dict]) -> list[dict]:
    output = []
    for stratum in STRATA:
        sub = [r for r in rows if stratum == "pooled" or r["scan_type"] == stratum]
        for source in ("member_mean", "stage2"):
            for score in SCORE_NAMES:
                value = aurc_batch(np.array([[r[score] for r in sub]]),
                                   np.array([[[r[source + "_merge_split"]] for r in sub]]))[0, 0]
                output.append({"stratum": stratum, "risk": "merge_split", "score": score, "risk_source": source,
                               "n_units": len(sub), "n_cases": len({r['case_id'] for r in sub}),
                               "aurc": float(value) if len(sub) > 1 else None})
    return output


def report(output: Path, collected: dict, cases: dict, datasets: dict, specs: dict,
           candidates: list[dict], selections: list[dict], config: dict, smoke: bool) -> dict:
    tables = {name: [] for name in ("normalization_scope", "normalization_scope_comparisons", "risk_source",
                                   "enrichment", "review_budget", "f1_volume_blindspots", "blindspot_summary",
                                   "scope_case_scores", "scope_pair_components", "outside_gt_disagreement",
                                   "fragment_coverage", "case_runtime")}
    total_units = 0
    for split, records in collected.items():
        ids = {r["case_id"] for r in cases[split]}
        data = datasets[split]
        rows = joined_rows(data, ids)
        total_units += len(rows)
        case_risks = {r["case_id"]: r for r in data["case_risks"]}
        scope = [{**r, **case_risks[r["case_id"]]} for record in records for r in record["scope_scores"]]
        estimates, contrasts = normalization_scope(scope, 32 if smoke else config["bootstrap_resamples"], config["bootstrap_seed"])
        generated = {"normalization_scope": estimates, "normalization_scope_comparisons": contrasts,
                     "scope_case_scores": scope,
                     "risk_source": risk_source_table(smoke_comparison(rows) if smoke else data["comparison"],
                                                       [] if smoke else data["paired_differences"]),
                     "enrichment": enrichment(rows, config), "review_budget": review_budget(rows, config),
                     "scope_pair_components": [r for record in records for r in record["scope_pairs"]],
                     "outside_gt_disagreement": [r for record in records for r in record["outside_gt"]],
                     "fragment_coverage": [r for record in records for r in record["fragments"]],
                     "case_runtime": [r for record in records for r in record["runtime"]]}
        model_rows = [r for r in data["model_risks"] if r["case_id"] in ids]
        blind = [r for r in model_rows if r["one_minus_f1"] <= config["blindspot_f1_tolerance"]
                 and r["missing_volume"] > config["blindspot_missing_volume"]]
        generated["f1_volume_blindspots"] = blind
        generated["blindspot_summary"] = []
        for stratum in STRATA:
            for model in ("all_models", *MODELS):
                sub = [r for r in blind if (stratum == "pooled" or r["scan_type"] == stratum)
                       and (model == "all_models" or r["model_id"] == model)]
                eligible = [r for r in model_rows if (stratum == "pooled" or r["scan_type"] == stratum)
                            and (model == "all_models" or r["model_id"] == model)]
                generated["blindspot_summary"].append({"stratum": stratum, "model_id": model,
                    "n_evaluated_model_units": len(eligible), "n_blindspot_model_units": len(sub),
                    "n_distinct_cases": len({r["case_id"] for r in sub}),
                    "n_distinct_bone_units": len({(r["case_id"], r["bone"]) for r in sub})})
        for name, rows_out in generated.items():
            tables[name].extend({"split": split, **r,
                                 **({"stratum": r["scan_type"]} if "stratum" not in r and "scan_type" in r else {})}
                                for r in rows_out)
    schemas = {"f1_volume_blindspots": ("split", "stratum", "case_id", "scan_type", "bone", "model_id",
                                        "one_minus_f1", "missing_volume")}
    for name, rows in tables.items():
        write_csv(output / "tables" / (name + ".csv"), rows, schemas.get(name, ()))
    hids = {r["case_id"] for r in cases["heldout"]}
    runtime = runtime_summary([r for r in datasets["heldout"]["inference_runtime"] if r["case_id"] in hids],
                              [r for r in datasets["heldout"]["evaluation_runtime"] if r["case_id"] in hids],
                              tables["case_runtime"])
    write_csv(output / "tables/runtime.csv", runtime)
    examples, visual_sources = [], {}
    for selected in selections:
        if selected["status"] != "selected":
            continue
        spec = specs["heldout", selected["case_id"]]
        image = spec["paths"]["image"]
        before = state(image)
        image_hash = sha256(image)
        volume_states = {k: state(spec["paths"][k]) for k in spec["hashes"]}
        for key, digest in spec["hashes"].items():
            require_hash(spec["paths"][key], digest)
        examples.append(prepare_example(selected, spec["paths"], {**config, **{k: datasets['parameters'][k]
                                        for k in ("prune_mm3", "connectivity", "iou_threshold")}}))
        if before != state(image) or volume_states != {k: state(spec["paths"][k]) for k in spec["hashes"]}:
            raise ValueError(f"input changed during qualitative rendering: {image}")
        visual_sources[selected["case_id"]] = {"path": str(image.resolve()), **before, "sha256": image_hash}
    write_csv(output / "heldout/qualitative_coordinates.csv", [r for e in examples for r in e["coordinates"]],
              ("category", "case_id", "bone", "z_index"))
    write_csv(output / "heldout/qualitative_display_mapping.csv", [r for e in examples for r in e["mapping"]],
              ("category", "case_id", "bone", "model_id", "original_id", "display_id"))
    plot_qualitative(output / "figures/qualitative", examples, selections, candidates,
                     datasets["heldout"]["model_risks"])
    return {"smoke_only": smoke, "n_cases": {s: len(r) for s, r in cases.items()}, "n_bone_units": total_units,
            "bootstrap_resamples": 32 if smoke else config["bootstrap_resamples"],
            "qualitative_selections": selections, "visual_sources": visual_sources,
            "n_fragment_records": len(tables["fragment_coverage"]),
            "n_blindspot_model_units": len(tables["f1_volume_blindspots"]),
            "runtime_limitations": ["E3 model load is included in inference_seconds; separate load timing was not recorded.",
                "E3 CPU scoring/evaluation is combined; historical read/cleanup/components/baselines timings were not recorded.",
                "E3 process peak RSS is a cumulative process high-water mark, not per-case allocation.",
                "E4 CPU timings are new analysis measurements, not a replay or estimate of E3 overhead."],
            "analysis_role": "secondary_sensitivity_and_qualitative; E3 primary method and decision unchanged",
            "table_rows": {name: len(rows) for name, rows in tables.items()}}


def run(config_path: Path, config: dict, smoke: bool, check_only: bool) -> None:
    # Lock upstream runs before reading their manifests and caches.
    base = ROOT / read_json(ROOT / config["e1_config"])["output_root"]
    heldout_root = ROOT / read_json(ROOT / config["e3_config"])["output_root"]
    with ExitStack() as stack:
        stack.enter_context(locked(base / ".e1.lock", shared=True))
        stack.enter_context(locked(heldout_root / ".evaluate.lock", shared=True))
        for model in MODELS:
            stack.enter_context(locked(heldout_root / "models" / model / ".lock", shared=True))
        cases, datasets, specs, inputs = load_inputs(config_path, config)
        print("E4/E5 PREFLIGHT PASS: complete E1/E3; 40 validation + 100 held-out cases; CPU only", flush=True)
        if check_only:
            return
        if smoke:
            cases = {split: [next(r for r in rows if r["scan_type"] == scan) for scan in ("pelvic", "femur")]
                     for split, rows in cases.items()}
        output = (ROOT / config["smoke_output_root" if smoke else "output_root"]).resolve()
        if (not output.is_relative_to(ROOT / "outputs") or output == ROOT / "outputs"
            or any(output == p or p.is_relative_to(output) or output.is_relative_to(p) for p in (
                ROOT / "outputs/stage3_h1", ROOT / "outputs/stage2_backend", heldout_root, base / "validation",
                base / "perturbations", base / "tables", base / "figures", base / "case_cache"))):
            raise ValueError("E4/E5 output overlaps protected existing artifacts")
        output.mkdir(parents=True, exist_ok=True)
        stack.enter_context(locked(output / ".e4_e5.lock"))
        identity = {**inputs["identity"], "smoke_only": smoke, "cases": cases}
        run_signature = signature(identity)
        manifest_path = output / "e4_e5_manifest.json"
        previous = read_json(manifest_path) if manifest_path.exists() else None
        if previous and previous["signature"] != run_signature:
            raise ValueError("E4/E5 identity changed; retain existing output and use a new output_root")
        if previous and previous["status"] == "complete":
            for filename, digest in previous["output_sha256"].items():
                require_hash(output / filename, digest)
            for filename, digest in previous["cache_sha256"].items():
                require_hash(output / filename, digest)
        manifest = {"status": "running", "signature": run_signature, "provenance": identity,
                    "started_at_unix": time.time(), "completed_cases": [], "input_sha256": {}}
        save_json(manifest_path, manifest)
        try:
            hids = {r["case_id"] for r in cases["heldout"]}
            candidates, selections = qualitative_candidates(joined_rows(datasets["heldout"], hids),
                                                            datasets["heldout"]["model_risks"], config)
            # Selection is persisted BEFORE any image is rendered; no visual cherry-picking.
            write_csv(output / "heldout/qualitative_candidates.csv", candidates,
                      ("category", "case_id", "scan_type", "bone", "priority", "selected"))
            write_csv(output / "heldout/qualitative_selections.csv", selections)
            collected = {split: [] for split in cases}
            for split, rows in cases.items():
                for index, case in enumerate(rows, 1):
                    cid = case["case_id"]
                    spec = specs[split, cid]
                    paths = spec["paths"]
                    states = {k: state(paths[k]) for k in spec["hashes"]}
                    for key, digest in spec["hashes"].items():
                        require_hash(paths[key], digest)
                    cache_path = output / "case_cache" / split / (cid + ".json")
                    if cache_path.exists():
                        cache = read_json(cache_path)
                        if cache["signature"] != run_signature or cache["input_states"] != states or cache["input_sha256"] != spec["hashes"]:
                            raise ValueError(f"cache identity/input drift: {split}/{cid}")
                        record = cache["result"]
                        if cache["result_sha256"] != signature(record):
                            raise ValueError(f"cache result checksum mismatch: {split}/{cid}")
                        print(f"[{split} {index}/{len(rows)}] {cid}: verified cache", flush=True)
                    else:
                        print(f"[{split} {index}/{len(rows)}] {cid}: scope -> GT fragments", flush=True)
                        record = process_case(case, paths, inputs["protocol"],
                                              [r for r in datasets[split]["model_risks"] if r["case_id"] == cid])
                        save_json(cache_path, {"signature": run_signature, "input_states": states,
                                              "input_sha256": spec["hashes"], "result": record,
                                              "result_sha256": signature(record)})
                    if states != {k: state(paths[k]) for k in spec["hashes"]}:
                        raise ValueError(f"input changed during analysis: {split}/{cid}")
                    collected[split].append(record)
                    manifest["completed_cases"].append(split + "/" + cid)
                    manifest["input_sha256"][split + "/" + cid] = spec["hashes"]
                    save_json(manifest_path, manifest)
            datasets["parameters"] = inputs["protocol"]
            summary = report(output, collected, cases, datasets, specs, candidates, selections, config, smoke)
            save_json(output / "summary.json", summary)
            manifest.update(status="complete", completed_at_unix=time.time(), summary=summary,
                output_sha256={str(p.relative_to(output)): sha256(p) for folder in ("tables", "figures", "heldout")
                               for p in sorted((output / folder).glob("*")) if p.is_file()},
                cache_sha256={str(p.relative_to(output)): sha256(p) for p in sorted((output / "case_cache").rglob("*.json"))})
            manifest["output_sha256"]["summary.json"] = sha256(output / "summary.json")
            save_json(manifest_path, manifest)
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_json(manifest_path, manifest)
            raise
        print(f"E4/E5 {'SMOKE' if smoke else 'FULL'} COMPLETE: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e4_e5.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="read-only small-artifact hashes and source-size check")
    group.add_argument("--smoke", action="store_true", help="one pelvic + one femur per split; separate output, 32 draws")
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.config.resolve(), read_json(args.config), args.smoke, args.check)


if __name__ == "__main__":
    main()
