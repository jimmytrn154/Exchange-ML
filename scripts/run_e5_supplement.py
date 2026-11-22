#!/usr/bin/env python3
"""Post-heldout E5 supplement; CPU bootstrap and fixed-case figures only.

Agent-typed: --check / --smoke (32 draws, two cases per split, one figure).
User-typed: full 10000-draw run through scripts/e5_supplement.sbatch.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
from pathlib import Path
import os
import time

import numpy as np

from run_e4_e5 import (ROOT, MEMBERS, MODELS, joined_rows, load_inputs, locked, read_csv,
                       read_json, require_hash, save_json, sha256, signature, state, write_csv, dataset_table_rows)
from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.e5_supplement_statistics import RISKS, SOURCES, budget_analysis
from exchange_ml.e5_supplement_figures import plot_budget_figures, plot_qualitative_supplement
from exchange_ml.e5_visualization import prepare_example

ROLE = "exploratory_post_heldout"


def check_config(config: dict) -> None:
    expected = {"analysis_role": ROLE, "bootstrap_seed": 20261119, "bootstrap_resamples": 10000,
                "retained_fractions": [.8, .9], "risks": list(RISKS), "risk_sources": list(SOURCES),
                "qualitative_selection": "copy_parent_without_reselection", "smoke_render_limit": 1}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"supplement setting {key} differs from the approved plan")
    if not isinstance(config["batch_size"], int) or not 1 <= config["batch_size"] <= 1024:
        raise ValueError("batch_size must be an integer in 1..1024")


def verify_parent(root: Path) -> dict:
    """Check the corrected source report while retaining its execution identity."""
    parent = read_json(root / "e4_e5_manifest.json")
    if parent["status"] != "complete" or parent["summary"]["smoke_only"]:
        raise ValueError("a complete non-smoke E4/E5 parent is required")
    if signature(parent["provenance"]) != parent["signature"]:
        raise ValueError("parent execution signature differs from provenance")
    for group in ("output_sha256", "cache_sha256"):
        for filename, digest in parent[group].items():
            require_hash(root / filename, digest)
    corrected_code = {}
    for revision in parent.get("report_corrections", []):
        receipt_path = root / revision["receipt"]
        require_hash(receipt_path, revision["receipt_sha256"])
        receipt = read_json(receipt_path)
        if receipt["original_compute_signature"] != parent["signature"]:
            raise ValueError("report correction belongs to a different compute run")
        for name, digest in receipt["archives"].items():
            require_hash(receipt_path.parent / name, digest)
        require_hash(receipt_path.parent / "e4_e5_manifest.before.json", receipt["parent_manifest_sha256"])
        require_hash(root / "tables/f1_volume_blindspots.csv", receipt["new_table_sha256"])
        corrected_code.update(receipt["correction_code_sha256"])
    for filename, digest in parent["provenance"]["code_sha256"].items():
        require_hash(ROOT / filename, corrected_code.get(filename, digest))
    return parent


def validate_inputs(parent: dict, inputs: dict, cases: dict, datasets: dict) -> None:
    for key in ("config_sha256", "upstream_manifests", "upstream_tables", "gt_caches", "software"):
        if parent["provenance"][key] != inputs["identity"][key]:
            raise ValueError(f"E1/E3 inputs differ from parent E4/E5: {key}")
    expected = {split + "/" + row["case_id"] for split, rows in cases.items() for row in rows}
    if len(parent["completed_cases"]) != len(expected) or set(parent["completed_cases"]) != expected:
        raise ValueError("parent E4/E5 has missing or duplicate case records")
    for split, rows in cases.items():
        joined = joined_rows(datasets[split], {r["case_id"] for r in rows})
        for row in joined:
            for score in SCORE_NAMES:
                if not np.isfinite(row[score]):
                    raise ValueError(f"nonfinite {score}: {split}/{row['case_id']}")
            for source in SOURCES:
                for risk in RISKS:
                    value = row[source + "_" + risk]
                    if value is None or not np.isfinite(value) or value < 0 or (risk != "merge_split" and value > 1):
                        raise ValueError(f"invalid {source}/{risk}: {split}/{row['case_id']}")


def check_output_path(output: Path, protected: list[Path]) -> None:
    if output == ROOT / "outputs" or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("supplement output must be a dedicated directory under outputs")
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("supplement output overlaps protected experiment artifacts")


def check_legacy_points(points: list[dict], split: str, parent_root: Path) -> None:
    """The original merge+split estimand must match for every original policy."""
    legacy = {(r["stratum"], r["risk_source"], r["score"], r["requested_coverage"]): r
              for r in read_csv(parent_root / "tables/review_budget.csv") if r["split"] == split}
    selected = [r for r in points if r["risk"] == "merge_split" and r["score"] != "random"]
    if len(selected) != len(legacy):
        raise ValueError("supplement does not cover the original merge+split budget grid")
    for row in selected:
        old = legacy[row["stratum"], row["risk_source"], row["score"], row["requested_coverage"]]
        for key in ("retained_mean_risk", "expected_reviewed_cases", "delta_vs_presence", "delta_vs_random",
                    "coverage", "retained_units", "reviewed_units"):
            if not np.isclose(row[key], old[key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"original merge+split budget changed: {split}/{row['score']}/{key}")


def validate_result_grid(result: dict, rows: list[dict]) -> None:
    """Require every policy/endpoint/budget cell, including in resumed caches."""
    from itertools import product
    if set(result) != {"fixed_budget", "paired_differences", "risk_coverage"}:
        raise ValueError("statistics cache has an unexpected set of tables")
    policies = (*SCORE_NAMES, "random")
    for table, extra_fields, extras in (
        ("fixed_budget", ("score", "requested_coverage"), list(product(policies, (.8, .9)))),
        ("paired_differences", ("score", "baseline", "requested_coverage"),
         list(product(SCORE_NAMES, ("presence", "random"), (.8, .9)))),
        ("risk_coverage", ("score", "coverage"), None)):
        fields = ("stratum", "risk_source", "risk", *extra_fields)
        expected = set()
        for stratum in ("pooled", "pelvic", "femur"):
            n = sum(stratum == "pooled" or r["scan_type"] == stratum for r in rows)
            tail = extras if extras is not None else list(product(policies, ((k + 1) / n for k in range(n))))
            expected.update((stratum, source, risk, *extra) for source, risk in product(SOURCES, RISKS) for extra in tail)
        keys = [tuple(row[key] for key in fields) for row in result[table]]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"statistics result has missing/duplicate cells: {table}")


def render_examples(parent_root: Path, parent: dict, specs: dict, config: dict,
                    protocol: dict, smoke: bool) -> tuple[list[dict], list[dict], list[dict], dict]:
    selections = read_csv(parent_root / "heldout/qualitative_selections.csv")
    candidates = read_csv(parent_root / "heldout/qualitative_candidates.csv")
    if selections != parent["summary"]["qualitative_selections"]:
        raise ValueError("parent qualitative selections differ from its manifest")
    selected = [r for r in selections if r["status"] == "selected"]
    if smoke:
        selected = selected[:config["smoke_render_limit"]]
    old_coordinates = read_csv(parent_root / "heldout/qualitative_coordinates.csv")
    old_mapping = read_csv(parent_root / "heldout/qualitative_display_mapping.csv")
    examples, sources = [], {}
    for choice in selected:
        cid = choice["case_id"]
        spec = specs["heldout", cid]
        before = {key: state(path) for key, path in spec["paths"].items()}
        for key, digest in spec["hashes"].items():
            require_hash(spec["paths"][key], digest)
        image_hash = parent["summary"]["visual_sources"][cid]["sha256"]
        require_hash(spec["paths"]["image"], image_hash)
        example = prepare_example(choice, spec["paths"], {**protocol, "qualitative":
                                  read_json(ROOT / config["parent_config"])["qualitative"]})
        expected = [r for r in old_coordinates if r["category"] == choice["category"]]
        if example["coordinates"] != expected:
            raise ValueError(f"qualitative crop/slices differ from the parent: {choice['category']}")
        if example["mapping"] != [r for r in old_mapping if r["category"] == choice["category"]]:
            raise ValueError(f"qualitative display mapping differs from the parent: {choice['category']}")
        if before != {key: state(path) for key, path in spec["paths"].items()}:
            raise ValueError(f"qualitative source changed while rendering: {cid}")
        examples.append(example)
        sources[cid] = {"image_sha256": image_hash, "volume_sha256": spec["hashes"]}
    return examples, selections, candidates, sources


def run(config_path: Path, *, smoke: bool, check_only: bool) -> None:
    config = read_json(config_path)
    check_config(config)
    parent_config_path = ROOT / config["parent_config"]
    parent_config = read_json(parent_config_path)
    parent_root = (ROOT / config["parent_output"]).resolve()
    if parent_root != (ROOT / parent_config["output_root"]).resolve():
        raise ValueError("parent_output differs from the E4/E5 config")
    base = (ROOT / read_json(ROOT / parent_config["e1_config"])["output_root"]).resolve()
    heldout = (ROOT / read_json(ROOT / parent_config["e3_config"])["output_root"]).resolve()
    with ExitStack() as stack:
        for path in (base / ".e1.lock", heldout / ".evaluate.lock", parent_root / ".e4_e5.lock",
                     *(heldout / "models" / m / ".lock" for m in MODELS)):
            stack.enter_context(locked(path, shared=True))
        parent = verify_parent(parent_root)
        cases, datasets, specs, inputs = load_inputs(parent_config_path, parent_config)
        validate_inputs(parent, inputs, cases, datasets)
        print("E5 SUPPLEMENT PREFLIGHT PASS: 40 validation + 100 held-out cases; source hashes verified; CPU only", flush=True)
        if check_only:
            return
        if smoke:
            cases = {split: [next(r for r in rows if r["scan_type"] == scan) for scan in ("pelvic", "femur")]
                     for split, rows in cases.items()}
        output = (ROOT / config["smoke_output_root" if smoke else "output_root"]).resolve()
        protected = [parent_root, heldout, ROOT / "outputs/stage3_h1", ROOT / "outputs/stage2_backend",
                     *(base / folder for folder in ("validation", "perturbations", "tables", "figures", "case_cache"))]
        check_output_path(output, protected)
        if (ROOT / config["output_root"]).resolve() == (ROOT / config["smoke_output_root"]).resolve():
            raise ValueError("smoke/full outputs must be separate")
        code = [Path(__file__), ROOT / "scripts/run_e4_e5.py", ROOT / "scripts/run_e1.py", ROOT / "scripts/run_e3.py",
                *(ROOT / "src/exchange_ml" / name for name in (
                    "e5_supplement_statistics.py", "e5_supplement_figures.py", "e5_visualization.py",
                    "e4_e5_analysis.py", "e4_e5_data.py", "e1_data.py", "e1_statistics.py", "e1_scoring.py",
                    "instances.py", "metrics.py"))]
        identity = {"config_sha256": sha256(config_path), "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code},
                    "parent_manifest_sha256": sha256(parent_root / "e4_e5_manifest.json"),
                    "parent_compute_signature": parent["signature"], "upstream": inputs["identity"],
                    "software": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "SimpleITK", "matplotlib")},
                    "cases": cases, "smoke_only": smoke, "analysis_role": ROLE}
        run_signature = signature(identity)
        output.mkdir(parents=True, exist_ok=True)
        stack.enter_context(locked(output / ".e5_supplement.lock"))
        path = output / "e5_supplement_manifest.json"
        previous = read_json(path) if path.exists() else None
        if previous and previous["signature"] != run_signature:
            raise ValueError("supplement identity changed; preserve output and use a new output_root")
        if previous and previous["status"] == "complete":
            for group in ("output_sha256", "cache_sha256"):
                for filename, digest in previous[group].items():
                    require_hash(output / filename, digest)
            print(f"E5 SUPPLEMENT {'SMOKE' if smoke else 'FULL'} COMPLETE (verified existing): {output}", flush=True)
            return
        manifest = {"status": "running", "signature": run_signature, "provenance": identity,
                    "started_at_unix": time.time(), "completed_splits": [], "cache_sha256": {}}
        save_json(path, manifest)
        try:
            resamples = 32 if smoke else config["bootstrap_resamples"]
            combined = {"fixed_budget": [], "paired_differences": [], "risk_coverage": []}
            for split, records in cases.items():
                rows = joined_rows(datasets[split], {r["case_id"] for r in records})
                cache_path = output / "cache" / (split + ".json")
                if cache_path.exists():
                    cached_digest = (previous or {}).get("cache_sha256", {}).get(str(cache_path.relative_to(output)))
                    if cached_digest is not None:
                        require_hash(cache_path, cached_digest)
                    cache = read_json(cache_path)
                    if cache["signature"] != run_signature or signature(cache["result"]) != cache["result_sha256"]:
                        raise ValueError(f"supplement cache mismatch: {split}")
                    result = cache["result"]
                    print(f"[{split}] verified statistics cache", flush=True)
                else:
                    points, paired, curves = budget_analysis(rows, resamples=resamples,
                        seed=config["bootstrap_seed"], retained_fractions=tuple(config["retained_fractions"]),
                        batch_size=config["batch_size"], progress=lambda msg: print(f"[{split}] {msg}", flush=True))
                    result = {"fixed_budget": points, "paired_differences": paired, "risk_coverage": curves}
                    save_json(cache_path, {"signature": run_signature, "result": result, "result_sha256": signature(result)})
                n_units = len(rows)
                if {name: len(table) for name, table in result.items()} != {
                    "fixed_budget": 468, "paired_differences": 864, "risk_coverage": 156 * n_units}:
                    raise ValueError(f"supplement output grid incomplete: {split}")
                validate_result_grid(result, rows)
                if not smoke:
                    check_legacy_points(result["fixed_budget"], split, parent_root)
                for name, table in result.items():
                    combined[name].extend({**r, "analysis_role": ROLE} for r in dataset_table_rows(table, split))
                manifest["completed_splits"].append(split)
                manifest["cache_sha256"][str(cache_path.relative_to(output))] = sha256(cache_path)
                save_json(path, manifest)
            for name, rows in combined.items():
                write_csv(output / "tables" / (name + ".csv"), rows)
            plot_budget_figures(output, combined["fixed_budget"], combined["paired_differences"], combined["risk_coverage"])
            examples, selections, candidates, visual_sources = render_examples(
                parent_root, parent, specs, config, inputs["protocol"], smoke)
            for filename in ("qualitative_candidates.csv", "qualitative_selections.csv"):
                target = output / "heldout" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((parent_root / "heldout" / filename).read_bytes())
            write_csv(output / "heldout/qualitative_coordinates.csv", [r for e in examples for r in e["coordinates"]],
                      ("category", "case_id", "bone", "z_index"))
            write_csv(output / "heldout/qualitative_display_mapping.csv", [r for e in examples for r in e["mapping"]],
                      ("category", "case_id", "bone", "model_id", "original_id", "display_id"))
            plot_qualitative_supplement(output, examples, selections, candidates, datasets["heldout"]["model_risks"])
            # End-of-run source audit; never replace parent reporting or caches.
            verify_parent(parent_root)
            require_hash(parent_root / "e4_e5_manifest.json", identity["parent_manifest_sha256"])
            summary = {"analysis_role": ROLE, "smoke_only": smoke, "bootstrap_resamples": resamples,
                       "n_cases": {s: len(r) for s, r in cases.items()},
                       "n_bone_units": sum(len(joined_rows(datasets[s], {r['case_id'] for r in c})) for s, c in cases.items()),
                       "table_rows": {name: len(rows) for name, rows in combined.items()},
                       "risks": list(RISKS), "risk_sources": list(SOURCES),
                       "original_qualitative_selections": selections,
                       "rendered_examples": [e["selection"] for e in examples], "visual_sources": visual_sources,
                       "limits": ["Added after inspection of held-out results; exploratory, no E3 primary change.",
                           "Fixed budgets remain bone-based; expected reviewed CT costs are descriptive point estimates.",
                           "Random is the expected random-ranking risk, not a sampled random-order simulation.",
                           "CIs are marginal percentile intervals; no familywise confirmatory claims.",
                           "No new policy/QC gate or inference; no change to historical timing measurements."]}
            save_json(output / "summary.json", summary)
            save_json(output / "source_parent_manifest.json", parent)
            manifest.update(status="complete", summary=summary, completed_at_unix=time.time(),
                output_sha256={str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and p.suffix in (".csv", ".pdf", ".png", ".json")
                               and p != path and not p.is_relative_to(output / "cache")})
            save_json(path, manifest)
            print(f"E5 SUPPLEMENT {'SMOKE' if smoke else 'FULL'} COMPLETE: {output}", flush=True)
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_json(path, manifest)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e5_supplement.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true")
    group.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.config.resolve(), smoke=args.smoke, check_only=args.check)


if __name__ == "__main__":
    main()
