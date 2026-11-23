#!/usr/bin/env python3
"""Descriptive positive/negative E6 cohorts from completed, verified CSVs.

Does not read images, change matching, compute official FROC, or touch E6 writers.
Run after the current external evaluation completes, in a separate output root.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ("member_01", "member_02", "member_03")
STRUCTURAL = ("merge_split", "missing_volume", "missing")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_sha(handle)


def _stream_sha(handle) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def number(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("finite values required")
    return result


def integer(value) -> int:
    result = number(value)
    if result < 0 or not result.is_integer():
        raise ValueError("nonnegative integer count required")
    return int(result)


def cohort_tables(risks: list[dict], models: list[dict]) -> tuple[list[dict], dict]:
    by_case = {str(r["case_id"]): r for r in risks}
    by_model = {(str(r["case_id"]), r["model_id"]): r for r in models}
    if not by_case or len(by_case) != len(risks) or len(by_model) != len(models):
        raise ValueError("unique nonempty case/model tables required")
    if set(by_model) != {(cid, member) for cid in by_case for member in MEMBERS}:
        raise ValueError("three model rows required for every case")
    metrics = ("false_positive_instances", "one_minus_f1", "one_minus_dice")
    for cid, row in by_case.items():
        n_gt = integer(row["n_GT"])
        selected = [by_model[cid, member] for member in MEMBERS]
        for item in selected:
            if integer(item["n_GT"]) != n_gt:
                raise ValueError("GT count mismatch across model rows")
            n_pred, fp = integer(item["n_pred"]), integer(item["false_positive_instances"])
            if fp > n_pred or (not n_gt and fp != n_pred):
                raise ValueError("invalid false-positive count")
            for metric in (*metrics, *STRUCTURAL):
                value = item[metric]
                if metric in STRUCTURAL and not n_gt:
                    if value not in (None, ""):
                        raise ValueError("negative structural risks must be undefined")
                    continue
                observed = number(value)
                if observed < 0 or (metric not in ("false_positive_instances", "merge_split") and observed > 1):
                    raise ValueError("risk outside expected range")
        for metric in (*metrics, *STRUCTURAL):
            expected = row["member_mean_" + metric]
            if metric in STRUCTURAL and not n_gt:
                if expected not in (None, ""):
                    raise ValueError("negative member-mean structural risk must be undefined")
            elif not math.isclose(number(expected), mean(number(item[metric]) for item in selected), abs_tol=1e-12):
                raise ValueError("member-mean risk differs from individual model rows")
    output = []
    for cohort in ("all", "positive", "negative"):
        ids = [cid for cid, r in by_case.items() if cohort == "all" or (integer(r["n_GT"]) > 0) == (cohort == "positive")]
        for member in (*MEMBERS, "member_mean"):
            def values(metric: str) -> list[float]:
                if member != "member_mean":
                    return [number(by_model[cid, member][metric]) for cid in ids]
                return [mean(number(by_model[cid, m][metric]) for m in MEMBERS) for cid in ids]

            fp = values("false_positive_instances")
            status = "descriptive" if ids else "undefined_no_cases"
            # member_mean is an expected per-model burden, never a consensus model.
            any_fp = ([float(integer(by_model[cid, member]["false_positive_instances"]) > 0) for cid in ids]
                      if member != "member_mean" else
                      [mean(float(integer(by_model[cid, m]["false_positive_instances"]) > 0) for m in MEMBERS) for cid in ids])
            row = {"cohort": cohort, "model": member, "n_cases": len(ids), "status": status,
                   "mean_fp_instances_per_CT": mean(fp) if ids else None,
                   "sum_fp_instances": sum(fp) if ids else None,
                   "fraction_CT_with_any_FP": mean(any_fp) if ids else None,
                   "mean_instance_f1": 1 - mean(values("one_minus_f1")) if ids else None,
                   "mean_binary_dice": 1 - mean(values("one_minus_dice")) if ids else None,
                   "interpretation": "per_model_burden" if member != "member_mean" else "mean_of_three_models_not_consensus"}
            positive = [cid for cid in ids if integer(by_case[cid]["n_GT"]) > 0]
            row["n_structural_eligible_cases"] = len(positive)
            for metric in STRUCTURAL:
                observed = [number(by_model[cid, member][metric]) if member != "member_mean"
                            else number(by_case[cid]["member_mean_" + metric]) for cid in positive]
                row["mean_" + metric] = mean(observed) if observed else None
            output.append(row)
    return output, {"n_cases": len(by_case),
                    "n_positive_cases": sum(integer(r["n_GT"]) > 0 for r in risks),
                    "n_negative_cases": sum(integer(r["n_GT"]) == 0 for r in risks),
                    "role": "descriptive_fixed_threshold_internal_matching_not_official_FROC",
                    "new_case_GT_used_for_prediction": False,
                    "new_inference_or_training": False,
                    "CI_computed": False}


def completed_sources(source: Path) -> tuple[list[dict], list[dict], dict]:
    path = source / "e6_manifest.json"
    if not path.exists():
        raise ValueError("E6 evaluation is not complete yet; run this report after evaluation finishes")
    state = json.loads(path.read_text())
    if state["status"] != "complete" or state["signature"] != signature(state["provenance"]):
        raise ValueError("complete hash-bound E6 evaluation required")
    hashes = {"e6_manifest.json": sha(path)}
    tables = []
    for name in ("risks.csv", "model_risks.csv", "summary.json"):
        expected = state["output_sha256"][name]
        if sha(source / name) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        hashes[name] = expected
        if name.endswith(".csv"):
            with (source / name).open(newline="") as handle:
                tables.append(list(csv.DictReader(handle)))
    summary = json.loads((source / "summary.json").read_text())
    if summary != state["summary"] or summary.get("smoke_only") is not False:
        raise ValueError("source summary is not the recorded full E6 summary")
    expected_cases = {f"RibFrac{i}" for i in range(421, 501)}
    if {r["case_id"] for r in tables[0]} != expected_cases or len(tables[0]) != 80:
        raise ValueError("all 80 official external cases required")
    _, derived = cohort_tables(*tables)
    if (derived["n_positive_cases"], derived["n_negative_cases"]) != (60, 20):
        raise ValueError("official positive/negative subject membership differs")
    return *tables, hashes


def validate_output(output: Path, source: Path) -> None:
    protected = [source.resolve(), (ROOT.parent / "dataset").resolve(),
                 *(ROOT / p for p in ("src", "scripts", "configs", "tests", "splits", "__docs__", "paper", "baselines", "environment")),
                 *(ROOT / "outputs" / p for p in ("stage3_h1_v2", "stage3_h1", "stage2_backend", "archive", "local_copy_v1"))]
    if output == ROOT or output == ROOT / "outputs" or output in ROOT.parents:
        raise ValueError("dedicated output root required")
    if any(output == p.resolve() or output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ValueError("output overlaps protected data, code or source artifacts")
    if any(part.startswith("Exchange-ML-E6") or part == "external_ribfrac" for part in output.parts):
        raise ValueError("E6 campaign output trees are read-only to this reporter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/research_followup_v1/ribfrac_cohorts")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    validate_output(output, source)
    risks, models, hashes = completed_sources(source)
    rows, summary = cohort_tables(risks, models)
    provenance = {"source_root": str(source), "source_hashes": hashes,
                  "output_root": str(output), "producer_sha256": sha(Path(__file__))}
    if args.check:
        print(json.dumps({"status": "ready", **summary}, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".report.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old.get("provenance") != provenance:
                raise ValueError("report identity drift; preserve output and use a new output root")
            if old.get("status") == "complete":
                for name, digest in old["output_sha256"].items():
                    if sha(output / name) != digest:
                        raise ValueError(f"report output hash mismatch: {name}")
                print("RIBFRAC COHORT REPORT ALREADY COMPLETE"); return
        elif any(p.name != ".report.lock" for p in output.iterdir()):
            raise ValueError("nonempty output without matching manifest")

        def save(path: Path, value) -> None:
            temp = path.with_name(path.name + ".tmp")
            temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
            temp.replace(path)

        state = {"status": "running", "provenance": provenance, "signature": signature(provenance)}
        save(manifest_path, state)
        temp = output / "cohorts.csv.tmp"
        with temp.open("w", newline="") as table:
            writer = csv.DictWriter(table, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        temp.replace(output / "cohorts.csv")
        save(output / "summary.json", summary)
        if any(sha(source / name) != digest for name, digest in hashes.items()):
            raise ValueError("E6 source changed during cohort reporting")
        state.update(status="complete", summary=summary,
                     output_sha256={name: sha(output / name) for name in ("cohorts.csv", "summary.json")})
        save(manifest_path, state)
    print("RIBFRAC COHORT REPORT COMPLETE (descriptive, no FROC or new experiment)")


if __name__ == "__main__":
    main()
