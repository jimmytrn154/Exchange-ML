#!/usr/bin/env python3
"""Audit completed E1–E5 report artifacts and extract manuscript evidence.

No inference, scoring, bootstrap, volume reads or frozen-output writes.
--write updates only outputs/stage3_h1_v2/reporting/e1_e5_evidence.json;
default verifies it. Experiment outputs remain read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/stage3_h1_v2"
TARGET = BASE / "reporting/e1_e5_evidence.json"
MANIFESTS = {"E1": BASE / "e1_manifest.json", "E2": BASE / "perturbations/e2_manifest.json",
             "E3": BASE / "heldout/e3_manifest.json", "E4_E5": BASE / "e4_e5/e4_e5_manifest.json",
             "E5_supplement": BASE / "e5_supplement_v1/e5_supplement_manifest.json"}
CONFIGS = {"E1": "configs/e1.json", "E2": "configs/e2.json", "E3": "configs/stage3_h1_v2.json",
           "E4_E5": "configs/e4_e5.json", "E5_supplement": "configs/e5_supplement.json"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def selected(path: Path, **filters) -> dict:
    matched = [r for r in rows(path) if all(r[k] in (v if isinstance(v, tuple) else (v,)) for k, v in filters.items())]
    if not matched:
        raise ValueError(f"no evidence rows: {path}: {filters}")
    return {"source": str(path.relative_to(ROOT)), "sha256": sha(path), "filters": filters, "rows": matched}


def execution_signature(name: str, manifest: dict) -> str:
    # Match each producer's signature contract; E2 also binds smoke/full mode.
    payload = ({"provenance": manifest["provenance"], "smoke_only": manifest["smoke_only"]}
               if name == "E2" else manifest["provenance"])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def audit() -> tuple[dict, dict]:
    evidence, manifests = {}, {}
    for name, path in MANIFESTS.items():
        m = read_json(path)
        if m["status"] != "complete" or m.get("smoke_only", False) or m["summary"].get("smoke_only", False):
            raise ValueError(f"not a complete full experiment: {path}")
        if m["signature"] != execution_signature(name, m):
            raise ValueError(f"provenance signature mismatch: {path}")
        config_hash = m["provenance"]["protocol_sha256" if name == "E3" else "config_sha256"]
        require(ROOT / CONFIGS[name], config_hash)
        for group in ("output_sha256", "cache_sha256"):
            for file, digest in m.get(group, {}).items():
                require(path.parent / file, digest)
        code_override = {}
        for correction in m.get("report_corrections", []):
            receipt_path = path.parent / correction["receipt"]
            require(receipt_path, correction["receipt_sha256"])
            receipt = read_json(receipt_path)
            if receipt["original_compute_signature"] != m["signature"]:
                raise ValueError("correction compute identity mismatch")
            for file, digest in receipt["archives"].items():
                require(receipt_path.parent / file, digest)
            for file, digest in receipt["correction_code_sha256"].items():
                require(ROOT / file, digest)
            code_override.update(receipt["correction_code_sha256"])
        recorded_code = m["provenance"].get("code_sha256", m["provenance"].get("code", {}))
        for file, digest in recorded_code.items():
            require(ROOT / file, code_override.get(file, digest))
        summary = path.parent / "summary.json"
        if summary.exists() and read_json(summary) != m["summary"]:
            raise ValueError(f"summary differs from manifest: {summary}")
        manifests[name] = m
        evidence[name] = {"manifest": str(path.relative_to(ROOT)), "manifest_sha256": sha(path),
                          "execution_signature": m["signature"], "summary": m["summary"],
                          "verified_config": CONFIGS[name], "config_sha256": config_hash,
                          "verified_output_hashes": len(m.get("output_sha256", {})),
                          "verified_cache_hashes": len(m.get("cache_sha256", {})),
                          "report_corrections": m.get("report_corrections", [])}
    # Preserve and check the chain, including corrected reporting vs execution identity.
    require(MANIFESTS["E1"], manifests["E2"]["provenance"]["e1_manifest_sha256"])
    protocol = read_json(ROOT / "configs/stage3_h1_v2.json")
    require(MANIFESTS["E1"], protocol["e1_manifest_sha256"])
    require(ROOT / "configs/stage3_h1_v2.json", manifests["E3"]["provenance"]["protocol_sha256"])
    require(MANIFESTS["E4_E5"], manifests["E5_supplement"]["provenance"]["parent_manifest_sha256"])
    val_ids, held_ids = manifests["E1"]["case_ids"], manifests["E3"]["case_ids"]
    if len(val_ids) != 40 or len(set(val_ids)) != 40 or len(held_ids) != 100 or len(set(held_ids)) != 100 or set(val_ids) & set(held_ids):
        raise ValueError("case membership/counts differ from the validation/held-out design")
    if not set(manifests["E2"]["response_case_ids"]).issubset(val_ids):
        raise ValueError("E2 response cases are not contained in validation")
    for model, digest in manifests["E3"]["provenance"]["models"].items():
        p = MANIFESTS["E3"].parent / "models" / model / "inference_manifest.json"
        m = read_json(p)
        if (hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest() != digest or m["status"] != "complete"
            or set(m["records"]) != set(held_ids) or len(m["records"]) != 100):
            raise ValueError(f"inference manifest mismatch: {model}")
    return evidence, manifests


def build() -> dict:
    audits, manifests = audit()
    e4 = BASE / "e4_e5/tables"
    e5 = BASE / "e5_supplement_v1/tables"
    extracted = {
        "E1_decomposition": selected(BASE / "tables/decomposition.csv"),
        "E1_validation_contrasts": selected(BASE / "validation/paired_differences.csv", stratum="pooled", risk="merge_split"),
        "E2_response_summary": selected(BASE / "perturbations/response_summary.csv"),
        "E2_replication_summary": selected(BASE / "perturbations/replication_summary.csv"),
        "E3_main_table": selected(BASE / "heldout/main_table.csv"),
        "E3_structural_contrasts": selected(BASE / "heldout/paired_differences.csv", risk="merge_split"),
        "E4_case_variants": selected(e4 / "normalization_scope.csv"),
        "E4_case_contrasts": selected(e4 / "normalization_scope_comparisons.csv"),
        "E4_enrichment": selected(e4 / "enrichment.csv"),
        "E4_risk_source": selected(e4 / "risk_source.csv"),
        "E4_blindspots": selected(e4 / "f1_volume_blindspots.csv"),
        "E4_blindspot_counts": selected(e4 / "blindspot_summary.csv"),
        "E5_fixed_budget": selected(e5 / "fixed_budget.csv", split="heldout", stratum="pooled",
                                     score=("fa_v2", "presence", "partition_v1", "partition_v2", "random")),
        "E5_paired_budget": selected(e5 / "paired_differences.csv", split="heldout", stratum="pooled", score="fa_v2"),
        "E5_curve_points": selected(BASE / "e5_supplement_v1/tables/risk_coverage.csv", split="heldout", stratum="pooled",
                                      risk_source="member_mean", risk="merge_split", score=("fa_v2", "presence"),
                                      coverage=("0.1", "0.2", "0.5", "0.8", "0.9")),
        "E5_qualitative": selected(BASE / "e5_supplement_v1/heldout/qualitative_selections.csv"),
        "E5_qualitative_model_risks": selected(BASE / "heldout/model_risks.csv", case_id=("176", "160", "187")),
        "E5_runtime": selected(e4 / "runtime.csv", stage=("E3_inference", "E3_CPU"), sampling="all_recorded"),
    }
    scope = rows(e4 / "scope_case_scores.csv")
    scope_groups = []
    for split in ("validation", "heldout"):
        for scan in ("pooled", "pelvic", "femur"):
            sub = [r for r in scope if r["split"] == split and (scan == "pooled" or r["scan_type"] == scan)]
            values = [float(r["fraction_outside_scope"]) for r in sub if r["fraction_outside_scope"]]
            scope_groups.append({"split": split, "stratum": scan, "n_cases": len(sub), "n_defined": len(values),
                                 "mean_of_case_mean_pair_ratios": mean(values), "median_of_case_mean_pair_ratios": median(values)})
    responses = rows(BASE / "perturbations/responses.csv")
    by_merge = {}
    for r in responses:
        if r["eligible"] == "True" and r["operation"] == "merge":
            by_merge.setdefault((r["case_id"], r["bone"]), {})[int(r["severity"])] = float(r["q_joint"])
    matched_merge = []
    for levels in ((1, 2), (1, 2, 3)):
        cohort = [r for r in by_merge.values() if all(level in r for level in levels)]
        matched_merge.append({"levels": levels, "n_bones": len(cohort),
                              "median_q_joint_by_level": [median(r[level] for r in cohort) for level in levels]})
    single = [r for r in responses if r["operation"] == "split" and r["eligible"] == "True" and r["gt_fragment_count"] == "1"]
    operation_summary = []
    for operation in sorted({r["operation"] for r in responses}):
        sub = [r for r in responses if r["operation"] == operation]
        eligible = [r for r in sub if r["eligible"] == "True"]
        operation_summary.append({"operation": operation, "eligible": len(eligible), "ineligible": len(sub) - len(eligible),
            **{quantity: {"min": min(float(r[quantity]) for r in eligible), "max": max(float(r[quantity]) for r in eligible)}
               for quantity in ("presence", "vi", "q_joint")}})
    git_tracked = subprocess.check_output(["git", "ls-files", "--", "configs/stage3_h1_v2.json"], cwd=ROOT, text=True).strip()
    figure_meta_path = BASE / "reporting/e2_matched_merge.json"
    figure_meta = read_json(figure_meta_path)
    require(ROOT / figure_meta["source_path"], figure_meta["source_sha256"])
    require(ROOT / figure_meta["source_manifest_path"], figure_meta["source_manifest_sha256"])
    require(ROOT / figure_meta["plot_script_path"], figure_meta["plot_script_sha256"])
    for file, digest in figure_meta["outputs"].items():
        require(ROOT / file, digest)
    for derived, figure in zip(matched_merge, figure_meta["cohorts"]):
        if (list(derived["levels"]) != figure["levels"] or derived["n_bones"] != figure["n_bones"]
            or derived["median_q_joint_by_level"] != figure["median_q_joint"]):
            raise ValueError("reporting figure differs from independent matched-cohort extraction")
    return {"snapshot_date": "2026-11-20", "artifact_role": "report_only_existing_results_no_experiment_rerun",
            "producer_sha256": sha(Path(__file__)), "audits": audits, "extracted": extracted,
            "derived": {"annotation_scope_case_summary": scope_groups,
                        "E2_operation_summary": {"source": "outputs/stage3_h1_v2/perturbations/responses.csv",
                                                 "sha256": sha(BASE / "perturbations/responses.csv"), "rows": operation_summary},
                        "matched_merge_q_joint": matched_merge,
                        "single_fragment_split": {"n_bones": len({(r['case_id'], r['bone']) for r in single}),
                                                  "n_rows": len(single), "all_q_joint_equal_one": all(float(r['q_joint']) == 1 for r in single)}},
            "reporting_figure": {"metadata_path": str(figure_meta_path.relative_to(ROOT)),
                                 "metadata_sha256": sha(figure_meta_path), "metadata": figure_meta},
            "provenance_limit": {"e3_config_tracked_in_current_git_index": bool(git_tracked),
                                 "formal_preregistration_or_timestamped_preheldout_commit_verified": False,
                                 "interpretation": "Run-bound hashes verified; chronology is not an independent registration proof."},
            "verification_scope": "Recorded report/cache/code hashes and table extraction, not raw-volume/checkpoint rereading."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the evidence sidecar under outputs/stage3_h1_v2/reporting")
    args = parser.parse_args()
    data = build()
    if args.write:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        tmp = TARGET.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        tmp.replace(TARGET)
    elif not TARGET.exists() or read_json(TARGET) != json.loads(json.dumps(data)):
        raise ValueError("evidence sidecar differs; inspect sources before explicitly regenerating with --write")
    print(f"E1–E5 evidence audit PASS: {sum(v['verified_output_hashes'] for v in data['audits'].values())} "
          f"output hashes; {sum(v['verified_cache_hashes'] for v in data['audits'].values())} cache hashes; "
          f"sidecar {'written' if args.write else 'verified'}")


if __name__ == "__main__":
    main()
