#!/usr/bin/env python3
"""CPU metadata audit of backend identity and candidate outcome exposure.

No volume, checkpoint, external evaluation, network request or model is read.
Absence of recorded exposure is never proof of an unseen confirmation cohort.
"""
from __future__ import annotations

import argparse
import configparser
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BACKEND = "baselines/pengwin2026-task1-abbc"
PIN = "ccc625fd000ec2dfce622b4621bf2d0b4ee12796"
URL = "https://github.com/RURUGURU/pengwin2026-task1-abbc"
PATCH = "patches/pengwin2026-task1-abbc_on_ccc625f.patch"
CANDIDATES = "outputs/partition_v3_development_v1/confirmation_candidates.json"
SCAN_ROOTS = ("outputs/stage2_backend", "outputs/stage3_h1",
              "outputs/stage3_h1_v2", "outputs/partition_v3_development_v1",
              "outputs/partition_v4_development_v1", "outputs/archive")
MAX_METADATA_BYTES = 8 * 1024**2
MODEL_OUTCOME_MARKERS = ("merge_split", "merge_errors", "split_errors", "missing_volume",
                         "one_minus_f1", "instance_f1", "binary_dice", "one_minus_dice",
                         "instance_precision", "instance_recall", "normalized_error")
VOLUME_ENDINGS = (".mha", ".nii", ".nii.gz", ".nrrd")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True,
                                   stderr=subprocess.PIPE).strip()


def case_id(value: Any) -> str | None:
    text = str(value)
    return text.zfill(3) if re.fullmatch(r"\d{1,3}", text) else None


def membership(root: Path) -> dict:
    """Validate candidate set arithmetic without opening risk outcomes."""
    inventory_path = root / CANDIDATES
    inventory = json.loads(inventory_path.read_text())
    config = json.loads((root / "configs/partition_v3.json").read_text())
    settings = config["confirmation"]
    split_path, e3_path = root / settings["split"], root / settings["e3_protocol"]
    split = json.loads(split_path.read_text())
    e3 = json.loads(e3_path.read_text())
    # Use the same split selector as the original inventory, without changing it.
    import sys
    sys.path.insert(0, str(root / "src"))
    from exchange_ml.stage2_backend import select_stage2_case_ids
    train = set(select_stage2_case_ids(split, settings["trajectory_seed"])["train"])
    expected = sorted(set(split["split"]["pool"]) - train - set(e3["case_ids"]))
    if inventory["case_ids"] != expected or len(expected) != 110:
        raise ValueError("candidate membership differs from frozen 110-case inventory")
    for key, path in (("split_sha256", split_path), ("e3_protocol_sha256", e3_path)):
        if inventory[key] != sha256(path):
            raise ValueError(f"candidate source SHA-256 mismatch: {path}")
    if set(expected) & set(split["split"]["validation"] + split["split"]["test"]):
        raise ValueError("candidates overlap validation/local-test holdback")
    return {"case_ids": expected, "n_candidate_cases": len(expected),
            "inventory_sha256": sha256(inventory_path),
            "split_sha256": sha256(split_path), "e3_protocol_sha256": sha256(e3_path),
            "source_inventory_confirmation_ready": inventory["confirmation_ready"],
            "source_inventory_exposure_audit_completed": inventory["exposure_audit_completed"],
            "local_test_role": "holdback_from_public_labeled_set_not_official_hidden_challenge_test"}


def backend_audit(root: Path) -> dict:
    module = root / BACKEND
    mapping_path = root / ".gitmodules"
    mapping = configparser.ConfigParser()
    mapping.read(mapping_path)
    section = f'submodule "{BACKEND}"'
    mapped = mapping.has_section(section) and mapping.get(section, "path", fallback="") == BACKEND
    mapped = mapped and mapping.get(section, "url", fallback="") == URL
    head = git(module, "rev-parse", "HEAD")
    index = git(root, "ls-files", "--stage", "--", BACKEND).split()
    index_ok = len(index) >= 3 and index[0] == "160000" and index[1] == PIN
    patch = root / PATCH
    check = subprocess.run(["git", "-C", str(module), "apply", "--reverse", "--check", str(patch)],
                           capture_output=True, text=True)
    observed_patch = subprocess.check_output(
        ["git", "-C", str(module), "diff", "--binary", PIN, "--"])
    exact_patch = observed_patch == patch.read_bytes()
    remote = subprocess.run(["git", "-C", str(module), "remote", "get-url", "origin"],
                            capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(root), "submodule", "status", "--", BACKEND],
                            capture_output=True, text=True)
    errors = []
    if not mapped: errors.append("missing_or_incorrect_gitmodules_mapping")
    if head != PIN: errors.append("backend_HEAD_differs_from_pinned_commit")
    if not index_ok: errors.append("backend_gitlink_differs_from_pinned_commit")
    if check.returncode: errors.append("recorded_project_patch_not_applied_as_expected")
    if not exact_patch: errors.append("backend_tracked_diff_differs_from_recorded_patch")
    if status.returncode: errors.append("git_submodule_status_failed")
    return {"path": BACKEND, "expected_commit": PIN, "observed_commit": head,
            "gitlink_matches": index_ok, "gitmodules_mapping_valid": bool(mapped),
            "upstream_url": URL, "gitmodules_sha256": sha256(mapping_path) if mapping_path.exists() else None,
            "patch_sha256": sha256(patch), "patch_reverse_check_passed": check.returncode == 0,
            "tracked_diff_exactly_matches_recorded_patch": exact_patch,
            "submodule_status": status.stdout.strip(), "submodule_status_ok": status.returncode == 0,
            "local_origin_url": remote.stdout.strip() if remote.returncode == 0 else None,
            "clean_clone_bootstrap_verified": False,
            "bootstrap_limit": "mapping_and_local_bytes_checked_no_network_clone_or_install_performed",
            "errors": errors}


def tracked_sources(root: Path) -> dict:
    folders = ("src", "scripts", "configs", "tests")
    tracked = set(git(root, "ls-files", "--", *folders).splitlines())
    summary = {}
    for folder in folders:
        files = [p for p in (root / folder).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")]
        untracked = sorted(str(p.relative_to(root)) for p in files if str(p.relative_to(root)) not in tracked)
        summary[folder] = {"n_files": len(files), "n_tracked": len(files) - len(untracked),
                           "untracked_paths": untracked}
    return {"commit": git(root, "rev-parse", "HEAD"), "folders": summary,
            "note": "working_tree_provenance_not_proof_of_preheldout_registration"}


def _manifest_records(value: Any) -> Iterable[dict]:
    """Yield explicit case records; generic case lists are not outcome evidence."""
    if isinstance(value, dict):
        if case_id(value.get("case_id")) is not None:
            yield value
        for child in value.values():
            yield from _manifest_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _manifest_records(child)


def inspect_metadata(path: Path, candidates: set[str]) -> list[dict]:
    """Read an already size-limited CSV/manifest, retaining only exposure evidence."""
    matches = []
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            keys = reader.fieldnames or []
            outcome = [key for key in keys if key.lower() in ("f1", "dice") or
                       any(marker in key.lower() for marker in MODEL_OUTCOME_MARKERS)]
            if "case_id" not in keys or not outcome:
                return []
            ids = set()
            for row in reader:
                cid = case_id(row["case_id"])
                if cid in candidates and any(row.get(key, "") not in ("", None) for key in outcome):
                    ids.add(cid)
            if ids:
                matches.append({"kind": "recorded_model_outcomes", "case_ids": sorted(ids),
                                "outcome_columns": outcome})
    else:
        data = json.loads(path.read_text())
        for row in _manifest_records(data):
            cid = case_id(row["case_id"])
            if cid not in candidates:
                continue
            outcome = [k for k in row if k.lower() in ("f1", "dice") or
                       any(marker in k.lower() for marker in MODEL_OUTCOME_MARKERS)]
            if any(row[k] is not None for k in outcome):
                matches.append({"kind": "recorded_model_outcomes", "case_ids": [cid],
                                "outcome_columns": outcome})
            elif any(k in row for k in ("prediction", "probability", "foreground_probability", "prediction_path")):
                matches.append({"kind": "prediction_record_not_proof_outcomes_viewed", "case_ids": [cid]})
    return matches


def exposure_audit(root: Path, candidates: set[str], scan_roots: Iterable[str] = SCAN_ROOTS) -> dict:
    """Bounded, metadata-only scan of known project output trees.

    Missing external/history records remain unknown. No symlinked directory is
    traversed, no image is decoded, and metadata over the bound is skipped.
    """
    findings, skipped, hashes = [], [], {}
    missing_roots = []
    for relative in scan_roots:
        folder = root / relative
        if not folder.is_dir():
            missing_roots.append(relative)
            continue
        for directory, directories, names in os.walk(folder, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in (".git", "__pycache__")
                                    and not (Path(directory) / d).is_symlink()
                                    and not d.startswith("documentation_"))
            for name in sorted(names):
                p = Path(directory) / name
                rel = str(p.relative_to(root))
                if p.is_symlink():
                    continue
                if name.endswith(VOLUME_ENDINGS) and any(
                        term in rel.lower() for term in ("predict", "probabilit", "inference")):
                    cid = case_id(name.split(".")[0])
                    if cid in candidates:
                        findings.append({"path": rel, "kind": "prediction_artifact_filename_only",
                                         "case_ids": [cid]})
                    continue
                is_meta = p.suffix == ".csv" or (p.suffix == ".json" and "manifest" in name)
                if not is_meta:
                    continue
                if p.stat().st_size > MAX_METADATA_BYTES:
                    skipped.append({"path": rel, "reason": "metadata_exceeds_size_bound"})
                    continue
                try:
                    before = sha256(p)
                    hits = inspect_metadata(p, candidates)
                    if sha256(p) != before:
                        raise ValueError("metadata changed during exposure scan")
                    hashes[rel] = before
                    findings.extend({"path": rel, **hit} for hit in hits)
                except (ValueError, UnicodeError, csv.Error, OSError) as error:
                    skipped.append({"path": rel, "reason": type(error).__name__ + ": " + str(error)})
    known = sorted({cid for hit in findings if hit["kind"] == "recorded_model_outcomes" for cid in hit["case_ids"]})
    predictions = sorted({cid for hit in findings if hit["kind"] != "recorded_model_outcomes" for cid in hit["case_ids"]})
    return {"scan_roots": list(scan_roots), "metadata_sha256": hashes,
            "metadata_files_read": len(hashes), "missing_roots": missing_roots,
            "skipped_metadata": skipped, "findings": findings,
            "known_recorded_outcome_case_ids": known,
            "prediction_evidence_case_ids": predictions,
            "case_status": {cid: "recorded_model_outcomes_present" if cid in known else
                            "prediction_evidence_only_outcome_exposure_unknown" if cid in predictions else
                            "no_record_found_exposure_unknown" for cid in sorted(candidates)},
            "confirmation_ready": False, "exposure_audit_completed": False,
            "unrecorded_history_unknown": True,
            "limit": "bounded_project_artifact_scan_cannot_establish_human_access_or_unrecorded_external_usage",
            "raw_GT_split_statistics_treated_as_model_outcome_exposure": False}


def output_path(root: Path, output: Path) -> Path:
    path = output.resolve()
    # Restrict this metadata report to its own namespace, with no exceptions for
    # absolute paths. It cannot replace datasets, sources, parent runs or E6.
    allowed = (root / "outputs/research_followup_v1").resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("preflight output must be a dedicated child of outputs/research_followup_v1")
    if path.name != "preflight":
        raise ValueError("preflight reports require a dedicated directory named preflight")
    return path


def run(root: Path, output: Path) -> dict:
    target = output_path(root, output)
    candidates = membership(root)
    backend = backend_audit(root)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "role": "technical_preflight_and_bounded_exposure_inventory_not_confirmation_authorization",
              "producer_sha256": sha256(Path(__file__)), "backend": backend,
              "source_tracking": tracked_sources(root), "candidate_membership": candidates,
              "exposure": exposure_audit(root, set(candidates["case_ids"])),
              "technical_integrity_passed": not backend["errors"], "confirmation_ready": False,
              "agent_permissions": "metadata_only_no_experiment_or_network_execution"}
    if backend["errors"]:
        raise ValueError("technical preflight failed: " + ", ".join(backend["errors"]))
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest = target / "reproducibility_audit.json"
        temporary = dest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(dest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/research_followup_v1/preflight")
    args = parser.parse_args()
    report = run(ROOT, args.output_root)
    print("REPRODUCIBILITY PREFLIGHT PASS: "
          f"{report['exposure']['metadata_files_read']} metadata files; "
          f"{len(report['exposure']['known_recorded_outcome_case_ids'])} candidates have recorded outcomes; "
          "confirmation_ready=false (unrecorded exposure remains unknown)")


if __name__ == "__main__":
    main()
