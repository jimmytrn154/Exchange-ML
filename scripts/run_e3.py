#!/usr/bin/env python3
"""Run frozen PENGWIN held-out inference and E3 evaluation."""
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
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exchange_ml.e1_data import MEMBERS, geometry, process_case, read_volume
from exchange_ml.stage2_backend import select_stage2_case_ids
from predict_stage2_backend import load_configured_backend
from run_e1 import save_json, sha256
from stage2_backend import check as check_backend_import
from stage3_h1 import inference_checkpoint_path, member_stage2_config

MODELS = (*MEMBERS, "stage2")
STAGE2 = Path("configs/stage2_backend.json")
STAGE3 = Path("configs/stage3_h1.json")
RESULT_ROOT = ROOT / "outputs/stage3_h1_v2/heldout"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def data_inventory(dataset: Path) -> dict[str, tuple[str, Path]]:
    found = {}
    for i in range(1, 5):
        folder = dataset / f"PENGWIN26_task1_2_train_part{i}"
        for case in folder.iterdir():
            if case.is_dir() and len(case.name) == 3 and case.name.isdecimal():
                if case.name in found:
                    raise ValueError(f"duplicate dataset case: {case.name}")
                found[case.name] = ("pelvic" if i <= 2 else "femur", case)
    return found


def load_protocol(path: Path) -> tuple[dict, list[dict], dict[str, Path]]:
    protocol = read_json(path)
    if (protocol["case_count"], protocol["selection_seed"], protocol["trajectory_seed"],
        protocol["connectivity"], protocol["prune_mm3"], protocol["iou_threshold"],
        protocol["bootstrap_seed"], protocol["bootstrap_resamples"]) != (100, 20261117, 20260920, 26, 1000, .1, 20261119, 10000):
        raise ValueError("E3 parameters differ from the frozen protocol")
    primary = protocol["primary"]
    if primary != {"score": "fa_v2", "risk_source": "member_mean", "risk": "merge_split",
                   "baselines": ["presence", "mean_entropy"], "max_delta_aurc": -.02,
                   "confidence_level": .95}:
        raise ValueError("E3 decision rule differs from the frozen protocol")
    if set(protocol["models"]) != set(MODELS):
        raise ValueError("E3 requires all three members and Stage 2")
    require_hash(STAGE2, protocol["stage2_config_sha256"])
    require_hash(STAGE3, protocol["stage3_config_sha256"])
    require_hash(Path(protocol["split"]), protocol["split_sha256"])
    require_hash(Path(protocol["case_list"]), protocol["case_list_sha256"])
    for filename, digest in protocol["training_split_sha256"].items():
        require_hash(Path(filename), digest)
    with Path(protocol["case_list"]).open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    split = read_json(Path(protocol["split"]))
    roles = select_stage2_case_ids(split, protocol["trajectory_seed"])
    excluded = set(roles["train"] + roles["validation"] + roles["test"])
    pool = set(split["split"]["pool"]) - excluded
    if len(pool) != 210:
        raise ValueError("expected 210 untouched candidates outside train/validation/test")
    inventory = data_inventory(Path(protocol["dataset_root"]))
    rng = np.random.default_rng(protocol["selection_seed"])
    expected = []
    for scan in ("pelvic", "femur"):
        candidates = sorted(cid for cid in pool if inventory[cid][0] == scan)
        if len(candidates) != 105:
            raise ValueError(f"{scan}: expected exactly 105 eligible candidates")
        sampled = sorted(rng.choice(candidates, 50, replace=False).tolist())
        expected.extend({"case_id": cid, "scan_type": scan} for cid in sampled)
    if cases != expected or [r["case_id"] for r in cases] != protocol["case_ids"]:
        raise ValueError("held-out membership or order differs from the frozen selection")
    paths = {row["case_id"]: inventory[row["case_id"]][1] for row in cases}
    for case in paths.values():
        for name in ("image.mha", "label.mha"):
            if not (case / name).is_file():
                raise FileNotFoundError(case / name)
    return protocol, cases, paths


def verify_backend(protocol: dict) -> None:
    source = protocol["backend"]
    checkout = Path(source["path"])
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != source["commit"]:
        raise ValueError(f"ABBC checkout must be {source['commit']}; observed {actual}")
    for name, digest in source["source_sha256"].items():
        require_hash(checkout / name, digest)
    extras = [path for folder in ("code_task1", "inference") for path in (checkout / folder).rglob("*.py")
              if str(path.relative_to(checkout)) not in source["source_sha256"]]
    if extras:
        raise ValueError(f"unexpected Python modules in ABBC checkout: {extras}")
    local_backend = checkout.resolve()
    # Remove caller overrides that could change routing, reconstruction or trainer defaults.
    for key in tuple(os.environ):
        if key.startswith("PENGWIN_"):
            os.environ.pop(key)
    check_backend_import(read_json(STAGE2))
    if not Path(sys.modules["core"].__file__).resolve().is_relative_to(local_backend):
        raise RuntimeError("ABBC trainer import resolved outside the verified checkout")


def verify_model(protocol: dict, model: str) -> dict:
    record = protocol["models"][model]
    config = read_json(STAGE2) if model == "stage2" else member_stage2_config(read_json(STAGE3), model)
    require_hash(Path(record["validation_manifest"]), record["validation_manifest_sha256"])
    if set(record["checkpoints"]) != {"anatomy", "fragment"}:
        raise ValueError(f"{model}: checkpoint inventory must include both stages")
    for stage, entry in record["checkpoints"].items():
        checkpoint = inference_checkpoint_path(config, stage)
        if checkpoint != Path(entry["path"]):
            raise ValueError(f"{model}/{stage}: checkpoint path mismatch")
        require_hash(checkpoint, entry["sha256"])
        for name in ("plans.json", "dataset.json"):
            require_hash(checkpoint.parent.parent / name, entry["model_metadata_sha256"][name])
    return config


def file_state(path: Path) -> dict:
    info = path.stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def fingerprint(path: Path) -> dict:
    state = file_state(path)
    digest = sha256(path)
    if file_state(path) != state:
        raise ValueError(f"file changed while reading: {path}")
    return {**state, "sha256": digest}


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def lock(path: Path, *, shared: bool = False):
    with path.open("r" if shared else "a") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is using {path.parent}") from error
        yield


def verify_record(record: dict, source: Path, files: dict[str, Path]) -> None:
    if record["image_state"] != file_state(source) or record["outputs"].keys() != files.keys():
        raise ValueError(f"source or output inventory changed for {source.parent.name}")
    for name, path in files.items():
        stored = record["outputs"][name]
        if file_state(path)["bytes"] != stored["bytes"]:
            raise ValueError(f"unexpected output size: {path}")
        require_hash(path, stored["sha256"])


def prediction_paths(root: Path, model: str, case_id: str) -> dict[str, Path]:
    paths = {"prediction": root / "models" / model / "predictions" / f"{case_id}.mha"}
    if model in MEMBERS:
        paths["probability"] = root / "models" / model / "probabilities" / f"{case_id}.mha"
    return paths


def write_checked(path: Path, array: np.ndarray, reference: sitk.Image, *, probability: bool) -> dict:
    shape = tuple(reversed(reference.GetSize()))
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid shape or non-finite voxels: {path}")
    if probability:
        if array.min() < 0 or array.max() > 1:
            raise ValueError(f"probabilities outside [0,1]: {path}")
    elif not np.issubdtype(array.dtype, np.integer) or array.min() < 0 or array.max() > 200 or not array.any():
        raise ValueError(f"invalid instance IDs or unexpected whole-volume empty result: {path}")
    image = sitk.GetImageFromArray(array.astype(np.float32 if probability else np.uint8, copy=False))
    image.CopyInformation(reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mha")
    sitk.WriteImage(image, str(temporary), useCompression=True)
    reread, _ = read_volume(temporary, geometry(reference), probability=probability)
    if not np.array_equal(reread, array):
        raise ValueError(f"volume changed on write/read: {path}")
    temporary.replace(path)
    return fingerprint(path)


def infer(protocol: dict, config_path: Path, cases: list[dict], source: dict[str, Path],
          root: Path, model: str, smoke: bool) -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E3 inference needs exactly one CUDA device visible to this process")
    torch.cuda.set_device(0)
    verify_backend(protocol)
    config = verify_model(protocol, model)
    backend = load_configured_backend(config)
    expected_module = ROOT / protocol["backend"]["path"] / "inference/inference.py"
    if Path(backend.__file__).resolve() != expected_module.resolve():
        raise RuntimeError("inference import did not resolve to the verified ABBC module")
    folder = root / "models" / model
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "inference_manifest.json"
    identity = {"protocol_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)),
                "stage2_loader_sha256": sha256(ROOT / "scripts/stage2_backend.py"),
                "inference_wrapper_sha256": sha256(ROOT / "scripts/predict_stage2_backend.py"),
                "model": model, "smoke": smoke, "case_ids": [r["case_id"] for r in cases],
                "software": {n: importlib.metadata.version(n) for n in ("torch", "nnunetv2", "numpy", "SimpleITK", "scipy")}}
    with lock(folder / ".lock"):
        manifest = read_json(path) if path.exists() else {"identity": identity, "records": {}}
        if manifest["identity"] != identity:
            raise ValueError(f"{model}: inference identity differs from existing run at {path}")
        records = manifest["records"]
        manifest["status"] = "running"
        save_json(path, manifest)
        try:
            for index, case in enumerate(cases, 1):
                cid = case["case_id"]
                image_path = source[cid] / "image.mha"
                files = prediction_paths(root, model, cid)
                if cid in records:
                    verify_record(records[cid], image_path, files)
                    print(f"[{model} {index}/{len(cases)}] {cid} verified, resuming", flush=True)
                    continue
                started = time.monotonic()
                image_state = file_state(image_path)
                reference = sitk.ReadImage(str(image_path))
                started_inference = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                print(f"[{model} {index}/{len(cases)}] {cid} {case['scan_type']}", flush=True)
                kwargs = {"prerouted_bone_masks": None, "anatomies": None}
                if model in MEMBERS:
                    prediction, probability = backend.run_per_anatomy(image_path, reference, return_foreground_probability=True, **kwargs)
                else:
                    prediction = backend.run_per_anatomy(image_path, reference, **kwargs)
                torch.cuda.synchronize()
                started_write = time.monotonic()
                outputs = {"prediction": write_checked(files["prediction"], prediction, reference, probability=False)}
                if model in MEMBERS:
                    outputs["probability"] = write_checked(files["probability"], probability, reference, probability=True)
                    del probability
                n_voxels = int(prediction.size)
                del prediction, reference
                if file_state(image_path) != image_state:
                    raise ValueError(f"CT changed during inference: {image_path}")
                records[cid] = {"image_state": image_state, "outputs": outputs, "scan_type": case["scan_type"],
                                "n_voxels": n_voxels, "read_seconds": started_inference - started,
                                "inference_seconds": started_write - started_inference,
                                "write_seconds": time.monotonic() - started_write,
                                "peak_vram_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(0)}
                save_json(path, manifest)
            manifest.update(status="complete", error=None)
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(path, manifest)
    print(f"E3 {model} COMPLETE: {len(records)}/{len(cases)}", flush=True)


def evaluate(protocol: dict, config_path: Path, cases: list[dict], source: dict[str, Path],
             root: Path, smoke: bool) -> None:
    from e3_reporting import report

    ids = [r["case_id"] for r in cases]
    if not smoke:
        e1_root = ROOT / "outputs/stage3_h1_v2"
        require_hash(e1_root / "e1_manifest.json", protocol["e1_manifest_sha256"])
        manifest = read_json(e1_root / "e1_manifest.json")
        comparison = "validation/comparison.csv"
        require_hash(e1_root / comparison, manifest["output_sha256"][comparison])
    table_names = ("scores", "pair_components", "model_risks", "risks", "case_scores", "case_risks", "case_model_risks")
    tables: dict[str, list[dict]] = {name: [] for name in table_names}
    with ExitStack() as stack:
        stack.enter_context(lock(root / ".evaluate.lock"))
        model_manifests = {}
        runtime = []
        for model in MODELS:
            folder = root / "models" / model
            if not (folder / "inference_manifest.json").is_file():
                raise FileNotFoundError(f"missing inference manifest: {model}")
            stack.enter_context(lock(folder / ".lock", shared=True))
            manifest = read_json(folder / "inference_manifest.json")
            identity = manifest["identity"]
            if (manifest["status"] != "complete" or set(manifest["records"]) != set(ids)
                or identity["protocol_sha256"] != sha256(config_path) or identity["case_ids"] != ids
                or identity["model"] != model or identity["smoke"] != smoke):
                raise ValueError(f"{model}: incomplete or mismatched inference")
            model_manifests[model] = manifest
            for cid, row in manifest["records"].items():
                runtime.append({"model": model, "case_id": cid, **{k: v for k, v in row.items() if k not in {"outputs", "image_state"}}})
        files = [Path(__file__), ROOT / "scripts/e3_reporting.py",
                 *(ROOT / "src/exchange_ml" / name for name in
                   ("e1_scoring.py", "e1_data.py", "e1_statistics.py", "instances.py", "metrics.py"))]
        provenance = {"protocol_sha256": sha256(config_path), "smoke": smoke,
                      "models": {m: signature(v) for m, v in model_manifests.items()},
                      "code": {str(p.relative_to(ROOT)): sha256(p) for p in files},
                      "software": {n: importlib.metadata.version(n) for n in ("numpy", "SimpleITK", "scipy", "matplotlib")}}
        evaluation_signature = signature(provenance)
        path = root / "e3_manifest.json"
        if path.exists() and read_json(path)["signature"] != evaluation_signature:
            raise ValueError(f"evaluation identity changed at {path}")
        status = {"status": "running", "signature": evaluation_signature, "provenance": provenance, "case_ids": ids}
        save_json(path, status)
        try:
            scoring_times = []
            for index, case in enumerate(cases, 1):
                cid, scan = case["case_id"], case["scan_type"]
                gt = source[cid] / "label.mha"
                paths = {"gt": gt}
                for model in MODELS:
                    outputs = prediction_paths(root, model, cid)
                    verify_record(model_manifests[model]["records"][cid], source[cid] / "image.mha", outputs)
                    for kind, file in outputs.items():
                        paths["stage2" if model == "stage2" else f"{model}_{kind}"] = file
                cache = root / "case_cache" / f"{cid}.json"
                gt_hash = fingerprint(gt)
                if cache.exists():
                    entry = read_json(cache)
                    if entry["signature"] != evaluation_signature or entry["gt"] != gt_hash:
                        raise ValueError(f"evaluation cache or GT changed: {cid}")
                else:
                    started = time.monotonic()
                    result = process_case(cid, scan, paths, protocol)
                    if file_state(gt) != {k: v for k, v in gt_hash.items() if k != "sha256"}:
                        raise ValueError(f"GT changed while reading: {gt}")
                    entry = {"signature": evaluation_signature, "gt": gt_hash, "tables": result,
                             "seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    save_json(cache, entry)
                for key in tables:
                    tables[key].extend(entry["tables"][key])
                scoring_times.append({**case, "scoring_and_evaluation_seconds": entry["seconds"], "process_peak_rss_kib": entry["peak_rss_kib"]})
                print(f"[E3 evaluate {index}/{len(cases)}] {cid} {entry['seconds']:.1f}s", flush=True)
            status["summary"] = report(root, tables, runtime, scoring_times, protocol, smoke)
            status["output_sha256"] = {str(p.relative_to(root)): sha256(p) for p in (*root.glob("*.csv"), *root.glob("*.pdf"), *root.glob("*.png"), root / "summary.json")}
            status["status"] = "complete"
        except BaseException as error:
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            save_json(path, status)
    print(f"E3 {'SMOKE' if smoke else 'HELD-OUT'} EVALUATION COMPLETE: {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage3_h1_v2.json")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check", help="CPU preflight and checkpoint verification")
    run = sub.add_parser("infer", help="one model on one assigned GPU; resumes automatically")
    run.add_argument("--model", required=True, choices=MODELS)
    run.add_argument("--smoke", action="store_true", help="one pelvic and one femur case from the prior validation split, separate output")
    evaluation = sub.add_parser("evaluate", help="score and evaluate the four completed model outputs on CPU")
    evaluation.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol, cases, source = load_protocol(args.config)
    if args.action == "check":
        verify_backend(protocol)
        for model in MODELS:
            verify_model(protocol, model)
            print(f"{model}: both checkpoints verified", flush=True)
        require_hash(ROOT / "outputs/stage3_h1_v2/e1_manifest.json", protocol["e1_manifest_sha256"])
        print(f"E3 PREFLIGHT PASS: 100 held-out cases, 50+50; free disk {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB")
        return
    root = (ROOT / protocol["output_root"]).resolve()
    if root != RESULT_ROOT.resolve() or not root.is_relative_to(ROOT / "outputs/stage3_h1_v2") or root == RESULT_ROOT.parent:
        raise ValueError("E3 requires its dedicated held-out output path")
    if args.smoke:
        # Use already scored validation examples, so smoke results cannot reach held-out analysis.
        cases = [{"case_id": "004", "scan_type": "pelvic"}, {"case_id": "254", "scan_type": "femur"}]
        inventory = data_inventory(Path(protocol["dataset_root"]))
        source = {r["case_id"]: inventory[r["case_id"]][1] for r in cases}
        root = ROOT / "outputs/smoke_e3"
    if args.action == "infer":
        infer(protocol, args.config, cases, source, root, args.model, args.smoke)
    else:
        evaluate(protocol, args.config, cases, source, root, args.smoke)


if __name__ == "__main__":
    main()
