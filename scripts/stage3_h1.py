#!/usr/bin/env python3
"""Validate, train, or print the frozen Stage 3 H1 runbook.

All training and dataset-wide inference/analysis commands printed here are
user-typed full runs under __docs__/rule.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from stage2_backend import (  # noqa: E402
    check as check_stage2_environment,
    load_config as load_stage2_config,
    patch_and_train,
    stage2_paths,
)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def member_record(config: dict[str, Any], member_id: str) -> dict[str, Any]:
    matches = [
        row for row in config["ensemble"]["members"] if row["id"] == member_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown or duplicate Stage 3 member: {member_id}")
    return matches[0]


def member_root(config: dict[str, Any], member_id: str) -> Path:
    return Path(config["workspace_root"]) / "members" / member_id


def member_stage2_config(
    config: dict[str, Any], member_id: str
) -> dict[str, Any]:
    member_record(config, member_id)
    stage2 = copy.deepcopy(load_stage2_config(Path(config["stage2_config"])))
    shared = stage2_paths(stage2)
    root = member_root(config, member_id)
    stage2["status"] = "stage3_h1_ensemble_member"
    stage2["paths"]["workspace_root"] = str(root)
    stage2["paths"]["raw_root"] = str(shared["raw"])
    stage2["paths"]["preprocessed_root"] = str(shared["preprocessed"])
    stage2["paths"]["results_root"] = str(root / "nnUNet_results")
    return stage2


def checkpoint_path(
    stage2: dict[str, Any], stage: str, checkpoint_name: str = "checkpoint_best.pth"
) -> Path:
    stage_key = "anatomy_stage" if stage == "anatomy" else "fragment_stage"
    stage_config = stage2[stage_key]
    software = stage2["software"]
    return (
        stage2_paths(stage2)["results"]
        / stage_config["dataset_name"]
        / (
            f"{stage_config['trainer']}__{software['plans']}__"
            f"{software['configuration']}"
        )
        / "fold_0"
        / checkpoint_name
    )


def validate_config(config: dict[str, Any]) -> None:
    members = config["ensemble"]["members"]
    if int(config["ensemble"]["size"]) != 3 or len(members) != 3:
        raise ValueError("Stage 3 main ensemble size must be exactly M=3")
    member_ids = [str(row["id"]) for row in members]
    seeds = [int(row["model_seed"]) for row in members]
    if len(set(member_ids)) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage 3 member IDs and model seeds must be unique")
    if config["ensemble"].get("reuse_stage2_checkpoint") is not False:
        raise ValueError("unseeded Stage 2 checkpoints must not be reused in H1")
    expected_reproducibility = {
        "python_hash_seed": "model_seed",
        "python_random_seed": "model_seed + local DDP rank",
        "numpy_seed": "model_seed + local DDP rank",
        "torch_seed": "model_seed + local DDP rank",
        "cuda_seed": "model_seed + local DDP rank",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if config.get("reproducibility") != expected_reproducibility:
        raise ValueError("Stage 3 reproducibility policy drifted")
    if config["analysis_role"] != "validation":
        raise ValueError("Stage 3 H1 analysis role must remain validation")
    if int(config["trajectory_seed"]) != 20260920:
        raise ValueError("Stage 3 must use the frozen initial labeled trajectory")


def check(config: dict[str, Any]) -> None:
    validate_config(config)
    stage2 = load_stage2_config(Path(config["stage2_config"]))
    check_stage2_environment(stage2)
    stage2_summary = Path("outputs/stage2_backend/validation_summary.json")
    stage2_metrics = Path(config["risk_contract"]["metrics_csv"])
    if not stage2_summary.is_file() or not stage2_metrics.is_file():
        raise FileNotFoundError("frozen Stage 2 validation artifacts are missing")
    paths = stage2_paths(stage2)
    for required in (paths["raw"], paths["preprocessed"]):
        if not required.is_dir():
            raise FileNotFoundError(f"shared Stage 2 dataset artifact missing: {required}")
    for row in config["ensemble"]["members"]:
        derived = member_stage2_config(config, row["id"])
        states = {
            stage: checkpoint_path(derived, stage).is_file()
            for stage in ("anatomy", "fragment")
        }
        print(
            f"{row['id']} seed={row['model_seed']} "
            f"anatomy_checkpoint={states['anatomy']} "
            f"fragment_checkpoint={states['fragment']}"
        )
    print("Stage 3 H1 implementation gate: PASS")


def train_member(
    config: dict[str, Any],
    member_id: str,
    stage: str,
    device: str,
    num_gpus: int,
    resume: bool = False,
) -> None:
    validate_config(config)
    member = member_record(config, member_id)
    derived = member_stage2_config(config, member_id)
    if stage == "fragment":
        anatomy_best = checkpoint_path(derived, "anatomy").resolve()
        if not anatomy_best.is_file():
            raise FileNotFoundError(f"Fragment warm-start checkpoint missing: {anatomy_best}")
        # PyTorch 2.1 cannot safely unpickle nnU-Net's NumPy metadata. Trust
        # only this member's locally produced Anatomy checkpoint.
        os.environ["PENGWIN_TRUSTED_WARMSTART_CHECKPOINT"] = str(anatomy_best)
    else:
        os.environ.pop("PENGWIN_TRUSTED_WARMSTART_CHECKPOINT", None)
    if resume:
        latest = checkpoint_path(derived, stage, "checkpoint_latest.pth").resolve()
        if not latest.is_file():
            raise FileNotFoundError(
                f"Resume requires checkpoint_latest.pth; refusing a fresh run: {latest}"
            )
        if member_id == "member_01" and stage == "anatomy":
            # Explicit approval covers only this locally produced checkpoint.
            os.environ["PENGWIN_TRUSTED_RESUME_CHECKPOINT"] = str(latest)
        else:
            os.environ.pop("PENGWIN_TRUSTED_RESUME_CHECKPOINT", None)
        print(f"Resuming {member_id} {stage} from {latest}", flush=True)
    else:
        os.environ.pop("PENGWIN_TRUSTED_RESUME_CHECKPOINT", None)
    patch_and_train(
        derived,
        stage,
        device,
        num_gpus,
        model_seed=int(member["model_seed"]),
        resume=resume,
    )


def print_runbook(config: dict[str, Any]) -> None:
    validate_config(config)
    execution = config["execution"]
    training_prefix = (
        f"CUDA_VISIBLE_DEVICES={execution['training_cuda_visible_devices']} "
        "conda run --no-capture-output -n exchange-stage2 "
        "python scripts/stage3_h1.py train"
    )
    print("# AGENT-TYPED lightweight preflight")
    print("conda run -n exchange-stage2 python scripts/stage3_h1.py check")
    print("# USER-TYPED full M=3 ensemble training")
    for row in config["ensemble"]["members"]:
        member_id = row["id"]
        seed = row["model_seed"]
        suffix = (
            f"--member {member_id} --device cuda "
            f"--num-gpus {execution['training_num_gpus']}"
        )
        print(f"# {member_id}: frozen model seed {seed}")
        print(f"{training_prefix} --stage anatomy {suffix}")
        print(f"{training_prefix} --stage fragment {suffix}")
    print("# USER-TYPED full validation inference with probabilities")
    for row in config["ensemble"]["members"]:
        print(
            f"CUDA_VISIBLE_DEVICES={execution['inference_cuda_visible_device']} "
            "conda run --no-capture-output -n exchange-stage2 "
            f"python scripts/predict_stage3_member.py --member {row['id']} --resume"
        )
    print("# USER-TYPED dataset-wide GT-free uncertainty scoring")
    print("conda run -n exchange-stage2 python scripts/score_stage3_h1.py")
    print("# USER-TYPED retrospective H1 evaluation against frozen Stage 2 risks")
    print("conda run -n exchange-stage2 python scripts/evaluate_stage3_h1.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage3_h1.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("print-runbook")
    train = subparsers.add_parser("train")
    train.add_argument("--member", required=True)
    train.add_argument("--stage", choices=("anatomy", "fragment"), required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--num-gpus", type=int, default=1)
    train.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "check":
        check(config)
    elif args.command == "print-runbook":
        print_runbook(config)
    elif args.command == "train":
        train_member(
            config, args.member, args.stage, args.device, args.num_gpus, args.resume
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
