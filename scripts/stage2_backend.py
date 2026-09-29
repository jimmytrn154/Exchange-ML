#!/usr/bin/env python3
"""Validate, train, or print the frozen Stage 2 runbook.

Full preprocessing, split installation, and training are user-typed commands.
``check`` is a lightweight read-only gate. No model fallback is implemented.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def stage2_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["paths"]["workspace_root"])
    return {
        "root": root,
        "raw": Path(config["paths"].get("raw_root", root / "nnUNet_raw")),
        "preprocessed": Path(
            config["paths"].get("preprocessed_root", root / "nnUNet_preprocessed")
        ),
        "results": Path(
            config["paths"].get("results_root", root / "nnUNet_results")
        ),
        "manifest": root / "stage2_preparation_manifest.json",
    }


def environment(config: dict[str, Any]) -> dict[str, str]:
    paths = stage2_paths(config)
    baseline = Path(config["paths"]["baseline_root"])
    env = dict(os.environ)
    env.update(
        {
            "nnUNet_raw": str(paths["raw"].resolve()),
            "nnUNet_preprocessed": str(paths["preprocessed"].resolve()),
            "nnUNet_results": str(paths["results"].resolve()),
            "PENGWIN_CODE_DIR": str((baseline / "code_task1").resolve()),
            "PENGWIN_ROOT": str(baseline.resolve()),
            "MPLCONFIGDIR": "/tmp/exchange-ml-matplotlib",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PENGWIN_NUM_EPOCHS": str(config["fragment_stage"]["max_epochs"]),
            "PENGWIN_INITIAL_LR": str(config["fragment_stage"]["initial_lr"]),
            "PENGWIN_ES_MIN_EPOCHS": str(
                config["fragment_stage"]["early_stop_min_epochs"]
            ),
            "PENGWIN_ES_PATIENCE": str(
                config["fragment_stage"]["early_stop_patience"]
            ),
            "PENGWIN_ES_MIN_DELTA": str(
                config["fragment_stage"]["early_stop_min_delta"]
            ),
        }
    )
    return env


def check(config: dict[str, Any]) -> None:
    expected = str(config["software"]["required_nnunetv2"])
    observed = importlib.metadata.version("nnunetv2")
    baseline = Path(config["paths"]["baseline_root"])
    required = [
        baseline / "code_task1" / "core.py",
        baseline / "code_task1" / "loss.py",
        baseline / "code_task1" / "model.py",
        baseline / "inference" / "inference.py",
        Path(config["split"]["manifest"]),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage 2 inputs: {missing}")
    print(f"nnunetv2 observed={observed} required={expected}")
    if observed != expected:
        raise RuntimeError(
            "Stage 2 environment mismatch. The ABBC trainer is pinned to "
            "nnunetv2==2.5.1; no compatibility fallback is allowed."
        )
    code_dir = (baseline / "code_task1").resolve()
    sys.path.insert(0, str(code_dir))
    import core  # type: ignore[import-not-found]  # noqa: F401

    print("ABBC trainer import: PASS")


def install_splits(config: dict[str, Any]) -> None:
    paths = stage2_paths(config)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("case_limit") is not None:
        raise RuntimeError("refusing to install smoke-test splits into production preprocessing")
    for stage_key, config_key in (
        ("anatomy", "anatomy_stage"),
        ("fragment", "fragment_stage"),
    ):
        dataset = config[config_key]["dataset_name"]
        target = paths["preprocessed"] / dataset / "splits_final.json"
        if not target.parent.is_dir():
            raise FileNotFoundError(f"preprocessed dataset missing: {target.parent}")
        payload = [
            {
                "train": manifest["split_samples"][stage_key]["train"],
                "val": manifest["split_samples"][stage_key]["validation"],
            }
        ]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"installed frozen split: {target}")


def _patch_trainer_discovery(stage_config: dict[str, Any]) -> None:
    """Expose the repository-local ABBC trainer to one nnU-Net process."""
    import core  # type: ignore[import-not-found]
    import nnunetv2.run.run_training as run_training

    original = run_training.recursive_find_python_class

    def resolve(folder: str, class_name: str, current_module: str):
        if class_name == stage_config["trainer"]:
            return getattr(core, class_name)
        return original(folder, class_name, current_module)

    run_training.recursive_find_python_class = resolve


def patch_inference_trainer_discovery(config: dict[str, Any]) -> None:
    """Expose both repository-local trainers to nnU-Net inference.

    nnU-Net 2.5.1 imports ``recursive_find_python_class`` directly into
    ``predict_from_raw_data``. Importing the custom trainer module is therefore
    insufficient: the resolver used while restoring a checkpoint must be
    patched in that module as well.
    """
    import core  # type: ignore[import-not-found]
    import nnunetv2.inference.predict_from_raw_data as predict_from_raw_data

    trainer_names = {
        config["anatomy_stage"]["trainer"],
        config["fragment_stage"]["trainer"],
    }
    original = predict_from_raw_data.recursive_find_python_class

    def resolve(folder: str, class_name: str, current_module: str):
        if class_name in trainer_names:
            return getattr(core, class_name)
        return original(folder, class_name, current_module)

    predict_from_raw_data.recursive_find_python_class = resolve


def _run_ddp_rank(
    rank: int,
    num_gpus: int,
    stage_config: dict[str, Any],
    software_config: dict[str, Any],
    model_seed: int | None,
) -> None:
    """Run one DDP rank while preserving custom trainer discovery in spawn children."""
    import nnunetv2.run.run_training as run_training

    _seed_training_process(model_seed, rank)
    _patch_trainer_discovery(stage_config)
    run_training.run_ddp(
        rank,
        str(stage_config["dataset_id"]),
        software_config["configuration"],
        0,
        stage_config["trainer"],
        software_config["plans"],
        False,  # use_compressed_data
        False,  # disable_checkpointing
        False,  # continue_training
        False,  # only_run_validation
        None,  # pretrained_weights; Stage B uses PENGWIN_STUNET_PRETRAINED
        False,  # export_validation_probabilities
        False,  # validate with best checkpoint
        num_gpus,
    )


def _seed_training_process(model_seed: int | None, rank: int = 0) -> None:
    """Seed one training process without changing the frozen optimizer or schedule."""
    if model_seed is None:
        return
    import random

    import numpy as np
    import torch

    effective_seed = int(model_seed) + int(rank)
    os.environ["PYTHONHASHSEED"] = str(int(model_seed))
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patch_and_train(
    config: dict[str, Any],
    stage: str,
    device: str,
    num_gpus: int,
    model_seed: int | None = None,
) -> None:
    check(config)
    env = environment(config)
    os.environ.update(env)
    stage_config = config[
        "anatomy_stage" if stage == "anatomy" else "fragment_stage"
    ]
    if stage == "fragment":
        anatomy = config["anatomy_stage"]
        anatomy_checkpoint = (
            stage2_paths(config)["results"]
            / anatomy["dataset_name"]
            / (
                f"{anatomy['trainer']}__{config['software']['plans']}__"
                f"{config['software']['configuration']}"
            )
            / "fold_0"
            / "checkpoint_best.pth"
        )
        if not anatomy_checkpoint.is_file():
            raise FileNotFoundError(
                "fragment training requires the frozen anatomy warm-start checkpoint: "
                f"{anatomy_checkpoint}"
            )
        os.environ["PENGWIN_STUNET_PRETRAINED"] = str(anatomy_checkpoint.resolve())
    sys.path.insert(0, env["PENGWIN_CODE_DIR"])
    import nnunetv2.run.run_training as run_training
    import torch

    if num_gpus < 1:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    torch_device = torch.device(device)
    if num_gpus > 1 and torch_device.type != "cuda":
        raise ValueError("multi-GPU training requires --device cuda")

    _seed_training_process(model_seed)
    _patch_trainer_discovery(stage_config)
    if num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(run_training.find_free_network_port())
        torch.multiprocessing.spawn(
            _run_ddp_rank,
            args=(num_gpus, stage_config, config["software"], model_seed),
            nprocs=num_gpus,
            join=True,
        )
        return

    run_training.run_training(
        str(stage_config["dataset_id"]),
        config["software"]["configuration"],
        0,
        trainer_class_name=stage_config["trainer"],
        plans_identifier=config["software"]["plans"],
        device=torch_device,
    )


def print_runbook(config: dict[str, Any]) -> None:
    paths = stage2_paths(config)
    anatomy = config["anatomy_stage"]
    fragment = config["fragment_stage"]
    execution = config["execution"]
    training_prefix = (
        f"CUDA_VISIBLE_DEVICES={execution['training_cuda_visible_devices']} "
        "conda run --no-capture-output -n exchange-stage2 "
        "python scripts/stage2_backend.py"
    )
    training_suffix = (
        f"--device cuda --num-gpus {execution['training_num_gpus']}"
    )
    print("# USER-TYPED: create the pinned environment")
    print("conda create -n exchange-stage2 python=3.10 -y")
    print(
        "conda run -n exchange-stage2 pip install -r "
        "baselines/pengwin2026-task1-abbc/requirements.txt"
    )
    print("# USER-TYPED: verify the pinned trainer environment before conversion")
    print("conda run -n exchange-stage2 python scripts/stage2_backend.py check")
    print("# USER-TYPED: build exactly 40 train + 40 validation cases")
    print("conda run -n exchange-stage2 python scripts/prepare_stage2_backend.py")
    env = (
        f"nnUNet_raw={paths['raw']} nnUNet_preprocessed={paths['preprocessed']} "
        f"nnUNet_results={paths['results']} MPLCONFIGDIR=/tmp/exchange-ml-matplotlib"
    )
    print("# USER-TYPED: fingerprint, plan, and preprocess")
    print(
        f"{env} conda run -n exchange-stage2 nnUNetv2_plan_and_preprocess "
        f"-d {anatomy['dataset_id']} {fragment['dataset_id']} "
        "-pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity"
    )
    print("# USER-TYPED: install the frozen split after preprocessing")
    print(
        "conda run -n exchange-stage2 python "
        "scripts/stage2_backend.py install-splits"
    )
    print("# USER-TYPED: train anatomy, then fragment backend")
    print(
        f"{training_prefix} train --stage anatomy {training_suffix}"
    )
    print("# Fragment training fails unless anatomy checkpoint_best.pth exists")
    print(
        f"{training_prefix} train --stage fragment {training_suffix}"
    )
    print("# USER-TYPED: validation inference; fails on any all-zero/error case")
    print(
        "conda run -n exchange-stage2 python scripts/predict_stage2_backend.py --resume"
    )
    print("# USER-TYPED: complete 40-case metric evaluation")
    print("conda run -n exchange-stage2 python scripts/evaluate_stage2_backend.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_backend.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("install-splits")
    subparsers.add_parser("print-runbook")
    train = subparsers.add_parser("train")
    train.add_argument("--stage", choices=("anatomy", "fragment"), required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--num-gpus", type=int, default=1)
    train.add_argument("--model-seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "check":
        check(config)
    elif args.command == "install-splits":
        install_splits(config)
    elif args.command == "print-runbook":
        print_runbook(config)
    elif args.command == "train":
        patch_and_train(
            config,
            args.stage,
            args.device,
            args.num_gpus,
            model_seed=args.model_seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
