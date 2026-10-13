# Stage 2 — Stable fixed PENGWIN backend

Status: **REOPENED (2026-10-12) — RETRAINING IN PROGRESS.** Stage A
retrained with the backend fixes and passed gate G1 (2026-10-13); Stage B,
validation inference, and metrics are pending, so the stage is not closed.
The 2026-09-26 PASS below is retained as history.

Date: 2026-09-26 (Asia/Ho_Chi_Minh). Reopened: 2026-10-12 by user decision.
Last updated: 2026-10-13.

## Reopening note (2026-10-12)

The Stage 3 diagnosis (`__docs__/reports/stage3/stage3_h1.md`, Post-hoc
diagnosis) found that Stage A was trained with axis-2 (L/R) mirror
augmentation on `Dataset701_PENGWINStage2Anatomy`, because the reference
trainer disables it only for `Dataset539`. LeftHip/RightHip labels were not
swapped with the image. On the 20 pelvic validation cases this backend covers
a median 0.51 of the GT left hip with any foreground (0.25 with the correct
anatomy label; 10/20 cases below 0.5). That does not meet this stage's gateway
("stable, meaningful instance predictions suitable for structural uncertainty
analysis"), and this model's metrics are the frozen H1 risk labels.

The backend fix and its tests are tracked in
`__docs__/reports/stage3/stage3_fix_progress.md` (F1). The Stage A + Stage B
backend must be retrained with the fix, re-evaluated, and re-closed before its
metrics are used as H1 risks.

Two further backend defects were found after reopening (details and evidence
in the tracker): the Anatomy partial-label case map was missing in every run,
so visible-but-unlabelled bones were trained as background (D3); and Fragment
validation/early stopping was computed per DDP rank (D2). Pre-fix Stage A
per-bone Dice on labelled validation cases (`scripts/check_anatomy_gate.py`):
Sacrum 0.973, LeftHip 0.613 (1/20 cases < 0.5), RightHip 0.371 (15/20),
Femur 0.978.

## Retraining with backend fixes (2026-10-12 to 2026-10-13)

### Changes to the frozen backend

All changes are corrections to the reference backend; architecture, loss
formulas, optimizer, schedule, split, preprocessing, and reconstruction are
unchanged. Changes to the ABBC code are recorded in
`patches/pengwin2026-task1-abbc_on_ccc625f.patch` (base commit `ccc625f`).

| ID | Change | Where |
|---|---|---|
| F1 | Axis-2 (L/R) mirroring disabled for `Dataset701_PENGWINStage2Anatomy` in training and inference; Fragment keeps all axes. | `code_task1/core.py` `DISABLE_X_MIRROR_DATASETS` |
| F2 | Fragment validation outputs are all-gathered across DDP ranks before F1, EMA, best-checkpoint, and early-stop decisions. | `code_task1/core.py` |
| F6 | Anatomy partial-label case map built per run from the `Dataset701` training labels (40 pelvic → `{1,2,3}`, 40 femur → `{4}`), checked against the frozen split, passed via `PENGWIN_CASE_LABELED_MAP`; an explicitly supplied map must load. | `scripts/stage2_backend.py`, `src/exchange_ml/stage2_backend.py`, `code_task1/core.py` |
| — | Fragment warm-start trusts only this run's own Anatomy `checkpoint_best.pth` (PyTorch 2.1.2 NumPy-pickle restriction). | `scripts/stage2_backend.py` |
| — | Explicit model seed `31000` (the 2026-09-26 run's seed was not recorded). | user-typed command |

Verification: 37/37 unit tests (including `tests/test_stage3_backend_fixes.py`);
a 2-GPU smoke run of both stages passed (tracker, F4).

Pre-fix artifacts were moved, not deleted, to
`outputs/archive/2026-10-12_pre_backend_fix/stage2_backend/`.

### Stage A result (C-0.1)

- Run: seed `31000`, GPUs `0,1`, 2026-10-13 07:36–15:25; log
  `outputs/logs/c0_stage2_anatomy.log`.
- Fixes active: case map loaded (80 cases); axis-2 mirror disabled
  (`inference_allowed_mirroring_axes (0, 1)`).
- Early stop at epoch 109 (best EMA pseudo-Dice 0.9721 at epoch 84); 40-case
  end-of-training validation completed; no DDP errors.
- **Gate G1: PASS.** Per-bone Dice on labelled validation cases (cases < 0.5):

| Bone | Pre-fix | Retrained |
|---|---:|---:|
| Sacrum | 0.973 (0) | 0.975 (0) |
| LeftHip | 0.613 (1) | **0.988 (0)** |
| RightHip | 0.371 (15) | **0.986 (0)** |
| Femur | 0.978 (0) | 0.984 (0) |

nnU-Net's own `Mean Validation Dice 0.498` is a class mean that counts
visible-but-unlabelled bones in the other scan type as Dice 0, so it is not
used. The hip improvement reflects F1 and F6 together; this run cannot
separate them. It is a Stage A result only, not an instance-level result.

### Pending

Stage B training (C-0.2), validation inference, metrics, and gate G2. The
evaluation stays the frozen whole-volume contract (decision D4); femur-scan
false positives from visible pelvic bone are recorded as a limitation, with a
labelled-anatomy-only view as a secondary analysis only.

## Configuration

- Frozen config: `configs/stage2_backend.json`.
- Training set: 40 initial labeled cases from trajectory seed `20260920`.
- Validation set: frozen 40-case validation split.
- Test set: untouched.
- Backend: two-stage ABBC/STU-Net reference.
  1. Five-class anatomy localization with `PengwinTrainerSTUNetBaseAnatomyV301`.
  2. Per-anatomy, CT-only, 13-output ABBC + learned-affinity model with `PengwinTrainerSTUNetBaseAffinityV308`.
- Reconstruction: average-linkage affinity decoding at `T=0.75`; conditional Femur retry at `T=0.15` only under the frozen large-single-instance condition.
- Software contract: `nnunetv2==2.5.1`, ResEnc-L plans, `3d_fullres`.
- Training execution: native nnU-Net DDP on two explicitly selected GPUs; the
  current runbook maps physical GPUs `5,6` to the two local DDP ranks.
- Initialization: Stage A trains from scratch because no trusted external STU-Net checkpoint is present; Stage B must warm-start from Stage A's `checkpoint_best.pth`, with its 13-channel output head reinitialized by the reference loader.

## What was checked or changed

- Read the complete project documentation and inspected the preserved ABBC source at commit `ccc625fd000ec2dfce622b4621bf2d0b4ee12796`.
- Confirmed that no ABBC/STU-Net checkpoint is present in the workspace.
- Added the split-safe Stage 2 config, conversion helpers, preprocessing entrypoint, version-pinned trainer runner, strict validation inference, evaluator, and synthetic tests.
- The converter uses only the frozen 40-case initial set and 40-case validation set. It records the 50 test IDs but does not read test volumes.
- Stage A maps subject-local instance labels to background/Sacrum/LeftHip/RightHip/Femur.
- Stage B creates per-anatomy bone-LUT CT crops and retains contiguous local instance IDs. The pinned V308 loss generates ABBC and affinity targets during training.
- Trainer discovery is patched in-process; site-packages are not modified.
- Validation inference raises on model/runtime errors and all-zero predictions. It does not use the challenge entrypoint's zero-mask fallback.
- A representative end-to-end conversion smoke used one frozen train case and one frozen validation case. It produced 2 full anatomy samples and 6 per-anatomy fragment samples; every image/label pair had matching size and spacing, the source cases were disjoint, and the manifest recorded `test_data_read=false`.

## Observed result

- An earlier base-environment preflight observed PyTorch `2.7.1+cu126`, SimpleITK `2.5.2`, and nnU-Net `2.8.1`.
- Preserved ABBC code requires nnU-Net `2.5.1`. Import under `2.8.1` fails with:

```text
ModuleNotFoundError: No module named 'nnunetv2.training.dataloading.data_loader_3d'
```

- GPU visibility from the agent sandbox failed because NVIDIA-SMI could not communicate with the driver. This is not evidence that the host has no usable GPU.
- Full preprocessing, training, validation inference and validation metrics were not run by the agent because `__docs__/rule.md` classifies them as user-typed commands.
- On 2026-09-22, the active agent base environment did not provide `SimpleITK`; importing the preparation, prediction, and evaluation CLIs therefore failed with `ModuleNotFoundError: No module named 'SimpleITK'`. This does not authorize a fallback or installation by the agent. The printed runbook creates the pinned environment and now requires `scripts/stage2_backend.py check` before conversion.
- The baseline Git commit could not be re-read through Git because Git rejected the shared checkout as dubious ownership. A command-scoped `safe.directory` override was also rejected. No global Git configuration was changed; the reference commit recorded in the frozen configuration remains a documented, not newly reverified, fact for this run.
- `ruff` was not available in the active environment (`zsh: command not found: ruff`). The Stage 2 unit tests and JSON validation passed, but lint was not rerun and is not reported as passing in this run.
- A cleanup briefly removed the `fragment` binding from `print_runbook`, and the runbook smoke failed at dataset-ID interpolation with `NameError: name 'fragment' is not defined`. The binding was restored; the affected checks were rerun after the fix.
- The first user-run dependency installation did not complete. Pip selected the correct pinned CUDA 11.8 wheel, `torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl`, but the connection was interrupted repeatedly and pip stopped after six attempts with `error: incomplete-download` at `869.6 MB/2325.9 MB`. Pip explicitly identified this as a network-connectivity failure. No package version or backend setting was changed.
- The subsequent user-run environment gate passed: `nnunetv2 observed=2.5.1 required=2.5.1` and `ABBC trainer import: PASS`. The earlier interrupted dependency download is therefore resolved. Dataset conversion and all later full runs remain pending.
- The user-run conversion then stopped at the final metadata write with `PermissionError: [Errno 13] Permission denied: 'outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy/dataset.json'`. A read-only permission audit showed mode `775` on the output directories and `664` on the existing metadata files, with no write permission for an unrelated user. This is an output-tree ownership/ACL problem, not a dataset or ABBC failure.
- The first two-GPU anatomy launch stopped before training because the wrapper
  patched the newer nnU-Net trainer-resolver name
  `recursive_find_trainer_class_by_name`, while pinned nnU-Net `2.5.1` exposes
  `recursive_find_python_class`. The wrapper now patches the exact pinned API
  inside both the parent and spawned DDP ranks; no checkpoint was written by
  the failed launch.
- The next two-GPU launch reached both DDP ranks but stopped during trainer
  construction because the ABBC warning-suppression shim assumed
  `torch.amp.GradScaler`, which is absent from the pinned PyTorch `2.1.2` even
  though `torch.amp` itself imports. The shim now feature-detects the class and
  retains nnU-Net's compatible `torch.cuda.amp.GradScaler` on PyTorch 2.1.2.
  This is an API compatibility correction; it does not alter the model, loss,
  optimizer, precision policy, or training schedule. No checkpoint was written
  by the failed launch.
- A read-only inventory found pre-existing, internally count-consistent conversion artifacts: 80 anatomy images, 80 anatomy labels, 160 fragment images, 160 fragment labels, `numTraining` values 80/160, 40 train plus 40 validation anatomy samples, 80 train plus 80 validation fragment samples, 160 fragment manifest rows, and `test_data_read=false`. Because the user-run command ended with an exception, conversion is not marked successful until output access is repaired and the command exits normally.

## Stage closure evidence (2026-09-26)

Superseded by the reopening. These artifacts now live under
`outputs/archive/2026-10-12_pre_backend_fix/stage2_backend/`.

The user completed preprocessing, both training stages, and validation inference.
A fresh read-only artifact validation passed the frozen internal contract:

- two non-empty `checkpoint_best.pth` files (anatomy: `465453162` bytes; fragment: `465420346` bytes);
- 40/40 validation predictions matching the frozen split;
- inference manifest: 39 `COMPLETE`, 1 validated `VALID_RESUME`;
- 120 metric rows: 40 cases × connectivity `{6,18,26}`;
- primary 26-connectivity mean binary Dice: `0.7639089542079369`;
- primary mean Instance F1: `0.7367580056395846`;
- total merge errors: `8`; total split errors: `115`;
- 31/40 cases contain at least one merge or split error;
- 29/40 cases have incomplete instance recall.

Artifact SHA-256 fingerprints:

- `configs/stage2_backend.json`: `86c6d12e432f736cc3b76f634d52c6b1dccbea346b1a1bdf5071b5dc600e80c6`;
- `validation_inference_manifest.json`: `ad6461838bf10c4e56b3e27a3f35a3ef670b26f20a2c1f58c578310530657ad0`;
- `validation_metrics.csv`: `a325174fb6ff5b9da0ca2256e1199835fa54d8cde839dc67edf749fe521a9bf0`;
- `validation_summary.json`: `910db1cc2738013f1929b554b6d363e9c12675f771e55433c45ab1981a835bf6`.

Exact closure command typed by the agent:

```bash
python3 scripts/validate_stage2_artifacts.py
```

## Gateway

**REOPENED (2026-10-12): not closed.** Stage A retrained and passed G1
(2026-10-13); the gateway is re-assessed at G2 after Stage B, inference, and
metrics.

Original 2026-09-26 decision: **PASS — internal research backend.** The model completed all 40 frozen validation cases with nonzero instance outputs and useful variation in structural error, so it is operationally suitable for Stage 3 uncertainty analysis. This is not evidence of multi-seed training stability, expert anatomical validity, hidden-test performance, or equivalence to the official PENGWIN evaluator.

## Agent-typed commands

The lightweight commands actually executed while preparing and continuing this stage were:

```bash
cat __docs__/rule.md
sed -n '184,235p' __docs__/plan.md
cat __docs__/reports/stage2_backend.md
cat configs/stage2_backend.json
sed -n '1,360p' scripts/prepare_stage2_backend.py
sed -n '1,360p' scripts/predict_stage2_backend.py
sed -n '1,360p' scripts/evaluate_stage2_backend.py
sed -n '1,360p' src/exchange_ml/stage2_backend.py
sed -n '1,320p' tests/test_stage2_backend.py
sed -n '1,180p' baselines/pengwin2026-task1-abbc/requirements.txt
find outputs/stage2_backend -maxdepth 4 -type f -print
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_stage2_backend.py' -v
python -B scripts/prepare_stage2_backend.py --output-root /tmp/exchange-stage2-smoke.eG3leO/nnUNet_raw --case-limit 1 --roles train validation
python -B scripts/prepare_stage2_backend.py --help
python -B scripts/stage2_backend.py --help
python -B scripts/predict_stage2_backend.py --help
python -B scripts/evaluate_stage2_backend.py --help
python -B scripts/stage2_backend.py print-runbook
python -m json.tool configs/stage2_backend.json
ruff check src/exchange_ml/stage2_backend.py scripts/prepare_stage2_backend.py scripts/stage2_backend.py scripts/predict_stage2_backend.py scripts/evaluate_stage2_backend.py tests/test_stage2_backend.py
git -C baselines/pengwin2026-task1-abbc rev-parse HEAD
git -c safe.directory=/mnt/sdb/shared/dang.cpm/Exchange-ML/baselines/pengwin2026-task1-abbc -C baselines/pengwin2026-task1-abbc rev-parse HEAD
python -c "from pathlib import Path; files=['src/exchange_ml/stage2_backend.py','scripts/prepare_stage2_backend.py','scripts/stage2_backend.py','scripts/predict_stage2_backend.py','scripts/evaluate_stage2_backend.py','tests/test_stage2_backend.py']; [compile(Path(f).read_text(), f, 'exec') for f in files]; print('stage2 syntax compile: PASS')"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_stage2_backend.py' -v
python -B scripts/stage2_backend.py print-runbook
namei -l /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy/dataset.json
stat -c 'mode=%A (%a) owner=%U:%G uid=%u gid=%g path=%n' outputs outputs/stage2_backend outputs/stage2_backend/nnUNet_raw outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy
getfacl -p outputs outputs/stage2_backend outputs/stage2_backend/nnUNet_raw outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy
find outputs/stage2_backend -maxdepth 3 -printf '%M %u:%g %p\n' | head -n 80
find outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy/imagesTr -maxdepth 1 -type f | wc -l
find outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy/labelsTr -maxdepth 1 -type f | wc -l
find outputs/stage2_backend/nnUNet_raw/Dataset702_PENGWINStage2Fragment/imagesTr -maxdepth 1 -type f | wc -l
find outputs/stage2_backend/nnUNet_raw/Dataset702_PENGWINStage2Fragment/labelsTr -maxdepth 1 -type f | wc -l
```

The representative smoke also ran read-only assertions over its temporary image/label geometry, label ranges, sample counts, split separation, and manifest. The two Git commands, the `ruff` command, and the three SimpleITK-dependent `--help` commands failed exactly as recorded under **Observed result**; they are not reported as successful checks.

## User-typed commands

**Retraining (2026-10-13).** Run by the user:

```bash
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29600 python scripts/stage2_backend.py train --stage anatomy --device cuda --num-gpus 2 --model-seed 31000 2>&1 | tee outputs/logs/c0_stage2_anatomy.log
```

Prepared, not yet run (GPU pair chosen at launch):

```bash
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=5,6 MASTER_PORT=29600 python scripts/stage2_backend.py train --stage fragment --device cuda --num-gpus 2 --model-seed 31000 2>&1 | tee outputs/logs/c0_stage2_fragment.log
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<one GPU> python scripts/predict_stage2_backend.py --resume
conda run -n exchange-stage2 python scripts/evaluate_stage2_backend.py
```

The original 2026-09-26 sequence follows for provenance.

Run the following canonical sequence from the repository root. These commands are prepared for the user and have not been executed by the agent:

```bash
conda create -n exchange-stage2 python=3.10 -y
conda run -n exchange-stage2 pip install -r baselines/pengwin2026-task1-abbc/requirements.txt
conda run -n exchange-stage2 python scripts/stage2_backend.py check
conda run -n exchange-stage2 python scripts/prepare_stage2_backend.py
nnUNet_raw=outputs/stage2_backend/nnUNet_raw nnUNet_preprocessed=outputs/stage2_backend/nnUNet_preprocessed nnUNet_results=outputs/stage2_backend/nnUNet_results MPLCONFIGDIR=/tmp/exchange-ml-matplotlib conda run -n exchange-stage2 nnUNetv2_plan_and_preprocess -d 701 702 -pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity
conda run -n exchange-stage2 python scripts/stage2_backend.py install-splits
CUDA_VISIBLE_DEVICES=5,6 conda run --no-capture-output -n exchange-stage2 python scripts/stage2_backend.py train --stage anatomy --device cuda --num-gpus 2
CUDA_VISIBLE_DEVICES=5,6 conda run --no-capture-output -n exchange-stage2 python scripts/stage2_backend.py train --stage fragment --device cuda --num-gpus 2
conda run -n exchange-stage2 python scripts/predict_stage2_backend.py --resume
conda run -n exchange-stage2 python scripts/evaluate_stage2_backend.py
```

The sequence creates a pinned `exchange-stage2` environment, builds the 40/40 datasets, runs nnU-Net planning/preprocessing, installs the frozen split, trains Anatomy then Fragment stages, performs strict validation inference, and computes metrics.

`install-splits` is user-typed because it writes `splits_final.json` into the preprocessed datasets. The agent must not execute it.

After the observed interrupted download, the prepared retry command is:

```bash
conda run -n exchange-stage2 pip install --timeout 120 --retries 10 --resume-retries 20 -r baselines/pengwin2026-task1-abbc/requirements.txt
```

This retries the same frozen dependencies; it does not substitute a different PyTorch build.

Before retrying conversion, the owner of the existing output tree must grant the experiment user access. The owner should run:

```bash
setfacl -R -m u:24chuong.ta:rwX /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage2_backend
find /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage2_backend -type d -exec setfacl -m d:u:24chuong.ta:rwx {} +
```

The experiment user should then verify and rerun the same frozen conversion without `--overwrite`:

```bash
test -w outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy/dataset.json && echo "Stage 2 metadata is writable" || echo "Stage 2 metadata is NOT writable"
conda run -n exchange-stage2 python scripts/prepare_stage2_backend.py
```

## Next step

Run Stage B (C-0.2), then validation inference and metrics, and assess gate G2
(stable, meaningful instance predictions; new frozen validation metrics CSV for
the H1 risks). Progress is tracked in
`__docs__/reports/stage3/stage3_fix_progress.md`.

Superseded 2026-10-12 text:

Execute Stage 3 exactly as printed by `python scripts/stage3_h1.py print-runbook`. Stage 3 remains scientifically blocked until all three independently seeded members are trained, all 40 validation cases have predictions and genuine Stage-B probabilities, and the retrospective H1 artifacts are reviewed against the manual two-of-three gateway.
