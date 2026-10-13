# Runbook — training Stage 3 ensemble member 01 (C-1)

You are training one model, `member_01` (seed 31001), for the Exchange-ML
project on `vishc-server-1`. Work only in your own clone. Read data only from
the shared copy, and never write to it. Do not run any other member or Stage 2.

Two runs, about 8–20 h each: **C-1.1 Anatomy**, a gate check, then **C-1.2
Fragment**. Each needs 2 GPUs with about 13 GB free per GPU.

## 0. Before you start (repo owner)

- Branch `fix` is pushed to `git@github.com:jimmytrn154/Exchange-ML.git`, and
  you have read access to that repository.
- Your account may use the GPUs you will pick, under the lab's rules.

## 1. Clone the project into a shared, readable folder

```bash
mkdir -p /mnt/sdb/shared/<your_user>
cd /mnt/sdb/shared/<your_user>
git clone -b fix git@github.com:jimmytrn154/Exchange-ML.git
cd Exchange-ML
chmod -R a+rX .   # lets the repo owner read your results
```

All later commands run from this `Exchange-ML` folder.

## 2. Install the patched ABBC backend

`baselines/pengwin2026-task1-abbc` is an empty folder in the clone. Fill it with
upstream commit `ccc625f`, then apply the project patch:

```bash
rmdir baselines/pengwin2026-task1-abbc
git clone https://github.com/RURUGURU/pengwin2026-task1-abbc.git baselines/pengwin2026-task1-abbc
git -C baselines/pengwin2026-task1-abbc checkout ccc625fd000ec2dfce622b4621bf2d0b4ee12796
git -C baselines/pengwin2026-task1-abbc apply ../../patches/pengwin2026-task1-abbc_on_ccc625f.patch
sha256sum baselines/pengwin2026-task1-abbc/code_task1/core.py baselines/pengwin2026-task1-abbc/code_task1/model.py baselines/pengwin2026-task1-abbc/inference/inference.py
```

The three hashes must be exactly:

```text
b7fe9b5d737856635e33d34c7164c8a210362fc23eba24d19c5967236c4e48a3  core.py
ed79fce233aa6d8fe16696693c6a6ee9e720c947cb51da4fa5f8ce56a8ec569d  model.py
7483138ca7953264f578e6bbf7e7f0b0f246ad22067688018b163a5266dbe615  inference.py
```

If the checkout, the patch, or a hash fails, stop and tell the repo owner.

## 3. Create the exact Python environment

```bash
conda create -n exchange-stage2 python=3.10 -y
conda run -n exchange-stage2 pip install --timeout 120 --retries 10 --extra-index-url https://download.pytorch.org/whl/cu118 -r environment/exchange-stage2.freeze.txt
conda run -n exchange-stage2 pip freeze | diff - environment/exchange-stage2.freeze.txt && echo "environment matches"
```

The `diff` must print nothing except `environment matches`. The PyTorch wheel
is large (~2.3 GB); if the download is interrupted, rerun the same `pip
install` line.

## 4. Point the data folders at the shared copy (read-only)

The clone holds only small metadata for the training data. Replace the two
data folders with links to the shared copy:

```bash
SHARED=/mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage2_backend
mv outputs/stage2_backend/nnUNet_raw outputs/stage2_backend/nnUNet_raw.git
mv outputs/stage2_backend/nnUNet_preprocessed outputs/stage2_backend/nnUNet_preprocessed.git
ln -s $SHARED/nnUNet_raw outputs/stage2_backend/nnUNet_raw
ln -s $SHARED/nnUNet_preprocessed outputs/stage2_backend/nnUNet_preprocessed
mkdir -p outputs/logs
```

Do not commit these changes. Git will show the two folders as changed; that is
expected.

## 5. Preflight checks (a minute, no GPU)

```bash
conda run -n exchange-stage2 python scripts/stage2_backend.py check
ls outputs/stage2_backend/nnUNet_raw/Dataset701_PENGWINStage2Anatomy/labelsTr | wc -l
ls outputs/stage2_backend/nnUNet_preprocessed/Dataset701_PENGWINStage2Anatomy/nnUNetPlans_3d_fullres/*.npy | wc -l
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest discover -s tests
```

Expected: `nnunetv2 observed=2.5.1 required=2.5.1` and `ABBC trainer import:
PASS`; `80` labels; `160` unpacked arrays; `OK` for the tests. Stop and report
anything else. Do not use `scripts/stage3_h1.py check` here: it checks the
Stage 2 scoring inputs, which are being regenerated, not training readiness.

## 6. C-1.1 — Anatomy training

Pick 2 GPUs with about 13 GB free each:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ss -ltn '( sport = :29601 )'   # must print nothing
```

Port `29601` is reserved for this run; ports `29600` and `29602` are used by
other runs of this project. Then, with your GPU numbers:

```bash
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<A>,<B> MASTER_PORT=29601 python scripts/stage3_h1.py train --stage anatomy --member member_01 --device cuda --num-gpus 2 2>&1 | tee outputs/logs/c1_member_01_anatomy.log
```

Early in the log you should see `Anatomy marginal case map: 80 cases`,
`[marginal] case→labeled 맵 로드: 80 cases`, and `disable axis-2 mirror ...
(0, 1, 2) → (0, 1)`. If any is missing, stop the run and report it.

The run ends with `Training done`, then a 40-case validation, then `Mean
Validation Dice`. That number is about 0.5 by design and is not the check.

## 7. Gate G1 — must pass before C-1.2

```bash
python3 scripts/check_anatomy_gate.py outputs/stage3_h1/members/member_01/nnUNet_results/Dataset701_PENGWINStage2Anatomy/PengwinTrainerSTUNetBaseAnatomyV301__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation/summary.json
```

Send the output to the repo owner. Continue only on `G1: PASS` (both hips
median Dice ≥ 0.90, at most 2 of 20 cases below 0.5). On `G1: FAIL`, stop.

## 8. C-1.2 — Fragment training (only after G1 PASS)

```bash
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<A>,<B> MASTER_PORT=29601 python scripts/stage3_h1.py train --stage fragment --member member_01 --device cuda --num-gpus 2 2>&1 | tee outputs/logs/c1_member_01_fragment.log
```

Expected: `[STU-Net warm-start] ... loaded=120 skipped_seg=10` near the start,
one `[instance]` line per epoch, then `Early stopping ...`, `Training done`, and
`perform_actual_validation SKIPPED`. The process should exit within seconds of
`Training done`. A hang of many minutes after an `Early stopping` line, or an
NCCL timeout, is the bug these fixes target: report it.

## 9. Hand over

```bash
chmod -R a+rX outputs/stage3_h1/members/member_01 outputs/logs
```

Tell the repo owner the full path of your `Exchange-ML` folder. They copy
`outputs/stage3_h1/members/member_01` and your two logs into their own
repository and run inference themselves.

## Rules

- Run only `member_01`, only from your own clone, only on port `29601`.
- Never write to `/mnt/sdb/shared/dang.cpm/...` or to the repo owner's folders.
- Never run two jobs on the same GPU, and do not restart a failed run without
  reporting it first: a fresh start overwrites the member's results.
- Do not edit configs, seeds, or code. If something fails, report the exact
  error and log; do not work around it.
