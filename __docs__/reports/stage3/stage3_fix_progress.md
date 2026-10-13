# Stage 3 — Backend fix progress tracker

Companion to `__docs__/reports/stage3/stage3_h1.md`. That report holds the
diagnosis and the H1 results; this file tracks fixing the backend defects that
confound H1. Only the backend is being fixed. The FA-IPD method, scores, split,
seeds, and evaluation protocol are unchanged.

Opened: 2026-10-12. Last updated: 2026-10-12.

Overall status: **IN PROGRESS — F1–F3 and F6 done and unit-tested; F4 smoke
passed on 2 GPUs (2026-10-13); D4 decided. Next: commit, archive (A), and
prepare the full retraining commands.**

Status values: `NOT STARTED` · `IN PROGRESS` · `DONE` · `BLOCKED` · `FAILED`.
A fix is `DONE` only when its lightweight test passes; a re-run is `DONE` only
when its outputs exist and pass the listed check.

## Defects being fixed

| ID | Defect | Evidence (see `stage3_h1.md`, Post-hoc diagnosis) |
|---|---|---|
| D1 | Stage A Anatomy training mirrors the L/R axis (axis 2) for `Dataset701_PENGWINStage2Anatomy`; LeftHip/RightHip labels are not swapped, so laterality cannot be learned. Whole hips are missed or mislabelled. | Observed fact 2 (per-bone coverage table) and 3 (`DISABLE_X_MIRROR_DATASETS` holds only `Dataset539`; `debug.json` mirror axes `(0, 1, 2)`). |
| D2 | Fragment validation F1, best-checkpoint selection, and early stopping are computed per DDP rank. Ranks can stop at different epochs. | Observed fact 4; `member_02` NCCL timeout; its inference used the epoch-61 crashed-run checkpoint. |
| D3 | **Found 2026-10-12; fixed by F6.** The Anatomy trainer's partial-label (marginal) loss needs `code_task1/result/reports/ds539_case_labeled_classes.json` (pelvic cases → Sacrum/LeftHip/RightHip, femur cases → Femur). The file does not exist in either workspace. Every Stage 2 and Stage 3 Anatomy run logged `[marginal] 맵 로드 실패 ... → 전부 labeled 폴백` and trained as if every class were labelled in every case, so bones that are present but unlabelled in a case are taught as background. | 8 Anatomy training logs (2026-09-22 to 2026-10-02) show the load failure; `find` over `/home/24chuong.ta` and `/mnt/sdb/shared/dang.cpm` found no copy. `Dataset701` labels: 40 cases `{0,1,2,3}` (pelvic) and 40 cases `{0,4}` (femur). In 12/20 femur validation cases the Stage 2 model predicts sacrum/hip (130k–707k voxels) where GT is background, and the CT there is bone-like (median 92–373 HU; 25–82% of voxels > 200 HU). So pelvic bone is visible but unlabelled in femur scans and was trained as background. Direct effect on hip coverage not measured. |
| D4 | **Found 2026-10-12, evaluation scope, not a training defect.** `scripts/evaluate_stage2_backend.py` compares the whole prediction to the whole label. In femur scans, predicted visible pelvic bone counts as false positives in Dice and Instance F1, and the H1 risks inherit this. | Same 12/20 femur cases as D3. Whether the official PENGWIN evaluator restricts by anatomy is not verified. |

## Fix items

| ID | Item | Status | Change | Lightweight verification (agent-typed) |
|---|---|---|---|---|
| F1 | Disable L/R mirror for Anatomy | DONE (2026-10-12) | Add `Dataset701_PENGWINStage2Anatomy` to `DISABLE_X_MIRROR_DATASETS` in `baselines/pengwin2026-task1-abbc/code_task1/core.py`. Fragment (`Dataset702`) mirroring stays as is: its labels are anatomy-local fragment IDs with no laterality. | Unit test: the trainer mirror config for Dataset701 excludes axis 2 for training and inference; Dataset702 keeps `(0, 1, 2)`. |
| F2 | Global DDP validation for Fragment | DONE (2026-10-12) | Gather validation statistics across ranks before computing F1/EMA; make the best-checkpoint and early-stop decisions once and share them with all ranks. | Unit test with simulated rank statistics: all ranks get identical F1, best flag, and stop flag. |
| F3 | Check Anatomy early-stop path | DONE (2026-10-12): no change needed | `PengwinTrainer.on_validation_epoch_end` calls the stock nnU-Net method first, which all-gathers `tp/fp/fn` and validation losses across ranks before logging `mean_fg_dice`; the early stop then reads the shared `ema_fg_dice`. Anatomy decisions are already global. | Inspected installed `nnunetv2` source (`nnUNetTrainer.on_validation_epoch_end`) and `core.py` `PengwinTrainer.on_validation_epoch_end`. |
| F6 | Anatomy partial-label case map (D3) | DONE (2026-10-12) | `exchange_ml.stage2_backend.case_labeled_class_map` records the anatomy classes present in each case's GT target (key parsed exactly as the trainer parses `batch['keys']`). `scripts/stage2_backend.py` `write_case_labeled_map` builds it from the run's `Dataset701` `labelsTr`, checks it covers every train/validation sample of the frozen split, writes `<results_root>/Dataset701_PENGWINStage2Anatomy/case_labeled_classes.json`, and sets `PENGWIN_CASE_LABELED_MAP` before DDP spawn (Anatomy stage only). `core.py` `_load_case_labeled_map`: an explicitly set map must load and be non-empty; the silent fallback remains only when the variable is unset. | 5 unit tests: per-case classes; invalid targets rejected; trainer mask for pelvic `[T,T,T,T,F]` and femur `[T,F,F,F,T]`; missing/empty explicit map raises; wrapper writes the map and rejects a split sample it does not cover. Without a map the femur mask is all `True` (old behaviour). Dry run on the real labels (output to scratch): 80 cases, 40 → `[1,2,3]`, 40 → `[4]`. Batches carry `keys` (`nnUNetDataLoader3D.generate_train_batch` returns them; `PengwinTrainer.get_dataloaders` defers to it). |
| F4 | Short end-to-end training smoke | DONE (B1 and B2 PASS, 2026-10-13) | Few-iteration 2-GPU run (e.g. `PENGWIN_TRAIN_ITERS`/`PENGWIN_VAL_ITERS`) to confirm the mirror log line and that both ranks finish together. | **User-typed** (uses GPUs); command to be prepared after F1–F3. |
| F5 | Docs and protocol record | IN PROGRESS (Stage 2 reopened in its report and `plan.md`; retraining results pending) | Record the mirror change as a backend correction in the Stage 2 and Stage 3 reports; reopen the Stage 2 status (user decision). | Report diff review. |

## Prepared user-typed commands

Run from the repository root. None of these has been run by the agent.

### A. Archive the pre-fix artifacts (before the full runs; moves only)

All paths are on the same filesystem, so `mv` is instant and nothing is
deleted. Shared raw/preprocessed data, the split manifest, and
`outputs/stage3_h1/diagnostics/` stay in place.

```bash
ARCH=outputs/archive/2026-10-12_pre_backend_fix
mkdir -p $ARCH/stage2_backend $ARCH/stage3_h1
mv outputs/stage2_backend/nnUNet_results outputs/stage2_backend/validation_predictions outputs/stage2_backend/validation_metrics.csv outputs/stage2_backend/validation_summary.json outputs/stage2_backend/validation_inference_manifest.json $ARCH/stage2_backend/
mv outputs/stage3_h1/members outputs/stage3_h1/h1_summary.json outputs/stage3_h1/h1_comparison.csv outputs/stage3_h1/risk_coverage.csv outputs/stage3_h1/risk_coverage.png outputs/stage3_h1/stratified_analysis.csv outputs/stage3_h1/uncertainty_scores.csv outputs/stage3_h1/uncertainty_scoring_manifest.json $ARCH/stage3_h1/
ls $ARCH/stage2_backend $ARCH/stage3_h1
```

Some moved CSV/JSON files are tracked by Git; stage the moves with
`git add -A outputs/` when committing. The copy under
`/mnt/sdb/shared/dang.cpm/Exchange-ML` is not touched.

### B. F4 smoke run (2 free GPUs, about 30–60 min)

Uses `outputs/smoke_f4/stage2_smoke_config.json` (agent-written): shared
raw/preprocessed data and split manifest, results only under
`outputs/smoke_f4/`, early stop after min 2 epochs / patience 1 / delta 1.0,
Fragment cap 4 epochs. `PENGWIN_TRAIN_ITERS`/`PENGWIN_VAL_ITERS` shorten each
epoch. nnU-Net's full 40-case validation after Anatomy training cannot be
skipped and takes most of the time. Replace `<A>,<B>` with two free GPUs. Run
B2 only if B1 ends with `Training done` and exit code 0.

```bash
# B1 — Anatomy smoke (checks F1 mirror, F6 case map, Anatomy DDP early stop)
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<A>,<B> MASTER_PORT=29611 PENGWIN_TRAIN_ITERS=5 PENGWIN_VAL_ITERS=2 python scripts/stage2_backend.py --config outputs/smoke_f4/stage2_smoke_config.json train --stage anatomy --device cuda --num-gpus 2 --model-seed 31001 2>&1 | tee outputs/smoke_f4/b1_anatomy.log

# B2 — Fragment smoke (checks F2 global validation/early stop and the trusted warm-start)
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<A>,<B> MASTER_PORT=29611 PENGWIN_TRAIN_ITERS=5 PENGWIN_VAL_ITERS=2 python scripts/stage2_backend.py --config outputs/smoke_f4/stage2_smoke_config.json train --stage fragment --device cuda --num-gpus 2 --model-seed 31001 2>&1 | tee outputs/smoke_f4/b2_fragment.log
```

Pass criteria, checked by the agent from the logs afterwards:

- B1: `disable axis-2 mirror` for `Dataset701_PENGWINStage2Anatomy`;
  `[marginal] case→labeled 맵 로드: 80 cases`; `debug.json` shows
  `inference_allowed_mirroring_axes (0, 1)`; early stop and `Training done`;
  no NCCL timeout.
- B2: warm-start loads the smoke Anatomy checkpoint; `[instance]` lines each
  epoch; early stop or the 4-epoch cap, then `Training done` and exit 0 with
  no NCCL `ALLREDUCE` timeout (the original D2 symptom).

### C. Full retraining (after commit and archive A)

Four independent runs, one GPU pair each. Replace `<GPU pair>` with two free
GPUs (e.g. `0,1`); keep one port per run. With fewer than four pairs, start as
many runs as there are pairs and start the rest as pairs free up. Never put two
runs on the same GPU. Each Anatomy run ends with nnU-Net's 40-case validation;
**stop after C-x.1 and wait for the agent's G1 result before C-x.2.**

| Run | Seed | Port | Anatomy (C-x.1) | Fragment (C-x.2) |
|---|---:|---:|---|---|
| C-0 Stage 2 | 31000 | 29600 | `python scripts/stage2_backend.py train --stage anatomy --device cuda --num-gpus 2 --model-seed 31000` | same with `--stage fragment` |
| C-1 `member_01` | 31001 (config) | 29601 | `python scripts/stage3_h1.py train --stage anatomy --member member_01 --device cuda --num-gpus 2` | same with `--stage fragment` |
| C-2 `member_02` | 31002 (config) | 29602 | `... --member member_02 ...` | same with `--stage fragment` |
| C-3 `member_03` | 31003 (config) | 29603 | `... --member member_03 ...` | same with `--stage fragment` |

Full form of each command (example C-0.1; the others differ only in the
script arguments, port, and log name in the table):

```bash
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<GPU pair> MASTER_PORT=29600 python scripts/stage2_backend.py train --stage anatomy --device cuda --num-gpus 2 --model-seed 31000 2>&1 | tee outputs/logs/c0_stage2_anatomy.log
```

Create the log folder once: `mkdir -p outputs/logs`. Log names:
`c0_stage2_{anatomy,fragment}.log`, `c1_member_01_{anatomy,fragment}.log`,
`c2_member_02_...`, `c3_member_03_...`. `PENGWIN_ALLOW_UNSAFE_TORCH_LOAD` is
no longer needed: both wrappers trust only the run's own Anatomy checkpoint.

**Gate G1 (agent-typed, after each C-x.1):**

```bash
python3 scripts/check_anatomy_gate.py <run results>/Dataset701_PENGWINStage2Anatomy/PengwinTrainerSTUNetBaseAnatomyV301__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation/summary.json
```

Scores each class only on cases whose GT labels it. Pass rule (**approved
2026-10-13**): LeftHip and RightHip median Dice ≥ 0.90 and at
most 2/20 labelled cases below 0.5. Pre-fix reference with this script: Stage 2
LeftHip 0.613 (1 below 0.5), RightHip 0.371 (15); `member_03` 0.547 (8) and
0.506 (10); all pre-fix runs have Sacrum ≈ 0.973 and Femur ≈ 0.98 → **FAIL**.
On G1 FAIL: do not start that Fragment run; stop and re-diagnose.

**After both stages of a run (single GPU from its pair):**

```bash
# Stage 2 (C-0): inference, then metrics → new frozen H1 risk CSV (gate G2)
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<one GPU> python scripts/predict_stage2_backend.py --resume
conda run -n exchange-stage2 python scripts/evaluate_stage2_backend.py
# Members (C-1..C-3)
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=<one GPU> python scripts/predict_stage3_member.py --member member_0X --resume
```

**After Stage 2 metrics and all three member inferences (sequential):**

```bash
conda run -n exchange-stage2 python scripts/score_stage3_h1.py
conda run -n exchange-stage2 python scripts/evaluate_stage3_h1.py
```

## Re-run plan (all user-typed)

Stage 2 and the three Stage 3 members are independent and can run in parallel
on separate GPU pairs. Within each row, Fragment warm-starts from that row's
Anatomy checkpoint, so it must be retrained after Anatomy.

| Run | Anatomy | Gate G1 | Fragment | Inference | Status |
|---|---|---|---|---|---|
| Stage 2 single model (seed 31000) | DONE (2026-10-13 07:36–15:25; early stop epoch 109, best 84) | **PASS** (hips 0.988 / 0.986, 0 cases < 0.5) | READY (C-0.2) | — | IN PROGRESS |
| `member_01` (seed 31001) | — | — | — | — | NOT STARTED |
| `member_02` (seed 31002) | — | — | — | — | NOT STARTED |
| `member_03` (seed 31003) | — | — | — | — | NOT STARTED |

Then, in order: Stage 2 re-evaluation → new frozen Stage 2 risk CSV →
`score_stage3_h1.py` → `evaluate_stage3_h1.py` → manual H1 gateway.

### Gates

- **G1 — Anatomy hip check (after each Anatomy run, before its Fragment
  run).** Rule status: **approved by the user on 2026-10-13** (proposed by the
  agent).
  - *Purpose:* confirm that the fixed Stage A model finds and correctly labels
    both hips before spending 9–21 h on its Fragment run. The D1/D3 defects
    showed up as lost or swapped hips while Sacrum and Femur stayed good.
  - *Input:* the run's own end-of-training validation,
    `fold_0/validation/summary.json` (nnU-Net predicts all 40 validation
    cases after every Anatomy run, so no extra inference is needed).
  - *Measure:* per-bone Dice, counted only on cases whose GT labels that bone
    (`n_ref > 0`): Sacrum and both hips on the 20 pelvic scans, Femur on the 20
    femur scans. nnU-Net's own class means also count visible-but-unlabelled
    bones in the other scan type as Dice 0, which roughly halves them (old
    Sacrum mean 0.49 vs 0.973 on labelled cases), so they are not used.
  - *Rule:* PASS if **LeftHip and RightHip** each have **median Dice ≥ 0.90**
    and **at most 2 of 20 labelled cases with Dice < 0.5**. Sacrum and Femur
    are reported but not gated.
  - *Why these numbers:* unaffected bones already reach 0.97–0.98, so a fixed
    model should bring the hips close to that; 0.90 leaves room for fractured
    hips being harder. The case count catches a whole hip being dropped or
    swapped in a few cases even when the median looks fine.
  - *Pre-fix reference (same script):* Stage 2 LeftHip 0.613 (1 case < 0.5),
    RightHip 0.371 (15); `member_03` 0.547 (8), 0.506 (10) → FAIL.
  - *On FAIL:* do not start that Fragment run; stop and re-diagnose. The
    thresholds are fixed before any new result is seen and are not adjusted
    afterwards to make a run pass.
  - *Tool:* `python3 scripts/check_anatomy_gate.py <summary.json>` (read-only,
    agent-typed; exit code 0 = PASS, 1 = FAIL).
- **G2 — Stage 2 re-evaluation.** The backend produces stable instance
  predictions (plan Stage 2 gateway) and a new frozen validation metrics CSV.
- **G3 — H1 gateway.** Unchanged from `__docs__/plan.md` Stage 3.

## Timing estimate (from 2026-09-29 to 2026-10-03 logs)

- Anatomy: about 13–20 h per run. Fragment: about 9–21 h. Inference: about 1 h per member.
- With four free GPU pairs, all trainings can run in parallel: Stage 3 results
  about 3 days after launch at best.
- Internal freeze is 2026-10-23. Stages 4–5 need many further retrains at this
  cost; the schedule impact is a user decision.

## Decision log

| Date | Decision | By |
|---|---|---|
| 2026-10-09 | Diagnosis recorded in `stage3_h1.md`; H1 FAIL of 2026-10-05 marked as confounded and Stage 3 set to BLOCKED. | Agent, at user request |
| 2026-10-12 | Fix plan (F1, F2, retraining) discussed; user asked for this tracker before any fix. | User |
| 2026-10-12 | F1–F3 approved; Stage 2 reopened. | User |
| 2026-10-12 | D3 fix (F6) approved before F4. | User |
| 2026-10-12 | Proposed: decide D4 before Stage 2 re-evaluation (G2); not yet confirmed. | Agent |
| 2026-10-13 | G1 pass rule approved: LeftHip and RightHip median Dice ≥ 0.90 and ≤ 2/20 labelled cases < 0.5. | User |
| 2026-10-13 | One GPU pair (`0,1`) available: launch C-0 (Stage 2) first, because its metrics are the H1 risk labels. | User GPUs; order proposed by agent |
| 2026-10-13 | Stage 2 retrain uses explicit model seed `31000` (distinct from members 31001–31003; old Stage 2 seed was unrecorded). | User (accepted agent proposal) |
| 2026-10-12 | D4: keep the frozen whole-volume evaluation; record femur-scan false positives as a limitation; report a labelled-anatomy-only view as a secondary analysis only. | User (accepted agent recommendation) |

## Open questions for the user

- **Commit the remaining files:** `patches/pengwin2026-task1-abbc_on_ccc625f.patch`,
  `scripts/check_anatomy_gate.py`, and this tracker.
- **GPU pairs:** only `0,1` is free (2026-10-13). Runs therefore go one at a
  time; more pairs would shorten the schedule by days.
- **Before launch:** archive the old, defective checkpoints so no run can pick
  them up. `stage3_h1.py check` still reports Anatomy + Fragment checkpoints
  for `member_02` and `member_03`, and the Stage 2 results tree is also old.
- **GPUs:** the 2026-10-12 `nvitop` screenshot shows no free pair. GPUs 1–6
  each hold ~18.5 GB under another user's job; GPU 7 is at 100%; GPU 0 has
  three smaller jobs. Four 2-GPU runs need coordination with other users.
- Observation, no action proposed: the Stage 2 report states Stage A trains
  from scratch (no trusted STU-Net checkpoint), while the reference trainer's
  early-stop settings (min 30 epochs, patience 25) were tuned for a warm start.

## Progress log

Append one entry per working session: date, item IDs, what changed, the exact
agent-typed commands, and any user-typed commands prepared.

### 2026-10-12

- Moved `__docs__/reports/stage3_h1.md` to `__docs__/reports/stage3/stage3_h1.md`
  (`git mv`); updated the two path references in `__docs__/plan.md`.
- Created this tracker. No code changed.
- Agent-typed: `mkdir -p __docs__/reports/stage3`; `git mv __docs__/reports/stage3_h1.md __docs__/reports/stage3/stage3_h1.md`; `sed -i` path update in `__docs__/plan.md`.
- User-typed commands prepared: None.

### 2026-10-12 (fixes)

- **F1:** added `Dataset701_PENGWINStage2Anatomy` to
  `PengwinTrainer.DISABLE_X_MIRROR_DATASETS` in
  `baselines/pengwin2026-task1-abbc/code_task1/core.py`.
- **F2:** added `_merge_rank_val_outputs` and an `all_gather_object` of the
  collated validation outputs at the start of
  `PengwinTrainerSTUNetBaseABBCPhase1V302.on_validation_epoch_end` (inherited
  by `PengwinTrainerSTUNetBaseAffinityV308`). Metric formulas are unchanged;
  they now average over all ranks' validation steps.
- **F3:** inspection only; no change (see table).
- Tests: new `tests/test_stage3_backend_fixes.py` (5 tests): Dataset701 mirror
  axes `(0, 1)` for training and inference; Dataset702 keeps `(0, 1, 2)`;
  Dataset539 entry kept; merge order; two simulated ranks with different local
  metrics reach the same F1, `num_epochs`, and stop flag. With each fix
  reverted in memory, the corresponding test fails (1 failure each).
- Full suite: 31/31 passed (26 previous + 5 new).
- Docs: Stage 2 reopened in `__docs__/reports/stage2_backend.md` and
  `__docs__/plan.md`; Stage 3 status line in `plan.md` changed from FAIL to
  BLOCKED to match `stage3_h1.md`.
- Found D3 (missing partial-label case map) while reading the Anatomy trainer.
- Agent-typed commands:

```bash
grep -n "def on_validation_epoch_end|Early stopping|all_gather|class .*Trainer.*(|patience" baselines/pengwin2026-task1-abbc/code_task1/core.py
conda run -n exchange-stage2 python -c "import inspect; from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as T; print(inspect.getsource(T.on_validation_epoch_end))"
grep -h "\[marginal\]|disable axis-2|mirror" <Stage 2 and member Dataset701 training_log_*.txt>
timeout 60 find /home/24chuong.ta /mnt/sdb/shared/dang.cpm -name 'ds539_case_labeled_classes.json'
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest tests.test_stage3_backend_fixes -v
conda run --no-capture-output -n exchange-stage2 python -B <scratch negative check: revert each fix in memory, rerun its test>
python3 -m py_compile baselines/pengwin2026-task1-abbc/code_task1/core.py tests/test_stage3_backend_fixes.py
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest discover -s tests -v
```

- User-typed commands prepared: None yet (F4 smoke waits for D3 decision and free GPUs).

### 2026-10-12 (D3 analysis and fix F6)

- D3 evidence: `Dataset701` label sets (40 pelvic `{0,1,2,3}`, 40 femur
  `{0,4}`); Stage 2 predicts pelvic bone in 12/20 femur validation scans at
  bone-like CT intensity. Recorded D4 (evaluation scope) from the same check.
- F6 implemented and tested as in the table. Full suite: 36/36 passed (31 + 5).
  `scripts/stage3_h1.py check`: PASS.
- Agent-typed commands:

```bash
grep -n "class MarginalDiceCELoss" -A45 baselines/pengwin2026-task1-abbc/code_task1/*.py
conda run --no-capture-output -n exchange-stage2 python <scratch d3_check.py: Dataset701 label sets; Stage 2 cross-anatomy predictions>
conda run --no-capture-output -n exchange-stage2 python <scratch d3_hu.py: CT HU where Stage 2 predicts pelvis in femur scans>
conda run -n exchange-stage2 python -c "<print nnUNetDataLoader3D.generate_train_batch source tail>"
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest tests.test_stage3_backend_fixes -v
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B <scratch d3_dry.py: no-map mask; real-label map written to scratch>
python3 -m py_compile baselines/pengwin2026-task1-abbc/code_task1/core.py scripts/stage2_backend.py src/exchange_ml/stage2_backend.py tests/test_stage3_backend_fixes.py
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest discover -s tests -v
conda run -n exchange-stage2 python scripts/stage3_h1.py check
```

- User-typed commands prepared: None (F4 waits for free GPUs).

### 2026-10-12 (wiring fixes, archive and F4 commands)

- Found while designing F4: (1) Stage 3 member configs pointed the F6 map
  builder at a non-existent per-member split manifest; added
  `paths.manifest_path` support to `stage2_paths` and set it to the shared
  manifest in `member_stage2_config`. (2) `stage2_backend.py` Fragment
  training did not trust its own Anatomy checkpoint for warm-start (only
  `stage3_h1.py` did), so a Stage 2 retrain would hit the NumPy-pickle refusal;
  it now sets `PENGWIN_TRUSTED_WARMSTART_CHECKPOINT` to that exact file.
- New test `Stage3MemberWiringTests`; full suite 37/37 passed.
- Wrote `outputs/smoke_f4/stage2_smoke_config.json`; `stage2_backend.py
  --config <smoke> check` passed.
- Prepared user-typed commands A (archive) and B (F4 smoke) above.
- D4: agent recommends keeping the frozen whole-volume evaluation, recorded
  as a limitation, with a labelled-anatomy-only view as a secondary analysis.
  Awaiting user decision (needed before G2).
- Agent-typed commands:

```bash
grep -n "_es_best_dice *=|PENGWIN_NUM_EPOCHS|PENGWIN_TRAIN_ITERS|PENGWIN_VAL_ITERS" baselines/pengwin2026-task1-abbc/code_task1/core.py
grep -n "PENGWIN_TRUSTED_WARMSTART|PENGWIN_ALLOW_UNSAFE" baselines/pengwin2026-task1-abbc/code_task1/*.py scripts/*.py
df -h outputs; du -sh outputs/stage2_backend/nnUNet_results outputs/stage2_backend/validation_predictions
python3 -m py_compile scripts/stage2_backend.py scripts/stage3_h1.py tests/test_stage3_backend_fixes.py
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest discover -s tests -v
mkdir -p outputs/smoke_f4 && python3 - <write smoke config from configs/stage2_backend.json>
conda run -n exchange-stage2 python scripts/stage2_backend.py --config outputs/smoke_f4/stage2_smoke_config.json check
```

- User-typed commands prepared: A (archive) and B1/B2 (F4 smoke).

### 2026-10-12 (D4 decision, GPU check for B1)

- D4 decided by the user (see decision log).
- User proposed GPUs 0 and 2 for B1. Read-only check: GPU 0 has 1.4 GB used
  (two small jobs) and is usable; GPU 2 has 18.5 GB used at 96% by another
  user's job (~6 GB free), so B1 was not started on it. GPUs 1–7 were all at
  16–18.7 GB. Port 29611 free.
- Agent-typed: `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader`; `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`; `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`; `ss -ltn '( sport = :29611 )'`.

### 2026-10-13 (F4 B1 result)

- User ran B1 on GPUs 0,1 (both idle at 2 MiB before launch). Log:
  `outputs/smoke_f4/b1_anatomy.log`.
- **B1: PASS.**
  - F6: wrapper wrote the map (80 cases); trainer logged
    `[marginal] case→labeled 맵 로드: 80 cases` (no fallback).
  - F1: `Dataset701_PENGWINStage2Anatomy → disable axis-2 mirror ...
    mirror_axes: (0, 1, 2) → (0, 1)`; `debug.json`
    `inference_allowed_mirroring_axes (0, 1)`.
  - DDP early stop: `Early stopping ... Stopping at epoch 3`, `Training done`;
    both ranks then ran the end-of-training validation (20 cases each,
    40 files + summary), `Validation complete`. No traceback, NCCL error, or
    timeout. `checkpoint_best.pth` and `checkpoint_final.pth` pass `unzip -t`.
    GPUs 0,1 released (2 MiB) afterwards.
  - Expected warnings only: cuDNN nvrtc workaround; DDP
    `find_unused_parameters` notice; nnU-Net `invalid value in scalar divide`
    when a class is absent from a validation batch (Femur pseudo-Dice `nan`
    at epoch 2), which `nanmean` ignores.
  - Smoke metrics (validation Dice 0.011 after 3 × 5 iterations) are not
    meaningful and are not results.
- User then ran B2 on GPUs 0,1. Log: `outputs/smoke_f4/b2_fragment.log`.
- **B2: PASS.**
  - Trusted warm-start: both ranks logged `[STU-Net warm-start] <smoke Anatomy
    checkpoint_best.pth>: loaded=120 skipped_seg=10` (no unsafe-load refusal;
    the 10 skipped keys are the reinitialized segmentation head, as specified).
  - F2: `[instance]` line each epoch; `Early stopping ... Stopping at epoch 2`;
    `Training done`; `perform_actual_validation SKIPPED` as designed. The run
    ended about 5 s after the stop with no NCCL `ALLREDUCE` timeout (the D2
    symptom was a 1,800 s hang on rank 0 followed by `SIGABRT`). GPUs 0,1
    released (2 MiB); no leftover `stage2_backend.py` or spawn processes.
  - Fragment mirroring unchanged: `debug.json`
    `inference_allowed_mirroring_axes (0, 1, 2)`.
  - Both checkpoints pass `unzip -t`. Warnings: cuDNN nvrtc workaround and
    DDP `find_unused_parameters` notice only.
  - Smoke artefact, not a defect: `Best EMA F1=-1.0000` is the trainer's
    initial sentinel, never replaced because the smoke `min_delta=1.0` makes
    improvement impossible; the frozen run uses `0.005`. Instance metrics
    (F1 0.0) after 2 × 5 iterations are not results.
  - Rank 1 writes no log lines, so its stop is evidenced by the clean exit and
    by the F2 unit test, not by a rank-1 log entry.
- Agent-typed (B2): `grep -nE "<criteria>" outputs/smoke_f4/b2_fragment.log`; `tail -8 outputs/smoke_f4/b2_fragment.log`; `ls -la <smoke Dataset702 fold_0>`; `unzip -tq` on both checkpoints; `grep -o inference_allowed_mirroring_axes <fold_0>/debug.json`; `grep -niE "traceback|error|nccl|timeout|warn"`; `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`; `ps -eo pid,etime,args | grep -E "stage2_backend.py|multiprocessing.spawn"`.
- Stage 2 seed `31000` agreed. Added `scripts/check_anatomy_gate.py` (G1,
  read-only) and prepared the full retraining commands (section C). Applying
  G1 to pre-fix runs: Stage 2 and `member_03` both FAIL (values in section C).
  nnU-Net's own `summary.json` class means (Sacrum 0.49–0.57) are halved by
  partial labels, so G1 scores only labelled cases.
- Agent-typed (planning): `python3 - <per-class labelled-case Dice from old Anatomy summary.json files>`; `python3 scripts/check_anatomy_gate.py <Stage 2 and member_03 pre-fix summary.json>`; `grep -n "add_argument" scripts/predict_stage2_backend.py scripts/predict_stage3_member.py scripts/evaluate_stage2_backend.py scripts/stage3_h1.py`.
- User approved G1 and reported committing the fixes (`dcc8832`, `94f9029`).
  Check found that those commits contain the wrapper, `src/`, tests, docs and
  smoke files, but **not** the ABBC trainer changes:
  `baselines/pengwin2026-task1-abbc` is a nested upstream repository (gitlink
  `ccc625f`, no `.gitmodules`), and its content changes remain uncommitted in
  its working tree. Content changes are limited to `code_task1/core.py`,
  `code_task1/model.py`, and `inference/inference.py`; the other files listed
  as modified there differ only in file mode (0644 → 0755).
- Saved those content changes as
  `patches/pengwin2026-task1-abbc_on_ccc625f.patch` (3 files; `git apply
  --check -R` confirms it matches the working tree). To reproduce: check out
  upstream `ccc625f`, then `git apply` the patch. SHA-256 of the code the full
  runs will use: `core.py` `b7fe9b5d…48a3`, `model.py` `ed79fce2…569d`,
  `inference.py` `7483138c…be615`, patch `66d61649…bb43`.
- Agent-typed: `git log --oneline -3`; `git status --short`; `git show --stat dcc8832 94f9029`; `git ls-files -s baselines/pengwin2026-task1-abbc`; `git -C baselines/pengwin2026-task1-abbc -c core.fileMode=false diff --stat`; `git -C baselines/pengwin2026-task1-abbc -c core.fileMode=false diff > patches/pengwin2026-task1-abbc_on_ccc625f.patch`; `git -C baselines/pengwin2026-task1-abbc apply --check -R ../../patches/pengwin2026-task1-abbc_on_ccc625f.patch`; `sha256sum <3 files + patch>`; `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`; `ss -ltn '( sport = :29600 )'`.
- User committed the patch, gate script and tracker (`e0eb76f`), ran archive A
  (verified: `outputs/archive/2026-10-12_pre_backend_fix/{stage2_backend,stage3_h1}`),
  and launched C-0.1 on GPUs 0,1 at 07:36 into an empty Stage 2 results tree
  (log `outputs/logs/c0_stage2_anatomy.log`).
- GPU check after launch: GPUs 3–6 each have ~22 GB free (another user's
  small jobs, ~45–53% util), so members can start in parallel (C-1 on 3,4;
  C-2 on 5,6) at the cost of shared compute; CPU load ~50 on 64 cores is the
  main contention risk. GPU 2 ~19 GB free (no free partner); GPU 7 busy.
- Agent-typed: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`; `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader` mapped to `ps -o user=,etime=,args=`; `nproc`; `uptime`; `free -g`; `tail -3 outputs/logs/c0_stage2_anatomy.log`; `ss -ltn '( sport = :29601 or sport = :29602 )'`; `ls outputs/archive/...`; `ls -la <new Stage 2 Anatomy fold_0>`; `git log --oneline -1`.
### 2026-10-13 (C-0.1 result and G1)

- C-0.1 (Stage 2 Anatomy, seed 31000, GPUs 0,1) finished: map loaded (80
  cases), axis-2 mirror disabled, `debug.json` axes `(0, 1)`; early stop at
  epoch 109 (best EMA pseudo-Dice 0.9721 at epoch 84); `Training done`; 40-case
  validation complete. No traceback, NCCL error, or timeout. Both checkpoints
  pass `unzip -t`. nnU-Net's `Mean Validation Dice 0.498` is the
  partial-label-confounded class mean and is not the gate.
- **G1: PASS.** Labelled-case median Dice (cases < 0.5): Sacrum 0.975 (0),
  LeftHip **0.988** (0), RightHip **0.986** (0), Femur 0.984 (0). Pre-fix
  Stage 2 with the same script: LeftHip 0.613 (1), RightHip 0.371 (15).
- Interpretation: the Stage A hip failure is resolved by the combined F1 + F6
  fixes. This run cannot separate the contribution of the mirror fix (D1) from
  the case-map fix (D3). Fragment, inference, and instance-level metrics are
  still pending.
- After the run, GPUs 0,1 carry another user's job (4.6 GB each); GPUs 2, 5, 6
  are idle; no member run has been launched yet.
- Agent-typed: `grep -nE "<criteria>" outputs/logs/c0_stage2_anatomy.log`; `ls -la <fold_0>`; `grep -c "Epoch time"`; `grep Yayy`; `grep -o inference_allowed_mirroring_axes <fold_0>/debug.json`; `ls <fold_0>/validation | wc -l`; `python3 scripts/check_anatomy_gate.py <fold_0>/validation/summary.json`; `unzip -tq` on both checkpoints; `nvidia-smi --query-compute-apps=...` + `ps -o user=,etime=,args=`; `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`; `ss -ltn '( sport = :29600 )'`.
- F5 docs: `__docs__/reports/stage2_backend.md` updated with the retraining
  section (fix table, C-0.1 result, G1 before/after, pending items, archive
  location, run and prepared commands); Stage 2 status line in `plan.md`
  updated to "retraining in progress; Stage A passed G1".
### 2026-10-13 (handover of member_01 to a second user)

- User plans to have a friend run C-1 (member_01) under the friend's account.
  Checked whether a fresh clone can do this without conflicts:
  - Branch `fix` is not on `origin` yet (local only).
  - A clone lacks: the ABBC code (`baselines/` are gitlinks with no
    `.gitmodules`), the dataset (gitignored, 134 GB), and the Stage 2 raw
    images/labels and preprocessed arrays (gitignored; 7.1 GB + 35 GB).
  - `/home/24chuong.ta` is mode 750, so another user cannot use this checkout,
    its data, or its conda env.
  - The shared copy `/mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage2_backend`
    is world-readable and matches this workspace: identical `labelsTr`
    hashes (80 and 160 files), identical preprocessed file names and sizes
    (already unpacked: 160 and 320 `.npy`), identical `splits_final.json`,
    plans, raw `dataset.json`, and split manifest.
  - `scripts/stage3_h1.py check` now fails here with `frozen Stage 2
    validation artifacts are missing`, as expected after archive A; it checks
    scoring inputs, not training readiness. `scripts/stage2_backend.py check`
    passes.
- Added `environment/exchange-stage2.freeze.txt` (110 exact package versions
  from this env; torch 2.1.2+cu118, nnunetv2 2.5.1, SimpleITK 2.3.1, numpy
  1.26.4; no local paths).
- Added `__docs__/reports/stage3/member_01_handover.md`: friend's own clone
  under `/mnt/sdb/shared/<user>`, ABBC upstream `ccc625f` + project patch with
  hash checks, env from the freeze file with a `pip freeze` diff, read-only
  symlinks to the shared data, preflight, C-1.1, G1, C-1.2, handover, rules.
  Port 29601 stays reserved for member_01.
- Agent-typed: `git remote -v`; `git status -sb`; `git fetch -q origin`; `git ls-files outputs/stage2_backend splits configs`; `git ls-files -s baselines`; `du -sh <raw, preprocessed, dataset>`; `stat -c '%A %U:%G %n' <home, repo, outputs, /mnt shared paths>`; `id -Gn`; `sha256sum` comparisons of splits, plans, manifest, raw `dataset.json`, and all `labelsTr` (here vs `/mnt`); `stat -c '%n %s'` inventory comparison of preprocessed `*.npz *.pkl`; `.npy` counts; `conda run -n exchange-stage2 pip freeze > environment/exchange-stage2.freeze.txt`; `grep` of the freeze file for local entries; `scripts/stage2_backend.py check`; `scripts/stage3_h1.py check` (failed as recorded); `grep` of tests for file dependencies.
- Agent-typed (B1): `grep -nE "<criteria>" outputs/smoke_f4/b1_anatomy.log`; `tail -15 outputs/smoke_f4/b1_anatomy.log`; `ls -la <smoke fold_0>`; `grep -o inference_allowed_mirroring_axes <smoke fold_0>/debug.json`; `ls <smoke fold_0>/validation | wc -l`; `grep -c "rank 0|rank 1"`; `grep -niE "traceback|error|nccl|timeout|warn"`; `unzip -tq` on both checkpoints; `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`.
