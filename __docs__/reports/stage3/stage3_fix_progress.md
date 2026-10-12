# Stage 3 — Backend fix progress tracker

Companion to `__docs__/reports/stage3/stage3_h1.md`. That report holds the
diagnosis and the H1 results; this file tracks fixing the backend defects that
confound H1. Only the backend is being fixed. The FA-IPD method, scores, split,
seeds, and evaluation protocol are unchanged.

Opened: 2026-10-12. Last updated: 2026-10-12.

Overall status: **IN PROGRESS — F1–F3 and F6 (D3) done and unit-tested; F4
smoke and all re-runs wait for free GPUs; D4 (evaluation scope) to be decided
before gate G2.**

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
| F4 | Short end-to-end training smoke | READY TO RUN (commands B1/B2 prepared; waiting for 2 free GPUs) | Few-iteration 2-GPU run (e.g. `PENGWIN_TRAIN_ITERS`/`PENGWIN_VAL_ITERS`) to confirm the mirror log line and that both ranks finish together. | **User-typed** (uses GPUs); command to be prepared after F1–F3. |
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

## Re-run plan (all user-typed)

Stage 2 and the three Stage 3 members are independent and can run in parallel
on separate GPU pairs. Within each row, Fragment warm-starts from that row's
Anatomy checkpoint, so it must be retrained after Anatomy.

| Run | Anatomy | Gate G1 | Fragment | Inference | Status |
|---|---|---|---|---|---|
| Stage 2 single model | — | — | — | — | NOT STARTED |
| `member_01` (seed 31001) | — | — | — | — | NOT STARTED |
| `member_02` (seed 31002) | — | — | — | — | NOT STARTED |
| `member_03` (seed 31003) | — | — | — | — | NOT STARTED |

Then, in order: Stage 2 re-evaluation → new frozen Stage 2 risk CSV →
`score_stage3_h1.py` → `evaluate_stage3_h1.py` → manual H1 gateway.

### Gates

- **G1 — Anatomy laterality check (after each Anatomy run, before its
  Fragment run).** Re-run `outputs/stage3_h1/diagnostics/bone_recall_by_model.py`
  on that run's validation output. Pass if both hips are found and correctly
  labelled at levels comparable to the sacrum. If it fails, stop and
  re-diagnose; do not start that Fragment run.
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

## Open questions for the user

- **D4 (before G2):** keep the frozen whole-volume evaluation and record the
  femur-scan false positives as a limitation, or evaluate each case only on its
  labelled anatomy (changes the frozen Stage 1/2 evaluation contract)?
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
