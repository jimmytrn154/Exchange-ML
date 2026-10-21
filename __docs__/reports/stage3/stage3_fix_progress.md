# Stage 3 — Backend fix progress tracker

Companion to `__docs__/reports/stage3/stage3_h1.md`. That report holds the
diagnosis and the H1 results; this file tracks fixing the backend defects that
confound H1. Only the backend is being fixed. The FA-IPD method, scores, split,
seeds, and evaluation protocol are unchanged.

Opened: 2026-10-12. Last updated: 2026-10-12.

Overall status: **IN PROGRESS — F1–F3, F6, F7 done and unit-tested; F4 smoke
passed; Stage 2 Anatomy and Fragment retrained (G1 PASS); Stage 2 inference
to rerun with F7; member_02 Anatomy running.**

Status values: `NOT STARTED` · `IN PROGRESS` · `DONE` · `BLOCKED` · `FAILED`.
A fix is `DONE` only when its lightweight test passes; a re-run is `DONE` only
when its outputs exist and pass the listed check.

## Defects being fixed

| ID | Defect | Evidence (see `stage3_h1.md`, Post-hoc diagnosis) |
|---|---|---|
| D1 | Stage A Anatomy training mirrors the L/R axis (axis 2) for `Dataset701_PENGWINStage2Anatomy`; LeftHip/RightHip labels are not swapped, so laterality cannot be learned. Whole hips are missed or mislabelled. | Observed fact 2 (per-bone coverage table) and 3 (`DISABLE_X_MIRROR_DATASETS` holds only `Dataset539`; `debug.json` mirror axes `(0, 1, 2)`). |
| D2 | Fragment validation F1, best-checkpoint selection, and early stopping are computed per DDP rank. Ranks can stop at different epochs. | Observed fact 4; `member_02` NCCL timeout; its inference used the epoch-61 crashed-run checkpoint. |
| D3 | **Found 2026-10-12; fixed by F6.** The Anatomy trainer's partial-label (marginal) loss needs `code_task1/result/reports/ds539_case_labeled_classes.json` (pelvic cases → Sacrum/LeftHip/RightHip, femur cases → Femur). The file does not exist in either workspace. Every Stage 2 and Stage 3 Anatomy run logged `[marginal] 맵 로드 실패 ... → 전부 labeled 폴백` and trained as if every class were labelled in every case, so bones that are present but unlabelled in a case are taught as background. | 8 Anatomy training logs (2026-09-22 to 2026-10-02) show the load failure; `find` over `/home/24chuong.ta` and `/mnt/sdb/shared/dang.cpm` found no copy. `Dataset701` labels: 40 cases `{0,1,2,3}` (pelvic) and 40 cases `{0,4}` (femur). In 12/20 femur validation cases the Stage 2 model predicts sacrum/hip (130k–707k voxels) where GT is background, and the CT there is bone-like (median 92–373 HU; 25–82% of voxels > 200 HU). So pelvic bone is visible but unlabelled in femur scans and was trained as background. Direct effect on hip coverage not measured. |
| D5 | **Found 2026-10-14; blocking since 2026-10-15 (case `030` all-zero on GPU); fixed by F7 on 2026-10-15.** The ABBC inference pipeline has a challenge-submission time budget (`inference/inference.py`: `V03_USE_TIME_BUDGET = True`, soft `TIME_BUDGET_SECONDS = 480`, hard `TIME_BUDGET_HARD_LIMIT = 540`). When exceeded, it logs `emit zero` and returns zeros for the current and remaining anatomies. `predict_stage2_backend.py` rejects only all-zero predictions, so a partially zeroed case is saved and marked `COMPLETE`; `--resume` would then keep it. `predict_stage3_member.py` uses the same pipeline (its guard not yet inspected). | 2026-10-14 Stage 2 inference (on CPU, see below): case `004` hit the hard limit at 654 s during RightHip; its file was written with labels `[0,1,2,51,52,54]` (no RightHip) and recorded `COMPLETE`. Case `008` timed out at Sacrum, came back all-zero, and the run stopped. Whether pre-fix inference ever hit the budget is not known; in pre-fix Stage 2 case `004` the right hip is present while the left hip is missing, which a budget cut (anatomy order Sacrum → LeftHip → RightHip → Femur) cannot produce. |
| D6 | **Found 2026-10-15; located 2026-10-15 in the retrained Stage B (Fragment) model output.** Retrained Stage 2 final predictions lose much of the sacrum in some pelvic cases, which the pre-fix Stage 2 did not. Stage A finds the sacrum (G1 Sacrum 0.975, 0/20 < 0.5) and routes a normal crop. **In the sacrum crop, the retrained Fragment model segments the hip parts as foreground and leaves most of the sacrum as background**; the paint step then correctly drops the hip voxels (other anatomies' Stage A masks), leaving a small sacrum. Decode and resampling lose nothing. Training labels are correct (sacrum crop = sacrum fragments only), the Fragment dataset is unchanged since 2026-09-22, and V308 does not read the case map. Why this checkpoint learned it is not yet established (see session log 2026-10-15, D6 diagnosis result). **Confirmed training-side 2026-10-15** by the per-bone Fragment check on the 80 Fragment validation crops (GT-routed, as trained): retrained Sacrum median Dice 0.964 with 4/20 crops < 0.5 and 8/20 < 0.90 (over-prediction into the hips: precision 0.10–0.56), pre-fix checkpoint 0.976 with 0/20 < 0.5 on the same crops; hips and femur equal for both. | Final-prediction GT-sacrum coverage, pre-fix → retrained: `030` 0.984 → 0.133, `047` 0.975 → 0.145, `060` 0.985 → 0.261, `074` 0.977 → 0.614, `172` 0.982 → 0.635, `154` 0.860 → 0.708, `046` 0.985 → 0.853. Median over 20 pelvic cases 0.982 → 0.959; cases < 0.5: 0 → 3. Log for `030`: `[Sacrum] bbox crop=(217,168,206) ... painted 74886 voxels, 8 fragments` (GT 352,258); `047`: 54,913 voxels, 8 fragments (GT 323,733); a good case: 294,886 voxels, 2 fragments. Fragment training: pre-fix best EMA F1 0.700 (epoch 86, rank-local), retrained 0.687 (epoch 83, global). Cause not identified. |
| D7 | **Merged into D4 (2026-10-16).** Raised 2026-10-15 as a new question about `one_minus_binary_dice` on femur-only scans; it is the D4 evaluation-scope issue, already decided 2026-10-12 (keep whole-volume evaluation; labelled-anatomy-only view as secondary analysis). No new decision needed. | Measured size of the D4 effect after the D3 fix: femur-only mean binary Dice 0.757 → 0.565 while femur-range Dice is unchanged (`317` 0.991 → 0.990, `305` 0.988 → 0.987, `417` 0.984 → 0.985); 0.2–1.2 M unlabelled pelvic voxels painted. Instance metrics are not affected: `instance_metrics` averages only anatomies present in the GT. |
| D8 | **Found 2026-10-20; inference pipeline, not training.** When Stage A finds no right hip in a femur-only scan, the ABBC pipeline re-adds RightHip from its bone skeleton and uses a `bone_fallback` mask. The paint step's confine-to-mask rule treats every other anatomy's mask as exclusive territory, so femur voxels inside that fallback mask are deleted. | `417` member_03 femur coverage 0.433 (others 0.97); `357` 0.57 in Stage 2 and all three members (pre-fix 0.970). Fallback fired in 4–10 femur-only cases per run; harmful only where the mask is large (`357` 1.4 %, `417` 2.7 % of the volume). See session log 2026-10-20 (C-3.3). |
| D4 | **Found 2026-10-12, evaluation scope, not a training defect.** `scripts/evaluate_stage2_backend.py` compares the whole prediction to the whole label. In femur scans, predicted visible pelvic bone counts as false positives in Dice and Instance F1, and the H1 risks inherit this. | Same 12/20 femur cases as D3. Whether the official PENGWIN evaluator restricts by anatomy is not verified. |

## Fix items

| ID | Item | Status | Change | Lightweight verification (agent-typed) |
|---|---|---|---|---|
| F1 | Disable L/R mirror for Anatomy | DONE (2026-10-12) | Add `Dataset701_PENGWINStage2Anatomy` to `DISABLE_X_MIRROR_DATASETS` in `baselines/pengwin2026-task1-abbc/code_task1/core.py`. Fragment (`Dataset702`) mirroring stays as is: its labels are anatomy-local fragment IDs with no laterality. | Unit test: the trainer mirror config for Dataset701 excludes axis 2 for training and inference; Dataset702 keeps `(0, 1, 2)`. |
| F2 | Global DDP validation for Fragment | DONE (2026-10-12) | Gather validation statistics across ranks before computing F1/EMA; make the best-checkpoint and early-stop decisions once and share them with all ranks. | Unit test with simulated rank statistics: all ranks get identical F1, best flag, and stop flag. |
| F3 | Check Anatomy early-stop path | DONE (2026-10-12): no change needed | `PengwinTrainer.on_validation_epoch_end` calls the stock nnU-Net method first, which all-gathers `tp/fp/fn` and validation losses across ranks before logging `mean_fg_dice`; the early stop then reads the shared `ema_fg_dice`. Anatomy decisions are already global. | Inspected installed `nnunetv2` source (`nnUNetTrainer.on_validation_epoch_end`) and `core.py` `PengwinTrainer.on_validation_epoch_end`. |
| F6 | Anatomy partial-label case map (D3) | DONE (2026-10-12) | `exchange_ml.stage2_backend.case_labeled_class_map` records the anatomy classes present in each case's GT target (key parsed exactly as the trainer parses `batch['keys']`). `scripts/stage2_backend.py` `write_case_labeled_map` builds it from the run's `Dataset701` `labelsTr`, checks it covers every train/validation sample of the frozen split, writes `<results_root>/Dataset701_PENGWINStage2Anatomy/case_labeled_classes.json`, and sets `PENGWIN_CASE_LABELED_MAP` before DDP spawn (Anatomy stage only). `core.py` `_load_case_labeled_map`: an explicitly set map must load and be non-empty; the silent fallback remains only when the variable is unset. | 5 unit tests: per-case classes; invalid targets rejected; trainer mask for pelvic `[T,T,T,T,F]` and femur `[T,F,F,F,T]`; missing/empty explicit map raises; wrapper writes the map and rejects a split sample it does not cover. Without a map the femur mask is all `True` (old behaviour). Dry run on the real labels (output to scratch): 80 cases, 40 → `[1,2,3]`, 40 → `[4]`. Batches carry `keys` (`nnUNetDataLoader3D.generate_train_batch` returns them; `PengwinTrainer.get_dataloaders` defers to it). |
| F7 | Disable the challenge time budget for validation inference; refuse CPU inference (D5) | DONE (2026-10-15) | `inference/inference.py`: `PENGWIN_DISABLE_TIME_BUDGET=1` sets `V03_USE_TIME_BUDGET = False` (upstream default unchanged when unset). New `scripts/stage2_backend.py` `import_inference_backend`: raises if `torch.cuda.is_available()` is False, sets the switch before importing `inference`, and raises if the imported module still has the budget on. `predict_stage2_backend.py` and `predict_stage3_member.py` both load the backend only through it. The legitimate `no usable mask` / `mask empty after CC filter` zero paths are unchanged. | 4 unit tests: switch on → budget off, unset → upstream `True`; loader refuses when CUDA is unavailable; loader returns a budget-off module; loader refuses a cached module imported with the budget on. Full suite 41/41. Output is unchanged for cases that never hit the budget, so valid `004`/`008` may be resumed. |
| F8 | Fragment inference uses `checkpoint_final.pth` and must pass the per-bone Fragment gate (D6) | DONE (2026-10-16) | `inference/inference.py`: `DS538_CHECKPOINT_NAME` from `PENGWIN_DS538_CHECKPOINT` (upstream default `checkpoint_best.pth` when unset), passed to the Stage B `build_predictor`; Stage A unchanged. `configs/stage2_backend.json` `fragment_stage.inference_checkpoint = checkpoint_final.pth` (members inherit it). `scripts/stage2_backend.py` `fragment_inference_checkpoint` (only best/final accepted); `import_inference_backend` sets the switch and refuses a module loaded with another checkpoint. `scripts/stage3_h1.py` `inference_checkpoint_path`; `predict_stage3_member.py` checks and hashes the checkpoint actually used. `check_fragment_gate.py --checkpoint` defaults to the configured one. Patch regenerated; runbook hash updated. | 6 unit tests: switch and upstream default; Stage B build uses the switch; loader sets and verifies; cached module with another checkpoint refused; unsupported name rejected; Stage 2 and member configs resolve Fragment → final, Anatomy → best. Full suite 52/52. `stage2_backend.py check` PASS. |
| F9 | Bone-fallback masks no longer block other anatomies at paint time (D8) | DONE (2026-10-20) | `inference/inference.py`: new `_paint_block_mask(merged, anatomy, img_shape, ds539_only)` used by the confine-to-mask paint step; `PENGWIN_CONFINE_DS539_ONLY=1` sets `V03_CONFINE_DS539_ONLY`, so only `source == 'ds539'` masks block (upstream default unchanged when unset: every mask blocks). Fallback masks still define their own anatomy's crop. `scripts/stage2_backend.py` `import_inference_backend` sets the switch and refuses a module loaded without it; used by Stage 2 and member inference. Training and the gates are unaffected. Patch regenerated; handover `inference.py` hash updated. | 6 unit tests: toy femur-only scan — upstream blocks the femur under the fallback, F9 frees it, a Ds539 neighbour still blocks; Ds539-only blocking identical to upstream; switch and upstream default; paint step calls the helper; loader sets the switch; cached module without it refused. Full suite 58/58. `stage2_backend.py check` PASS. Patch applies to a clean `ccc625f` export and reproduces all three hashes. |
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
| Stage 2 single model (seed 31000) | DONE (2026-10-13 07:36–15:25; early stop epoch 109, best 84) | **PASS** (hips 0.988 / 0.986, 0 cases < 0.5) | DONE (2026-10-13 19:33 – 10-14 06:17, GPUs 5,6; early stop epoch 83, best EMA F1 0.6869) | DONE 2026-10-15 (40/40, F7; sacrum regression D6) | Evaluation DONE 2026-10-15 17:07 with `checkpoint_best`; **G2 not passed** (D6). **C-0.3 re-inference with F8 DONE 2026-10-16** (40/40 COMPLETE, `checkpoint_final` verified); re-evaluation DONE 2026-10-16 15:53; **G2 proposed PASS, awaiting user confirmation** |
| `member_01` (seed 31001) | DONE (2026-10-17 10:15 → crash at epoch 74 → resumed from epoch 50, 21:28 – 10-18 00:34; early stop epoch 96) | **PASS** (hips 0.988 / 0.985) | DONE (2026-10-18 09:07 – 10-19 00:07; early stop epoch 123, best EMA F1 0.7123); **GF PASS** 2026-10-19 (`checkpoint_final`) | DONE 2026-10-19 (40/40 COMPLETE, `checkpoint_final` verified) | DONE |
| `member_02` (seed 31002) | DONE (2026-10-15 07:17–14:15; early stop epoch 96) | **PASS** (hips 0.989 / 0.986) | DONE (fresh restart 2026-10-16 13:37 – 10-17 07:00; early stop epoch 138, best EMA F1 0.7068); **GF PASS** 2026-10-17 (`checkpoint_final`) | DONE 2026-10-17 (40/40 COMPLETE, `checkpoint_final` verified) | DONE |
| `member_03` (seed 31003) | DONE (2026-10-19 07:22–14:54; early stop epoch 110, best 85) | **PASS** (hips 0.984 / 0.985) | DONE (crashed 2026-10-19 at epoch 27; fresh restart 2026-10-20 06:31–16:32; early stop epoch 77, best EMA F1 0.6893); **GF PASS** 2026-10-20 (`checkpoint_final`) | DONE 2026-10-20 (40/40 COMPLETE); **D8** femur loss on `417`/`357` → re-run after F9 | IN PROGRESS |

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
- **GF — Fragment gate (approved 2026-10-16).** Before any inference with a
  Fragment checkpoint: `scripts/check_fragment_gate.py` on the 80 Dataset702
  validation crops; every bone median foreground Dice ≥ 0.90 and at most 2
  of 20 crops < 0.5 (exit 0 = PASS). Reference: Stage 2 retrained
  `checkpoint_best` FAIL (Sacrum 4 crops < 0.5), `checkpoint_final` PASS,
  pre-fix PASS. On FAIL: no inference with that checkpoint; stop and
  re-diagnose (same rule as G1; proposed, confirm).
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
| 2026-10-16 | F8: Fragment inference uses `checkpoint_final.pth` for Stage 2 and all members, each gated by GF. | User (accepted agent recommendation) |
| 2026-10-16 | GF thresholds approved: every bone median Dice ≥ 0.90, at most 2/20 crops < 0.5. | User |
| 2026-10-16 | D7 merged into D4; the D4 decision stands. | Agent (correction) |

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
### 2026-10-20 (F9 implemented)

- User approved F9. Implemented as in Fix items (switch
  `PENGWIN_CONFINE_DS539_ONLY`, helper `_paint_block_mask`, loader check).
- Tests: 6 new (`PaintBlockMaskFixTests`); full suite 58/58;
  `stage2_backend.py check` PASS (`nnunetv2 2.5.1`, ABBC trainer import).
- Patch `patches/pengwin2026-task1-abbc_on_ccc625f.patch` regenerated (3
  files: `core.py`, `model.py`, `inference.py`; `git apply --check -R` OK),
  SHA-256 `4d23431ef38667d7b9e5077e8dd4bf916d652ecad806254c311a1c2b3ed08976`.
  Applied to a clean `git archive ccc625f` export it gives `core.py`
  `b7fe9b5d…48a3` and `model.py` `ed79fce2…569d` (unchanged) and
  `inference.py` `7e64a66949eb306d373b6e3b62bd660d77b4998f28da8999dccd5f967d65705d`
  (was `3371f523…3361`); `member_01_handover.md` updated.
- Not yet verified on real data: needs GPU inference (user-typed). Expected:
  only cases where a fallback fired change; `357` and `417` femur coverage
  return to ≈0.97.
- Re-run plan (inference only, no retraining): archive the current Stage 2
  and member inference outputs (moves only; `--resume` would otherwise keep
  the old `COMPLETE` cases), re-run all four inferences (all 40 cases), check
  the F9 effect, re-evaluate Stage 2, re-propose G2.
- Agent-typed: edits; `py_compile`; F9 tests; full suite; `stage2_backend.py
  check`; `git diff > patch`; `git apply --check -R`; `git archive` + `git
  apply` + `sha256sum` in scratch; `grep` of recorded hashes; `nvidia-smi`.

### 2026-10-20 (C-3.3 `member_03` inference done; new defect D8 found)

- **C-3.3:** `predict_stage3_member.py --member member_03 --resume` (log
  `outputs/logs/c3_member_03_predict.log`, manifest 2026-10-20 21:20):
  `complete: 40/40`; 40 rows `COMPLETE`, seed 31003; Anatomy SHA-256
  `becbca3b…` (= member_03 Anatomy `checkpoint_best`), Fragment `ecb63a1e…`
  (= `checkpoint_final`). Fingerprints 40× Stage A `1.0194e+02`, 40× Stage B
  `1.1607e+02`. No errors / `emit zero` / empty-mask lines. Outputs on sdb
  (`/mnt/sdb/shared/chuong/m03_infer/`): 40 + 40 files, 1.8 + 1.5 GB.
- Coverage (median / < 0.5 / < 0.9), old member_03 → **new**: Sacrum
  0.973/0/1 → 0.971/0/3; LeftHip 0.732/8/12 → 0.979/0/0; RightHip
  0.545/9/20 → 0.980/0/0; Femur 0.982/0/1 → 0.976/**1**/3. Below 0.9: `047`
  Sac 0.893, `074` Sac 0.798, `154` Sac 0.752, `292` Femur 0.564, `357`
  Femur 0.568, **`417` Femur 0.433** (other runs 0.966–0.984).
- The coverage script is now saved as
  `outputs/stage3_h1/diagnostics/member_bone_coverage.py` (the session
  scratchpad was wiped by the reboot).
- **D8 diagnosis (`417`, member_03).** Stage A femur is correct (762,714
  voxels, as member_02; nnU-Net validation map: 748k of GT femur labelled
  Femur), and the Fragment model scores 0.977 on the GT-routed 417 femur crop
  (GF). Stage A finds no right hip (126 raw → 0 after cleanup), but the
  pipeline's bone-recall step re-adds RightHip from the bone skeleton;
  `Ds539 sanity fail (covers 0.0%), bone-skeleton fallback (2.7%)` → a
  `bone_fallback` RightHip mask (crop 62.7% of the volume) containing femur
  bone. The paint step (`inference.py` ~1938–1956, `PENGWIN_CONFINE_TO_MASK`,
  default on) forbids each anatomy from painting into every other anatomy's
  `mask`, assuming these are disjoint Stage A argmax masks; the bone fallback
  mask is not. The femur is painted last, so its voxels inside the RightHip
  fallback mask are dropped: 332,642 painted vs 747,560 for member_02.
- Fallback events (all RightHip, all femur-only scans): Stage 2 7, member_01
  9, member_02 4, member_03 10. GT femur coverage on these cases (`*` =
  fallback fired; Stage 2 / m01 / m02 / m03 / pre-fix m02): `357` 0.566* /
  0.566* / 0.568* / 0.568* / 0.970 and `417` 0.979 / 0.966 / 0.971 /
  0.433* / 0.984; the other fallback cases (masks 0.1–0.6 % of the volume)
  are within 0.01 of the no-fallback runs. `292` (≈0.57 in all runs,
  including pre-fix) has no fallback and is a different issue.
- **D8 explains the unexplained `357` femur loss** (noted 2026-10-15) and
  the `417` member_03 loss. Effects: the Stage 2 risk CSV contains `357`
  femur loss from a pipeline bug; on `417`, member disagreement comes from
  the bug, not from model uncertainty. Training and the gates are not
  affected (GF uses GT-routed crops).
- **Proposed F9 (awaiting approval):** in the confine-to-mask block, build
  `other` only from masks whose `source == 'ds539'` (bone fallback masks
  still define their own crop but no longer block other anatomies). Opt-in
  switch like F7/F8, unit test, patch regenerated. Then re-run validation
  inference for Stage 2 and all three members (no retraining; ~1 h each),
  re-evaluate Stage 2, and re-propose G2. **G2 must not be confirmed on the
  current CSV.**
- Agent-typed: `tail`/`grep` of the predict logs; manifest read; `ls`/`du`;
  `member_bone_coverage.py`; Stage A validation-map and pipeline log
  comparison for `417`; read of `inference.py` paint and fallback code;
  fallback count over the four predict logs; femur coverage on fallback
  cases across five runs; `uptime -s`; `quota -s`.

### 2026-10-20 (GF for `member_03`: PASS)

- User ran `check_fragment_gate.py --model member_03=...` (log
  `outputs/logs/fragment_gate_member_03.log`, finished 19:02; outputs
  `outputs/stage3_h1/diagnostics/fragment_gate/member_03/`). Checkpoint
  `checkpoint_final.pth`; Stage B `w0sum=1.1607e+02` (distinct from Stage 2
  `1.1402e+02`, member_01 `1.1432e+02`, member_02 `1.1190e+02`).
- Median Dice / crops < 0.5: Sacrum 0.974 / 0, LeftHip 0.981 / 0, RightHip
  0.981 / 0, Femur 0.980 / 0 → **GF PASS**. The short training (77 epochs)
  did not leave an under-trained sacrum in `checkpoint_final`.
- Crops < 0.90, member_03 / member_01 / member_02 / Stage 2 / pre-fix:
  Sacrum 2 / 1 / 3 / 3 / 0, LeftHip 2 / 1 / 1 / 1 / 5, RightHip
  0 / 0 / 2 / 0 / 3, Femur 1 / 1 / 0 / 1 / 1. Lowest: `004` LeftHip 0.794
  (≈0.80 in every model, 0.772 pre-fix), `357` Femur 0.813, `154` Sacrum
  0.836, `074` Sacrum 0.862, `008` LeftHip 0.865 (others ≥ 0.97). `074`
  Sacrum is < 0.87 in all four retrained models (0.981 pre-fix).
- All four Fragment models (Stage 2 + 3 members) now pass GF on
  `checkpoint_final`.
- Next: C-3.3 `member_03` inference, outputs linked to
  `/mnt/sdb/shared/chuong/m03_infer/`.
- Agent-typed: `tail`/`grep` of the gate log; read of the summary; scratch
  per-crop comparison of the five gate CSVs; `nvidia-smi`; `ls`.

### 2026-10-20 (C-3.2 `member_03` Fragment restarted and finished)

- User moved the crashed fold to `/mnt/sdb/shared/chuong/m03_frag_crashed_2026-10-19`
  and restarted fresh: 2026-10-20 06:31:48 → 16:32:56 (log
  `outputs/logs/c3_member_03_fragment.log`), world size 2, epoch 0 at 06:32.
  Warm-start from member_03's own Anatomy `checkpoint_best.pth`
  (`loaded=120 skipped_seg=10`; SHA-256 `becbca3b…`, as at G1). No reboot
  since 2026-10-19 21:50.
- Early stop (global, F2) at epoch 77: trainer best EMA F1 0.6893 (no
  improvement > delta for 25 epochs, so last such gain ~epoch 52); nnU-Net's
  best EMA updated at epoch 76. `Training done`. Shortest Fragment run so far
  (Stage 2 83 epochs / 0.6869; member_02 138 / 0.7068; member_01 123 /
  0.7123), close to the Stage 2 run whose `checkpoint_best` failed GF (D6).
  GF on `checkpoint_final` decides.
- No `Traceback`/NCCL/OOM/quota errors. SHA-256: `checkpoint_final.pth`
  `ecb63a1e…9232`, `checkpoint_best.pth` `6ae4971c…3ff9`.
- Quota 91.7 GB (+0.4 GB since 2026-10-19): a VS Code Claude Code extension
  update in `~/.vscode-server` (~420 MB, 18:38), not project files.
- Agent-typed: `tail`/`grep` of the logs and fold; `readlink -f`; `ls` of
  sdb; `sha256sum`; `uptime -s`; `quota -s`; `find` of large new files;
  `nvidia-smi`; `pgrep`.

### 2026-10-20 (third server reboot; C-3.2 `member_03` Fragment lost)

- C-3.2 started 2026-10-19 17:39:13 (epoch 0 at 17:40). Trainer and `tee`
  logs stop at 21:25:13, start of epoch 27 (after epoch 26's
  `checkpoint_best` save). No error line (no Traceback, NCCL, OOM, quota).
- Server rebooted: `uptime -s` 2026-10-19 21:50:28 (`last -x` boot 21:46),
  kernel unchanged 5.15.0-198. Third unplanned stop of a long run (2026-10-16
  11:14, 2026-10-17 16:32, 2026-10-19 21:50); cause not visible to this user.
- Fold has only `checkpoint_best.pth` (epoch ~26), no
  `checkpoint_latest.pth` (first save at epoch 50) → `--resume` impossible;
  fresh restart, as for `member_02` on 2026-10-16. The crashed fold is moved
  aside on sdb, not deleted.
- Data links and quota (91.3 GB) intact. GPUs 0, 1, 2, 4, 5 idle at 06:29.
- Agent-typed: `date`; `uptime -s`; `last -x`; `tail`/`grep` of the logs;
  `ls` of the fold; `pgrep`; `quota -s`; `nvidia-smi` ×3; `ss`; `df`.

### 2026-10-19 (C-1.3 `member_01` inference done; C-3.1 `member_03` Anatomy G1 PASS)

- **C-1.3:** `predict_stage3_member.py --member member_01 --resume` (log
  `outputs/logs/c1_member_01_predict.log`, manifest 15:07): `complete:
  40/40`; manifest 40 rows `COMPLETE`, seed 31001; Anatomy SHA-256
  `97ae624d…` (= member_01 Anatomy `checkpoint_best`), Fragment `bdb5251c…`
  (= `checkpoint_final`, F8). Log fingerprints 40× Stage A
  `w0sum=1.0227e+02`, 40× Stage B `1.1432e+02` (same as GF). No errors, no
  `emit zero`, no empty-mask lines. Outputs on sdb
  (`/mnt/sdb/shared/chuong/m01_infer/`): 40 + 40 files, 1.8 + 1.4 GB.
- Retrospective correct-label GT coverage (median / < 0.5 / < 0.9), old
  member_01 → Stage 2 retrained → **new member_01**: Sacrum 0.974/0/3 →
  0.978/0/4 → **0.978/0/2**; LeftHip 0.123/17/19 → 0.987/0/0 →
  **0.980/0/0**; RightHip 0.829/4/14 → 0.982/0/0 → **0.981/0/0**; Femur
  0.978/1/3 → 0.980/0/2 → **0.978/0/2**. Below 0.9: `074` Sacrum 0.750,
  `154` Sacrum 0.893, `292` Femur 0.571, `357` Femur 0.566 (same weak cases
  as the gate and the other retrained models).
- **C-1 (`member_01`) complete.**
- **C-3.1:** started 07:22:23 on GPUs 5,7 from epoch 0; early stop at epoch
  110 (14:54), best EMA Dice 0.9720 at epoch 85; `Training done`; 40
  validation predictions (15:02). No errors. **G1
  (`check_anatomy_gate.py`, exit 0): PASS** — Sacrum 0.977 / 0, LeftHip
  0.984 / 0, RightHip 0.985 / 0, Femur 0.985 / 0. `checkpoint_best.pth`
  SHA-256 `becbca3b…`.
- Quota 91.3 GB. The session scratchpad was cleared by the 2026-10-17 reboot;
  the coverage script was rewritten generically (`member_cov.py <member>`).
- Agent-typed: `tail`/`grep` of the predict and C-3.1 logs; manifest read;
  `ls`/`du`; `check_anatomy_gate.py`; `sha256sum`; scratch coverage script;
  `uptime -s`; `quota -s`; `nvidia-smi`; `ss`.

### 2026-10-19 (GF for `member_01`: PASS; C-3.1 running)

- User ran `check_fragment_gate.py --model member_01=...` (log
  `outputs/logs/fragment_gate_member_01.log`, finished 08:17; outputs
  `outputs/stage3_h1/diagnostics/fragment_gate/member_01/`). Checkpoint
  `checkpoint_final.pth`; Stage B `w0sum=1.1432e+02` (distinct from Stage 2
  final `1.1402e+02` and member_02 `1.1190e+02`).
- Median Dice / crops < 0.5: Sacrum 0.975 / 0, LeftHip 0.983 / 0, RightHip
  0.982 / 0, Femur 0.984 / 0 → **GF PASS**.
- Crops < 0.90, member_01 / member_02 / Stage 2 final / pre-fix: Sacrum
  1 / 3 / 3 / 0, LeftHip 1 / 1 / 1 / 5, RightHip 0 / 2 / 0 / 3, Femur
  1 / 0 / 1 / 1. Weakest: `074` Sacrum 0.674 (0.817 / 0.810 / 0.981),
  `004` LeftHip 0.799 (0.795 / 0.800 / 0.772), `357` Femur 0.758 (0.954 /
  0.759 / 0.865). `074` Sacrum is below 0.82 in all three retrained Fragment
  models but 0.981 pre-fix: a consistent retrained-model weakness on one
  crop, not a gate failure.
- C-3.1 (`member_03` Anatomy) started 2026-10-19 07:22 on GPUs 5,7 (port
  29603): case map 80 cases, axis-2 mirror disabled, world size 2; at epoch
  15 at 08:28.
- Next: `member_01` validation inference with output folders linked to
  `/mnt/sdb/shared/chuong/m01_infer/`.
- Agent-typed: `tail`/`grep` of the gate log; read of the summary; scratch
  per-crop comparison of the four gate CSVs; `pgrep`; `tail`/`grep` of the
  C-3.1 log; `nvidia-smi`.

### 2026-10-19 (C-1.2 `member_01` Fragment finished)

- Run 2026-10-18 09:07:59 → 2026-10-19 00:07:59 on GPUs 4,5 (log
  `outputs/logs/c1_member_01_fragment.log`), world size 2, results on sdb
  (`/mnt/sdb/shared/chuong/m01_results`). Warm-start from member_01's own
  Anatomy `checkpoint_best.pth` (`loaded=120 skipped_seg=10`; SHA-256
  `97ae624d…`, same as at G1). Started at epoch 0; no server reboot since
  2026-10-17 16:32.
- Early stop (global, F2) at epoch 123: no instance F1 improvement > delta
  for 25 epochs; trainer best EMA F1 0.7123. nnU-Net's best EMA 0.7171 was
  updated at the last epoch (gain below the early-stop delta), so
  `checkpoint_best` and `checkpoint_final` are both from epoch ~123.
  `Training done`. Reference: Stage 2 0.6869 (epoch 83), member_02 0.7068
  (epoch 138).
- No `Traceback`/`RuntimeError`/NCCL/quota errors. Quota 91.3 GB.
- SHA-256: `checkpoint_final.pth` `bdb5251c…3877`, `checkpoint_best.pth`
  `b51dfa41…b2a7`.
- Next: GF on `checkpoint_final` (user-typed). `member_03` folder and sdb
  link exist (2026-10-18 09:17) but C-3.1 was not started (results folder
  empty, no log).
- Agent-typed: `ls`/`tail`/`grep` of the logs and fold; `readlink -f`;
  `sha256sum`; `uptime -s`; `quota -s`; `nvidia-smi`; `pgrep`.

### 2026-10-18 (C-1.1 `member_01` Anatomy resumed and finished; G1 PASS)

- User resumed with `--resume` on GPUs 5,6 (log
  `outputs/logs/c1_member_01_anatomy_resume.log`, new trainer log
  `training_log_2026_10_17_21_28_53.txt`): `Resuming member_01 anatomy from
  .../checkpoint_latest.pth`, first epoch `Epoch 50` (21:29). Case map
  loaded (80 cases); `disable axis-2 mirror` → `(0, 1)`; world size 2.
- Early stop at epoch 96 (00:34): `no EMA-Dice improvement for 25 epochs`
  (counter restarted at epoch 50 as expected). nnU-Net's best EMA pseudo Dice
  0.9735 (`checkpoint_best.pth` 00:22; small gains below the trainer's
  early-stop delta still update it). `Training done`; 40-case validation
  complete 00:43 (nnU-Net all-class mean 0.523 counts unlabelled bones as 0,
  not used). No `Traceback`/NCCL/quota errors.
- **G1 (`check_anatomy_gate.py`, exit 0): PASS.** Median Dice / cases < 0.5
  on labelled cases: Sacrum 0.970 / 0, LeftHip 0.988 / 0, RightHip 0.985 /
  0, Femur 0.982 / 0 (Stage 2 hips 0.988/0.986; member_02 0.989/0.986).
- SHA-256: `checkpoint_best.pth` `97ae624d…c12d` (Fragment warm-start and
  Stage A inference), `checkpoint_final.pth` `ab702813…a2e2`. As for the
  earlier G1 runs, the validation is nnU-Net's end-of-training prediction.
- Next: C-1.2 (`member_01` Fragment). GPUs at check: 2, 4, 5 idle.
- Agent-typed: `ls`/`tail`/`grep` of the logs and fold;
  `check_anatomy_gate.py`; `sha256sum`; `uptime -s`; `quota -s`;
  `nvidia-smi`; `pgrep`; `ss`.

### 2026-10-17 (second server crash; C-1.1 `member_01` Anatomy interrupted)

- C-1.1 started 10:15:58 on GPUs 4,5 (results on sdb via
  `outputs/stage3_h1/members/member_01/nnUNet_results` →
  `/mnt/sdb/shared/chuong/m01_results`). Both the trainer log and the `tee`
  log stop at 15:28:05, at the start of epoch 74 (last line `Current learning
  rate: 0.00093`), right after a new best EMA pseudo Dice 0.9698. No error
  line. ~276 s per epoch.
- Server rebooted: `uptime -s` 2026-10-17 16:32:16; `last -x` marks the
  other users' sessions as `crash`; same kernel 5.15.0-198. Cause not visible
  to this user (`journalctl` and `/var/log/kern.log` need `adm`). The
  1-hour gap between the last log line and the reboot suggests the machine
  hung before it was restarted.
- Fold contents: `checkpoint_latest.pth` 13:48 (end of epoch 50; nnU-Net
  saves it every 50 epochs), `checkpoint_best.pth` 15:28, `debug.json`,
  `progress.png`, one training log. Data links and quota (91.2 GB) intact.
- Resume path: `stage3_h1.py train --resume` requires `checkpoint_latest.pth`
  and, for `member_01` Anatomy only, trusts that file for full unpickling
  (`PENGWIN_TRUSTED_RESUME_CHECKPOINT`). nnU-Net restores weights, optimizer,
  logger (EMA history) and `_best_ema`, so `checkpoint_best` selection
  continues correctly. The trainer's early-stop counters (`_es_best_dice`,
  `_es_no_improve`) are not saved and restart at the resume epoch; the run
  can only last longer than an uninterrupted run, never stop earlier. A
  resumed run is not bit-identical to an uninterrupted one (data-loader RNG).
- GPUs at 21:23: 5 and 6 idle (0 MiB, 0 %); 0–4 and 7 used by other users.
- Agent-typed: `date`; `uptime -s`; `last -x`; `quota -s`; `ls -l` of links
  and fold; `tail` of logs; `grep` of `stage3_h1.py`, `stage2_backend.py`,
  `core.py` (resume and early-stop code); `journalctl` (no access);
  `nvidia-smi` ×4; `ps` of GPU processes; `ss` for port 29601.

### 2026-10-17 (C-2.3 `member_02` validation inference done)

- User ran `predict_stage3_member.py --member member_02 --resume` on GPU 4
  (log `outputs/logs/c2_member_02_predict.log`, manifest written 10:01):
  `member_02 validation predictions complete: 40/40`; manifest 40 rows, all
  `COMPLETE`; seed 31002. Outputs on sdb via links
  (`/mnt/sdb/shared/chuong/m02_infer/`): 40 predictions (1.8 GB), 40
  probability maps (1.4 GB). Quota unchanged at 91.3 GB.
- Checkpoints: manifest Fragment SHA-256 `c72dba94…dd53` = `checkpoint_final`
  (F8); Anatomy `cd0f1376…3fd4` = member_02 Anatomy `checkpoint_best`
  (`/mnt/sdb/shared/chuong/b/021`). Log fingerprints: 40× Stage A
  `w0sum=1.0065e+02`, 40× Stage B `1.1190e+02` (same as the GF run).
- No `Traceback`/`Error`, no `emit zero`, no `no usable mask` / `mask empty
  after CC filter` lines.
- Retrospective correct-label GT coverage (median / cases < 0.5 / < 0.9;
  pelvic n=20 for Sac/hips, femur-only n=20), old member_02 (2026-10-05 H1)
  → Stage 2 retrained → **new member_02**: Sacrum 0.981/0/0 → 0.978/0/4 →
  **0.981/0/0**; LeftHip 0.869/4/11 → 0.987/0/0 → **0.984/0/0**; RightHip
  0.460/12/20 → 0.982/0/0 → **0.982/0/0**; Femur 0.984/0/1 → 0.980/0/2 →
  **0.979/0/2**. The D1 hip losses are gone in this member.
- Below 0.9: femur `292` 0.565 (old member_02 0.605, so a hard case) and
  `357` 0.568 (Stage 2 retrained 0.566, old member_02 0.970). `357` femur
  loss now appears in both retrained models; still not examined (noted
  2026-10-15). Not a gate item.
- **C-2 (`member_02`) complete.**
- Agent-typed: `tail`/`grep` of the predict log; read of the manifest;
  `ls`/`du` of outputs; `sha256sum` of the Anatomy checkpoint; scratch
  three-way coverage script (reads GT retrospectively; not part of scoring);
  femur check of `292`/`357`; `quota -s`; `nvidia-smi`; `ss` for ports.

### 2026-10-17 (GF for `member_02`: PASS)

- User ran `check_fragment_gate.py --model member_02=...` (log
  `outputs/logs/fragment_gate_member_02.log`, 07:39–07:50; outputs
  `outputs/stage3_h1/diagnostics/fragment_gate/member_02/`). Checkpoint
  `checkpoint_final.pth`; Stage B `w0sum=1.1190e+02` (Stage 2 final
  `1.1402e+02`, best `1.1307e+02`), so the member_02 model was loaded.
- Median Dice / crops < 0.5: Sacrum 0.973 / 0, LeftHip 0.984 / 0, RightHip
  0.980 / 0, Femur 0.981 / 0 → **GF PASS**.
- Crops < 0.90 (member_02 / Stage 2 final / pre-fix): Sacrum 3 / 3 / 0,
  LeftHip 1 / 1 / 5, RightHip 2 / 0 / 3, Femur 0 / 1 / 1. Weakest: `154`
  Sacrum 0.804 (0.916 / 0.971), `074` Sacrum 0.817 (0.810 / 0.981), `167`
  Sacrum 0.879, `004` LeftHip 0.795 (0.800 / 0.772), `030` RightHip 0.837
  (0.970 / 0.772). The D6 crop `047` Sacrum is 0.971 (Stage 2 final 0.558).
- Next: `member_02` validation inference with `checkpoint_final`. Expected
  size ~3 GB (old member_02: predictions 1.8 GB, probabilities 1.1 GB), so the
  output folders are linked to `/mnt/sdb/shared/chuong/m02_infer/` first
  (quota 91.3 of 100 GB).
- Agent-typed: `tail` of the gate log; read of `summary_checkpoint_final.json`;
  scratch per-crop comparison with `retrained_final.csv` and `prefix.csv`;
  `du` of the old member_02 outputs on sdb; `grep` of
  `predict_stage3_member.py` output paths.

### 2026-10-17 (C-2.2 `member_02` Fragment finished)

- Fresh run 2026-10-16 13:37:59 → 2026-10-17 07:00:25, 2 GPUs (world size
  2), results on sdb (`.../Dataset702_PENGWINStage2Fragment` →
  `/mnt/sdb/shared/chuong/b/m02_frag`). Warm-start from
  `/mnt/sdb/shared/chuong/b/021` = `member_02` Anatomy `checkpoint_best.pth`
  (`MAP.tsv`), `loaded=120 skipped_seg=10`.
- Early stop (global, F2) at epoch 138: no instance F1 improvement for 25
  epochs; best EMA F1 0.7068, last best save at epoch ~112 (2026-10-17
  03:57). `Training done`. Stage 2 reference: best EMA F1 0.6869, early stop
  epoch 83. Per-epoch patch F1 max 0.749.
- No `Traceback`, `RuntimeError`, NCCL timeout/watchdog or quota errors in
  the `tee` log. Quota after the run: 91.3 GB of 100 GB.
- SHA-256: `checkpoint_final.pth` `c72dba94…dd53`, `checkpoint_best.pth`
  `d789e031…b88f`.
- Training-side checks pass. **GF (per-bone Fragment gate on
  `checkpoint_final`) is next** (user-typed, one GPU); inference waits for it.
- Agent-typed: `tail`/`grep` of the `tee` and trainer logs; `ls` of the fold;
  `readlink -f`; `grep` of `MAP.tsv`; `sha256sum`; `quota -s`; `nvidia-smi`.

### 2026-10-16 (Stage 2 re-evaluation with F8 `checkpoint_final`; G2)

- `scripts/evaluate_stage2_backend.py` finished 15:53 (log
  `outputs/logs/c0_stage2_evaluate_f8.log`, ends normally):
  `validation_metrics.csv` 121 lines (40 cases × 3 connectivities),
  `validation_summary.json`. Connectivities 6/18/26 agree to 1e-4.
- Connectivity 26, pre-fix → `checkpoint_best` → `checkpoint_final` (F8):

  | Group | Binary Dice | Inst. P | Inst. R | Inst. F1 | Merges | Splits | Failure cases |
  |---|---|---|---|---|---|---|---|
  | All 40 | 0.764 → 0.721 → **0.729** | 0.868 → 0.960 → 0.948 | 0.695 → 0.812 → 0.821 | 0.737 → 0.854 → **0.850** | 8 → 10 → 14 | 115 → 233 → 214 | 31 → 38 → 37 |
  | Pelvic 20 | 0.770 → 0.877 → **0.895** | 0.754 → 0.933 → 0.908 | 0.589 → 0.799 → 0.816 | 0.617 → 0.833 → 0.824 | 6 → 9 → 10 | 52 → 164 → 159 | 13 → 19 → 18 |
  | Femur-only 20 | 0.757 → 0.565 → 0.563 | 0.983 → 0.988 → 0.988 | 0.801 → 0.826 → 0.826 | 0.857 → 0.876 → 0.876 | 2 → 1 → 4 | 63 → 69 → 55 | 18 → 19 → 19 |

- D6 cases recovered at the semantic level (binary Dice best → final):
  `030` 0.790 → 0.887 (F1 0.411 → 0.687), `047` 0.753 → 0.827 (F1 0.639
  unchanged), `060` 0.783 → 0.848, `074` 0.843 → 0.867 (F1 → 1.000), `172`
  0.869 → 0.935.
- Instance F1 changes between the two checkpoints of the same run go both
  ways by ~0.1 on single cases (up: `030`, `031` 0.806 → 0.933, `074`; down:
  `036` 0.889 → 0.778, `060` 1.000 → 0.889, `076` 0.848 → 0.737, `177` 0.952
  → 0.841, `008` 0.944 → 0.909). Cases have ~5–15 instances, so one
  merge/split moves F1 by ~0.1. The pelvic mean is unchanged within that noise
  (0.833 vs 0.824).
- Femur-only binary Dice remains 0.563 (D4 limitation, decided 2026-10-12).
  Lowest instance F1 is unchanged across all three runs (`336`, `355`, `365`
  0.57–0.67, `417` 0.444, `112` 0.515), i.e. not checkpoint-related.
- **G2: proposed PASS, awaiting user confirmation.** The D6 blocker is
  resolved (GF PASS, no sacrum case < 0.5, pelvic binary Dice best of the
  three runs); instance metrics equal `checkpoint_best` within per-case noise
  and well above pre-fix. On confirmation, this CSV becomes the frozen H1 risk
  source. Recorded caveats: high split count (pelvic 159; the pre-fix 52 was
  partly hidden by swapped hip IDs), femur-only binary Dice (D4), and
  ±0.1 per-case F1 sensitivity to checkpoint choice.
- Agent-typed: `head`/`wc` of the log and CSV; read of new and archived
  `validation_summary.json`; scratch three-way group/per-case comparison
  (connectivity 26).

### 2026-10-16 (disk quota incident; C-2.2 restart crashed; bulk data moved to /mnt/sdb)

- The fresh C-2.2 restart (11:48) died at the start of epoch 2 (~12:08):
  `RuntimeError: One or more background workers are no longer alive`; the
  `tee` log is cut off mid-word. Cause: user disk quota on `/dev/sda3`
  (`/`, includes `/tmp`): 282 GB used vs 100 GB soft quota, grace expired,
  so every new write failed (also the agent's shell and VS Code's zsh
  start-up files in `/tmp`). Not a user change: `whoami` 24chuong.ta, uid
  1049. Most likely enforced since the 11:14 reboot.
- Usage: `Exchange-ML/dataset` 134 GB (PENGWIN 61, RibFrac 74),
  `Exchange-ML/outputs` 58 GB, `miniconda3` 50 GB, other projects ~27 GB.
- User moved data to `/mnt/sdb/shared/chuong/` (chmod 700), each item copied →
  `diff -rq` verified → original deleted → short symlink left in place:
  `dataset -> /mnt/sdb/shared/chuong/dataset`; 39 gitignored `outputs/`
  items (`nnUNetPlans_3d_fullres`, `imagesTr`/`labelsTr`,
  `gt_segmentations`, `validation_predictions`, `*.pth`/`*.npy` > 50 MB) →
  `/mnt/sdb/shared/chuong/b/NNN`, mapping in `/mnt/sdb/shared/chuong/b/MAP.tsv`;
  the crashed member_02 Fragment folder kept as
  `/mnt/sdb/shared/chuong/b/m02_frag_crashed`, and
  `outputs/stage3_h1/members/member_02/nnUNet_results/Dataset702_PENGWINStage2Fragment`
  now links to the empty `/mnt/sdb/shared/chuong/b/m02_frag` for the restart.
  The running Stage 2 evaluation was stopped first (it would have failed at
  its CSV write).
- Result: quota 88.4 GB (< 100 GB). Agent checks: 40 symlinks, 0 broken;
  `MAP.tsv` 39 rows; tests 52/52; `stage2_backend.py check` PASS; dataset
  readable. `.gitignore`: added `/dataset` (the `dataset/` pattern does not
  match a symlink). Nested `baselines/*` repos show only file-mode changes
  (`100644 → 100755`; 0 changes with `core.fileMode=false`, ABBC 3 real files).
- Headroom is ~11 GB: member inference writes predictions/probabilities under
  `outputs/stage3_h1/members/<member>/` on `/dev/sda3`; link those folders to
  sdb before member inference.
- Agent-typed: Read of the crash log and `.gitignore`; `whoami`; `find -L`
  broken-link check; `wc -l MAP.tsv`; `git status`; nested-repo
  `git status`/`diff --summary`; unit suite; `stage2_backend.py check`;
  `nvidia-smi`.

### 2026-10-16 (C-0.3 Stage 2 re-inference validated; C-2.2 interrupted)

- C-0.3 (`predict_stage2_backend.py`, fresh, after archive A2 to
  `outputs/archive/2026-10-16_fragment_checkpoint_best/`): log ends
  `validation predictions complete: 40/40` (09:09); manifest 40 × `COMPLETE`;
  40 `.mha`; no `emit zero`, no `no usable mask`, no traceback. Per-case
  pipeline time median 118 s, max 478 s.
- Checkpoint used: every Stage B `build_predictor` line shows
  `w0sum=1.1402e+02`, identical to the `checkpoint_final` gate run
  (`checkpoint_best` is `1.1307e+02`) → F8 active. Stage A `9.7851e+01`.
- Correct-label GT coverage, 20 pelvic cases (median / cases < 0.5 / < 0.9):

  | Bone | Pre-fix | Retrained `best` | Retrained `final` (F8) |
  |---|---|---|---|
  | Sacrum | 0.982 / 0 / 1 | 0.959 / 3 / 7 | **0.978 / 0 / 4** |
  | LeftHip | 0.244 / 15 / 20 | 0.984 / 0 / 0 | **0.987 / 0 / 0** |
  | RightHip | 0.718 / 6 / 19 | 0.980 / 0 / 0 | **0.982 / 0 / 0** |

  Sacrum per case (pre-fix / best / final): `030` 0.984 / 0.133 / 0.980,
  `047` 0.975 / 0.145 / 0.568, `060` 0.985 / 0.261 / 0.727, `074` 0.975 /
  0.614 / 0.788, `172` 0.982 / 0.635 / 0.979, `154` 0.860 / 0.706 / 0.865,
  `046` 0.984 / 0.853 / 0.978. D6 resolved for the gate (no case < 0.5);
  `047`, `060`, `074` remain below pre-fix.
- C-2.2 (`member_02` Fragment, GPUs 4,5, started 07:26): killed by a server
  reboot at 11:14 (`last -x`: kernel 5.15.0-164 → 5.15.0-198). Trainer log
  reached epoch 27 (11:05). Only `checkpoint_best.pth` exists; nnU-Net writes
  `checkpoint_latest.pth` every 50 epochs, so `--resume` is impossible. The
  `tee` log stopped at 09:41 (buffering). After reboot: `torch.cuda` OK on 8
  GPUs, driver 595.91.07.
- Agent-typed: `grep`/`tail` of `c0_stage2_predict_f8.log`; manifest count;
  `build_predictor` fingerprints vs gate logs; scratch three-way coverage
  script; `pgrep`/`ps`; `ls` of the member_02 fold; `tail` of its trainer
  log; `date`; `uptime -s`; `last -x`; `nvidia-smi`; one-line CUDA test.

### 2026-10-16 (F8 implemented; GF approved; D7 merged into D4)

- User approved F8 and the Fragment gate thresholds (decision log).
- F8 implemented and tested (see Fix items). Patch
  `patches/pengwin2026-task1-abbc_on_ccc625f.patch` regenerated (3 files,
  `git apply --check -R` OK), SHA-256
  `47207f145bf7fa642ef49722e79deb36b31cb5a7266e9e88793ad7519d80ae82`;
  `inference.py` `3371f5238ae7c2cb31796f2a939f0861770153b5d0f9d371a9fb20f2ea353361`;
  `core.py` and `model.py` unchanged. `member_01_handover.md`: inference.py
  hash updated; hand-over now asks to keep both Fragment checkpoints.
- Correction: D7 (2026-10-15) restated D4, which the user already decided on
  2026-10-12. Merged into D4; the measured femur-only Dice drop is recorded
  there as the size of the known limitation. The 2026-10-15 G2 note "D7 needs
  decision" is withdrawn; G2 now waits only on F8 re-inference.
- Existing Stage 2 predictions/metrics were made with `checkpoint_best`; they
  must be archived before re-inference (prepared command A2 in reply).
- Agent-typed: `grep` of `inference.py`, member/backend scripts; edits;
  `py_compile`; full unit suite (52/52); `stage2_backend.py check`;
  `git -C baselines/pengwin2026-task1-abbc -c core.fileMode=false diff >
  patch`; `git apply --check -R`; `sha256sum`; `nvidia-smi`.

### 2026-10-15 (Fragment check of `checkpoint_final`)

- User ran `check_fragment_gate.py --checkpoint checkpoint_final.pth` on the
  retrained Stage 2 Fragment run (log `outputs/logs/fragment_gate_final.log`,
  80/80 crops; `retrained_final.csv`, `summary_checkpoint_final.json`).
  Result: Sacrum median 0.971, 0 crops < 0.5; LeftHip 0.984, RightHip 0.982,
  Femur 0.984 → proposed gate **PASS**.
- Sacrum crops, `checkpoint_best` → `checkpoint_final` (pre-fix best):
  `047` 0.119 → 0.558 (0.974), `154` 0.260 → 0.916 (0.971), `060` 0.419 →
  0.967 (0.975), `074` 0.431 → 0.810 (0.981), `167` 0.587 → 0.932 (0.929),
  `046` 0.659 → 0.892 (0.977), `172` 0.814 → 0.973, `193` 0.870 → 0.971; the
  other 12 are unchanged or slightly better.
- From the training log, the last `checkpoint_best` save was around epoch 58;
  early stopping ended at epoch 83 (patience 25). The pooled validation F1
  (random patches, all bones together) peaked while the sacrum was not yet
  converged, so `checkpoint_best` captured an under-trained sacrum.
- Conclusion: **D6 is a checkpoint-selection defect**, not a failure of the
  trained weights. `checkpoint_final` is still weaker than pre-fix on `047`
  and `074`.
- Proposed fix F8 (awaiting approval): see reply 2026-10-15.
- Agent-typed: `grep`/`tail` of the log; three-way per-crop comparison of
  the CSVs; read of `core.py` 2069–2158 (Fragment validation step, ES); `grep`
  of the best-checkpoint epochs in both training logs.

### 2026-10-15 (Stage 2 evaluation result; G2)

- `scripts/evaluate_stage2_backend.py` finished 17:07: 40 cases × 3
  connectivities, `validation_metrics.csv` (121 lines) and
  `validation_summary.json` written; log ends normally. Pre-fix files are in
  the archive.
- Connectivity 26, pre-fix → retrained:

  | Group | Binary Dice | Instance P | Instance R | Instance F1 | Merges | Splits |
  |---|---|---|---|---|---|---|
  | All 40 | 0.764 → 0.721 | | | 0.737 → 0.854 | 8 → 10 | 115 → 233 |
  | Pelvic 20 | 0.770 → 0.877 | 0.754 → 0.933 | 0.589 → 0.799 | 0.617 → 0.833 | 6 → 9 | 52 → 164 |
  | Femur-only 20 | 0.757 → 0.565 | 0.983 → 0.988 | 0.801 → 0.826 | 0.857 → 0.876 | 2 → 1 | 63 → 69 |

  Failure event (≥ 1 merge or split): 31/40 → 38/40.
- Pelvic: large gain from the hip fix. D6 cases still weak (`030` F1 0.642 →
  0.411, `047` 0.633 → 0.639). Splits rose 52 → 164; part of the pre-fix count
  was hidden because swapped hips fell outside the scored ID range.
- Femur-only: instance metrics unchanged or better, but **binary Dice fell
  0.757 → 0.565**. Checked `255`, `305`, `317`, `357`, `417`, `254`:
  femur-range Dice is unchanged (e.g. `317` 0.991 → 0.990), but the retrained
  pipeline now paints 0.2–1.2 M voxels of pelvic bone, which is visible in
  these scans but not labelled. Binary Dice (`pred > 0` vs `gt > 0`, whole
  volume) counts that as false positive. Expected effect of the D3 fix (Stage A
  no longer learns to suppress visible unlabelled bone). Exception: `357`
  femur-range Dice 0.976 → 0.721 (real femur loss; not yet examined).
- **D7 (new, needs decision):** H1's semantic-control risk
  `one_minus_binary_dice` (configs/stage3_h1.json `risk_contract`) is, for
  femur-only cases, now dominated by unlabelled-bone predictions, not by
  segmentation quality. The instance metrics already score only GT-present
  anatomies; binary Dice does not.
- **G2 verdict: not passed.** The instance predictions are not yet stable:
  the Stage 2 Fragment checkpoint fails the per-bone check (D6), so this CSV
  must not be frozen as the H1 risk source. Re-evaluate after D6 is fixed and
  D7 decided.
- Agent-typed: `tail` of `c0_stage2_evaluate.log`; read of new and archived
  `validation_summary.json`; `wc -l`; read of `instances.py` metric code;
  scratch group/per-case comparison of the two CSVs; scratch femur-range vs
  whole-volume Dice on 6 femur-only cases; `awk` failure-event count; `grep`
  of `configs/stage3_h1.json` and `plan.md` risk contract.

### 2026-10-15 (per-bone Fragment check result)

- User ran `scripts/check_fragment_gate.py` on the retrained and the
  archived pre-fix Fragment `checkpoint_best.pth` (log
  `outputs/logs/fragment_gate.log`; `retrained.csv`, `prefix.csv`,
  `summary.json` under `outputs/stage3_h1/diagnostics/fragment_gate/`). The
  final conda `ERROR` line is only the script's exit code 1 (gate FAIL).

  | Bone | Retrained median / < 0.5 | Pre-fix median / < 0.5 |
  |---|---|---|
  | Sacrum | 0.964 / **4** | 0.976 / 0 |
  | LeftHip | 0.983 / 0 | 0.982 / 0 |
  | RightHip | 0.979 / 0 | 0.977 / 0 |
  | Femur | 0.984 / 0 | 0.984 / 0 |

  Proposed gate: retrained **FAIL**, pre-fix **PASS**.
- Sacrum crops, retrained vs pre-fix Dice: `047` 0.119 vs 0.974, `154` 0.260
  vs 0.971, `060` 0.419 vs 0.975, `074` 0.431 vs 0.981, `167` 0.587 vs 0.929,
  `046` 0.659 vs 0.977, `172` 0.814 vs 0.973, `193` 0.870 vs 0.972; the other
  12 are ≥ 0.936 for both. Failing crops over-predict (pred 1.5–2.4× GT,
  precision 0.10–0.56), i.e. hip bone inside the sacrum crop is segmented.
  The same cases are the ones that failed in validation inference (D6 table);
  `030` passes here (0.936) but failed in the pipeline, where the crop comes
  from the Stage A mask instead of the GT bbox, so the model is fragile on it.
- Conclusion: D6 is in the retrained Fragment checkpoint itself (training
  side). The same crops, preprocessing, and code give 0/20 failures with the
  pre-fix checkpoint. Open: whether the cause is checkpoint selection
  (`checkpoint_best` by pooled EMA F1, F2 now pooling all ranks) or the
  trained weights (warm-start from the fixed Stage A, or seed).
- Added `--checkpoint` to the script (default `checkpoint_best.pth`;
  summary now `summary_<checkpoint>.json`) to score `checkpoint_final.pth`.
  Full suite 46/46.
- Agent-typed: `grep`/`tail` of the gate log; read of `summary.json`;
  per-crop comparison of the two CSVs; `python3` edit of the script; full
  unit suite.

### 2026-10-15 (per-bone Fragment check script)

- Added `scripts/check_fragment_gate.py` (user-typed, 1 GPU). For each
  Fragment checkpoint given as `--model NAME=RESULTS_ROOT` (repeatable;
  default the config's results root), it predicts the 80 Dataset702 fold-0
  validation crops as trained (bone-LUT crops from `nnUNet_raw`, same
  `build_predictor` / `predict_logits_full` / `resample_label_map_to_original`
  as validation inference) and scores foreground (`1 − p(bg) > 0.5`) against
  the crop label: Dice, recall, precision per crop, grouped by bone. Writes
  `<out>/<NAME>.csv` and `<out>/summary.json` under
  `outputs/stage3_h1/diagnostics/fragment_gate/`.
- **Proposed Fragment gate (not yet approved)**, mirroring G1: every bone
  median foreground Dice ≥ 0.90 and at most 2 crops < 0.5. Exit code 0 = PASS.
  Thresholds are flags; the rule is to be approved after the first results.
- Purpose: compare the retrained and the archived pre-fix checkpoint per bone
  (does the retrained model fail sacrum crops in its own validation; did the
  pre-fix one not), and then check every member Fragment checkpoint before
  inference.
- Tests: `FragmentGateTests` (3): foreground metrics, gate fails on one bad
  bone, crop-name parsing and the 80-crop split. Full suite 46/46.
- Agent-typed: Read of `inference.py` (`build_predictor`,
  `predict_logits_full`, `resample_label_map_to_original`, Stage B loop);
  `py_compile`; full unit suite; `--help`; `nvidia-smi`.

### 2026-10-15 (D6 diagnosis result)

- User ran `scripts/diagnose_d6_sacrum.py` (cases `030`, `047`, `060`, control
  `004`; 15:48–15:58). Sacrum voxels per step (original-grid estimates):

  | Case | Model fg (`1−p(bg)>0.5`) | After decode | After resample | Painted | GT sacrum |
  |---|---|---|---|---|---|
  | `030` | 668,880 | 813,409 (pp) | 669,506 | 74,880 | 352,258 |
  | `047` | 469,875 | 672,633 (pp) | 469,252 | 54,913 | 323,733 |
  | `060` | 808,610 | 974,344 (pp) | 807,373 | 119,070 | 395,305 |
  | `004` (control) | 216,026 | 314,441 (pp) | 215,467 | 215,401 | 210,162 |

  Decode and resampling keep every voxel; ~89% is removed at the paint step,
  which only drops voxels inside other anatomies' Stage A masks (Sacrum is
  painted first). Hips/Femur painted ≈ resampled in all four cases.
- Overlay of the decoded sacrum crop on GT (`030`, `047`; identity direction,
  so array overlay is valid; `060`/`004` are RAS-flipped and were not
  overlaid): the largest Stage B instances lie on the hips — `030`: 278k
  (99% LeftHip), 223k (98% RightHip), 97k (97% RightHip), sacrum only from a
  37k instance; `047`: 220k (99% RightHip), 179k (98% LeftHip), 40k (98%
  Sacrum). GT-sacrum recall of all Stage B foreground: 0.133 / 0.145. Flipping
  the crop on any axis does not improve the overlap (no orientation error).
- Conclusion: the loss is in the **retrained Fragment model's output**, not in
  decode, resampling, or painting. Ruled out: training labels (sacrum crop
  labels only sacrum fragments: `002`, `004`, `008` checked), dataset change
  (raw/preprocessed Dataset702 dated 2026-09-22/23, before the pre-fix run),
  case map (V308 → V302 does not call `_load_case_labeled_map`), warm-start
  mechanics (both runs: 120 keys loaded, 10 seg keys skipped). Remaining
  differences between the runs: warm-start source (fixed Stage A), seed, and
  F2 (checkpoint_best now chosen on all 80 validation crops instead of rank
  0's half). The Fragment validation metric is a pooled instance F1 with no
  per-anatomy split, so this failure is invisible in training (0.687 vs 0.700).
- Consequence for members: member Fragment runs use the same recipe, so the
  same failure can occur and would not be seen until inference. C-2.2 stays on
  HOLD.
- Agent-typed: `grep` of `outputs/logs/d6_diagnosis.log`; Read of
  `d6_report.json`; Read of `inference.py` 1840–1954 (paint/confine);
  scratch overlay script (`decoded_crop` vs GT, with axis flips); scratch read
  of Dataset702 `dataset.json` and 4 training crops/labels; `ls`
  timestamps of Dataset702; `tail` of new and archived Fragment training
  logs; `grep` of `_load_case_labeled_map` and trainer classes in `core.py`.

### 2026-10-15 (D6 diagnosis script)

- Reading `inference/inference.py`: our validation inference runs
  `PENGWIN_AFFINITY_DECODE=1`, so every anatomy uses `decode_affinity_agglo`;
  the Sacrum-only size-ratio (0.10) / min-250-voxel merge exists only in the
  fusion and non-affinity branches and is **not** on our path. Candidate loss
  points: Stage B foreground itself, the affinity decode, resampling, and the
  paint step (`PENGWIN_CONFINE_TO_MASK=1` forbids painting into other
  anatomies' Stage A masks).
- Refactored `scripts/predict_stage2_backend.py`: backend setup moved into
  `load_configured_backend(config)` (same environment and settings; `main`
  calls it). Added `scripts/diagnose_d6_sacrum.py` (user-typed, 1 GPU): reruns
  that exact pipeline on chosen cases with read-only hooks on the backend's
  `log`, `bbox_from_mask`, and `resample_label_map_to_original`, plus the
  pipeline's own `PENGWIN_DUMP_PROBS`. Per anatomy it records model foreground
  (`1 − p(bg) > 0.5`, ABBC argmax class counts, mean short-range affinity),
  voxels/instances after decode and after resampling, and painted voxels; for
  the sacrum it adds GT coverage. Writes only under
  `outputs/stage3_h1/diagnostics/d6/` (report JSON, Sacrum 13-channel dump,
  Sacrum decoded arrays, per-case prediction). Default cases: `030`, `047`,
  `060` (bad) and `004` (control, sacrum coverage 0.981).
- Tests: `D6DiagnosisRecorderTests` (2): step attribution to the right
  anatomy with the original log still called; foreground counts. Full suite
  43/43. `--help` runs.
- Stage 2 evaluation was still running (CPU) at the time.
- Agent-typed: `sed -n`/Read of `inference/inference.py` (Stage B loop, lines 1749–1983) and of `predict_stage2_backend.py`; `grep -n` for hook functions; scratch check of sacrum coverage for `004`, `008`, `053`; `python3 -m py_compile`; full unit suite; `scripts/diagnose_d6_sacrum.py --help`; `ps`; `nvidia-smi`.
- User-typed command prepared: the D6 diagnosis run (see reply).

### 2026-10-15 (Stage 2 inference done; C-2.1 G1; D6 found)

- Stage 2 inference rerun with F7 (`outputs/logs/c0_stage2_predict_r2.log`):
  `validation predictions complete: 40/40`; manifest 38 `COMPLETE` + 2
  `VALID_RESUME` (`004`, `008`); 0 `emit zero`, 0 budget lines, 0
  `device=cpu`. Per-case time 49.8–1,479.3 s (median 554.9 s, above the old
  480 s budget). Evaluation not yet run.
- Final-prediction per-bone GT coverage on the 20 pelvic cases (new script
  `outputs/stage3_h1/diagnostics/bone_recall_stage2_retrained.py`), median
  any-foreground / correct-label (cases any-fg < 0.5), pre-fix → retrained:
  Sacrum 0.982/0.982 (0) → 0.959/0.959 (3); LeftHip 0.504/0.244 (10) →
  **0.985/0.984 (0)**; RightHip 0.734/0.718 (4) → **0.984/0.980 (0)**.
  Hips fixed end to end; Sacrum regressed (D6).
- C-2.1 (`member_02` Anatomy, seed 31002, GPUs 3,4) finished: map loaded (80
  cases), axis-2 mirror disabled, `debug.json` `(0, 1)`; early stop at epoch 96
  (best EMA 0.9700 at 71); `Training done`; 40-case validation complete; no
  DDP errors; checkpoints pass `unzip -t`. **G1: PASS** — Sacrum 0.975 (0),
  LeftHip 0.989 (0), RightHip 0.986 (0), Femur 0.986 (0).
- Agent-typed: `grep -c` checks and `grep -oE "pipeline complete in ..."` on the inference log; `ls ... | wc -l`; manifest status count; `ls -la` of Stage 2 metrics paths (absent); `ls -la`, `grep`, `unzip -tq`, `scripts/check_anatomy_gate.py` on member_02 Anatomy; `conda run ... python outputs/stage3_h1/diagnostics/bone_recall_stage2_retrained.py`; scratch scripts for low-sacrum cases (new vs archived predictions; Stage A validation output — its `060` value 0.033 contradicts G1 and is likely an orientation mismatch in the scratch comparison, not used); `awk`/`grep` extraction of `[Sacrum]` log lines per case; `grep` of both Fragment training logs.

### 2026-10-15 (D5 fix F7)

- User approved the D5 fix. Implemented F7 as in the fix table; removed the
  now-unused `importlib` imports from both inference wrappers.
- Regenerated `patches/pengwin2026-task1-abbc_on_ccc625f.patch` (3 files;
  `git apply --check -R` passes). New SHA-256: `inference.py`
  `01073e7b10b81311e88e5b5f5b9733fe88712da94b0c5a3e6b501c2ee9484e2e`, patch
  `4fb8420b08927511662838568afab9ff24ab6b6237b83bffcc45ff05befd9ba9`.
  `core.py` and `model.py` hashes are unchanged, so training code (including
  the running C-2.1) is unaffected. Updated the `inference.py` hash in
  `member_01_handover.md`; the earlier `7483138c…` value is superseded.
- Agent-typed: `sed -n` reads of `inference/inference.py`, `predict_stage2_backend.py`, `predict_stage3_member.py`; `grep -n "emit zero|V03_USE_TIME_BUDGET|TIME_BUDGET|..."`; `sed -i '/^import importlib$/d' scripts/predict_stage2_backend.py scripts/predict_stage3_member.py`; `python3 -m py_compile <4 edited files>`; `conda run ... python -B -m unittest tests.test_stage3_backend_fixes -v`; `conda run ... python -B -m unittest discover -s tests`; `git -C baselines/pengwin2026-task1-abbc -c core.fileMode=false diff > patches/...patch`; `git -C ... apply --check -R ...`; `sha256sum <3 files + patch>`; `ps -eo pid,etime,args | grep "stage3_h1.py train"`.
- User-typed commands prepared: Stage 2 inference rerun (`--resume`), the
  log checks, and evaluation (see reply / section C).

### 2026-10-15 (Stage 2 inference attempt 2: D5 blocks inference on GPU)

- CUDA restored (agent check: `torch.cuda.is_available()` True; GPU 1 visible
  again, admin confirmation pending). The user moved the invalid CPU outputs
  to `outputs/archive/2026-10-14_invalid_cpu_inference/` and reran inference
  on GPU 2 (`device=cuda` in the log).
- Case `004`: 97.7 s, labels `[0,1,2,51,52,54,101,102,103,151]`. Case `008`:
  316.6 s, 12 labels. Neither logged `emit zero`; both valid.
- Case `030` (285×512×512): the Anatomy stage took ~671 s; `[Sacrum] HARD time
  budget hit at 671s, emit zero for this and remaining anatomies`; all-zero →
  `RuntimeError: 030: backend returned an all-zero prediction`.
- Contention at the time: load average ~20–30 on 64 cores (another user's
  process at 280% CPU, several at ~85%; our C-2.1 ranks at ~145% each). The
  pipeline resamples Anatomy probabilities on the CPU in float64 (default;
  `PENGWIN_LOWMEM_RESAMPLE` off). Per-case time therefore varies widely
  (98–671+ s) with server load.
- Conclusion: with the 480/540 s challenge budget, validation inference cannot
  reliably complete on this shared server, and a budget hit after the first
  anatomy would save a partially zeroed case as `COMPLETE`. D5 is now blocking.
- C-2.1 (member_02 Anatomy) is running on GPUs 3,4 (started ~07:15).
- Agent-typed: `wc -l`, `grep -nE "device=|CUDA|emit zero|budget|pipeline complete|write_label_map|Traceback|Error"`, `tail -25` on `outputs/logs/c0_stage2_predict.log`; `ls outputs/stage2_backend/validation_predictions`; `ls outputs/archive`; `uptime`; `nproc`; `nvidia-smi --query-gpu=...`; `ps -eo pid,user,etime,pcpu,args --sort=-pcpu | head -8`; `ls -la outputs/logs`; `tail -2 outputs/logs/c2_member_02_anatomy.log`; `grep -n "lowmem-resample|PENGWIN_LOWMEM|stock float64" baselines/pengwin2026-task1-abbc/inference/inference.py`.

### 2026-10-14 (Stage 2 inference attempt: CUDA down, D5 found)

- User ran Stage 2 validation inference on `CUDA_VISIBLE_DEVICES=0`
  (`outputs/logs/c0_stage2_predict.log`). PyTorch warned `CUDA driver
  initialization failed`; nnU-Net fell back to `device=cpu`. Case `004` took
  654 s and hit the hard time budget at RightHip; case `008` hit it at Sacrum,
  returned all-zero, and the script raised `RuntimeError: 008: backend
  returned an all-zero prediction`.
- On disk: `outputs/stage2_backend/validation_predictions/004.mha` (missing
  RightHip) and a manifest listing `004` as `COMPLETE`. Both are invalid and
  must not be resumed from.
- Agent check: `torch.cuda.is_available()` is `False` with
  `CUDA_VISIBLE_DEVICES=0` and with `=2`. CUDA is unusable server-wide,
  consistent with the GPU 1 (`0000:1B:00.0`) fault. No GPU work (inference,
  C-2.1, member_01 handover run) can run until the administrator restores
  the GPU/driver. C-2.1 had not been started.
- Recorded D5 (time-budget zeroing accepted as complete).
- Agent-typed: `wc -l`, `sed -n 1,12p`, `grep -niE "traceback|error|...|zero|COMPLETE"`, `tail -15` on `outputs/logs/c0_stage2_predict.log`; `ls -la outputs/stage2_backend/validation_predictions`; `python3 -c <print inference manifest>`; `CUDA_VISIBLE_DEVICES={0,2} conda run -n exchange-stage2 python -c "import torch; print(torch.cuda.is_available())"`; `ps -eo ... | grep`; `ls outputs/logs`; `grep -rn "HARD time budget|TIME_BUDGET|time_budget" baselines/pengwin2026-task1-abbc/inference/*.py`.

### 2026-10-14 (C-0.2 result; GPU 1 fault)

- C-0.2 (Stage 2 Fragment, seed 31000, GPUs 5,6) finished. Both ranks logged
  the trusted warm-start from the new C-0.1 `checkpoint_best.pth`
  (`loaded=120 skipped_seg=10`, 2×). 83 epochs; `Early stopping: no instance F1
  improvement for 25 epochs. Best EMA F1=0.6869. Stopping at epoch 83.`;
  `Training done` 06:17:11; `perform_actual_validation SKIPPED` as designed.
  No traceback, NCCL error, or timeout; no leftover processes. Last epoch:
  IoU-F 0.689, recall 0.724, precision 0.643, F1 0.681. `checkpoint_best.pth`
  (03:21) and `checkpoint_final.pth` pass `unzip -t`; `debug.json` mirroring
  `(0, 1, 2)`. F2 confirmed on a full-length run: the two ranks stopped together.
  These are per-epoch validation-patch proxies, not the G2 instance metrics.
- **GPU 1 fault (2026-10-14 07:30):** `nvidia-smi` reports `Unable to determine
  the device handle for GPU 0000:1B:00.0: Unknown Error` for GPU 1; all other
  GPUs respond. None of this project's runs was using GPU 1 at the time
  (C-0.1 had finished on GPUs 0,1 on 2026-10-13 15:25). Needs the server
  administrator; do not schedule GPU 1.
- No member run had been launched; GPUs 0, 2–6 idle, GPU 7 ~7.3 GB used.
- Agent-typed: `ls -la <Fragment fold_0>`; `grep -nE "<criteria>" outputs/logs/c0_stage2_fragment.log`; `grep -c "Epoch time"`; `grep "[instance]"`; `grep Yayy`; `unzip -tq` on both checkpoints; `grep -o inference_allowed_mirroring_axes`; `ps -eo pid,etime,args | grep ...`; `nvidia-smi -L`; `nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader`; `nvidia-smi -i <0..7> --query-gpu=pci.bus_id,memory.used --format=csv,noheader`; `grep -o "loaded=... skipped_seg=..."`; `ls outputs/logs`; `date`.

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
