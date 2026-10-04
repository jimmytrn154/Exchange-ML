# Stage 3 — FA-IPD + H1 Round-0 benchmark

Status: **BLOCKED (partial user-run training complete; DDP validation/early-stop defect must be fixed before H1 execution continues)**

Opened: 2026-09-26

Last updated: 2026-10-04

Stage 2 prerequisite: **PASS — internal research backend complete.** A fresh
read-only `python3 scripts/validate_stage2_artifacts.py` run on 2026-09-26
confirmed both non-empty checkpoints, 40/40 frozen validation predictions, and
120 metric rows. This is the Stage 2 internal gate, not a claim of hidden-test
or expert anatomical validity.

## Configuration

- Frozen config: `configs/stage3_h1.json` (SHA-256 `64501bae8a620fd0b9a0eeb3f3d0f8951cf766bac965135f0ff5d6fd09098106`).
- Analysis set: the frozen 40-case PENGWIN validation split; test remains untouched.
- Ensemble: `M=3`, model seeds `31001`, `31002`, `31003`; rank-local stochastic streams use `seed + local DDP rank`, deterministic cuDNN kernels are requested, and cuDNN benchmarking is disabled.
- Stage 2 checkpoints are not reused because their model seed was not explicitly recorded.
- Each member retrains the unchanged two-stage ABBC/STU-Net backend.
- Foreground probability: Stage-B `1-p(background)`, linear resampling to the input grid, voxelwise maximum across routed anatomies, float32 on disk.
- Shared score ROI: union of the three cleaned member foreground maps.
- Prediction cleanup: 26-connectivity, remove per-ID components below `1000 mm3`, no morphology.
- H1 risks: frozen Stage 2 single-model `1-Instance F1`, normalized merge+split count, and `1-Dice`.

## What was changed

- Added explicit process seeding to the Stage 2 trainer wrapper, including spawned DDP ranks, without changing the optimizer, schedule, loss, architecture, split, or preprocessing.
- Added optional genuine Stage-B foreground-probability return to the preserved ABBC inference path. The default Stage 2 return remains the deterministic label map.
- Added strict per-member validation inference with geometry/range checks, checkpoint fingerprints, and resumable manifests; scoring rejects member/seed/case/probability-contract drift and records manifest/checkpoint hashes.
- Added GT-free scoring for mean/p95 entropy, pairwise semantic Dice disagreement, foreground Jaccard disagreement, union-NVI, fragment-count variance, anatomy-restricted Hungarian disagreement, and FA-IPD.
- Added retrospective risk–coverage/AURC, top-10%/20% enrichment, Spearman, controlled Spearman, tie-safe confounder stratification, and selection of a true low-entropy/high-FA-IPD merge/split failure example (explicitly null if none qualifies).
- Separated acquisition scoring from retrospective evaluation: the score program does not read ground truth or risk artifacts, and records `ground_truth_loaded=false`.

## Validation performed by the agent

- JSON parsing and Python syntax compilation passed.
- The final full Stage 1–3 unit suite passed: 26/26 tests; the focused Stage 3 suite contains seven passing tests.
- Ruff was attempted in the pinned environment but is not installed (`ruff: command not found`), so lint is not reported as passing. Python syntax compilation passed.
- Pinned environment preflight passed with `nnunetv2==2.5.1` and both ABBC trainer imports.
- A 40-case synthetic file-level smoke passed through GT-free scoring and retrospective evaluation, producing 40 score rows, 24 score/risk comparison rows, risk–coverage data, stratified analysis, a PNG figure, and a pending-scientific-assessment summary.
- The first tiny synthetic fixture failed exactly because every component was below the frozen `1000 mm3` cleanup threshold; only the synthetic voxel spacing was corrected. The scientific method was not relaxed.
- A second attempted smoke invocation used unsupported direct path flags and was rejected by the config-only CLI; the corrected run used a temporary config.

These checks establish implementation behavior, not H1 results.

## Observed user-run progress

Read-only artifact inspection on 2026-10-04 found:

| Member | Anatomy | Fragment | Interpretation |
|---|---|---|---|
| `member_01` | Missing | Missing | Training has not produced a checkpoint in the current workspace. |
| `member_02` | `checkpoint_best.pth` and `checkpoint_final.pth` present | `checkpoint_best.pth` and `checkpoint_latest.pth` present; no final checkpoint | Fragment training failed after one DDP rank early-stopped and the other rank entered the next epoch. The best checkpoint archive is structurally readable, but its scientific acceptance is unresolved because checkpoint selection used rank-local validation metrics. |
| `member_03` | `checkpoint_best.pth` and `checkpoint_final.pth` present | `checkpoint_best.pth` and `checkpoint_final.pth` present | The fragment command reached early stopping at epoch 85 and logged `Training done`. Code inspection shows that the same rank-local validation path was used, so checkpoint-selection validity must be resolved together with member 02. |

All eight present Stage 3 checkpoint archives for members 02 and 03 passed
read-only ZIP integrity checks. `scripts/stage3_h1.py check` reports Anatomy and
Fragment checkpoints for both members and no checkpoints for member 01. This
presence check does not establish correct DDP validation aggregation or H1
scientific validity.

The first Fragment warm-start attempt for member 03 was rejected by PyTorch's
restricted checkpoint loader because the locally generated Anatomy checkpoint
contains a NumPy scalar. The repository's explicit trusted-source opt-in,
`PENGWIN_ALLOW_UNSAFE_TORCH_LOAD=1`, was then scoped to the user-run Fragment
commands. This opt-in is appropriate only for the locally generated, trusted
checkpoints and is not a general setting.

No Stage 3 member-level `validation_predictions` or
`validation_probabilities` directories, complete member inference manifests,
uncertainty table, H1 comparison table, risk-coverage outputs, or H1 summary
were observed. The `.mha` files under the Anatomy trainer's internal
`fold_0/validation` directories are training validation artifacts; they are
not the required Stage 3 two-stage member inference products.

## Blocking issue

Member 02 Fragment training exposed a defect in the custom DDP validation and
early-stopping path. Rank 1 reported no F1 improvement for 25 epochs and exited
at epoch 70 while rank 0 entered epoch 70. Rank 0 then timed out after 1,800
seconds in NCCL `ALLREDUCE`, and the user-run command ended with
`ProcessExitedException`/`SIGABRT`.

The standard nnU-Net `on_validation_epoch_end` gathers validation statistics
across all ranks. The custom Fragment override in
`baselines/pengwin2026-task1-abbc/code_task1/core.py` instead computes F1, EMA,
best-checkpoint selection, and early stopping independently on each rank. This
explains the divergent stop decisions. Re-running the unchanged command is not
a fix: the wrapper starts fresh and can reproduce the same failure. Silently
switching to one GPU would change the frozen two-GPU training protocol and is
not allowed.

Member 02's `checkpoint_best.pth` was written at epoch 61, before the timeout,
and its archive integrity check passed. The downstream Stage 3 code references
`checkpoint_best.pth`, not `checkpoint_final.pth`. It is therefore a readable
recovery artifact, but it is not accepted as a final scientific artifact while
the rank-local checkpoint-selection defect remains unresolved.

## Historical runtime handoff observation

At the 2026-09-26 read-only check, GPUs 0, 1, 3, 4, 5, and 6 were idle (2 MiB each); GPUs 2 and 7 were busy. This is a snapshot, so recheck before reserving GPUs. The same 18 orphaned `multiprocessing.spawn` workers plus one resource tracker from the old `exchange-stage2` environment still had `PPID=1` and retained about 20 GiB RSS. They had no reported GPU allocation and were not terminated by the agent. Have their owner verify and clear these exact stale processes before launching the concurrent jobs.

## Gateway

**BLOCKED.** Partial real training artifacts now exist, but the M=3 ensemble is
incomplete, the Fragment DDP validation/early-stop path is defective, and no
required two-stage inference or H1 result exists. Fix and validate the main
method; do not substitute a single-GPU run, reduce the ensemble, or promote the
readable member 02 checkpoint without resolving the defect. The manual gateway
cannot be evaluated until all three corrected members complete inference and
the frozen H1 analyses exist. Proceed to H2 only if FA-IPD then shows meaningful
advantage in at least two of AURC, top-k enrichment, and controlled association,
without collapsing to the Jaccard-only term or fragment-size/count confounding.

## Agent-typed commands

```bash
cat __docs__/rule.md
conda run -n exchange-stage2 python scripts/stage3_h1.py check
python3 -m py_compile scripts/stage2_backend.py scripts/stage3_h1.py scripts/predict_stage3_member.py scripts/score_stage3_h1.py scripts/evaluate_stage3_h1.py src/exchange_ml/stage3_h1.py tests/test_stage3_h1.py
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest discover -s tests -v
conda run --no-capture-output -n exchange-stage2 env PYTHONPATH=src:. python -B -m unittest discover -s tests -p 'test_stage3_h1.py' -v
conda run --no-capture-output -n exchange-stage2 python scripts/score_stage3_h1.py --config /tmp/exchange-stage3-smoke40.KO63G2/stage3_smoke.json
conda run --no-capture-output -n exchange-stage2 python scripts/evaluate_stage3_h1.py --config /tmp/exchange-stage3-smoke40.KO63G2/stage3_smoke.json
python3 scripts/validate_stage2_artifacts.py
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
free -h
ps -eo pid,ppid,etime,rss,args | rg 'multiprocessing.spawn|resource_tracker|stage3_h1.py|stage2_backend.py train|predict_stage3_member.py' | head -45
ss -ltn '( sport = :29601 or sport = :29602 or sport = :29603 )'
conda run -n exchange-stage2 ruff check scripts/validate_stage2_artifacts.py scripts/stage2_backend.py scripts/stage3_h1.py scripts/predict_stage3_member.py scripts/score_stage3_h1.py scripts/evaluate_stage3_h1.py src/exchange_ml/stage3_h1.py tests/test_stage3_h1.py
```

The `/tmp` commands used generated arrays only. They did not train a model or read validation images/labels during acquisition scoring.

Additional lightweight, read-only update checks performed on 2026-10-04:

```bash
cat __docs__/plan.md
cat __docs__/rule.md
cat __docs__/reports/stage3_h1.md
git status --short
find outputs/stage3_h1 -type f \( -name 'checkpoint_*.pth' -o -name '*.mha' -o -name '*.csv' -o -name '*manifest*.json' -o -name 'h1_summary.json' -o -name 'risk_coverage.png' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS\t%s\t%p\n' | sort
for f in $(find outputs/stage3_h1/members/member_02 outputs/stage3_h1/members/member_03 -type f -name 'checkpoint_*.pth' | sort); do unzip -t "$f"; done
for f in $(find outputs/stage3_h1/members/member_02 outputs/stage3_h1/members/member_03 -type f -name 'training_log_*.txt' | sort); do rg -n 'Early stopping|Training done|perform_actual_validation SKIPPED|Yayy! New best' "$f"; done
rg -n 'def on_validation_epoch_end|all_gather_object|checkpoint_best|checkpoint_final' baselines/PENGWIN_Challenge/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py baselines/pengwin2026-task1-abbc/code_task1/core.py scripts/predict_stage3_member.py scripts/stage2_backend.py
PYTHONDONTWRITEBYTECODE=1 conda run -n exchange-stage2 python scripts/stage3_h1.py check
git diff --check -- __docs__/reports/stage3_h1.md
for member in member_01 member_02 member_03; do for name in validation_predictions validation_probabilities; do artifact_dir="outputs/stage3_h1/members/$member/$name"; if [ -d "$artifact_dir" ]; then find "$artifact_dir" -type f | wc -l; else echo "MISSING $artifact_dir"; fi; done; done
```

The first form of the final directory loop used `path` as a zsh variable,
which temporarily replaced `PATH` and caused only the trailing `rg` command to
fail with `command not found`. The corrected `artifact_dir` form above passed;
the failed diagnostic made no file changes.

## User-typed commands

The following trusted-checkpoint Fragment commands were prepared for and run
by the user. They are recorded as execution history, not as commands for the
agent to run:

```bash
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=5,6 MASTER_PORT=29603 PENGWIN_ALLOW_UNSAFE_TORCH_LOAD=1 python scripts/stage3_h1.py train --stage fragment --member member_03 --device cuda --num-gpus 2
conda run --no-capture-output -n exchange-stage2 env CUDA_VISIBLE_DEVICES=2,4 MASTER_PORT=29602 PENGWIN_ALLOW_UNSAFE_TORCH_LOAD=1 python scripts/stage3_h1.py train --stage fragment --member member_02 --device cuda --num-gpus 2
```

The member 03 command completed. The member 02 command failed with the DDP
divergent-early-stop/NCCL timeout described above. Do not rerun either command
unchanged as a scientific remedy. No new full-run command is prepared until
the DDP aggregation defect is fixed and its lightweight validation passes.

The original frozen execution handoff follows for provenance; it is currently
superseded by the blocker above.

Run from the repository root. This schedule keeps the frozen two-GPU DDP
training and member seeds. The `CUDA_VISIBLE_DEVICES` and `MASTER_PORT`
values only assign separate hardware and DDP rendezvous ports; do not edit the
frozen config or change `--num-gpus 2`. First run the lightweight preflight
and recheck GPU ownership:

```bash
conda run -n exchange-stage2 python scripts/stage3_h1.py check
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

If the three GPU pairs are still available, start **these three blocks at the
same time in three separate terminals**. Within each block, the second command
runs only if anatomy training exits successfully; Fragment warm-starts from that
same member's Anatomy checkpoint. Each member writes to its own results tree.
The separate `MASTER_PORT` values prevent concurrent DDP jobs from sharing a
rendezvous port; ensure these ports are free on the host.

Terminal 1 — member 01, GPUs 0–1:

```bash
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29601 conda run --no-capture-output -n exchange-stage2 python scripts/stage3_h1.py train --stage anatomy --member member_01 --device cuda --num-gpus 2 &&
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29601 conda run --no-capture-output -n exchange-stage2 python scripts/stage3_h1.py train --stage fragment --member member_01 --device cuda --num-gpus 2
```

Terminal 2 — member 02, GPUs 3–4:

```bash
CUDA_VISIBLE_DEVICES=3,4 MASTER_PORT=29602 conda run --no-capture-output -n exchange-stage2 python scripts/stage3_h1.py train --stage anatomy --member member_02 --device cuda --num-gpus 2 &&
CUDA_VISIBLE_DEVICES=3,4 MASTER_PORT=29602 conda run --no-capture-output -n exchange-stage2 python scripts/stage3_h1.py train --stage fragment --member member_02 --device cuda --num-gpus 2
```

Terminal 3 — member 03, GPUs 5–6:

```bash
CUDA_VISIBLE_DEVICES=5,6 MASTER_PORT=29603 conda run --no-capture-output -n exchange-stage2 python scripts/stage3_h1.py train --stage anatomy --member member_03 --device cuda --num-gpus 2 &&
CUDA_VISIBLE_DEVICES=5,6 MASTER_PORT=29603 conda run --no-capture-output -n exchange-stage2 python scripts/stage3_h1.py train --stage fragment --member member_03 --device cuda --num-gpus 2
```

After **both** training commands for a member finish, start its inference on
one GPU from its now-free pair. The three inference commands may run together
in separate terminals if all three pairs are free; each writes only to its own
member directory. A finished member may start inference while other members
are still training, provided its inference GPU is not used by those jobs.

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n exchange-stage2 python scripts/predict_stage3_member.py --member member_01 --resume
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n exchange-stage2 python scripts/predict_stage3_member.py --member member_02 --resume
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n exchange-stage2 python scripts/predict_stage3_member.py --member member_03 --resume
```

Run these **sequentially** after all three inference commands exit successfully:

```bash
conda run -n exchange-stage2 python scripts/score_stage3_h1.py
conda run -n exchange-stage2 python scripts/evaluate_stage3_h1.py
```

All training, full inference, scoring, and evaluation above are **user-typed**
under `__docs__/rule.md`; the agent has not run them. Parallel jobs share the
same read-only preprocessed data and may contend for CPU, RAM, or disk, so
monitor throughput and use fewer concurrent pairs if needed. Never run two
jobs on the same GPU or two jobs for the same member at once. If any training
or inference command fails, resolve that member before scoring; do not reduce
the ensemble or reuse a Stage 2 checkpoint. `score_stage3_h1.py` requires all
three complete 40-case member manifests, and `evaluate_stage3_h1.py` requires
the completed score file and manifest.

## Expected artifacts

- `outputs/stage3_h1/members/member_0{1,2,3}/nnUNet_results/.../checkpoint_best.pth`
- `outputs/stage3_h1/members/member_0{1,2,3}/validation_predictions/*.mha`
- `outputs/stage3_h1/members/member_0{1,2,3}/validation_probabilities/*.mha`
- `outputs/stage3_h1/uncertainty_scores.csv`
- `outputs/stage3_h1/uncertainty_scoring_manifest.json`
- `outputs/stage3_h1/h1_comparison.csv`
- `outputs/stage3_h1/risk_coverage.csv`
- `outputs/stage3_h1/stratified_analysis.csv`
- `outputs/stage3_h1/risk_coverage.png`
- `outputs/stage3_h1/h1_summary.json`

## Next step

Fix the custom Fragment validation path so all DDP ranks use the same globally
aggregated metrics and stop decision, then validate that fix with lightweight
tests before preparing any further user-run training command. After corrected
M=3 training and real inference complete, run the frozen scoring/evaluation,
review the H1 table and figure manually, and record PASS or FAIL without
automatic promotion to Stage 4.
