# Stage 3 — FA-IPD + H1 Round-0 benchmark

Status: **BLOCKED — the 2026-10-05 H1 FAIL is confounded by backend defects
found on 2026-10-09; gateway decision suspended pending user-approved fixes.**

Opened: 2026-09-26. H1 assessed: 2026-10-05 (Asia/Ho_Chi_Minh). Diagnosis
added: 2026-10-09.

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

The checks above established implementation behavior before the real H1 run.

## Real H1 result and manual gateway assessment

The three seeded members each produced 40 validation instance maps, 40
foreground-probability maps, and a complete inference manifest. The full
GT-free score file has 40 rows. Retrospective evaluation produced 24
score/risk comparisons, 960 risk-coverage rows, 288 stratified-analysis rows,
the risk-coverage figure, and `h1_summary.json`. The scoring CSV, scoring
manifest, frozen Stage 3 config, and frozen Stage 2 risk CSV match the SHA-256
values recorded in the manifests and summary. The generated summary retains
`PENDING_SCIENTIFIC_ASSESSMENT`; the decision here is the manual assessment.

| Structural risk | Score | AURC (lower is better) | Controlled Spearman | Top-10% / top-20% failure enrichment |
|---|---|---:|---:|---:|
| `1 - Instance F1` | FA-IPD | 0.1866 | 0.1467 | 0.968 / 1.129 |
| `1 - Instance F1` | Foreground Jaccard | 0.1865 | 0.1473 | 0.968 / 1.129 |
| `1 - Instance F1` | Mean entropy | 0.1911 | 0.1812 | 0.968 / 1.129 |
| `1 - Instance F1` | P95 entropy | 0.1742 | 0.1415 | 0.968 / 0.968 |
| Normalized merge + split | FA-IPD | 0.8219 | -0.3349 | 0.968 / 1.129 |
| Normalized merge + split | Foreground Jaccard | 0.8447 | -0.3394 | 0.968 / 1.129 |
| Normalized merge + split | Mean entropy | 0.8078 | -0.3053 | 0.968 / 1.129 |

FA-IPD gives no meaningful advantage in at least two required dimensions:
top-k enrichment is tied with Jaccard and mean entropy, controlled association
does not improve on the relevant baselines, and AURC is mixed. The FA-IPD and
Jaccard scores have Spearman rank correlation 0.998 across the 40 cases; their
top-4 sets are identical and their top-8 sets overlap in seven cases. This
does not establish a structural benefit beyond foreground disagreement. The
failure prevalence is 31/40, so the top-10% enrichment of 0.968 is below the
cohort prevalence. Case `076` meets the predeclared illustrative-example rule,
but one example does not reverse the aggregate gateway result.

**Manual gateway decision (2026-10-05): FAIL.** This was recorded as a
completed negative H1 result. The approved Stage 4 prerequisite is not met. No
Stage 4 AL pilot was implemented or run. The 2026-10-09 diagnosis below shows
that this result does not isolate FA-IPD, so it is not treated as a valid
negative test of the method.

## Post-hoc diagnosis (2026-10-09)

All checks below are read-only. Ground truth was read only retrospectively for
debugging; no score or acquisition rule was changed. Evidence and scripts are in
`outputs/stage3_h1/diagnostics/`. The member prediction volumes were read from
the copy at `/mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage3_h1/members/`,
which produced the H1 artifacts in this workspace.

### Observed facts

1. **FA-IPD reduces to foreground Jaccard.** In `uncertainty_scores.csv`,
   member foreground Jaccard disagreement has median 0.365 (range 0.021–0.619).
   The coverage-weighted NVI term (`fa_weighted_partition_contribution`) has
   median 0.0034 and is a median 1.1% of FA-IPD (max 62.5%). The FA-IPD/Jaccard
   Spearman correlation is 1.00 to two decimals (0.998 above).
2. **The foreground disagreement is whole-bone presence, not fragment
   partition.** The 40-case validation set has 20 pelvic and 20 femur-only
   cases. For the 20 pelvic cases, the median fraction of each GT bone covered
   by predicted foreground is:

   | Model | Sacrum | Left hip | Right hip |
   |---|---:|---:|---:|
   | Stage 2 single model | 0.98 | 0.51 (10/20 cases < 0.5) | 0.74 (4/20) |
   | `member_01` | 0.97 | 0.24 (13/20) | 0.86 (2/20) |
   | `member_02` | 0.98 | 0.87 (4/20) | 0.73 (8/20) |
   | `member_03` | 0.97 | 0.73 (7/20) | 0.66 (7/20) |

   Coverage by the correctly labelled anatomy block is lower still (Stage 2
   left hip median 0.25; `member_02` right hip 0.46). Member coverage of one
   bone spans at least 0.5 in 16/20 cases for the left hip and 9/20 for the
   right hip. Example case `004`: GT has both hips (~315k voxels each); Stage 2
   and `member_02` have no left-hip foreground, and `member_02` labels the
   right hip with the left-hip block (51–100). Case `177`: `member_02` covers
   only the right hip and `member_03` only the left.
3. **Likely root cause: L/R mirror augmentation in Stage A.**
   `PengwinTrainerSTUNetBaseAnatomyV301` was written for `Dataset539`. The
   upstream trainer disables axis-2 (L/R) mirroring only for datasets in
   `DISABLE_X_MIRROR_DATASETS` (`code_task1/core.py`), which contains only
   `Dataset539_PelvicFemurAnatomyV3`. The upstream comment attributes 87.6% of
   its hip errors to L/R swaps caused by this augmentation. Our Anatomy dataset
   is `Dataset701_PENGWINStage2Anatomy`, so it is not covered: the Stage 3
   Anatomy `debug.json` records `inference_allowed_mirroring_axes (0, 1, 2)`
   and `transpose_forward [0, 1, 2]`, so axis 2 is L/R and training applied
   L/R flips without swapping LeftHip/RightHip labels. No project doc or config
   mentions this. Because Stage 2 uses the same dataset and trainer, the defect
   also applies to the Stage 2 backend and therefore to the H1 risk labels.
4. **`member_02` used the crashed-run checkpoint.** Its inference manifest
   records Fragment SHA-256 `f5b99a60…`, identical to the epoch-61
   `checkpoint_best.pth` from the run that ended in the DDP divergent
   early-stop/NCCL timeout. The rank-local Fragment validation/early-stop
   defect in `code_task1/core.py` was not fixed before inference. `member_01`
   (stopped at epoch 78) and `member_03` (epoch 85) used the same rank-local
   path.

### Interpretation

The ensemble mostly disagrees about whether a whole hip exists, which the
presence (Jaccard) term captures and the partition (NVI) term cannot. The H1
comparison therefore measures a Stage A laterality defect, with risk labels
from a model that has the same defect. It does not test FA-IPD on a backend
that segments every bone.

### Unresolved

- The causal link between the mirror setting and the hip losses is inferred
  from code, logs, and outputs; it is not yet confirmed by a retrained model.
- The 20 femur-only cases were not diagnosed.
- NVI is normalized by `log(number of active voxels)` (≈14 at 10^6 voxels),
  which keeps the partition term small. This is part of the frozen contract;
  changing it after seeing validation results would be tuning on the analysis
  set, so it is recorded only as an observation.
- Whether Stage 2's PASS must be reopened is a user decision.

## Runtime handoff observation

At the 2026-09-26 read-only check, GPUs 0, 1, 3, 4, 5, and 6 were idle (2 MiB each); GPUs 2 and 7 were busy. This is a snapshot, so recheck before reserving GPUs. The same 18 orphaned `multiprocessing.spawn` workers plus one resource tracker from the old `exchange-stage2` environment still had `PPID=1` and retained about 20 GiB RSS. They had no reported GPU allocation and were not terminated by the agent. Have their owner verify and clear these exact stale processes before launching the concurrent jobs.

## Gateway

**BLOCKED.** The 2026-10-05 outputs did not meet the two-of-three advantage
requirement, and FA-IPD ranked cases almost identically to foreground Jaccard.
The 2026-10-09 diagnosis shows that the result is confounded by a Stage A L/R
mirror defect, shared by the backend that supplies the H1 risks, and by the
unfixed DDP early-stop defect (`member_02` used the crashed-run checkpoint).
The gateway cannot be validly assessed until these are fixed and the ensemble
is retrained. Stage 4 and H2 remain stopped.

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

Read-only diagnosis checks run on 2026-10-09 (inline scratch scripts; the two
reusable ones are saved in `outputs/stage3_h1/diagnostics/`):

```bash
diff -rq --exclude=.git --exclude=baselines --exclude=outputs --exclude=data --exclude=__pycache__ /home/24chuong.ta/chun/Exchange-ML /mnt/sdb/shared/dang.cpm/Exchange-ML
cp -p /mnt/sdb/shared/dang.cpm/Exchange-ML/<6 differing code/doc files> <same paths here>
cp -p /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage3_h1/{h1_summary.json,h1_comparison.csv,risk_coverage.csv,risk_coverage.png,stratified_analysis.csv,uncertainty_scores.csv,uncertainty_scoring_manifest.json} outputs/stage3_h1/
cp -p /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage3_h1/members/member_0{1,2,3}/validation_inference_manifest.json <same paths here>
conda run --no-capture-output -n exchange-stage2 python outputs/stage3_h1/diagnostics/fa_ipd_decomposition.py
conda run --no-capture-output -n exchange-stage2 python outputs/stage3_h1/diagnostics/bone_recall_by_model.py
sha256sum outputs/stage3_h1/members/member_02/nnUNet_results/Dataset702_PENGWINStage2Fragment/*/fold_0/checkpoint_best.pth
grep -nE "Early stopping|Training done|Yayy" /mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/stage3_h1/members/member_0*/nnUNet_results/Dataset702_*/*/fold_0/training_log_*.txt
grep -n "mirror" baselines/pengwin2026-task1-abbc/code_task1/core.py outputs/stage3_h1/members/member_03/nnUNet_results/Dataset701_*/*/fold_0/debug.json
```

Inline scripts also read case `004` geometry, labels, per-anatomy overlaps,
and GT for cases `004`, `254`, `336`, and `417`. A first run of the bone
recall script failed because a scratch file named `grp.py` shadowed the Python
standard-library `grp` module; it was renamed and the rerun succeeded.

## User-typed commands

The commands below are the original execution handoff, retained for provenance.
They are superseded by the completed H1 evaluation and are not a new run plan.

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

Awaiting user approval to fix the backend, not the FA-IPD method:

1. disable axis-2 (L/R) mirroring for `Dataset701_PENGWINStage2Anatomy`,
   matching the upstream intent for its anatomy dataset;
2. aggregate Fragment validation F1, best-checkpoint selection, and early
   stopping across DDP ranks;
3. validate both with lightweight tests, then prepare user-typed retraining of
   the Stage 2 backend and all three Stage 3 members, followed by inference,
   scoring, and evaluation.

No new full-run command is prepared until the fixes are approved and tested.
