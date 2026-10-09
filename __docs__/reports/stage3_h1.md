# Stage 3 — FA-IPD + H1 Round-0 benchmark

Status: **FAILED (H1 scientific gateway; full experiment completed)**

Opened: 2026-09-26. Last assessed: 2026-10-05 (Asia/Ho_Chi_Minh).

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

**Manual gateway decision: FAIL.** This is a completed negative H1 result, not
a missing-artifact or runtime failure. The approved Stage 4 prerequisite is
not met. No Stage 4 AL pilot was implemented or run.

## Runtime handoff observation

At the 2026-09-26 read-only check, GPUs 0, 1, 3, 4, 5, and 6 were idle (2 MiB each); GPUs 2 and 7 were busy. This is a snapshot, so recheck before reserving GPUs. The same 18 orphaned `multiprocessing.spawn` workers plus one resource tracker from the old `exchange-stage2` environment still had `PPID=1` and retained about 20 GiB RSS. They had no reported GPU allocation and were not terminated by the agent. Have their owner verify and clear these exact stale processes before launching the concurrent jobs.

## Gateway

**FAIL.** The real H1 outputs are complete, but FA-IPD does not meet the
predeclared two-of-three advantage requirement and remains nearly identical to
foreground Jaccard in case ranking. Stage 4 and H2 are stopped at this gateway.

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

Review the negative H1 result and decide how to investigate the main FA-IPD
method under the approved research direction. The current plan does not
authorize Stage 4 after an H1 gateway failure. No new full-run command is
prepared.
