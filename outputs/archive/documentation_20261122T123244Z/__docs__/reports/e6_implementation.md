# E6 — external score transfer to RibFrac

**Status, 21/11/2026:** data contract, backend, runner and Slurm chain implemented; **160 tests pass**, and native nnU-Net synthetic CPU smoke is complete. Full preprocessing, three-member training and external inference **not run**. GPU smoke submission is **blocked by the scheduler submit-host mapping**, before any job ID was issued (details below). No E6 performance result is available. Implementation validation is distinct from an external scientific result.

## Material Passport

| Field | Value |
|---|---|
| Origin skill / scope | `academic-pipeline`; implementation and reproducibility bridge requested by the user |
| Version | E6 implementation v1, 21 November 2026 |
| Inputs | E6 in `exps.md`, frozen E3 score definitions, official RibFrac release, local train/validation pairs |
| Verified | Primary label semantics; metadata inventory; representative mask/geometry checks; data/scoring unit checks |
| Not verified by these checks | Full training convergence, full GPU runtime, external effects, clinical review utility |
| Manuscript stages | No paper Stage 2.5/4.5 integrity PASS or submission readiness is claimed |

## Scope and mathematical choices

This is transfer of the **fixed disagreement scores** to **rib-fracture lesion instances**, not transfer of the PENGWIN checkpoints and not whole-rib/bone-fragment segmentation. [Verified primary sources and data checks](e6_protocol_sources.md) explain this distinction. All nonzero GT IDs, including `label_code=-1`, remain positive; −1 excludes fracture-type classification only. GT IDs are never component-pruned or assigned a PENGWIN anatomy range.

The new backend is binary nnU-Net 3D full resolution (`nnunetv2==2.5.1`), trained from scratch. Three models use the same preprocessing, architecture plan and internal split, with seeds **32001, 32002, 32003**. The 420 official training subjects form the development set: a fixed seeded split gives **336 gradient-training and 84 internal-validation subjects**. None of the 80 official validation subjects enters the nnU-Net raw training dataset, fingerprint, planning, internal validation or checkpoint selection. The 160 test images are not used.

The config fixes 1,000 epochs, 250 training and 50 internal-validation iterations per epoch, and selects the **final checkpoint**. Those are the configured training budget, not measured time or a claim that models have converged. A coarse header-based storage check precedes preparation. No new model is downloaded and no PENGWIN model is used as an initialization.

True softmax foreground probabilities are saved on the native image grid. Threshold **0.5** and **26-connectivity** reconstruct connected-component lesion instances. There is **no minimum-component-volume filter**: a representative valid lesion is smaller than the PENGWIN 1,000 mm³ threshold. Touching lesions may merge and disconnected predictions may split; these are measured limitations, not repaired using GT. The same instance maps are used for all scores.

All 12 E3 scores are reported case-wise. Shared/union-foreground VI formulas, mean three-pair aggregation, entropy on the ensemble foreground ROI and empty-ROI rules remain fixed. The Hungarian baseline uses generic instance IDs with no anatomy constraints. Its sparse assignment is equivalent to maximum summed IoU with unmatched objects and denominator `max(K_i,K_j)`.

Scoring uses only the three predictions and their probabilities, before label voxels are read for evaluation. Primary risk source for this extension is the mean of separately evaluated model risks, rather than risk of a consensus segmentation. Matching retains the internal E1 best-IoU rule at **IoU≥0.1** and its merge/split definitions. It is **not the official RibFrac FROC evaluator**. Internal F1 may conceal merge/split or missing-volume errors, so those endpoints stay separate.

For zero-GT cases, merge, split, missing-instance and missing-volume endpoints are N/A. Such cases remain in F1/Dice/false-positive evaluation, with Dice=1 for both empty and Dice=0 for nonempty prediction against empty GT. Structural tables explicitly count excluded undefined cases. Reporting uses **10,000 paired case-bootstrap draws**, exact-tie AURC and marginal 95% intervals, without pooling PENGWIN and RibFrac AURCs. Constant-risk endpoints are labeled uninformative. There is no newly invented success threshold or replacement of the PENGWIN primary endpoint.

## Files and output ownership

| File | Responsibility |
|---|---|
| `configs/e6.json` | Split, seeds, training budget, fixed reconstruction, scoring/evaluation and storage settings |
| `src/exchange_ml/e6_data.py` | Strict filenames/CSV contract, native geometry and label conversion |
| `src/exchange_ml/e6_backend.py` | New nnU-Net backend, reconstruction, storage estimate, hashing and locks |
| `src/exchange_ml/e6_evaluation.py` | Generic-ID scores, internal risks, paired bootstrap and tables |
| `scripts/run_e6.py` | `check`, `prepare`, `preprocess`, `train`, `infer`, `evaluate`, synthetic `smoke` |
| `scripts/e6_*.sbatch`, `scripts/submit_e6.sh` | Scheduler resources and dependency chain |
| `tests/test_e6_*.py` | Data, mathematical contracts and runner/backend regressions |

Suggested full output root: `/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6`. Code and source data stay in their current locations. The agent has not written full artifacts there. The default repo output filesystem has limited free space; use the explicit larger destination and inspect `check` before submission. The estimate depends on header voxel counts and fixed byte factors; compression/resampling can change actual usage.

The read-only preflight inspected all 500 source image headers and returned an estimated requirement of **459,327,668,224 bytes (approximately 428 GiB)**, with approximately **1.6 TiB** free on the proposed disk1 filesystem at inspection time. This is a planning estimate, not measured full-run storage or a lasting free-space guarantee.

Per-stage manifests bind config, code, software, source/parent hashes and output root. Locks reject duplicate writers. Preparation preserves source data and writes new binary targets; external inference saves true probabilities and integer maps separately. Checkpoints record optimizer and RNG state for resume, without claiming bitwise reproducibility across hardware/library changes. Identity/hash drift is an error; preserve old outputs and select a new root if the intended implementation/config changes.

Expected final outputs include `scores.csv`, `pair_components.csv`, `model_risks.csv`, `risks.csv`, `comparison.csv`, `paired_differences.csv`, `risk_coverage.csv`, `summary.json`, `e6_manifest.json`, and model/preparation/preprocessing manifests. A complete final run requires all 80 official external cases, three distinct trained checkpoints and **240 inference records**. Merely having files or a successful synthetic smoke is insufficient.

## Validation and research-integrity review

- Strict inventory confirms 420 development and 80 external filename/CSV pairs. GT masks `RibFrac421`, `RibFrac435`, `RibFrac491` match CSV IDs and image geometry; the latter two are representative negatives. The reversed X/Y direction of case491 is retained. This is not a voxel audit of all 500 cases.
- Six data tests pass: duplicate/unknown metadata, missing pairs, split leakage, untyped positives, negative cases, generic IDs, probability range and native-grid geometry round trips. An initial SimpleITK reader method mismatch was corrected before the passing run.
- Mathematical tests compare sparse Hungarian with a dense solver, generic-ID invariance, frozen single-anatomy E1 components, internal matching, negative-case N/A rules, tie handling and paired bootstrap. Independent CPU checks by the root agent also compared all 12 scores on 12 random cases to the frozen single-anatomy implementation within `1e-12`, and internal F1/merge+split against the old evaluator.
- The complete suite passes **160 tests in 13.948 seconds**: 41 E6 tests and 119 existing tests. Shell syntax checks pass. The existing E1–E5 evidence audit still passes **111 output hashes and 142 cache hashes**. These checks do not imply that the GPU training path has run.
- Native nnU-Net CPU smoke completed in **29.334 seconds**, recorded at `/tmp/exchange-e6-cpu-smoke-v2/smoke_manifest.json` with `status=complete` and `smoke_device=cpu`. It exercised preprocessing of four synthetic 32³ cases, two training steps, one validation step, checkpoint writing, the manually initialized predictor, finite native-grid 32³ probabilities within [0,1], connected components, and scoring/evaluation APIs. Six recorded current-producer hashes and three output hashes were independently verified. A repeated single network in this synthetic API check is explicitly labeled and is not a three-member external experiment.
- The first CPU smoke at `/tmp/exchange-e6-cpu-smoke` failed when the planner inherited augmentation worker count zero and called `torch.set_num_threads(0)`. Planning now uses a positive thread count independently from single-process augmentation; a regression test covers this fix. The failed attempt remains separate from the successful v2 smoke.
- A separate check loaded the actual smoke checkpoint at epoch 1, confirmed nonempty SGD optimizer state, initialized native nnU-Net train/validation loaders and verified that Python/NumPy/Torch RNG states matched the checkpoint after loader warmup. A mismatched `e6_identity` was rejected. This exercised resume loading without another training step; it does not establish bitwise continuation on different hardware.
- Full E1–E5 evidence is preserved; E6 files are separate and no E1–E5 result is recomputed or relabeled.

Seven failure-mode scope checks: implementation contracts are tested; no E6 results are fabricated; scoring cannot access GT; bug fixes are reported as fixes rather than findings; dataset/model provenance is recorded rather than invented; no favorable outcome is promised; and failure/constant-risk/negative outcomes are retained for interpretation. These implementation checks do not constitute a manuscript integrity-stage clearance.

## Agent-typed checks

CPU tests and metadata/synthetic checks are lightweight implementation validation; no full data conversion, training or inference is authorized for agent execution by `rule.md`.

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONPATH=src:.:tests:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest discover -s tests -p 'test_e6_*.py' -v

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 MPLCONFIGDIR=/tmp/exchange-e6-cpu-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e6.py smoke \
  --device cpu --output-root /tmp/exchange-e6-cpu-smoke-v2
```

The commands above are for checking the implementation. Full external results must not be inferred from them.

### Scheduler blocker observed

The root agent attempted this authorized synthetic-smoke submission:

```bash
sbatch --parsable scripts/e6_smoke.sbatch
```

Slurm rejected it **before issuing a job ID**, reporting that it could not identify which server the job came from: `VU-PROD-GPU-01`. Read-only inspection found that `/etc/slurm/job_submit.lua` maps the older `vishc-server-1`/`server1` names and IPs, while the submit host now identifies itself as `VU-PROD-GPU-01` in `/etc/hosts` and `/etc/slurm/slurm.conf`. The submission origin is obtained from the authenticated submit-node information. An administrator must update the submit-host mapping. No system file was modified and no host-identity bypass was attempted.

The Slurm command chain below remains the intended production entry point, but it **cannot submit successfully until the administrator repairs that mapping**. No GPU smoke, production preparation, training, inference or external evaluation job was started by this attempt. The completed CPU synthetic smoke exercises native nnU-Net preprocessing, forward/backward and serialization; it does not validate CUDA placement, GPU memory or full-run throughput. CPU mode is accepted only for synthetic smoke; production training/inference still require one Slurm-allocated CUDA device.

## Full — user-typed commands

Metadata/storage preflight (read-only, CPU):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B scripts/run_e6.py check \
  --output-root /mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6
```

Submit the whole chain:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
E6_OUTPUT_ROOT=/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6 \
  bash scripts/submit_e6.sh full
```

This submits synthetic GPU smoke → CPU preparation → CPU preprocessing → three-member GPU training → three-member GPU inference → CPU scoring/evaluation. Training and inference each run **at most two concurrent tasks, one allocated GPU per task** (`0-2%2`), preserving Slurm's device assignment. Preparation, preprocessing and final evaluation do not request a GPU. The configured 8 GiB planner target is not a measured GPU-memory guarantee. A first production training segment is needed before estimating full runtime.

Standalone synthetic pipeline smoke, without full jobs:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash scripts/submit_e6.sh smoke
```

Monitor `squeue -u "$USER"` and `outputs/logs/e6-*.log`. A timeout/failure leaves downstream `afterok` jobs unable to run. Inspect the cause, cancel only stale dependent jobs from that submission, then resubmit the same full command to resume verified stages. Do not start two chains writing to the same root. Resume is bounded by the last valid recorded checkpoint; it is not an automatic guarantee that a failed job completes.

## Next step

Have the administrator repair the submit-host mapping, run GPU synthetic smoke, then run the user-typed chain and audit all manifests/counts/hashes, reporting every baseline and paired interval. Until then E6 is implemented and passes CPU implementation checks, but **GPU validation and external validation remain unrun**. The historical `configs/study_protocol.json` is frozen and still records its original deferred dependency snapshot; this report and the current data inventory describe current readiness without rewriting that history.
