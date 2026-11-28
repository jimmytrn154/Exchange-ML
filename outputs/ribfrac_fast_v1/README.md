# RibFrac E6 fast: completed external results

**Completed and audited — 28/11/2026.** Three independently seeded members reached fixed final epoch 1000; inference completed for 3 × 80 CTs. Evaluation job `3532` and audit/cohort job `3533` completed. The full audit verified checkpoint, source image/GT, prediction/probability, cache, and output hashes. All 80 cases (RibFrac421–500) are present: 60 positive and 20 negative CTs. Structural/missing endpoints use only positive cases; F1, Dice and FP burden retain negatives.

| Score | Merge/split (60 positive CTs) | Missing volume (60 positive CTs) | 1−F1 (80 CTs) |
|---|---:|---:|---:|
| FA-v2 / joint entropy | 0.1079 | 0.2129 | 0.2927 |
| FA-v1 / log voxel count | 0.1101 | 0.2123 | 0.2914 |
| Coverage | 0.1110 | 0.2118 | 0.2919 |
| Mean entropy | 0.1110 | 0.2120 | 0.2905 |
| Hungarian | 0.1006 | 0.2042 | 0.2926 |
| Partition-only v2 | 0.0598 | 0.2949 | 0.4749 |

| Endpoint | Baseline | ΔAURC FA-v2−baseline [95% CI] |
|---|---|---:|
| merge_split | presence | -0.0031 [-0.0109; 0.0005] |
| merge_split | mean_entropy | -0.0031 [-0.0259; 0.0124] |
| missing_volume | presence | 0.0011 [-0.0008; 0.0044] |
| missing_volume | mean_entropy | 0.0009 [-0.0099; 0.0111] |
| one_minus_f1 | presence | 0.0007 [-0.0007; 0.0035] |
| one_minus_f1 | mean_entropy | 0.0021 [-0.0039; 0.0094] |
| one_minus_dice | presence | 0.00297 [-0.000019; 0.00857] |
| one_minus_dice | mean_entropy | 0.0068 [0.0012; 0.0148] |

Lower AURC is better. CIs use 10,000 paired case-bootstrap draws (seed 20261121); these are marginal 95% intervals without familywise multiplicity adjustment. Results are secondary external transfer evidence, with no new primary success threshold. Changing normalization alone gives structural ΔAURC FA-v2−FA-v1 = **-0.0022 [-0.0081; 0.0004]**.

| Cohort | CTs | Mean instance F1 | Mean binary Dice | FP instances/CT |
|---|---:|---:|---:|---:|
| all | 80 | 0.5250 | 0.5129 | 3.3125 |
| positive | 60 | 0.6389 | 0.6228 | 3.7333 |
| negative | 20 | 0.1833 | 0.1833 | 2.0500 |

Cohort values average the three individual models, not a consensus segmentation. Partition-v2 is nonzero in only **9/80** CTs; FA-v2 equals coverage (tolerance 1e-12) in **71/80** CTs. Connected components of binary lesion predictions are the instances in this backend, which limits how often shared-foreground grouping differs.

**Full-result assessment:** FA-v2 does not establish a structural-ranking advantage over coverage or mean entropy: both paired confidence intervals include zero. The normalization-only contrast also includes zero. For Dice error, FA-v2 is worse than mean entropy (ΔAURC +0.0068, marginal 95% CI [0.0012; 0.0148]). Structural AURC for FA-v2 (0.1079) is worse than the expected random ordering (0.0588, computed as mean risk × (n−1)/n under the frozen integration convention). Partition-only AURC (0.0598) is close to that random reference, so its lower AURC than FA-v2 does not establish useful structural discrimination. Only 17/60 positive CTs have nonzero member-mean structural risk. These external results do **not** validate a general transfer benefit for FA-v2; the completed PENGWIN evidence remains a separate result.

**Interpretation boundary:** this experiment tests fracture-lesion instances, not whole-bone fragments, and does not transfer PENGWIN checkpoints. A benefit on a specific endpoint does not establish general superiority, clinical review savings, or universal cross-dataset transfer. Preserve unfavorable and uncertain contrasts in `paired_differences.csv`; do not pool PENGWIN and RibFrac AURCs. Internal best-IoU matching uses IoU ≥ 0.1 and is **not official RibFrac FROC**.

## Evidence and reproduction

- `comparison.csv`: all 12 scores, all 8 endpoints, AURC/CI/correlation and exact-tie diagnostics.
- `paired_differences.csv`: FA-v2 versus each control with paired confidence intervals.
- `scores.csv`, `risks.csv`, `model_risks.csv`, `pair_components.csv`, `risk_coverage.csv`: full scalar tables.
- `summary.json`, `e6_manifest.json`, `models/*/*_manifest.json`: frozen configuration, software/code identity, completion counts, and source hashes.
- `cache/*.json`: the exact scalar evaluation and prediction-only score caches for all 80 cases; these support statistics inspection without medical images.
- `statistics_reproduction_receipt.json`: PASS after recomputing all 96 comparison rows, 88 paired contrasts and 6,480 risk-coverage rows from the exported caches using the frozen 10,000-draw bootstrap.
- `audit_receipt.txt`: completed full-source audit and cohort-report receipt from Slurm job 3533.
- `export_receipt.json`: original bulk source path and SHA-256 for every byte-preserved exported source file.
- [Positive/negative cohorts](../research_followup_v1/ribfrac_cohorts/cohorts.csv) and its [manifest](../research_followup_v1/ribfrac_cohorts/manifest.json).

Bulk source: `/mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1`. Original paths inside manifests intentionally remain unchanged. Images, probability volumes, predictions, and checkpoints stay at that source; this export contains scalar research evidence and provenance.

Full source audit (read-only; requires the original data and environment):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B \
  scripts/check_ribfrac_fast.py audit
```
