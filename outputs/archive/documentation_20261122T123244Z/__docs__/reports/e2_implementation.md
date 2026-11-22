# E2 implementation and run command

Status: **full E2 complete; artifacts and numerical results audited on 2026-11-19**.

The existing full run has `status=complete`, `smoke_only=false` in
`outputs/stage3_h1_v2/perturbations/e2_manifest.json`. It covers 20 selected
GT cases (10 pelvic + 10 femur), 40 bone units and all 240 E1 prediction pairs.
The audit below inspected this completed run; it did not rerun full E2.

## Full-run findings

| Check | Verified result |
|---|---|
| E2a phantom | 32/32 numerical assertions passed |
| E2b responses | 527 unique rows; 452 eligible; 75 ineligible merge rows because too few fragments |
| Merge / split | All 45 eligible merge and 120 split rows have presence=0 and partition>0 |
| Missing / boundary | All 87 missing and 120 boundary rows have shared-partition VI=0; presence detects support change |
| Boundary targets | 120/120 within tolerance 0.005; maximum absolute Dice error `9.33e-7` |
| E2c replication | 154 positive-VI E1 pairs + one phantom, five factors each → 775 rows; 86 zero-VI pairs → 430 rows |
| Numerical reconstruction | Independently recomputed from saved sparse counts and CSVs: 12,131 comparisons, maximum absolute discrepancy `6.66e-16` |
| Integrity | All 25 checked hashes match: 12 output files, 9 source files, config/E1 manifest/metadata/split. Metadata of 140 used input volumes matches; input-volume bytes were not rehashed in this audit |
| Current checks | 51/51 relevant unit tests and `run_e2.py --check` pass; E1 output/code hashes also pass preflight |

At replication factor 16, median `Q1(r)/Q1(1)` across positive-VI E1 pairs is
0.822683 (about 17.7% reduction), while VI, joint entropy, presence, overlap
weight and Q2 remain unchanged. This verifies the denominator mechanism;
it does not by itself explain all of E1's observed score differences.

**Interpretation and figure limitations before paper use:**

- The current merge curve joins medians from **29, 12 and 4 different eligible
  bone cohorts** at severities 1, 2 and 3. Its declining pooled median must not
  be presented as a within-bone severity response. On the same four bones
  eligible for all levels, median Q2 increases
  **0.005722 → 0.017858 → 0.032355**; on the same 12 eligible for levels 1/2,
  it increases **0.015032 → 0.033021**. For the paper, use a matched-cohort
  panel or within-bone trajectories alongside eligibility counts. The
  existing CSVs suffice; no new inference or perturbation run is needed.
- For the 11 single-fragment bones, splitting into 2, 3 or 4 parts gives
  **Q2=1 at every level**. This is expected saturation of VI/H for a one-part
  reference versus its refinement, not a code failure. The score detects
  disagreement but does not universally measure the number of split errors.
- The missing panel has overlapping `n` labels and connects distinct
  missing-rank groups, including deduplicated ranks. Prefer per-unit scatter
  against removed-volume fraction or explicitly labelled groups for the
  paper. These are presentation issues; source measurements remain valid.
- E2 supports controlled construct validity. It does not establish held-out
  failure-detection performance, clinical utility or model invariance to CT
  resolution. E3 remains necessary for C2.

Audit commands (agent-typed, CPU/read-only apart from test temporary files):

```bash
env PYTHONPATH=src:.:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/dang.cpm/miniconda3/bin/python -B -m unittest test_e2 test_e1 test_stage1_metrics test_stage2_backend test_stage3_h1 -v
env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/dang.cpm/miniconda3/bin/python -B scripts/run_e2.py --check
```

Additional read-only checks parsed the output CSVs/JSONL, recomputed entropy
from sparse counts independently, verified selections/identities and hashes,
and inspected both PNG figures. No full-run command was executed during the
audit. The implementation and original launch instructions follow.

E2 follows [exps.md](../exps.md) E2a–E2c. Entry point:
`scripts/run_e2.py`, configuration: [`configs/e2.json`](../../configs/e2.json).
It runs on CPU using existing GT and E1 member predictions, without training,
inference, CT resampling or the upstream ABBC checkout.

## Method choices

- E2a: the 64-voxel/two-fragment phantom has 32 numerical assertions, including
  identity, ID permutation, merge and its reversal, missing, both empty, and
  both directions of the empty/nonempty comparison. Tolerance is `1e-10`.
- E2b: sort validation IDs per scan type, then use one NumPy RNG seeded
  `20261118` to select 10 pelvic followed by 10 femur cases, without replacement.
  The selected set is independent of scores and risks. This gives 40 bone units.
- Every operation and severity starts from the original, unpruned GT bone.
  Only zero padding is cropped; up to 8 voxels of padding are retained for the
  existing boundary helper, clipped at the original image edge. No additional
  cleanup, pruning, smoothing or GT filtering is performed.
- Merge: sort centroid distances in mm, break ties by ascending ID pairs,
  and greedily choose disjoint pairs. Use the first 1/2/3 pairs independently.
- Split: largest fragment, ties by lowest ID; physical voxel-cell extent is
  `(max_index - min_index + 1) * spacing`, with axis ties z/y/x. Sort coordinates
  by that axis, then remaining coordinates in z/y/x order; split into nearly
  equal chunks. The first retains its ID, other chunks receive ascending unused
  IDs in the same anatomy. Insufficient voxels or IDs are marked ineligible.
- Missing: order by `(volume, ID)`, use the smallest, upper-median rank and
  largest fragment (largest-volume ties use lowest ID). Deduplicate identical
  selected IDs, recording all their rank roles in `severity` and metadata.
- Boundary: reuse the existing erosion/dilation helper on the largest fragment
  for Dice targets `.99/.95/.90`, shell limit 8. Record its selected operation
  and actual achieved Dice. Errors above `.005` are `target_not_reached`;
  cases with no possible nonempty candidate are explicitly ineligible.
  Figure x coordinates use actual Dice, with missed targets marked.
- `changed_volume_fraction` is the fraction of original bone voxels whose
  **instance label** changed. `foreground_changed_volume_fraction` separately
  records foreground symmetric difference. `affected_volume_mm3` records the
  original volume of selected fragments. These quantities differ for ID-only
  merge/split/permutation interventions.
- E2c: reconstruct all **240 E1 pair contingency tables**, using exactly E1
  prediction cleanup. Check their counts, VI, H and score components against
  the saved E1 rows. Scale the entire union table, including foreground
  mismatch counts, by `r=1,2,4,8,16`. Assert that VI, H, presence, overlap weight
  and VI/H stay invariant and `Q1=VI/log(rN)`. No large replicated image is made.
  Positive-VI pairs and pairs with `VI<=1e-12` are reported separately.

Summary figures show median/IQR across available units, with `n` at each
merge/split level, including `n=0`. No confidence intervals are computed;
repeated perturbations/pairs are not treated as independent patients.
This is a mechanism check, not evidence of clinical or held-out performance.

## Outputs

Full root: `outputs/stage3_h1_v2/perturbations/`.

| File | Content |
|---|---|
| `selected_cases.csv` | 20 GT cases, 10 per scan type |
| `phantoms.csv` | 32 exact assertions |
| `responses.csv` | One row per unit/operation/severity, including ineligible rows |
| `response_summary.csv` | Median/IQR and eligibility by severity and scan type |
| `replication.csv` | Positive-VI E1 pairs and phantom, five factors each |
| `replication_zero_vi.csv` | Zero-VI E1 pairs, five factors each |
| `contingencies.jsonl` | Sparse original union counts, sufficient to reproduce E2c |
| `replication_summary.csv` | Invariant residuals and case/pair counts |
| `response_curves.{png,pdf}`, `replication_curves.{png,pdf}` | Scientific figures |
| `e2_manifest.json`, `summary.json`, `case_cache/` | Provenance, completion and resume |

The current E1 has 154 positive-VI and 86 zero-VI pairs. Full E2c therefore
expects **775 positive-VI rows** (including five phantom rows) and **430 zero-VI
rows**, representing 240 E1 tables plus one phantom table. E2b has up to 560
response rows; missing-rank deduplication reduces that count. Ineligible
merge/split rows remain in the table. Full caches cover 40 prediction cases,
with GT responses present only for the 20 selected cases.

E1 output/code hashes are checked first. Used input volumes must also match
their E1 SHA-256. A shared E1 lock prevents a simultaneous E1 writer; a separate
exclusive E2 lock prevents duplicate E2 writers. Every new cache binds input
state/hashes, code, config, E1 identity, package versions and selected cases.
Rerun the same command to resume unchanged cases. An identity mismatch stops
the run instead of overwriting a conflicting experiment.

## Validation performed (agent-typed)

```bash
PYTHONPATH=src:.:tests python -B -m unittest test_e2 test_e1 test_stage1_metrics test_stage2_backend test_stage3_h1 -v
env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/dang.cpm/miniconda3/bin/python -B scripts/run_e2.py --check
env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e2-mpl /home/dang.cpm/miniconda3/bin/python -B -u scripts/run_e2.py --smoke
git diff --check
```

51 relevant unit tests pass, including 12 E2 tests. Smoke cases are `004`
(pelvic) and `255` (femur), producing 4 bone units, 54 response rows (48
eligible), 12 reconstructed E1 pairs, 65 replication rows and 32 passed
phantom assertions. Four PNG/PDF figure files and output hashes are checked.
An initial smoke figure version is retained at `outputs/smoke_e2_initial_plot/`;
the current smoke is `outputs/smoke_e2/`. Neither is the full E2 experiment.

No E1/v1 implementation, scientific outputs or source dataset was modified.

## Full-run command (reference; existing run is now complete)

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/bin/python -B -u scripts/run_e2.py --config configs/e2.json
```

The current full output already has `status: complete`; rerunning it is not
needed for this audit. Next prepare the matched-cohort figure described above
from existing CSVs and proceed with E3 preparation alongside the E1 findings.
