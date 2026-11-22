# E1: implementation and run command

Status: **full E1 complete and audited: 40 cases, 80 bone units, 10,000 case-bootstrap draws**.

Current status is verified from `outputs/stage3_h1_v2/e1_manifest.json`; the
implementation/smoke checks below describe the earlier preparation stage.

The E1a/E1b/E1c path is `scripts/run_e1.py`, configured by
[`../../configs/e1.json`](../../configs/e1.json). It uses the existing v1
validation predictions, probabilities and GT. No model inference, training,
GPU or ABBC checkout is needed. Validation results remain exploratory; this
config is not the held-out v2 protocol freeze.

## Method and outputs

- `e1_scoring.py`: raw VI and joint entropy on shared foreground; `VI/log N`
  versus `VI/H`, weighted and unweighted terms, and all 12 planned scores.
  Probability entropy uses the mean probability on the fixed per-bone union
  ROI. The scorer never receives GT. The scan-type CSV contains only case IDs
  and the existing pelvic/femur metadata, extracted from the Stage 0 audit.
- `e1_data.py`: unchanged 26-connectivity / 1000 mm³ per-ID prediction cleanup
  and unchanged instance matching. Cropping each ID's zero padding before
  cleanup is numerically equivalent to the existing operation, tested on
  anisotropic synthetic volumes. GT is not pruned. Member risks are evaluated
  separately and then averaged; Stage 2 risks are kept as a second source.
- `e1_statistics.py`: tie-expectation prefix risk, trapezoid AURC on coverage
  `1/N..1`, rank-residualized partial Spearman, and paired case bootstrap
  stratified by scan type. Duplicate cases retain all their bone rows.
- `e1_reporting.py`: E1a decomposition summaries, E1b comparisons for both
  risk sources and all strata, and E1c rough precision projections for
  100/210 cases against every alternative score. Projection is not a power
  calculation or held-out confirmation.

Full-run output root: `outputs/stage3_h1_v2/`.

| Output | Expected rows / content |
|---|---|
| `validation/pair_components.csv` | 240 case/bone/member-pair rows |
| `validation/scores.csv` | 80 units; 12 scores plus averaged components |
| `validation/model_risks.csv` | 320 unit/model rows |
| `validation/risks.csv` | 80 units, member-mean and Stage 2 risks |
| `validation/case_scores.csv`, `case_risks.csv` | 40 case-level rows each |
| `validation/case_model_risks.csv` | 160 additional case/model rows |
| `validation/comparison.csv`, `paired_differences.csv` | All scores, both risk sources, pooled/pelvic/femur |
| `tables/decomposition.csv`, `precision_projection.csv` | E1a and E1c |
| `figures/` | Decomposition, signal–risk scatter and risk–coverage, PNG/PDF |
| `e1_manifest.json`, `summary.json`, `case_cache/` | Provenance, completion status and resumable case results |

All source volumes are fingerprinted with SHA-256. A file lock prevents
duplicate writers. Rerunning the same command reuses completed case caches
only if code, config, provenance and input file metadata match. If these
identities change, the command stops; retain the old run and choose a new
`output_root` in a separate config. Frozen v1 files are not changed.

## Validation performed (agent-typed)

```bash
PYTHONPATH=src:.:tests python -B -m unittest test_e1 test_stage1_metrics test_stage2_backend test_stage3_h1 -v
python -B scripts/run_e1.py --check
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e1-mpl /home/dang.cpm/miniconda3/bin/python -B -u scripts/run_e1.py --smoke
git diff --check
```

Observed: **39 tests passed** (13 E1 and 26 existing metric/backend-helper/H1
tests). The preflight found all 40 cases and their required files. The smoke
completed case `004` (pelvic) and `254` (femur), including geometry, ID range,
finite probabilities in `[0,1]`, cleanup, scoring, four-model risk evaluation,
bootstrap and PNG/PDF export. It produced 4 unit rows, 12 pair rows and 16
model-risk rows. Rerunning the smoke with the same inputs reused both case
caches successfully. Smoke output is separate at `outputs/smoke_e1/`; it uses
32 bootstrap draws and must not be reported as the full E1 result.

The upstream backend-fix integration tests are outside this change and still
require the missing ABBC checkout. No model/training code was modified.

## Full experiment (user-typed; not executed by the agent)

Run from the verified checkout with the verified local Python interpreter:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/bin/python -B -u scripts/run_e1.py --config configs/e1.json
```

This processes all 40 validation cases and runs 10,000 paired case-bootstrap
draws (seed `20261119`). It is CPU-only. Wait for `E1 VALIDATION COMPLETE` and
`status: complete` in the manifest; partially written tables or caches do not
establish completion. The next step is to inspect E1a raw-VI spread, E1b
comparisons and E1c precision before deciding the held-out budget in E3.
