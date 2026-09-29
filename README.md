# Exchange-ML

**Foreground-Aware Instance-Partition Disagreement for active learning in 3D fracture segmentation**

Exchange-ML is a research implementation for studying whether disagreement over predicted fracture-instance partitions is a better uncertainty signal than voxel-level uncertainty. The project targets whole-volume active learning for 3D CT fracture segmentation and is being developed toward the paper:

> *Beyond Voxel Uncertainty: Instance-Partition Disagreement for Active Learning in 3D Fracture Segmentation*

The primary benchmark is PENGWIN 2026 Task 1, with RibFrac planned as a cross-anatomy transfer study. This repository is an active research workspace: the backend and metric implementation are available, but the main multi-seed Stage 3 experiment has not been completed and no final scientific claim should be inferred from the current artifacts.

## Motivation

Voxel entropy can be low even when a segmentation contains a clinically meaningful structural error, such as:

- merging two fracture fragments;
- splitting one fragment into several predictions;
- missing a fragment entirely; or
- predicting an extra fragment.

These failures are naturally expressed as disagreement between variable-cardinality instance partitions. Exchange-ML measures that disagreement while keeping the segmentation backend, preprocessing, training policy, and reconstruction procedure fixed across acquisition strategies.

```text
3D CT
  -> fixed two-stage segmentation backend
  -> ensemble probabilities and instance maps
  -> entropy and structural disagreement scores
  -> whole-volume acquisition ranking
  -> active-learning trajectories
  -> failure detection, label-efficiency, and transfer evaluation
```

## FA-IPD

For two predicted instance maps `P` and `Q`, let `F_P` and `F_Q` be their foreground sets, `U = F_P union F_Q`, and `C = F_P intersection F_Q`. The pairwise Foreground-Aware Instance-Partition Disagreement is

```text
d_FA(P, Q) = |F_P symmetric_difference F_Q| / |U|
             + |C| / |U| * NVI(P restricted to C, Q restricted to C).
```

The case-level score averages `d_FA` across every unordered pair in an ensemble:

```text
FA-IPD(x) = 2 / (M(M - 1)) * sum_{i < j} d_FA(P_i(x), P_j(x)).
```

The first term captures missing or extra foreground. The normalized Variation of Information term captures disagreement about how shared foreground is divided into instances. The frozen implementation defines both-empty predictions as `0`, exactly-one-empty as `1`, and NVI as `0` when the shared foreground contains at most one voxel.

FA-IPD is bounded, symmetric, and invariant to instance-ID permutations. It is a **dissimilarity score, not a mathematical metric**, because triangle inequality is not guaranteed.

The main hybrid acquisition score combines percentile ranks of entropy and FA-IPD with equal weight:

```text
A(x) = 0.5 * rank_pct(entropy(x)) + 0.5 * rank_pct(FA-IPD(x)).
```

See [`__docs__/paper.md`](__docs__/paper.md) for the complete definition, assumptions, novelty boundary, and falsification criteria.

## Research questions

The study is organized around three primary hypotheses:

1. **H1 — Signal validity:** FA-IPD ranks true merge, split, and missing-instance failures better than predictive entropy and semantic ensemble disagreement.
2. **H2 — Active-learning utility:** entropy plus FA-IPD improves instance-level label efficiency at equal whole-volume annotation budget without materially degrading semantic Dice.
3. **H3 — Cross-task transfer:** the same frozen FA-IPD definition remains useful on RibFrac without dataset-specific score redesign.

A controlled perturbation study, H4, checks that FA-IPD responds to merge, split, missing-fragment, and boundary changes as designed.

## Current status

| Stage | Scope | Status |
|---|---|---|
| 0 | Data readiness, static split, and protocol freeze | **PASS** |
| 1 | Instance rules, metrics, connectivity, and perturbation audit | **PASS** |
| 2 | Fixed PENGWIN ABBC/STU-Net backend | **PASS** |
| 3 | Three-member Round-0 FA-IPD/H1 benchmark | **BLOCKED** — implementation complete; ensemble execution incomplete |
| 4–8 | AL pilot, full trajectories, RibFrac transfer, ablations, and paper | Pending Stage 3 gateway |

The completed Stage 2 validation contains 40 cases and reports:

- mean binary Dice: `0.7639`;
- mean Instance F1: `0.7368`;
- 8 merge errors and 115 split errors;
- 31/40 cases with at least one merge or split error.

These are internal research-backend results, not hidden-test performance or a claim of equivalence to the official PENGWIN evaluator. Stage 3 currently has no completed ensemble member, uncertainty table, or H1 result.

Detailed evidence is recorded in [`__docs__/reports/`](__docs__/reports/).

## Data protocol

### PENGWIN

The frozen study uses the 340 public labeled PENGWIN cases:

- 50-case static test set;
- 40-case validation set;
- 250-case active-learning pool;
- 40 initially labeled cases;
- five whole-volume acquisition rounds of 20 cases;
- budgets `40 -> 60 -> 80 -> 100 -> 120 -> 140`;
- at least three paired active-learning trajectories.

Ground truth may define the static split and initial sets, but it must never influence acquisition scoring. Subject-local fragment IDs are preserved; they are not global semantic classes.

### RibFrac

RibFrac is reserved for H3 cross-anatomy validation. The same FA-IPD definition and evaluation protocol must be used without post-hoc score tuning.

Datasets and generated outputs are excluded from Git. Expected local paths are:

```text
dataset/
├── PENGWIN26_task1_2/
└── RibFrac/
```

See [`__docs__/data_description.md`](__docs__/data_description.md) for the verified local inventory, label semantics, geometry notes, and known missing assets.

## Frozen evaluation contract

- Preserve all valid ground-truth instance IDs.
- Clean predictions only, using 26-connectivity.
- Remove per-ID predicted components smaller than `1000 mm3`.
- Apply no additional morphology.
- Match instances within frozen anatomical ID ranges at IoU `>= 0.1`.
- Use `1 - Instance F1` and normalized merge-plus-split error as structural risks.
- Use `1 - Dice` as the semantic control.
- Evaluate H1 with risk-coverage/AURC, top-10% and top-20% enrichment, and confounder-controlled association.

The machine-readable contracts are in [`configs/study_protocol.json`](configs/study_protocol.json), [`configs/stage2_backend.json`](configs/stage2_backend.json), and [`configs/stage3_h1.json`](configs/stage3_h1.json).

## Repository layout

```text
Exchange-ML/
├── src/exchange_ml/       # Instance processing and uncertainty metrics
├── scripts/               # Stage preparation, training, inference, and evaluation CLIs
├── configs/               # Frozen study and execution contracts
├── tests/                 # Unit tests for metrics and Stage 2/3 behavior
├── splits/                # Frozen split and audit manifests
├── baselines/             # External reference implementation checkouts
├── __docs__/              # Research blueprint, roadmap, rules, and stage reports
├── dataset/               # Local data; ignored by Git
└── outputs/               # Generated artifacts; ignored by Git
```

The external repositories under `baselines/` are local reference checkouts. Their upstream URLs and intended roles are listed in [`__docs__/plan.md`](__docs__/plan.md); they are not currently configured through a root `.gitmodules` file.

## Environment

The Stage 2/3 backend contract expects:

- Python `3.10`;
- `nnunetv2==2.5.1`;
- a CUDA-capable PyTorch environment for full training;
- SimpleITK and the dependencies required by the preserved PENGWIN ABBC implementation.

There is currently no top-level environment lock file. Use the project runbook to inspect the pinned environment and commands:

```bash
python scripts/stage2_backend.py print-runbook
```

If the prepared Conda environment already exists, validate it with:

```bash
conda run -n exchange-stage2 python scripts/stage2_backend.py check
conda run -n exchange-stage2 python scripts/stage3_h1.py check
```

Do not silently upgrade nnU-Net: the preserved backend is tied to version `2.5.1` APIs.

## Tests

Run the complete lightweight test suite from the repository root:

```bash
conda run --no-capture-output -n exchange-stage2 \
  env PYTHONPATH=src:. python -B -m unittest discover -s tests -v
```

Validate the completed Stage 2 artifacts with:

```bash
python3 scripts/validate_stage2_artifacts.py
```

The last documented Stage 1–3 suite completed with 26 passing tests. Unit and synthetic tests validate implementation behavior; they do not substitute for the full scientific experiments.

## Stage 3 workflow

Stage 3 trains three independent members with seeds `31001`, `31002`, and `31003`. Each member trains the anatomy stage first and then warm-starts its fragment stage from the matching anatomy checkpoint. Stage 2 checkpoints are not reused because their model seed was not explicitly recorded.

Print the authoritative commands before launching any full job:

```bash
python scripts/stage3_h1.py print-runbook
```

After every member has completed training, inference, and its 40-case manifest, scoring and retrospective evaluation are run sequentially:

```bash
conda run -n exchange-stage2 python scripts/score_stage3_h1.py
conda run -n exchange-stage2 python scripts/evaluate_stage3_h1.py
```

Under the project rules, full preprocessing, training, dataset-wide inference, scoring, and multi-seed experiments are operator-run jobs. Check GPU ownership, memory, output permissions, and rendezvous ports before launching them. Do not reduce the ensemble, reuse a Stage 2 checkpoint, or change the frozen protocol to bypass a failed run.

The H1 gateway passes only if FA-IPD shows a meaningful advantage in at least two of:

1. AURC;
2. top-k failure enrichment;
3. controlled association.

The effect must also remain meaningful relative to the Jaccard-only ablation and after controlling for fragment count and size.

## Documentation

- [`__docs__/paper.md`](__docs__/paper.md) — complete research blueprint and literature context.
- [`__docs__/plan.md`](__docs__/plan.md) — fixed experimental roadmap and stage gates.
- [`__docs__/data_description.md`](__docs__/data_description.md) — dataset inventory and label semantics.
- [`__docs__/rule.md`](__docs__/rule.md) — command ownership, evidence, and reporting rules.
- [`__docs__/reports/`](__docs__/reports/) — stage-specific configurations, evidence, results, and blockers.

## Data, licensing, and claims

The datasets are not distributed by this repository. Obtain PENGWIN and RibFrac from their official sources and comply with their licenses, challenge rules, citation requirements, and publication restrictions. PENGWIN publication/embargo clarification remains an external issue documented in the Stage 0 report.

External implementations under `baselines/` retain their own licenses and attribution requirements. This repository does not currently declare a root software license.

Until the frozen experiments are complete, cite this repository only as research software or a study design—not as evidence that FA-IPD improves active learning.
