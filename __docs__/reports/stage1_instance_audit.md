# Stage 1 — Instance and metric audit

Status: **PASS — INTERNAL RESEARCH CONTRACT**

Date: 2026-09-21 (Asia/Ho_Chi_Minh)

## Frozen contract

- Preserve every GT instance ID in the documented `1..200` range. Do not filter
  GT by size and do not apply connected-component cleanup to GT.
- For predictions only, use 26-connectivity and remove each per-ID connected
  component smaller than `1000 mm3`.
- Apply no additional morphology.
- Report the cleanup-disabled setting as a required ablation.
- Match instances within the four frozen anatomy ID ranges at IoU `>=0.1`.
- Treat this as the project's internal evaluation contract. Equivalence to an
  unverified official PENGWIN evaluator is not claimed.

The contract is encoded in `configs/study_protocol.json`,
`configs/stage2_backend.json`, and `src/exchange_ml/instances.py`.

## Dataset-wide audit

The user completed the full audit over all 340 discovered training labels. The
artifact then passed an independent JSON-level integrity validator.

- Artifact: `splits/pengwin_stage1_connectivity_audit.json`
- SHA-256: `76b6a58018fc3a2254275c3522d1eef887e8ef52e11dfa73410d556876491378`
- Cases: `340/340`, with 340 unique case IDs and existing source label paths
- Annotated GT instances: `2,427`

| Connectivity | Components | Disconnected GT IDs | Small components | Cases with disconnected IDs | Cases with small components |
|---:|---:|---:|---:|---:|---:|
| 6 | 4,925 | 559 | 3,202 | 217 | 268 |
| 18 | 3,575 | 357 | 1,852 | 168 | 249 |
| 26 | 3,431 | 337 | 1,708 | 161 | 246 |

Component counts were monotonic for every case (`6 >= 18 >= 26`). The selected
26-connectivity produces the fewest artificial separations under the audited
adjacency conventions and is the maximal standard 3D neighborhood.

The audit also exposes why cleanup must never be applied to GT:

| Complete GT instance volume | Count | Fraction of 2,427 instances |
|---|---:|---:|
| `<500 mm3` | 513 | 21.14% |
| `<1000 mm3` | 704 | 29.01% |

Instances below `1000 mm3` account for about `0.143%` of total annotated GT
foreground volume. Their small volume fraction does not make them disposable:
missing-fragment behavior is a target of this study. The `500` and `1000 mm3`
fields in the audit artifact are descriptive reference thresholds; the script
does not rewrite GT.

## FA-IPD construct checks

Foreground-Aware Instance-Partition Disagreement is

\[
d_{FA}(P,Q)=\frac{|F_P\triangle F_Q|}{|F_P\cup F_Q|}
+\frac{|F_P\cap F_Q|}{|F_P\cup F_Q|}NVI(P|_C,Q|_C).
\]

It is bounded, symmetric, and instance-ID permutation invariant. Both-empty
foreground scores `0`; exactly-one-empty scores `1`. Ensemble FA-IPD is the
unweighted mean over unordered pairs, each with its own union and intersection.
It is a dissimilarity, not a mathematical metric: a retained finite
counterexample violates triangle inequality.

| Controlled case | Union-foreground NVI | FA-IPD | Expected behavior |
|---|---:|---:|---|
| Identity | 0.000000 | 0.000000 | identity preserved |
| Merge | 0.125537 | 0.125537 | equal support reduces to NVI |
| Split | 0.062766 | 0.062766 | equal support reduces to NVI |
| Missing one of two equal fragments | 0.000000 | 0.500000 | presence penalty is positive |
| Empty vs non-empty | 0.125537 | 1.000000 | frozen empty-mask rule |
| Boundary perturbation | 0.063931 | 0.183007 | foreground term detects support change |

The matched boundary perturbation achieved whole-foreground Dice `0.899281`
for target `0.9`. These are construct tests, not H1–H3 empirical results.

## Gateway decision

**PASS** for Stage 1 because:

1. GT-vs-GT, label-permutation, empty-mask, ROI, pairwise aggregation,
   missing/extra-fragment, and non-metric counterexample tests pass.
2. The completed 340-case artifact is internally consistent and checksum-bound.
3. The dataset evidence supports 26-connectivity as the least fragmenting
   standard neighborhood.
4. Code and configs preserve all GT IDs and restrict cleanup to predictions.
5. Cleanup-disabled evaluation remains required, so the `1000 mm3` choice is
   testable rather than hidden.

This PASS authorizes progression to Stage 2 under the internal contract. It
does not establish anatomical validity, model performance, H1–H3, or official
evaluator equivalence.

## Reproduction and validation

User-executed full audit:

```bash
PYTHONPATH=src python -B scripts/stage1_audit_instances.py \
  --dataset-root dataset/PENGWIN26_task1_2 \
  --output splits/pengwin_stage1_connectivity_audit.json \
  --component-prune-min-volume-mm3 1000 \
  --gt-fragment-min-volume-mm3 500
```

Lightweight artifact and code validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B scripts/validate_stage1_artifacts.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest discover -s tests -v
ruff check src scripts tests
python -B -m json.tool configs/study_protocol.json >/dev/null
python -B -m json.tool configs/stage2_backend.json >/dev/null
```
