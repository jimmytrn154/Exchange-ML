# plan.md — ISBI 2027 Implementation Roadmap

**Paper:** *Beyond Voxel Uncertainty: Instance-Partition Disagreement for Active Learning in 3D Fracture Segmentation*  
**Target:** IEEE ISBI 2027 main conference, four-page paper  
**Official submission:** **26 October 2026**  
**Internal freeze:** **23 October 2026**

## 1. Fixed study contract

### PENGWIN
- Primary dataset: 340 public labeled cases.
- Static split: **50 test / 40 validation / 250 AL pool**.
- Initial labeled set: 40 cases.
- Five whole-volume query rounds, 20 cases/round.
- Budget: **40 → 60 → 80 → 100 → 120 → 140**.
- Minimum **3 independent AL trajectories**.
- Ground truth may define the static split but may never influence acquisition.

### RibFrac
- Secondary cross-anatomy dataset.
- Mandatory use: H3 failure-detection transfer with the exact PENGWIN FA-IPD definition.
- Optional use: short 3–4-round AL study after PENGWIN is complete.

### Common model protocol
- Same segmentation backend, preprocessing, training policy and instance reconstruction for every acquisition strategy.
- Main ensemble size: **M = 3**.
- Test `M ∈ {2,3,4}` only in Round-0 ablation.
- Contribution is the acquisition/uncertainty signal, not a new segmentation architecture.

### Main method

For each unordered ensemble pair, construct the foreground union `U_ij` and
intersection `C_ij` inside the fixed target ROI, where
`U_ij = F_i union F_j` and `C_ij = F_i intersection F_j`. Define

\[
d_{FA}(P^{(i)},P^{(j)})=
\frac{|F_i\triangle F_j|}{|U_{ij}|}
+\frac{|C_{ij}|}{|U_{ij}|}
NVI(P^{(i)}|_{C_{ij}},P^{(j)}|_{C_{ij}}),
\]

\[
FA\text{-}IPD(x)=\frac{2}{M(M-1)}\sum_{i<j}d_{FA}(P^{(i)},P^{(j)}).
\]

\[
A(x)=\frac{1}{2}R_U(x)+\frac{1}{2}R_{FA\text{-}IPD}(x)
\]

The implementation defines both-empty as `0`, exactly-one-empty as `1`, and
NVI as `0` for an intersection of at most one voxel. FA-IPD is a bounded,
symmetric, permutation-invariant disagreement score; it is **not** called a
mathematical metric because triangle inequality is not guaranteed.

Before H1, freeze NVI normalization, pairwise ROI/domain and empty-mask rules,
connectivity, component filtering, morphology, entropy aggregation and instance
reconstruction.

---

## 2. Implementation structure

```text
data + static split
      ↓
fixed segmentation backend
      ↓
ensemble probability / prediction cache
      ↓
deterministic instance reconstruction
      ↓
uncertainty scores
  entropy | pairwise Dice | count | Hungarian | FA-IPD
      ↓
whole-volume acquisition
      ↓
AL loop
      ↓
H1 failure detection + H2 learning curves + H3 transfer
```

All acquisition methods must operate on the same predictions. Do not maintain separate end-to-end training pipelines for different baselines.

---

## 3. Baselines

| Priority | Method | H1 | H2 | Decision |
|---|---|---:|---:|---|
| Must | Uniform Random | — | ✓ | Required |
| Must | Foreground-/prediction-aware Random | — | ✓ | Required strong random |
| Must | Predictive Entropy | ✓ | ✓ | Required voxel uncertainty |
| Must | Mean pairwise semantic Dice disagreement | ✓ | ✓ | Required ensemble baseline |
| Must | Foreground Jaccard disagreement | ✓ | ablation | Isolates the FA-IPD presence term |
| Must | Union-foreground NVI | ✓ | ablation | Isolates the original partition term and missing-fragment failure |
| Must | Fragment-count variance | ✓ | ablation | Tests cardinality confounding |
| Must | Hungarian fragment disagreement | ✓ | ablation | Original structural baseline |
| Must | **FA-IPD** | ✓ | ✓ | Proposed structural signal |
| Must | **Entropy + FA-IPD** | — | ✓ | Main acquisition method |
| Strong | Stochastic Batch | — | ✓ | Preferred external AL comparator |
| Optional | PAAL | — | ✓ | Add only if acquisition logic integrates cleanly |
| Discuss | ClaSP PE / nnActive | — | — | Patch/partial-query protocol mismatch |
| Discuss | CSAL-3D | — | — | SSL/cold-start protocol mismatch |
| Discuss | SBC-AL | — | — | Changes training architecture |
| Discuss | RegAL | — | — | Coupled AL + SSL/registration |
| Discuss | USIM | — | — | Relevant, but no verified official public repo found in repository check |

For ISBI, a fair six-method core comparison is more important than reproducing every related paper.

---

## 4. Repositories to reuse

| Repository | URL | Use |
|---|---|---|
| nnU-Net v2 | https://github.com/MIC-DKFZ/nnUNet | Common segmentation infrastructure |
| nnActive | https://github.com/MIC-DKFZ/nnActive | AL framework ideas, entropy, foreground-aware random, ClaSP reference |
| PENGWIN 2026 ABBC | https://github.com/RURUGURU/pengwin2026-task1-abbc | **Preferred PENGWIN backend reference**: boundary/core + instance reconstruction |
| PENGWIN Challenge | https://github.com/yuepeiyan/PENGWIN_Challenge | Secondary PENGWIN nnU-Net/inference reference |
| StochasticBatchAL | https://github.com/Minimel/StochasticBatchAL | Preferred external acquisition baseline |
| PAAL-MedSeg | https://github.com/shijun18/PAAL-MedSeg | Optional predictive-accuracy acquisition baseline |
| CSAL-3D | https://github.com/HiLab-git/CSAL-3D | Optional recent 3D AL reference |

Do not compare repositories by running their original end-to-end models independently. Reuse/adapt their **acquisition logic** inside the common backend whenever a fair comparison is possible.

---

# 5. Stage roadmap

## Stage 0 — Readiness and frozen protocol
**Target: 20–22 Sep**

### Work
- Resolve PENGWIN 2026 publication/embargo permission for an ISBI submission.
- Confirm all PENGWIN data and label semantics.
- Obtain RibFrac validation images if using the official validation split.
- Generate the fixed PENGWIN split.
- Freeze H1/H2 metrics and the semantic Dice non-inferiority margin.

### Report
`__docs__/reports/stage0_readiness.md`: data/permission status, split, frozen protocol, blockers.

### Gateway
**PASS** only when PENGWIN is publication-usable and the data/split/evaluation contract is fixed.

---

## Stage 1 — Instance and metric audit
**Target: 20–25 Sep**

**Status: PASS (2026-09-21) — internal research contract.**

### Work
- Load PENGWIN while preserving geometry and subject-local instance IDs.
- Audit fragment count and instance size.
- Compare 6/18/26 connectivity and freeze the cleanup rule.
- Preserve every valid GT instance ID. Apply cleanup only to predictions using
  26-connectivity and remove per-ID components below `1000 mm3`; apply no
  additional morphology. Retain cleanup-disabled evaluation as an ablation.
- Implement Dice, Instance F1, merge/split error, surface distances, and FA-IPD.
- Build H4 controlled perturbations: merge, split, missing fragment, matched boundary perturbation.
- Unit-test FA-IPD identity, symmetry, label-permutation invariance, ROI handling,
  empty masks, pairwise ensemble aggregation, and missing/extra fragments.
- Preserve a finite triangle-inequality counterexample so the paper cannot
  silently relabel FA-IPD as a mathematical metric.

### Report
`__docs__/reports/stage1_instance_audit.md`: chosen instance rules + metric/perturbation sanity table.

### Gateway
**PASS** if:
- GT vs GT gives perfect instance matching and zero merge/split error;
- instance-ID relabeling does not change FA-IPD;
- missing/extra foreground gives a positive presence penalty while equal-support
  merge/split cases reduce to NVI;
- empty-mask and ROI behavior matches the frozen contract;
- the 340-case connectivity/cleanup audit is complete and its rule is frozen.

The synthetic metric sub-gate and the 340-case artifact validation pass. The
frozen rule is recorded in `configs/study_protocol.json`; the full evidence and
artifact checksum are recorded in the Stage 1 report. This is an internal
evaluation contract, not a claim of equivalence to an unverified official
PENGWIN evaluator.

---

## Stage 2 — Stable fixed PENGWIN backend
**Target: 23–30 Sep**

**Status: PASS (2026-09-26) — internal research backend.**

### Work
1. Inspect/adapt the **PENGWIN 2026 ABBC boundary/core** implementation.
2. Use nnU-Net v2 components where possible.
3. Freeze architecture, target representation, preprocessing, loss, reconstruction, post-processing and training schedule.
4. Evaluate one model on validation data.

### Report
`__docs__/reports/stage2_backend.md`: Dice, Instance F1, merge/split error, final backend specification.

### Gateway
**PASS** only if the model produces stable, meaningful instance predictions suitable for structural uncertainty analysis.

---

## Stage 3 — FA-IPD + H1 Round-0 benchmark
**Target: 29 Sep–4 Oct**

**Status: FAIL (2026-10-05) — M=3 execution complete; the H1 scientific gateway was not met. See `__docs__/reports/stage3_h1.md`.**

**This is the main scientific go/no-go gate.**

### Work
Train one `M=3` ensemble from the initial labeled subset using frozen model
seeds `31001`, `31002`, and `31003`. Apply each seed to Python, NumPy,
PyTorch, and CUDA, with `seed + local DDP rank` for stochastic rank-local
streams; request deterministic cuDNN kernels and disable cuDNN benchmarking. Do
not reuse the Stage 2 checkpoints, because their model seed was not explicitly
recorded.

For each member, retain the deterministic ABBC instance map and the genuine
Stage-B foreground probability `1-p(background)`. Use the union of the three
cleaned member foreground predictions as one shared case ROI. Apply the Stage 1
prediction-only cleanup rule before every structural score.

Compute for every analysis case:
- mean entropy;
- 95th-percentile entropy;
- pairwise semantic Dice disagreement;
- foreground Jaccard disagreement;
- union-foreground NVI;
- fragment-count variance;
- Hungarian structural disagreement;
- FA-IPD.

Evaluate against:
- `1 - Instance F1`;
- normalized merge + split error;
- `1 - Dice` as semantic control.

Required analyses:
- risk–coverage / AURC;
- top-10% and top-20% failure enrichment;
- fragment-count, foreground-volume, and minimum-fragment-volume stratification;
- count/volume/minimum-fragment-volume-controlled association;
- empirical confounder quartiles without splitting ties; omit empty strata;
- Spearman correlation as secondary.

Acquisition scoring is GT-free. Retrospective H1 evaluation joins the frozen
Stage 2 single-model validation risks only after all ensemble scores are written.

### Report
`__docs__/reports/stage3_h1.md`: one comparison table, risk–coverage figure, one low-entropy/high-FA-IPD failure example.

### Gateway
Proceed to H2 only if FA-IPD shows a meaningful advantage in at least **2 of 3**:
1. AURC;
2. top-k enrichment;
3. count/volume/minimum-fragment-volume-controlled association.

The effect must not reduce to foreground Jaccard disagreement or “cases with
more/larger fragments receive higher FA-IPD.”

---

## Stage 4 — AL engine + two-round pilot
**Target: 4–8 Oct**

**Status: BLOCKED — Stage 3 H1 gateway failed; the pilot has not been implemented or run.**

### Work
Implement a whole-volume AL loop with identical training/evaluation for all methods.

Pilot:
- Random;
- Entropy;
- FA-IPD;
- Entropy + FA-IPD;
- 3 seeds;
- first 2 rounds.

Track Instance F1, merge/split error, Dice and selected-case characteristics.

### Report
`__docs__/reports/stage4_al_pilot.md`: early learning curves + implementation stability.

### Gateway
**PASS** if Stage 3 passed, trajectories are stable, and Hybrid does not show obvious semantic degradation beyond the frozen Dice guardrail.

---

## Stage 5 — Full PENGWIN H2 study
**Target: 8–15 Oct**

### Main comparison
1. Uniform Random
2. Foreground-aware Random
3. Predictive Entropy
4. Pairwise Dice disagreement
5. FA-IPD
6. Entropy + FA-IPD
7. Stochastic Batch, if cleanly adapted

Add PAAL only if integration is straightforward and does not change the common backend.

### Protocol
- ≥3 paired trajectory seeds.
- Five rounds.
- Identical initial set within each trajectory.
- Same fixed test set and training policy.
- Save selected case IDs and acquisition scores each round.

### Primary endpoints
- **AULC Instance F1**
- **AULC merge + split error**

### Statistics
- trajectory-paired AULC comparison;
- paired permutation test or Wilcoxon signed-rank;
- bootstrap CIs over test cases;
- effect sizes.

### Report
`__docs__/reports/stage5_h2_pengwin.md`: learning curves + compact AULC/final-budget table.

### Gateway
**PASS as an experiment** when every mandatory method completes the same budgets and seeds and H2 can be answered without cherry-picking. A negative H2 result is still a valid result.

---

## Stage 6 — RibFrac H3 transfer
**Target: 12–18 Oct**

### Mandatory work
Train a RibFrac ensemble and evaluate:
- Entropy;
- Pairwise Dice disagreement;
- count/Hungarian where meaningful;
- FA-IPD.

Use the **same** FA-IPD definition, aggregation and H1 evaluation protocol as PENGWIN. No dataset-specific retuning after seeing RibFrac results.

### Optional
If time remains, run a short RibFrac AL study:
- Random / Entropy / FA-IPD / Hybrid;
- 3 seeds;
- 3–4 rounds.

### Report
`__docs__/reports/stage6_h3_ribfrac.md`: PENGWIN vs RibFrac failure-detection table/figure.

### Gateway
**PASS** when H3 has been evaluated without score redesign. A negative result narrows the claim; it does not invalidate the stage.

---

## Stage 7 — Focused ablations and mechanism analysis
**Target: 16–20 Oct**

### Required
- count variance vs Hungarian vs union-NVI vs foreground Jaccard vs FA-IPD;
- Entropy vs FA-IPD vs Hybrid;
- `M = 2,3,4` on Round 0;
- cleanup off vs fixed cleanup;
- fusion sensitivity `γ ∈ {0,.25,.5,.75,1}` on non-test data;
- selected-case distributions: fragment count, foreground volume, merge/split prevalence, baseline error.

Only run diversity/query-budget sensitivity if time permits.

### Report
`__docs__/reports/stage7_ablation.md`: one compact ablation table and, if useful, one selection-behavior figure.

### Gateway
**PASS** when headline gains cannot reasonably be explained only by count, cleanup, ensemble size or unequal diversity handling.

---

## Stage 8 — Paper freeze and ISBI submission
**Target: 20–26 Oct**

### Four-page structure
- **Page 1:** problem, gap, closest work, contributions.
- **Page 2:** fixed backend, FA-IPD, hybrid acquisition.
- **Page 3:** protocol + main H1/H2 results.
- **Page 4:** H3, focused ablation/qualitative example, limitations, conclusion.

### Required paper assets
- Method figure: `CT → ensemble → entropy + partitions → FA-IPD → query`.
- Main Instance-F1 / merge-split learning curve.
- Compact H1/H2/H3 result table.
- Low-entropy/high-FA-IPD structural-failure example.
- Reproducibility details and compliance/data-use statement.

### Claim gate

| Results | Manuscript framing |
|---|---|
| H1 + H2 + H3 positive | Full FA-IPD active-learning + transfer paper |
| H1 + H2 positive, H3 negative | PENGWIN-focused AL paper with explicit transfer boundary |
| H1 + H3 positive, H2 negative | Structural failure-detection paper; AL is a negative downstream result |
| H1 negative | Do not claim FA-IPD as a validated structural-uncertainty mechanism |

### Final gateway
Submit only if:
- all reported numbers come from frozen outputs;
- all H2 methods share the same backend and training policy;
- no test-set tuning occurred;
- novelty is framed as **instance-partition disagreement for structural uncertainty/acquisition**, not invention of VI or generic topology-aware AL;
- all technical content fits the ISBI four-page limit.

**Target upload: 25 October 2026.**

---

## 6. Critical path

```text
Stage 0: permission + protocol
            ↓
Stage 1: instances + metrics
            ↓
Stage 2: stable backend
            ↓
Stage 3: H1 GO/NO-GO
            ↓
Stage 4: AL pilot
            ↓
Stage 5: full PENGWIN H2
         ↙             ↘
Stage 6: H3       Stage 7: ablations
         ↘             ↙
          Stage 8: paper
```

**Critical path:** stable backend → convincing H1 → repeated PENGWIN AL → RibFrac transfer → paper.

Do not spend deadline-critical time reproducing every related AL method.

---

## 7. Minimal stage report

Every stage report needs only:

```markdown
# Stage X

Status: PASS / FAIL / BLOCKED

## Configuration
Key data split and frozen settings.

## Result
Main table/figure and short interpretation.

## Gateway
Why the stage passed or failed.

## Blocking issue
Only if it affects the next stage.
```
