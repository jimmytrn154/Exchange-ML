# Beyond Voxel Uncertainty: Instance-Partition Disagreement for Active Learning in 3D Fracture Segmentation

## Research blueprint for an IEEE ISBI 2027 main-track submission

> **Status:** research design, not a results manuscript.  
> **Target:** IEEE ISBI 2027 full paper.  
> **Design principle:** the paper must not depend on a single fragile hypothesis. It is deliberately built around three partially independent scientific claims—signal validity, active-learning utility, and cross-task transfer—so that a negative result on one axis does not automatically invalidate the entire study.

---

## 1. Executive thesis

The original proposal starts from a strong observation: for fracture-instance segmentation, voxel-level uncertainty can be low even when a model makes a clinically meaningful **merge, split, or missing-fragment error**. The core idea should be retained, but the novelty claim needs to be sharpened substantially.

The paper should **not** be framed as another generic “topology-aware active learning” method. Recent work already occupies that space:

- **SBC-AL (MICCAI 2024)** selects samples using structure and boundary consistency [R8].
- **RegAL (2026)** explicitly combines voxel uncertainty, feature diversity, and topological consistency inside an active/semi-supervised learning framework [R9].
- **Topology-Aware Uncertainty (NeurIPS 2023)** already argues that pixel-wise uncertainty can be the wrong granularity for structurally meaningful segmentation failures [R10].

The defensible gap is narrower and more specific:

> **Existing medical active-learning methods mostly reason about voxel/class uncertainty, boundary consistency, representation diversity, or fixed anatomical connectedness. They do not explicitly measure disagreement over a variable-cardinality predicted instance partition, where important errors are merge/split/missing-instance failures.**

Pelvic fracture segmentation is a particularly suitable test case because the target is a variable set of fragments rather than a fixed semantic class set. PENGWIN evaluates instance F1, merge errors, split errors, and topology consistency in addition to conventional overlap metrics [R3, R4]. RibFrac provides a second anatomically distinct fracture-instance task [R25, R26].

The scientific question should therefore become:

> **When segmentation is fundamentally instance-structured, does disagreement over the predicted instance partition provide useful information that voxel-wise uncertainty misses?**

This yields three primary hypotheses:

1. **H1 — Signal validity:** instance-partition disagreement detects true structural failures better than standard voxel uncertainty and semantic ensemble disagreement.
2. **H2 — Active-learning utility:** using this structural signal improves case-level label-budget efficiency for instance segmentation.
3. **H3 — Cross-task transfer:** the same structural score remains useful on a second fracture-instance task without dataset-specific re-tuning.

This is intentionally failure-resistant. If H2 fails but H1 and H3 are strong, the work can still be reframed as a structural uncertainty / failure-detection paper rather than collapsing completely.

---

# 2. What should change from the current proposal

## 2.1 Replace the broad “topology-aware” claim

The current formulation uses:

- fragment-count variance;
- Hungarian matching between predicted fragments;
- pairwise fragment IoU;
- a weighted topology score;
- voxel entropy;
- morphology;
- k-center diversity.

That is understandable and implementable, but after SBC-AL and RegAL it is difficult to defend the novelty as simply “adding structure/topology to active learning” [R8, R9].

### Proposed reframing

Use the term:

- **Foreground-Aware Instance-Partition Disagreement (FA-IPD)** — frozen main method; or
- **Fragment-Partition Disagreement (FPD)** — if the paper remains fracture-specific.

Avoid using “topology-aware” as the central novelty term unless a genuine topological invariant is introduced.

---

## 2.2 Reduce heuristic complexity

SBC-AL is a useful warning. Public MICCAI reviews praised its motivation but also criticized the number of interacting modules and asked for:

- repeated experiments with different initial labeled sets;
- standard deviations;
- stronger diversity/hybrid baselines;
- the same model across acquisition strategies;
- active-learning curves;
- explicit separation between gains from training architecture and gains from sample selection [R8].

For a four-page ISBI paper, method simplicity is a strength.

### Design rule

Keep the segmentation model and instance post-processing fixed. Put the paper contribution almost entirely in the **acquisition / uncertainty score**.

---

## 2.3 Fix the instance-segmentation formulation

PENGWIN instance IDs are patient-specific fragment labels; they are not globally meaningful semantic classes. A network cannot simply learn “fragment 1”, “fragment 2”, ..., across subjects.

Recent pelvic-fracture work instead uses instance-compatible representations such as:

- primary/secondary or major/minor fragment representations;
- boundary/core separation;
- deterministic instance reconstruction after semantic prediction [R19, R20, R24].

### Consequence

The paper should adopt an **established fragment-instance backend** and explicitly state that the active-learning contribution is **post-hoc and backbone-agnostic**.

---

## 2.4 Raise the active-learning evaluation bar

Recent work has made weak AL baselines unacceptable:

- Burmeister et al. showed that simple random/strided strategies can be surprisingly strong in 3D medical segmentation [R6].
- nnActive identified recurring evaluation problems and showed that common AL methods do not reliably outperform a stronger foreground-aware random baseline [R12].
- ClaSP Predictive Entropy later showed that carefully designed entropy can be much stronger than naïve entropy/random strategies [R13].
- USIM, PAAL, SBC-AL, and CSAL-3D all reinforce that contemporary AL comparisons need both uncertainty and diversity/representativeness controls [R8, R14, R15, R21].

### Consequence

The paper cannot use only `Random vs Entropy vs Ours`.

---

## 2.5 Do not validate uncertainty only by correlation

A 2024 MIDL paper demonstrated that uncertainty–quality correlations can be confounded by structure size [R16]. A 2025 Medical Image Analysis benchmark further found that simple ensemble pairwise Dice is a very strong failure-detection baseline and advocated risk–coverage analysis [R17].

### Consequence

H1 should use:

- risk–coverage / AURC;
- top-k failure enrichment;
- size/count-controlled analysis;
- correlation only as a secondary descriptor.

---

# 3. Literature review — priority 2021–2026

## 3.1 Active learning for medical image segmentation

### Nath et al., TMI 2021 — ensemble uncertainty + diversity

Nath et al. proposed a query-by-committee strategy for medical image segmentation, combining epistemic uncertainty with a diversity term. The study remains relevant because it established ensemble disagreement as a practical AL mechanism in medical segmentation [R5].

**Implication:** using an ensemble itself is not novel. The novelty must be **what kind of disagreement is measured**.

---

### Burmeister et al., 2022 — simple baselines can be strong

“Less Is More” compared active-learning strategies for 3D medical image segmentation and showed that simple random/strided sampling can perform competitively, with AL gains depending strongly on dataset and training conditions [R6].

**Implication:** repeated trajectories and strong random controls are necessary.

---

### Gaillochet et al., 2023 — stochastic batch selection

Gaillochet et al. proposed stochastic batches to combine informativeness with diversity and explicitly noted that random sampling can be hard to beat consistently [R7].

**Implication:** if diversity is used, it must be controlled. A proposed structural method should not get a diversity module that the entropy baseline does not receive.

---

### TAAL, 2023 — transformation-consistency uncertainty

TAAL uses test-time augmentation consistency as an uncertainty source and couples active learning with semi-supervised learning [R18].

**Implication:** “a different uncertainty score” is no longer enough. The paper must show that instance-partition disagreement captures a **specific failure type** that conventional uncertainty misses.

---

### USIM, MIDL 2024 — uncertainty + representativeness with nnU-Net

USIM combines uncertainty-aware selection with submodular mutual-information-based representativeness in an nnU-Net setting [R14].

**Implication:** hybrid uncertainty/diversity methods are now a standard comparison class. If USIM is not reproduced because of query-granularity or implementation mismatch, this should be explained rather than ignored.

---

### PAAL, IJCAI 2024 — predict segmentation accuracy

PAAL adds an accuracy predictor and weighted polling strategy to identify informative samples and maintain diversity [R15].

**Implication:** the field has moved beyond raw entropy. A structural uncertainty paper should show that its score predicts **instance-specific error**, not merely that it is numerically different from entropy.

---

### SBC-AL, MICCAI 2024 — structure and boundary consistency

SBC-AL is one of the closest precedents to the original proposal. It queries data based on structure and boundary consistency [R8].

Public MICCAI reviewer feedback is particularly instructive:

- repeat experiments across different initial labeled sets;
- report standard deviations;
- compare with diversity-oriented methods such as CoreSet/BADGE-style baselines;
- use the same segmentation architecture for random and proposed selection;
- show full active-learning curves, not isolated budget points;
- reduce architectural complexity and explain generalizability [R8].

**Novelty boundary for this paper:** SBC-AL reasons about anatomical feature/boundary consistency. The proposed work should target **disagreement over a variable-cardinality instance partition**, particularly merge/split ambiguity.

---

### CSAL-3D, MICCAI 2025 — cold-start 3D AL

CSAL-3D combines SSL-derived uncertainty and diversity in a cold-start 3D setting and evaluates multiple organs/datasets [R21].

**Implication:** 3D label-efficient learning is a mature and competitive space. A niche dataset alone will not establish novelty.

---

### nnActive, 2025 — evaluation rigor and the strong-random problem

nnActive evaluates multiple AL strategies across four 3D biomedical datasets and highlights recurring pitfalls:

1. too few datasets/budgets;
2. evaluation designs that do not match 3D annotation;
3. weak random baselines;
4. unrealistic annotation-cost claims [R12].

Its central warning is that AL gains over a well-designed random baseline are not guaranteed.

**Implication:** this paper should claim **case-level / whole-volume label-budget efficiency**, not real annotation-time savings unless annotation time is actually measured.

---

### ClaSP Predictive Entropy, 2026 — entropy is stronger than it used to be

ClaSP PE uses class stratification and scheduled power noising of predictive entropy and reports broad gains across multiple 3D settings, including transfer to unseen datasets [R13].

**Implication:** the paper should not assume “entropy is weak.” The hypothesis should be narrower:

> Voxel/class uncertainty may not rank **instance-partition failures** optimally.

Also note that ClaSP’s strongest evaluation uses patch-level/partial labeling. Our task is intentionally different because fragment identity/count are global instance properties.

---

### RegAL, 2026 — topology-aware AL already exists

RegAL combines voxel uncertainty, feature diversity, and topological consistency in an active + semi-supervised framework [R9]. Its topological criterion focuses on connectedness / structurally implausible predictions.

**Critical novelty implication:** a paper that only proposes “entropy + topology + diversity” would now appear incremental.

The proposed distinction should be:

- **RegAL:** fixed anatomical connectedness / topology consistency.
- **Ours:** uncertainty over **variable-cardinality instance partitions**, where many disconnected components are valid and the question is whether ensemble members agree on how the object decomposes into instances.

---

## 3.2 Structural uncertainty and segmentation failure detection

### Gupta et al., NeurIPS 2023 — topology-aware uncertainty

Gupta et al. argue that pixel-wise uncertainty is the wrong unit for curvilinear structures and estimate uncertainty over topological units using discrete Morse theory [R10].

**Relation:** strong conceptual support for moving uncertainty from voxels to structure.

**Difference:** the proposed work targets variable fracture instances and uses the score for pool-level acquisition and failure detection.

---

### Geißler et al., MIDL 2024 — structure size is a confounder

This work demonstrates that both segmentation quality and uncertainty estimates may depend strongly on the size of the segmented structure [R16].

**Required consequence:** H1 must control for:

- total foreground volume;
- fragment/instance count;
- optionally mean or minimum instance size.

Without this, a high FA-IPD score may simply mean “more/larger fragments” rather than “more useful uncertainty.”

---

### Berger et al., 2024 — pitfalls of topology-aware segmentation

Berger et al. discuss practical evaluation pitfalls including connectivity definitions, artifacts in topological ground truth, and metric selection [R22].

**Required consequence:** explicitly pre-specify:

- 3D connectivity convention;
- minimum-component filtering;
- morphology policy;
- sensitivity analysis for small components.

---

### Yang et al., Medical Image Analysis 2025 — structural uncertainty

SU-ASM estimates structural uncertainty using shape modeling rather than relying only on pixel uncertainty [R23].

**Relation:** additional evidence that global structural uncertainty can be useful.

**Difference:** fracture anatomy is poorly represented by a single coherent fixed-shape template because the number and geometry of fragments vary; a partition formulation is more natural.

---

### Zenk et al., Medical Image Analysis 2025 — failure-detection benchmark

Zenk et al. benchmark segmentation failure-detection methods across five 3D public datasets, advocate **risk–coverage analysis**, and find mean pairwise Dice disagreement between ensemble predictions to be a strong simple baseline [R17].

**Required baseline:** mean pairwise Dice disagreement must be included in H1.

---

## 3.3 Fracture instance segmentation

### PENGWIN

The PENGWIN benchmark demonstrates that pelvic fracture segmentation is an instance-structured task requiring specialized representations and post-processing. PENGWIN 2026 provides 500 clinical cases in total, with **340 public training cases and 160 hidden test cases**, and assigns unique IDs to connected fracture fragments [R3].

Official metrics include:

- Dice / fracture-local Dice;
- HD95 / ASSD;
- Instance F1, precision, recall;
- merge errors;
- split errors;
- topology consistency [R4].

This evaluation is unusually well matched to the scientific question of structural uncertainty.

---

### Recent pelvic-fracture segmentation methods

Recent work uses nnU-Net-derived pipelines, anatomical cropping, major/minor fragment representations, fracture-distance information, boundary/core separation, and post-processing to recover individual fragments [R19, R20, R24].

**Design implication:** do not spend ISBI novelty budget on a new segmentation architecture. Use an established backend and isolate the acquisition contribution.

---

### RibFrac

RibFrac provides approximately **5,000 rib-fracture instances in 660 CT scans**, with 420 training, 80 validation, and 160 hidden evaluation cases. The detection task is explicitly formulated in an instance-segmentation fashion [R25, R26].

**Why it is a good second dataset:** it changes anatomy, object scale, and fracture morphology while preserving the key requirement for H3: multiple fracture instances within a 3D CT volume.

---

# 4. Research gap

The literature supports four premises:

1. **Active learning in 3D segmentation is difficult to validate**, and strong random/entropy baselines matter [R12, R13].
2. **Structure-aware active learning already exists**, so “topology/structure-aware querying” alone is not sufficient novelty [R8, R9].
3. **Voxel uncertainty can be misaligned with structurally meaningful errors**, but structural uncertainty must be evaluated rigorously [R10, R16, R17, R23].
4. **Fracture segmentation is naturally an instance-partition problem** with meaningful merge/split failures [R3, R4, R19, R26].

The missing question is:

> **Can disagreement over the predicted instance partition serve as a reliable and transferable uncertainty signal for active selection in variable-instance 3D medical segmentation?**

This is narrower than “topology-aware AL,” but more defensible and experimentally falsifiable.

---

# 5. Datasets

## 5.1 Primary dataset: PENGWIN 2026 Task 1 — conditional on embargo clearance

**Public labeled data:** 340 clinical 3D CT scans; 160 test scans are hidden and cannot be treated as a simulated labeled AL pool [R3].

**Instance labels:** connected fragments receive separate IDs within bone-specific ranges [R3].

### Recommended static split

From the 340 public labeled scans:

- **50 fixed test** cases;
- **40 validation** cases;
- **250 active-learning pool** cases.

Within the 250-case AL pool:

- initial labeled seed: 40;
- five rounds × 20 newly labeled volumes;
- final labeled set: 140 / 250 cases.

Budget sequence:

`40 → 60 → 80 → 100 → 120 → 140`

This provides more learning-curve resolution than `40 + 5×30` and keeps the final budget far below the pool size.

### Split stratification

Use ground truth only to define the static train/validation/test split, never for acquisition. Stratify by:

- fragment count;
- affected bone group;
- femur involvement if applicable;
- scanner/site if metadata allow.

Report fragment-count and volume distributions for all splits.

### Publication-embargo contingency

PENGWIN 2026 states a 12-month publication embargo after challenge results [R2]. The exact applicability to independent work using only public Task-1 training data must be confirmed in writing with organizers.

If publication is not allowed for ISBI 2027:

- make RibFrac the primary dataset;
- use PENGWIN 2024 public data only if its publication restrictions permit;
- narrow or alter the paper claim accordingly.

---

## 5.2 Secondary dataset: RibFrac

Official public benchmark:

- 420 training CTs;
- 80 labeled validation CTs;
- 160 hidden evaluation CTs;
- approximately 5,000 fracture instances [R25, R26].

### Minimum required use

**Cross-task failure-detection validation:**

- train one ensemble on the public training data;
- evaluate FA-IPD vs entropy vs pairwise Dice for predicting instance failures on labeled validation cases;
- use the same FA-IPD definition and fusion rule as PENGWIN;
- do not retune FA-IPD on RibFrac.

This is sufficient to test H3 at low computational cost.

### Preferred stronger use

If compute permits, run a shorter simulated whole-volume AL study on RibFrac:

- 3–4 query rounds;
- 3 independent trajectories;
- Random, Entropy, FA-IPD, Hybrid.

The second dataset does **not** need the full ablation matrix used on PENGWIN.

---

# 6. Segmentation backend

## 6.1 Principle

All acquisition strategies must use the **same segmentation model, preprocessing, training schedule, and instance reconstruction**. Only the acquisition score changes.

This is essential to prevent the reviewer from attributing performance gains to a better network rather than better sample selection.

---

## 6.2 PENGWIN backend

Recommended pipeline:

1. anatomical localization / cropping for pelvic structures;
2. nnU-Net-style 3D segmentation with an **instance-compatible target representation**;
3. deterministic conversion to an instance map.

Two practical target representations:

- primary/secondary (major/minor) fragment representation;
- boundary/core representation [R19, R20].

Choose the one that can be reproduced most robustly and freeze it before AL experiments.

---

## 6.3 RibFrac backend

Use a strong, stable fracture segmentation/instance baseline.

If the network outputs a binary fracture mask, connected-component post-processing can create instances, but the following must be fixed in advance:

- connectivity;
- minimum component volume;
- any morphology;
- handling of touching predictions.

---

# 7. Proposed method

## 7.1 Ensemble

At AL round `t`, train an ensemble of `M` models on the current labeled set `D_L^t`.

Recommended:

- **M = 3** for all repeated AL trajectories;
- sensitivity check `M ∈ {2, 3, 4}` only in Round 0.

Three members are preferable to four if the saved compute enables three or more independent AL trajectories.

Each model returns:

- voxel probabilities `p^(m)(v)`;
- a deterministic predicted instance partition `P^(m)(x)`.

For the frozen PENGWIN Round-0 implementation, all three members are retrained
from scratch with model seeds `31001`, `31002`, and `31003`; each rank uses the
member seed plus its local DDP rank for Python, NumPy, PyTorch, and CUDA streams,
with deterministic cuDNN kernels requested and cuDNN benchmarking disabled. The
earlier Stage 2 checkpoints are not reused because their model seed was not
explicitly recorded. The voxel signal is the Stage-B binary foreground probability
`1-p(background)`, linearly resampled to the input grid and combined across
routed anatomies by voxelwise maximum. The deterministic partition retains the
unchanged ABBC affinity decoder and label-range reconstruction.

---

## 7.2 Baseline voxel uncertainty

For each voxel `v`, ensemble predictive entropy is computed from the mean class probability:

\[
\bar p_c(v)=\frac{1}{M}\sum_{m=1}^{M}p_c^{(m)}(v)
\]

\[
H(v)=-\sum_c \bar p_c(v)\log \bar p_c(v)
\]

Aggregate into a case-level score `U(x)` within a fixed anatomical/target ROI.

The frozen H1 aggregations are:

- mean entropy;
- 95th-percentile entropy.

Both are computed within one case-level ROI: the union of foreground predicted
by the three members after the frozen prediction-only cleanup (26-connectivity,
remove per-ID components below `1000 mm3`, no morphology). The same ROI is used
for every H1 score. This is a fixed protocol, not a hyperparameter search.

---

## 7.3 Foreground-Aware Instance-Partition Disagreement (FA-IPD)

### Core representation

For each ensemble member:

\[
P^{(m)}(x)=\{I^{(m)}_1,\ldots,I^{(m)}_{K_m}\},
\]

where the partition contains a variable number of predicted instances.
Instance IDs are arbitrary, so the comparison must be permutation-invariant.

### Pairwise disagreement

Variation of Information (VI) compares two partitions without aligning their
instance IDs:

\[
VI(P,Q)=H(P\mid Q)+H(Q\mid P).
\]

VI reacts to merge/split changes and variable cardinality. However, NVI on the
union of predicted foreground has a specific failure: a completely missing
instance can be renamed as background without changing the induced partition.
NVI on the full CT has the opposite problem because shared background can
dominate small foreground errors. We therefore separate foreground presence
from instance decomposition on the foreground shared by a prediction pair.

For predicted instance maps `P` and `Q`, restricted to the fixed target
ROI, define

\[
F_P=\{v:P(v)\neq0\},\qquad F_Q=\{v:Q(v)\neq0\},
\]

\[
U=F_P\cup F_Q,\qquad C=F_P\cap F_Q.
\]

The pairwise foreground-aware disagreement is

\[
d_{FA}(P,Q)=
\frac{|F_P\triangle F_Q|}{|U|}
+
\frac{|C|}{|U|}
NVI\!\left(P|_C,Q|_C\right).
\]

The first term is foreground Jaccard disagreement. The second measures how the
shared foreground is decomposed into instances, using `NVI = VI/log(|C|)`.
We define NVI as zero for `|C| <= 1`, `d_FA = 0` when both foregrounds are
empty, and `d_FA = 1`
when exactly one foreground is empty. Each unordered ensemble pair constructs
its own `U` and `C`; no ground-truth mask participates in acquisition.

The case-level score is

\[
FA\text{-}IPD(x)=\frac{2}{M(M-1)}\sum_{i<j}
d_{FA}\!\left(P^{(i)}(x),P^{(j)}(x)\right).
\]

All later references use the explicit name FA-IPD for this frozen definition.

### Properties and scope

FA-IPD is bounded in `[0,1]`, symmetric, and invariant to permutations of
instance IDs. If `F_P = F_Q`, the presence term is zero and the score reduces
to NVI on the common foreground, preserving pure merge/split sensitivity. A
completely missing or extra foreground region instead receives direct presence
penalty without adding agreed-background CT voxels.

FA-IPD is a **dissimilarity score, not a mathematical metric**. Its
pair-dependent intersection can violate the triangle inequality. It is also a
composite rather than pure NVI: boundary displacement contributes through the
Jaccard term, and missing-fragment penalty is voxel-volume weighted. These are
testable limitations, so H1/H4 must include pairwise Dice or Jaccard, union-NVI,
matched-boundary perturbations, and fragment-size stratification.

Pre-register:

- fixed target ROI;
- connectivity convention;
- minimum-component filtering;
- morphology policy;
- NVI normalization;
- pairwise empty-mask behavior and ensemble aggregation.

### Why this is cleaner than the original score

Original concept:

\[
T=\lambda\,\widetilde{Var(K)}+(1-\lambda)\,\widetilde{HungarianIoUDisagreement}.
\]

Proposed:

\[
FA\text{-}IPD=\text{mean pairwise foreground-aware partition disagreement}.
\]

Advantages:

- one deterministic two-level score instead of a tuned scalar mixture;
- no fragment-ID alignment;
- explicitly handles missing/extra foreground and merge/split structure;
- fewer hyperparameters;
- the same definition transfers to PENGWIN and RibFrac.

The original count/Hungarian score, foreground Jaccard disagreement alone, and
union-foreground NVI alone remain required **ablation baselines**. This isolates
whether gains come from presence disagreement, partition disagreement, or their
foreground-aware composition.

---

## 7.4 Hybrid acquisition score

Do not rely on a heavily tuned scalar combination.

Convert both scores to percentile ranks in the current unlabeled pool:

\[
R_U(x)=rank_{pct}(U(x)),\qquad R_S(x)=rank_{pct}(FA\text{-}IPD(x))
\]

Primary method:

\[
A(x)=\frac{1}{2}R_U(x)+\frac{1}{2}R_S(x)
\]

Why rank fusion:

- entropy and FA-IPD live on different scales;
- min–max normalization is sensitive to extreme outliers;
- equal weighting avoids post-hoc tuning.

Small sensitivity only:

\[
A_\gamma=\gamma R_U+(1-\gamma)R_S,
\quad \gamma\in\{0,.25,.5,.75,1\}
\]

The primary value should remain `γ = 0.5` regardless of test results.

---

## 7.5 Diversity control

For the main scientific comparison, **do not let diversity be unique to the proposed method**.

Primary comparison:

- Entropy: top-`b` by `U`;
- FA-IPD: top-`b` by `FA-IPD`;
- Hybrid: top-`b` by `A`.

Optional controlled extension:

- Entropy + same diversity module;
- Hybrid + same diversity module.

If k-center or stochastic-batch diversity is used, apply it identically to both. Otherwise the paper cannot isolate the value of FA-IPD.

---

# 8. Primary hypotheses

## H1 — Instance-partition disagreement is a better detector of structural failure

### Statement

> **H1:** On unseen volumes, FA-IPD ranks true merge/split/instance-segmentation failures more accurately than voxel predictive entropy and semantic ensemble disagreement.

This is the **measurement-validity hypothesis**. It has scientific value independently of active-learning performance.

### Ground-truth risk definitions

The frozen Round-0 structural risks are:

1. `1 − Instance F1`;
2. normalized number of merge + split errors;
3. `1 − Dice` as a semantic control.

These risks come from the frozen Stage 2 single-model predictions on the same
40 validation cases. They are joined only in retrospective evaluation; ground
truth and Stage 2 risk values are never read by the acquisition-score program.

### Scores compared

- mean predictive entropy;
- 95th-percentile predictive entropy;
- mean pairwise semantic Dice disagreement;
- foreground Jaccard disagreement;
- union-foreground NVI;
- fragment-count population variance;
- anatomy-restricted Hungarian fragment-matching disagreement;
- **FA-IPD — proposed**.

### Primary H1 evaluation

#### A. Risk–coverage / AURC

Sort cases from most to least uncertain. At each retained coverage, compute remaining segmentation risk. Lower AURC indicates a better failure detector.

#### B. Top-k failure enrichment

For the highest-uncertainty 10% and 20% of cases, report:

- prevalence of at least one merge/split failure;
- mean structural risk;
- enrichment relative to dataset base rate.

#### C. Spearman correlation

Use only as a secondary descriptive analysis.

#### D. Confounder control

Because structure size can confound uncertainty-quality relations [R16]:

- stratify by fragment-count bin;
- stratify by foreground-volume quartile;
- stratify missing-fragment analyses by removed-fragment volume;
- fit a simple residualized/partial analysis controlling for fragment count,
  total foreground volume, and minimum retained-fragment volume;
- form empirical confounder quartiles without splitting equal values across
  strata, and omit empty strata rather than manufacturing rank-only bins.

### Falsification criterion

H1 fails if FA-IPD is consistently no better than entropy/pairwise Dice or its
Jaccard-only ablation in AURC and top-k enrichment after confounder control.

### Why H1 can carry the paper

If H1 is strong on two datasets, the study establishes a meaningful **instance-level structural uncertainty / failure-detection result**, even if downstream AL gains are modest.

---

## H2 — Structural disagreement improves active-learning label efficiency

### Statement

> **H2:** At equal whole-volume labeling budget, hybrid entropy + FA-IPD selection improves instance-level learning efficiency relative to entropy-based and strong non-structural acquisition baselines, without materially degrading semantic segmentation performance.

This is the **downstream utility hypothesis**.

### Primary endpoints

Area under the active-learning learning curve (**AULC**) for:

- Instance F1 — primary;
- merge + split error — co-primary.

### Secondary endpoints

- Dice / local fracture Dice;
- HD95 / ASSD;
- PQ if used;
- annotation budget needed to reach 90% or 95% of full-data Instance-F1 performance.

### Semantic non-inferiority guardrail

A structural selection method that improves instance metrics while damaging semantic segmentation is not convincing.

Pre-specify a small semantic non-inferiority margin, e.g. no more than **1 absolute Dice point** below the best non-structural uncertainty baseline at matched budget. The exact margin must be frozen before test evaluation.

### Required baselines

**Mandatory**

1. Uniform Random.
2. Stronger prediction/foreground-aware random variant.
3. Predictive Entropy.
4. Pairwise Dice ensemble disagreement.
5. FA-IPD only.
6. Hybrid Entropy + FA-IPD.

**Highly desirable if stable**

7. Stochastic Batch [R7].
8. PAAL or USIM [R14, R15].

**Related work to cite/discuss, not necessarily force into a like-for-like reproduction**

- SBC-AL: changes the segmentation/training pipeline [R8].
- RegAL: combines AL with semi-supervised registration-guided learning [R9].
- ClaSP PE: strongest evidence uses patch queries / partial labels, whereas this paper studies whole-volume global instance structure [R13].

### Whole-volume query justification

Generic 3D AL often benefits from patch querying, but fragment identity and fragment count are global properties. A partial patch may cut through an object and cannot define the full-patient instance partition.

Therefore the query unit should remain **whole CT volume / case**.

Do **not** claim human annotation-time savings unless annotation time is measured. Claim:

- whole-volume label-budget efficiency;
- case-level annotation efficiency.

---

## H3 — The structural signal transfers across fracture-instance tasks

### Statement

> **H3:** Without re-tuning the FA-IPD definition, FA-IPD retains superior structural-failure ranking on RibFrac and, if compute permits, improves a limited RibFrac active-learning trajectory.

This is the **generalization hypothesis**.

### Mandatory H3 experiment

Train a RibFrac ensemble and compare:

- FA-IPD;
- predictive entropy;
- pairwise Dice disagreement;
- count/Hungarian structural baselines where meaningful.

Use the exact same H1 risk–coverage / top-k enrichment protocol.

Do not re-tune:

- FA-IPD pairwise domain, NVI normalization, and empty-mask rules;
- rank-fusion weight;
- score aggregation;
- component filtering except physically necessary resolution scaling declared in advance.

### Preferred extension

Run a 3–4-round RibFrac simulated AL experiment.

This is highly valuable, but **H3 does not depend on completing it**. Signal-transfer evidence already protects the paper from being a one-dataset observation.

### Falsification criterion

H3 fails if FA-IPD loses its structural-failure advantage after changing anatomy/task or requires dataset-specific score redesign.

### If H3 fails

If H1 + H2 succeed on PENGWIN but H3 fails:

- narrow the claim to pelvic fracture / fragment segmentation;
- present RibFrac as a boundary condition rather than hiding it;
- do not claim general instance-structured medical AL.

That can still be a viable ISBI paper if the primary PENGWIN evidence is strong.

---

# 9. Auxiliary construct-validation hypothesis

## H4 — FA-IPD responds to presence and partition errors as designed

H4 should be a low-cost mechanistic analysis, not a headline claim.

Create controlled instance-map perturbations:

- merge two adjacent instances;
- split one instance;
- remove a small instance;
- apply erosion/dilation or boundary jitter with approximately matched Dice degradation.

Compare how strongly the following react:

- pairwise semantic Dice;
- foreground Jaccard disagreement;
- union-foreground NVI;
- original count/Hungarian score;
- FA-IPD.

Expected interpretation:

- when foreground support is fixed, FA-IPD should equal intersection-NVI and
  remain sensitive to merge/split changes;
- a completely missing or extra fragment should produce positive FA-IPD even
  when union-foreground NVI is zero;
- FA-IPD may react strongly to boundary perturbation and may underweight a
  clinically important small fragment because its presence term is voxel-weighted.

This validates **what the score measures and where it can fail**. It is not
sufficient alone to prove clinical or active-learning utility.

---

# 10. Intended contributions

Only claim contributions directly supported by experiments.

## Contribution 1 — Instance-partition uncertainty formulation

A simple, permutation-invariant **Foreground-Aware Instance-Partition
Disagreement** score that combines foreground-presence disagreement with NVI
over the shared predicted foreground.

**Do not claim invention of VI, Jaccard disagreement, or a new mathematical
metric.** The novelty claim is limited to the foreground-aware ensemble
uncertainty/acquisition formulation and its evaluation for variable-instance
medical segmentation.

## Contribution 2 — Structural-failure validation

A direct test of whether instance-partition uncertainty identifies merge/split and low-instance-F1 failures better than:

- voxel entropy;
- semantic ensemble disagreement;
- the original count/matching heuristic.

This contribution depends on risk–coverage and confounder-controlled analysis.

## Contribution 3 — Active-learning evidence

A paired repeated simulated AL study determining whether adding instance-partition information improves case-level labeling efficiency.

## Contribution 4 — Cross-task evidence

Evidence on a second fracture-instance benchmark showing whether the same structural score transfers without re-tuning.

For a four-page ISBI paper, Contributions 1–3 are core; Contribution 4 should be compact but high-value.

---

# 11. Experiment plan

## Experiment 0 — Dataset / label audit

Before training:

1. verify PENGWIN label ranges and instance reconstruction;
2. compute fragment-count distribution;
3. compute smallest/largest instance volume;
4. quantify behavior under 6-, 18-, and 26-connectivity;
5. determine whether tiny components require a fixed minimum-size rule;
6. reproduce the frozen internal instance metrics on ground truth and controlled
   perturbations.

**Go/no-go:** do not proceed until instance extraction and evaluation are stable.

The frozen internal evaluation contract preserves every annotated subject-local
GT instance ID, independent of volume or connectedness. Cleanup is
prediction-only: within each predicted ID, 26-connected components smaller than
`1000 mm3` are removed, with no additional morphology. Primary evaluation uses
this rule and reports a cleanup-disabled ablation. The completed 340-case audit
contains 2,427 annotated instances, including 513 below `500 mm3` and 704 below
`1000 mm3`; filtering GT at either reference threshold would therefore alter the
task. These are dataset-audit observations, not model-performance results, and
the internal contract is not claimed to reproduce an unverified official
evaluator.

---

## Experiment 1 — Stable backend reproduction

Train a strong fixed segmentation backend before any AL study.

Report:

- Dice;
- Instance F1;
- merge/split errors;
- post-processing behavior.

This is not a paper contribution, but unstable instance predictions make any AL conclusion meaningless.

---

## Experiment 2 — Round-0 uncertainty benchmark — highest priority

This is the first serious **go/no-go experiment**.

Train one ensemble on the initial labeled subset.

For every remaining case compute:

- entropy;
- pairwise Dice disagreement;
- foreground Jaccard disagreement;
- union-foreground NVI;
- fragment-count variance;
- Hungarian structural disagreement;
- FA-IPD.

Reveal GT only for retrospective analysis.

Evaluate:

- AURC against structural risk;
- top-10/top-20 failure enrichment;
- Spearman correlation;
- fragment-count / total-foreground-size / minimum-fragment-size-controlled analysis.

### Practical go criterion

Proceed to the expensive full AL study only if FA-IPD shows a meaningful advantage in at least **two of three**:

1. AURC;
2. top-k enrichment;
3. confounder-controlled association.

If FA-IPD does not outperform entropy/pairwise Dice and its Jaccard-only
ablation here, do **not** burn hundreds of GPU-hours on full AL trajectories.
Redesign or pivot.

---

## Experiment 3 — PENGWIN full active-learning trajectories

Protocol:

- same fixed test set for every method;
- same initial labeled set within each trajectory;
- at least **3 independent trajectory seeds**;
- ideally 5 if compute permits;
- five query rounds.

Methods:

- Random;
- stronger random;
- Entropy;
- pairwise-Dice disagreement;
- FA-IPD;
- Hybrid Entropy + FA-IPD;
- one recent hybrid baseline if stable.

At every round:

- use the same train-from-scratch or warm-start policy for all methods;
- evaluate on the same fixed test set;
- save selected-case metadata.

Report:

- mean ± SD / 95% CI;
- Instance-F1 learning curve;
- merge/split learning curve;
- Dice learning curve;
- trajectory-level AULC.

### Statistical comparison

Use trajectory-paired comparisons:

- paired permutation test or Wilcoxon signed-rank on trajectory-level AULC;
- bootstrap confidence intervals over test cases for endpoint differences;
- report effect sizes, not only p-values.

Avoid many uncorrected per-round significance tests.

---

## Experiment 4 — Selection-behavior analysis

For each acquisition strategy, characterize selected cases by:

- fragment count;
- foreground volume;
- affected anatomy;
- baseline model error;
- prevalence of merge/split failure;
- feature-space redundancy if embeddings are available.

This answers:

> **What type of case is each strategy buying with one additional annotation?**

This analysis is especially valuable if Dice changes are small but instance-level metrics improve.

---

## Experiment 5 — RibFrac cross-task validation

### Minimum version

Evaluate H3 failure detection using the labeled RibFrac validation set:

- identical FA-IPD formula;
- identical rank fusion;
- no post-hoc parameter search.

Report the same H1 metrics.

### Strong version

Run a shortened simulated AL study:

- 3–4 rounds;
- 3 trajectories if feasible;
- Random, Entropy, FA-IPD, Hybrid.

Do not reproduce every PENGWIN ablation on RibFrac.

---

## Experiment 6 — Focused ablations

### A. Structural score

- count variance;
- Hungarian matching disagreement;
- union-foreground NVI;
- foreground Jaccard disagreement;
- FA-IPD.

### B. Signal combination

- Entropy only;
- FA-IPD only;
- rank-fused Entropy + FA-IPD.

### C. Ensemble size

- `M = 2, 3, 4` on Round 0 only.

### D. Instance cleanup

- no component filtering/morphology;
- fixed cleanup.

### E. Diversity control

Only if diversity is included:

- Entropy + diversity;
- Hybrid + identical diversity.

### F. Query budget sensitivity

If feasible, use two acquisition batch sizes in a small sensitivity analysis. Otherwise state query-budget sensitivity as a limitation.

---

# 12. Baseline priority under limited compute

## Tier 1 — must have

1. Uniform Random
2. Predictive Entropy
3. Mean pairwise Dice disagreement
4. Original count + Hungarian structural score
5. FA-IPD
6. Hybrid Entropy + FA-IPD

These directly isolate the scientific question.

## Tier 2 — strong paper

7. Foreground/prediction-aware Random
8. Stochastic Batch
9. PAAL **or** USIM

Choose one recent hybrid method based on stable implementation rather than attempting every paper.

## Tier 3 — cite/discuss unless a fair reproduction is possible

- SBC-AL
- RegAL
- ClaSP PE

Reasons:

- SBC-AL changes the segmentation/training architecture;
- RegAL couples active and semi-supervised learning;
- ClaSP PE’s strongest modern results use patch queries and partial labeling.

If adapting any of them to whole-volume querying, label it clearly as an **adaptation**, not the published protocol.

---

# 13. Metrics

## 13.1 Segmentation metrics

PENGWIN-aligned:

- **Instance F1** — primary structural segmentation metric;
- Instance precision / recall;
- merge errors;
- split errors;
- topology consistency;
- Dice / local Dice;
- HD95 / ASSD [R4].

Optional:

- 3D Panoptic Quality if implementation is stable and clearly defined.

For the four-page paper, headline only:

1. Instance F1;
2. merge/split error;
3. Dice as semantic control.

---

## 13.2 Active-learning metrics

- AULC for Instance F1;
- AULC for structural error;
- final-budget performance;
- budget needed to reach a fixed fraction of full-data performance.

---

## 13.3 Failure-detection metrics

- AURC;
- top-k failure enrichment;
- Spearman correlation;
- optional AUROC/AUPRC only if a structural-failure threshold is pre-defined.

---

# 14. Failure-resistant paper design

The study should not be built so that one failed result destroys every contribution.

| Outcome | Scientific interpretation | Paper action |
|---|---|---|
| H1 ✓, H2 ✓, H3 ✓ | Structural signal is valid, useful, transferable | Full AL story; strongest version |
| H1 ✓, H2 ✓, H3 ✗ | Works for pelvic fragments but not generally | Narrow claim; still viable ISBI paper |
| H1 ✓, H2 ✗, H3 ✓ | Structural uncertainty is real/transferable but does not improve learning | Pivot to failure detection / quality control; AL becomes negative downstream result |
| H1 ✓, H2 ✗, H3 ✗ | PENGWIN-specific structural signal | Weak generic method; anatomy-specific framing only |
| H1 ✗, H2 ✓ | AL gain likely due confounding/diversity rather than claimed mechanism | Do not claim structural mechanism until explained |
| H1 ✗, H2 ✗ | Core premise unsupported | Stop; do not force submission |
| Results unstable across seeds | AL variance dominates | Increase repeats or narrow claim; never cherry-pick seed |

This matrix should guide decisions throughout experimentation.

---

# 15. Reviewer-attack checklist

## Attack 1 — “Structure-aware AL already exists.”

Required response:

- SBC-AL = structure/boundary consistency;
- RegAL = connectedness/topological consistency + AL/SSL;
- proposed method = uncertainty over **variable-cardinality instance partitions** and merge/split structure.

If experiments do not empirically support that distinction, do not claim it.

---

## Attack 2 — “VI is old. Where is the novelty?”

Correct response:

- VI and Jaccard disagreement are not novel.
- FA-IPD is their deterministic foreground-aware composition for ensemble
  variable-instance disagreement.
- Novelty depends on evidence that this formulation detects useful structural
  failures or improves acquisition; the formula alone is not enough.

Do not oversell the mathematics.

---

## Attack 2b — “You call this a metric, but triangle inequality fails.”

Do not call FA-IPD a mathematical metric or distance. Describe it consistently
as a bounded symmetric dissimilarity or disagreement score, and report the
known finite counterexample in the Stage 1 construct audit.

---

## Attack 3 — “Why not just use pairwise Dice?”

This is why pairwise Dice must be a direct H1 baseline.

The paper only survives if FA-IPD is demonstrably better for:

- merge/split risk;
- low Instance F1;
- top-k structural-failure enrichment;
- or downstream instance-level AL efficiency.

---

## Attack 4 — “FA-IPD only selects cases with more/larger fragments.”

Address with:

- fragment-count stratification;
- foreground-volume stratification;
- minimum-fragment-volume and missing-fragment-size stratification;
- partial/residualized association;
- selected-case distribution plots.

Also compare FA-IPD against foreground Jaccard disagreement alone. Otherwise a
gain cannot be attributed to instance-partition structure.

---

## Attack 4b — “The foreground term mostly measures boundary noise.”

Use matched-Dice boundary perturbations in H4, report the presence and partition
components separately, and avoid claiming structural specificity if FA-IPD is
dominated by the foreground-presence component on real predictions.

---

## Attack 5 — “The gain comes from diversity.”

Do not give Ours a diversity module that Entropy does not receive.

If diversity is used, apply the identical selection mechanism to both.

---

## Attack 6 — “The gain comes from a better segmentation network.”

Use the same network and post-processing across all acquisition strategies.

---

## Attack 7 — “One dataset is not enough.”

At minimum, complete the RibFrac H3 signal-transfer experiment.

A full second-dataset AL study is preferable but secondary to a rigorous primary PENGWIN study.

---

## Attack 8 — “Whole-volume annotation is unrealistic.”

Do not claim actual annotation-time savings.

Justify whole-volume querying because fragment identity/count are global properties, and report **case-level label budget**.

---

## Attack 9 — “Your uncertainty correlation is confounded by object size.”

Use the H1 size/count controls explicitly recommended above [R16].

---

# 16. ISBI four-page manuscript design

The actual paper must be much narrower than this research plan.

## Page 1 — Introduction + concise related work

Three paragraphs:

1. annotation problem + limits of voxel uncertainty;
2. variable-instance structural failure gap + closest recent work;
3. contributions + hypotheses.

One compact related-work paragraph covering:

- medical AL;
- structural/topological uncertainty;
- fracture instance segmentation.

---

## Page 2 — Method

One figure:

`CT → ensemble → voxel entropy + instance partitions → FA-IPD → rank fusion → query`

Only essential equations:

1. entropy;
2. FA-IPD;
3. hybrid score.

---

## Page 3 — Experiments

Compact table:

- PENGWIN split;
- RibFrac role;
- baselines;
- metrics;
- number of trajectories.

Main figure:

- PENGWIN Instance-F1 learning curve;
- merge/split curve or compact second panel.

---

## Page 4 — Results + analysis

One compact table:

- AULC / final Instance F1 / Dice;
- H1 AURC or top-20 enrichment;
- RibFrac transfer result.

One qualitative panel:

- low entropy + high FA-IPD case with a true merge/split error.

End with limitations:

- whole-volume query unit;
- simulated annotation;
- limited second-dataset scope;
- no claim of measured human annotation time.

---

# 17. Recommended working titles

## Preferred if H1–H3 succeed

**Beyond Voxel Uncertainty: Instance-Partition Disagreement for Active Learning in 3D Fracture Segmentation**

## More conservative

**Instance-Structural Disagreement for Active Learning in 3D Fracture Segmentation**

## If H2 fails but H1/H3 succeed

**Detecting Structural Segmentation Failures Through Ensemble Instance-Partition Disagreement**

## Avoid

**Fragment-Topology-Aware Active Learning...**

Reason: “topology-aware active learning” is now too close to SBC-AL / RegAL framing.

---

# 18. Immediate go/no-go sequence

## Gate 0 — Data permission

Obtain written clarification from PENGWIN organizers regarding the publication embargo.

## Gate 1 — Stable instance backend

A single model must produce meaningful instance labels and non-trivial Instance F1.

## Gate 2 — H1 Round-0 test

Before full AL, compare FA-IPD with entropy, pairwise Dice, foreground Jaccard,
and union-NVI for structural failure detection.

## Gate 3 — First two AL rounds

Run Random / Entropy / FA-IPD / Hybrid for two rounds and three seeds.

If H1 is weak and the curves are indistinguishable, stop rather than spending the full compute budget.

## Gate 4 — Full PENGWIN AL

Only proceed after H1 or early-H2 evidence is promising.

## Gate 5 — RibFrac

At minimum complete H3 failure-detection transfer before paper freeze.

This sequencing minimizes wasted GPU time and prevents a late discovery that the central signal is not informative.

---

# 19. Intended abstract skeleton — fill only after experiments

**Background:** Active learning for medical image segmentation commonly ranks unlabeled cases using voxel-wise uncertainty, but variable-instance targets can fail through merges and splits that are poorly summarized by voxel confidence.

**Method:** We introduce Foreground-Aware Instance-Partition Disagreement
(FA-IPD), a permutation-invariant ensemble score combining foreground Jaccard
disagreement with NVI over shared predicted foreground, and combine it with
predictive entropy using fixed rank fusion. We study pelvic fracture fragments
in PENGWIN and test cross-task transfer on rib-fracture instances in RibFrac.

**Evaluation:** We evaluate (i) whether FA-IPD detects structural failures using
risk–coverage and merge/split enrichment while controlling for fragment
size/count, and (ii) whether it improves simulated active-learning efficiency
under paired repeated trajectories.

**Results:** `[ONLY ACTUAL RESULTS]`

**Conclusion:** `[CLAIM ONLY WHAT H1–H3 SUPPORT]`

---

# 20. References

**[R1]** IEEE ISBI 2027. *Conference / Author Information*.  
https://biomedicalimaging.org/2027/

**[R2]** PENGWIN 2026. *Rules*. Publication embargo statement.  
https://pengwin2026.grand-challenge.org/rules/

**[R3]** Liu, Y. et al. *PENGWIN 2026 Task 1 and Task 2: Peripelvic Fracture Segmentation Dataset*. Zenodo, 2026.  
https://zenodo.org/records/19732767

**[R4]** PENGWIN 2026. *Task-1 Evaluation*. Includes Dice, HD95, ASSD, Instance F1/precision/recall, merge/split errors, and topology consistency.  
https://pengwin2026.grand-challenge.org/evaluation/ab7bce15-77de-4627-b41c-ba60128f5289/

**[R5]** Nath, V. et al. *Diminishing Uncertainty Within the Training Pool: Active Learning for Medical Image Segmentation*. IEEE Transactions on Medical Imaging, 2021.  
https://pubmed.ncbi.nlm.nih.gov/33373298/

**[R6]** Burmeister, J.-M. et al. *Less Is More: A Comparison of Active Learning Strategies for 3D Medical Image Segmentation*. 2022.  
https://arxiv.org/abs/2207.00845

**[R7]** Gaillochet, M., Desrosiers, C., Lombaert, H. *Active Learning for Medical Image Segmentation with Stochastic Batches*. 2023.  
https://arxiv.org/abs/2301.07670

**[R8]** Zhou, T. et al. *SBC-AL: Structure and Boundary Consistency-based Active Learning for Medical Image Segmentation*. MICCAI 2024. Public reviews available on the paper page.  
https://papers.miccai.org/miccai-2024/670-Paper3047.html

**[R9]** Jafrasteh, B. et al. *Unifying Active Learning and Semi-Supervised Learning for Medical Image Segmentation*. RegAL, 2026.  
https://arxiv.org/abs/2607.25014

**[R10]** Gupta, S. et al. *Topology-Aware Uncertainty for Image Segmentation*. NeurIPS 2023.  
https://proceedings.neurips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html

**[R11]** Singh, Y. et al. *Topological data analysis in medical imaging: current state of the art*. Insights into Imaging, 2023.  
https://link.springer.com/article/10.1186/s13244-023-01413-w

**[R12]** Lüth, C. T. et al. *nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation*. 2025.  
https://arxiv.org/abs/2511.19183

**[R13]** Lüth, C. T. et al. *Finally Outshining the Random Baseline: A Simple and Effective Solution for Active Learning in 3D Biomedical Imaging*. 2026.  
https://arxiv.org/abs/2601.13677

**[R14]** Föllmer, B. et al. *Active Learning with the nnUNet and Sample Selection with Uncertainty-Aware Submodular Mutual Information Measure*. MIDL 2024.  
https://proceedings.mlr.press/v250/follmer24a.html

**[R15]** Shi, J. et al. *Predictive Accuracy-Based Active Learning for Medical Image Segmentation*. IJCAI 2024.  
https://www.ijcai.org/proceedings/2024/540

**[R16]** Geißler, K. et al. *Structure Size as Confounder in Uncertainty Based Segmentation Quality Prediction*. MIDL 2024.  
https://proceedings.mlr.press/v250/geissler24a.html

**[R17]** Zenk, M. et al. *Comparative Benchmarking of Failure Detection Methods in Medical Image Segmentation: Unveiling the Role of Confidence Aggregation*. Medical Image Analysis, 2025.  
https://pubmed.ncbi.nlm.nih.gov/39657400/

**[R18]** Gaillochet, M., Desrosiers, C., Lombaert, H. *TAAL: Test-time Augmentation for Active Learning in Medical Image Segmentation*. 2023.  
https://arxiv.org/abs/2301.06624

**[R19]** Sang, Y. et al. *Benchmark of Segmentation Techniques for Pelvic Fracture in CT and X-ray: Summary of the PENGWIN 2024 Challenge*. 2025.  
https://arxiv.org/abs/2504.02382

**[R20]** *Automatic pelvic fracture segmentation: a deep learning approach and benchmark dataset*. Frontiers in Medicine, 2025.  
https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1511487/full

**[R21]** Zhu, N. et al. *CSAL-3D: Cold-start Active Learning for 3D Medical Image Segmentation via SSL-driven Uncertainty-Reinforced Diversity Sampling*. MICCAI 2025.  
https://papers.miccai.org/miccai-2025/0198-Paper2315.html

**[R22]** Berger, A. H. et al. *Pitfalls of topology-aware image segmentation*. 2024.  
https://arxiv.org/abs/2412.14619

**[R23]** Yang, B. et al. *Structural uncertainty estimation for medical image segmentation*. Medical Image Analysis, 2025.  
https://www.sciencedirect.com/science/article/pii/S1361841525001495

**[R24]** *Fragment distance-guided dual-stream learning for automatic pelvic fracture segmentation*. Computerized Medical Imaging and Graphics, 2024.  
https://www.sciencedirect.com/science/article/pii/S0895611124000892

**[R25]** RibFrac Grand Challenge. *Dataset and Task Description*.  
https://ribfrac.grand-challenge.org/tasks/

**[R26]** Yang, J. et al. *Deep Rib Fracture Instance Segmentation and Classification From CT on the RibFrac Challenge*. IEEE Transactions on Medical Imaging, 2025.  
https://doi.org/10.1109/TMI.2025.3565514

---

# 21. Bottom-line recommendation

For ISBI 2027, the strongest feasible version is:

> **One simple acquisition innovation + one rigorous structural-failure hypothesis + one downstream active-learning hypothesis + one cross-task transfer hypothesis.**

Do **not** spend the remaining time inventing a more complex segmentation architecture.

Highest-value sequence:

1. obtain written PENGWIN publication permission;
2. stabilize an instance-compatible segmentation backend;
3. run the Round-0 H1 benchmark;
4. if H1 is positive, run repeated PENGWIN AL trajectories;
5. run RibFrac H3 failure-detection transfer;
6. complete only the ablations required to isolate FA-IPD.

The paper should survive because the evidence is **triangulated** rather than because one fragile result has to be true.

---

# 22. Current claim–evidence ledger and self-review

This section tracks design evidence and must not be copied into a results
section as if H1–H3 had already been observed.

## Claim–evidence map

| Claim | Current evidence | Status |
|---|---|---|
| FA-IPD is bounded, symmetric, and instance-ID permutation invariant | Definition plus Stage 1 synthetic tests | Supported at construct level |
| FA-IPD detects a completely missing equal-sized fragment that union-NVI misses | Synthetic score changes from `0.0` to `0.5` | Supported at construct level |
| Equal-support merge/split cases reduce to intersection-NVI | Synthetic merge/split tests | Supported at construct level |
| FA-IPD is a mathematical metric | Finite triangle-inequality counterexample | Rejected; must not be claimed |
| The Stage 1 internal GT/cleanup contract is frozen without deleting small annotated instances | Complete 340-case audit, artifact validator, and prediction-only cleanup test | Supported at implementation/audit level |
| FA-IPD is robust to boundary and fragment-size confounding | No dataset-wide evidence yet | Needs H1/H4 evidence |
| FA-IPD improves failure detection over entropy, pairwise Dice, and Jaccard-only disagreement | Not yet run | Needs H1 evidence |
| Entropy + FA-IPD improves active-learning efficiency | Not yet run | Needs H2 evidence |
| The frozen score transfers to RibFrac | Not yet run | Needs H3 evidence |

## Five-dimension self-review

1. **Contribution — needs evidence.** The foreground-aware composition is
   explicit and reproducible, but novelty depends on H1/H2 rather than the
   formula alone.
2. **Writing clarity — pass for method contract.** Pairwise domains,
   normalization, edge cases, aggregation, and terminology are stated.
3. **Experimental strength — pending.** No Round-0 or AL result is claimed.
4. **Evaluation completeness — planned, not complete.** Jaccard-only,
   union-NVI, pairwise Dice, count/Hungarian, size controls, and matched-boundary
   tests are mandatory.
5. **Method soundness — pass for the internal Stage 1 contract.** Synthetic
   behavior and the 340-case connectivity audit are verified, and the
   prediction-only cleanup rule is frozen. Official-evaluator equivalence and
   H1–H3 empirical claims remain unverified.
