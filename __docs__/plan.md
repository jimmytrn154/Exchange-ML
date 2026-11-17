# plan.md — EMBC 2027 plan (replaces the ISBI roadmap for this submission)

**Written:** 2026-11-17. **Status:** DRAFT until the §8 decisions are recorded.
**Supersedes for this submission:** `__docs__/plan.md` (ISBI 2027 roadmap). That file stays as the record of the original study design; Stages 4–8 there are deferred, not cancelled.
**Frozen artifacts not touched by this plan:** `configs/stage3_h1.json` (v1, SHA-256 `64501bae…6106`), the Stage 0 split, the three member checkpoints, every v1 output under `outputs/stage3_h1/`.

---

## 1. Venue decision: EMBC 2027 full contributed paper (main track)

### Options in the window (now → early Feb 2027)

| Option | Deadline | Indexed | Fit | Verdict |
|---|---|---|---|---|
| **EMBC 2027 full contributed paper** | **24 Jan 2027** | IEEE Xplore | 4 technical pages + 1 page refs (≤ 20). Broad biomedical-engineering audience. | **Chosen** |
| EMBC 2027 "workshop" / mini-symposium | organiser proposals | — | Not a paper track. Content is invited by organisers; no individual peer-reviewed submission, no proceedings. | Not applicable |
| EMBC 2027 1-page research poster | 23 Apr 2027 | **Not** in Xplore | Too thin for C1–C3. | No |
| ISBI 2027 workshop paper | 26 Jan 2027 | varies | None of the eight listed workshops covers segmentation uncertainty, active learning, or musculoskeletal CT. | No fit |
| ISBI 2027 1-page abstract | 1 Feb 2027 | — | Companion only; in-person Lausanne; dual-submission rule unverified. | Optional (§8 Q1) |
| MIDL 2027 full paper | 4 Dec 2026 | proceedings | No time for held-out confirmation; would rest on 40 cases. | No |
| MIDL short / MICCAI workshops (UNSURE) | Apr / Jun 2027 | — | Outside the requested window. | Extended version later |

### Why main and not a workshop

1. **At EMBC, "workshop" is not a paper venue.** The only peer-reviewed, indexed route with a January deadline is the full contributed paper.
2. **Every workshop that would fit topically (UNSURE, MIDL short) is outside the window.** The only in-window workshop track (ISBI) has no topical match.
3. **The content is main-track content.** Three of four contributions are already in hand with numbers (§3). A 4+1 IEEE paper is the right size: one method figure, one main table, one risk–coverage figure, one qualitative example.
4. **The main-track risk is the audience, not the format.** EMBC reviewers are general biomedical engineers. The paper must motivate fracture fragments clinically in two sentences and avoid active-learning vocabulary entirely.

**Decision: EMBC 2027 full contributed paper. Deadline 24 Jan 2027, 23:59 AoE. Onsite presentation in Singapore required.**
Open check before WP0 closes: EMBC author guidelines for a dual-submission / arXiv rule (the FAQ does not state one).

---

## 2. Where the project stands (facts, 2026-11-17)

| Area | Fact | Source |
|---|---|---|
| Backend | Six ABBC defects (D1–D8, D4 accepted as limitation, D7 merged into D4) fixed as F1–F9; patch on `ccc625f`; 58/58 tests. Stage 2 single model: instance F1 0.737 → 0.850; binary Dice 0.764 → 0.730 (femur-only Dice drops because unlabelled pelvic bone is now segmented — D4). | `__docs__/reports/stage3/stage3_fix_progress.md`, `patches/` |
| Members | Three members (seeds 31001–31003) trained, gated (G1, GF), inferred on the 40 validation cases with F9. | `outputs/stage3_h1/members/*/validation_inference_manifest.json` |
| H1 v1 | Re-run 2026-10-21 on the fixed backend: **no FA-IPD advantage in any of the three criteria.** Recommendation FAIL; not yet recorded by the user. | `outputs/stage3_h1/h1_comparison.csv`, fix-progress log 2026-10-21 |
| Why v1 failed | FA-IPD ≈ foreground Jaccard (Spearman 0.997). Partition term: median 2 % share, max 0.0198 absolute. Cause: NVI normalised by `log N_voxels` ≈ 12–14 nats; a 7-fragment partition has VI of at most a few nats. | `uncertainty_scores.csv`; `src/exchange_ml/metrics.py:41-80` |
| Hidden signal | Union-NVI alone is best on merge+split AURC (0.739 vs FA-IPD 0.798, entropy 0.865) and the only score with positive controlled association (+0.157). | `h1_comparison.csv` |
| Evaluation flaws | E1 score/risk model mismatch; E2 92.5 % failure prevalence caps enrichment at 1.081; E3 n = 40 in two scan types; E4 98.5 % of femur-only disagreement on unlabelled bone; E5 IoU ≥ 0.1 matching gives F1 = 1.0 with 43 % of an instance missing (`357`). | 2026-10-21 diagnostics |
| Open gates | G2 (Stage 2 risk CSV) proposed PASS, unconfirmed. H1 v1 gateway decision unrecorded. | fix-progress log |
| Docs | `README.md` quotes pre-fix numbers; `stage3_h1.md` stops at 2026-10-16. | — |
| Infrastructure | Three server reboots in October; GPU 1 faulted; home-disk quota forced bulk data to `/mnt/sdb/shared/chuong/`. | fix-progress log, `__docs__/storage_layout.md` |
| Not started | AL engine (H2), RibFrac (H3). | — |

---

## 3. Refined direction and contributions

### 3.1 Thesis the evidence supports

The original thesis ("instance-partition disagreement is a better acquisition signal than entropy") is not supported. What two H1 runs do support:

> Ensemble disagreement over 3D instance maps is a **presence** component (is the bone/fragment there?) plus a **partition** component (how is shared bone divided?). Under the v1 normalisation the presence component is ~50× larger, so FA-IPD was a presence score. The partition component is the one that tracks merge/split error, and the v1 formulation hid it.

**Working title:** *Presence or partition? Decomposing ensemble disagreement for structural-failure detection in 3D pelvic fracture segmentation.*

**Question:** When an ensemble of 3D fracture-instance segmenters disagrees, which component flags merge, split and missing-fragment failures, and under what evaluation conditions is the difference measurable at all?

**Scope:** H1 (structural-failure detection) on PENGWIN pelvic/femur CT, with H4 (controlled perturbations) as the mechanism check. **Not in this paper:** active learning (H2), RibFrac (H3), the hidden PENGWIN test set, any "new metric" claim.

### 3.2 Contributions

| # | Claim | Evidence | Status |
|---|---|---|---|
| **C1 Decomposition (we *show*, not *propose*)** | `d_FA = presence + weight·partition`. On 40 validation cases the partition term has median share 2 % and never exceeds 0.02 absolute; FA-IPD v1 is rank-identical to foreground Jaccard (ρ = 0.997). The cause is the `VI / log N` scale: Vinh, Epps & Bailey (2010, §3.3, fn 4) show `VI/log N` and `VI/2 log K` use looser bounds than `H(U,V) ≤ 2 log K`; applied here (N ≈ 10⁵–10⁶, K ≤ 12) the v1 term cannot exceed ≈ 0.36 and is observed ≤ 0.02. The decomposition itself has precedent (Arganda-Carreras et al. 2015 split/merge, foreground-restricted scoring); the new content is the voxel-scale regime and the empirical dominance. **Rival explanation to rule out in WP1:** members may simply agree on partitions (raw VI ≈ 0), in which case no re-normalisation helps. | `uncertainty_scores.csv`, `outputs/stage3_h1/diagnostics/fa_ipd_decomposition.py`; `__docs__/research/phase3_analysis/synthesis.md` Theme 1 | **In hand** (bound); WP1 check pending |
| **C2 Scale-corrected partition disagreement** | FA-IPD v2: normalise by joint entropy, `VI / H(P,Q)` (bounded [0,1]; Vinh, Epps & Bailey 2010), scored per annotated bone. **Pre-registered claim:** on held-out cases, v2 has lower merge+split AURC than foreground Jaccard and than mean entropy, with 95 % case-cluster bootstrap CIs on both differences excluding 0. Presence term, partition term, union-NVI and v1 reported as separate rows. | WP1–WP3 | **Needs held-out run** |
| **C3 Evaluation conditions** | Four conditions under which a structural-failure comparison is invalid or uninformative, each with our numbers: (a) scoring model A's disagreement against model B's errors; (b) a binary failure label at 92.5 % prevalence caps enrichment at 1.08; (c) partially annotated scans put 98.5 % of disagreement on unlabelled bone; (d) IoU ≥ 0.1 matching reports F1 = 1.0 with 43 % of an instance missing. | 2026-10-21 diagnostics | **In hand** |
| **C4 Reproducible PENGWIN backend** | Six documented defects in the public ABBC pipeline, fixes as a patch with 58 tests, per-bone gates. One paragraph + repository link; supports C1–C3, not a headline. | `patches/`, fix-progress log | **In hand** |

**Explicit non-claims:** not a mathematical metric (keep the triangle-inequality counterexample); no invention of VI or Jaccard disagreement; no AL benefit; no transfer; no equivalence to the official PENGWIN evaluator.

**Relation to `__docs__/paper.md` §10:** Contribution 1 kept, re-centred on the decomposition; Contribution 2 becomes C2 + C3; Contributions 3 (AL) and 4 (transfer) dropped from this paper.

### 3.3 Framing fixed in advance by outcome

| Held-out result | Framing |
|---|---|
| C2 primary passes | Partition disagreement detects structural failures that presence disagreement and entropy miss. v1 reported as the normalisation lesson. |
| C2 primary fails | Analysis paper: C1 + C3 are the findings; v2 reported as a negative confirmation. No method change after seeing the held-out result. |
| Held-out run does not finish by 10 Jan | **Reporting-scope decision, not a method change:** C1 + C3 + C4 on the 40 validation cases; v2 reported on the same 40 cases, labelled exploratory, with no confirmation claim. Recorded in the paper's limitations. |
| WP1 shows the v2 partition term has no spread on validation (raw VI ≈ 0 across units) | The members agree on partitions; C2 is moot. Paper = C1 + C3 + C4 analysis; held-out inference may still run for C3(a)–(d) but is not needed for C2. Decided before any held-out inference is launched. |

---

## 4. Protocol v2 (frozen before any held-out case is scored)

v1 stays frozen and is reported unchanged. v2 changes are motivated by the diagnosis on the 40 validation cases and are confirmed only on cases that played no part in that diagnosis. The v2 config is committed **before** held-out inference starts; its SHA-256 goes in the paper.

| # | Item | v2 rule | Fixes |
|---|---|---|---|
| V1 | Partition normalisation | `NVI_H = VI(P,Q) / H(P,Q)` on the shared foreground of each bone (Vinh et al. 2010 Theorem 3: a metric); both-empty = 0; one-empty = 1; single-voxel shared foreground = 0. **Stated reason for NVI over NID** (which Vinh et al. call "generally more preferable"): continuity with the VI-based v1 term, and VI = H(P\|Q) + H(Q\|P) separates into split and merge components. NID is listed as an untested alternative; it is **not** run on validation. Wording: "specified before held-out scoring", never "a priori". | S3 |
| V2 | Unit of analysis | (case, bone) for the bones annotated in that scan type: pelvic → sacrum, left hip, right hip; femur-only → femur. Bone from frozen ID ranges (`instances.py` `ANATOMY_RANGES`); scan type from dataset metadata. No voxel GT enters a score. 40 validation cases → 80 units. | E3, E4 |
| V3 | Risk source | Primary: mean over the three members of their own per-bone risks (`per_anatomy` output of `instance_metrics`). Secondary: Stage 2 single-model per-bone risk (v1 source). | E1 |
| V4 | Failure definition | Continuous risks for AURC and Spearman. For enrichment, failure = top quartile of risk among units (prevalence fixed at 25 %, ceiling 4.0). Per-error-type breakdown: merge present, split present, missing fragment (recall < 1). | E2 |
| V5 | Held-out set | Random 100 cases from the 210 pool cases outside trajectory-1's initial set, drawn with a recorded seed, stratified 50 pelvic / 50 femur-only, listed in the v2 config. Test set (50) stays sealed. Pool GT is read only retrospectively for risk; if H2 is ever run this is recorded. | E3 |
| V6 | Gate | Primary: C2 claim above. Secondary: `1 − F1`, top-quartile enrichment, controlled Spearman (fragment count, bone volume, min-fragment volume), per-error-type AURC. All computed on validation (40) and held-out (100), reported side by side. | S1 |
| — | Unchanged | 26-connectivity, < 1000 mm³ prune, no morphology, IoU ≥ 0.1 matching (E5 recorded as limitation), M = 3, seeds 31001–31003, Stage-B probabilities, shared ROI = union of cleaned member foregrounds. | — |

---

## 5. Work packages and schedule

Target: submit 24 Jan 2027. Internal freeze 21 Jan. About one week of slack total; the server rebooted three times in October, so WP2 starts as early as possible and the §3.3 scope fallback is pre-declared.

| WP | Dates | Work | Owner / command type |
|---|---|---|---|
| **WP0 Close the books** | 17–21 Nov | Record H1 v1 = FAIL in `stage3_h1.md`; confirm/decline G2; update `README.md` numbers and status; add this plan's decisions to the reports. Check EMBC dual-submission rule. | User decides; agent edits docs (agent-typed) |
| **WP1 v2 implementation + freeze** | 21 Nov – 4 Dec | `metrics.py`: `NVI_H`; per-bone score units; member-risk evaluation; top-quartile enrichment; case-cluster bootstrap; per-error-type tables; GT-volume-fraction secondary risk. Unit tests (identity, symmetry, permutation, empty cases, bound). Run v2 scoring + evaluation on the existing 40-case member outputs. **Three pre-declared checks before freeze (DA-2 and DA-Ph3-1):** (a) distribution of the raw VI numerator and of the v2 term across the 80 validation units — no spread → §3.3 row 4; (b) union-NVI reported as the *mixed* variant (union foreground) vs the v2 term as the *pure* variant (shared foreground); (c) bootstrap precision projection to n = 100 and 210. Write `configs/stage3_h1_v2.json` with the held-out case list, seed, NVI-vs-NID rationale and the projection result; **commit; record SHA-256.** Member-level risk on validation needs member predictions vs GT: one user-typed evaluation run. | Agent: code, tests, smoke on synthetic data. User: 40-case member evaluation |
| **WP2 Held-out inference** | 4 – 20 Dec | 3 members × 100 cases = 300 inferences. October logs: 100–670 s per case on GPU → ~8–56 GPU-hours total; run members in parallel on separate GPUs. Link `outputs/stage3_h1_v2/members/` to `/mnt/sdb` **before** writing. Resumable manifests already exist in `predict_stage3_member.py`. | **User-typed** |
| **WP3 Score, evaluate, decide** | 20 Dec – 6 Jan | Score (GT-free) → evaluate (retrospective) on held-out; manual C2 gate assessment; H4 perturbation check of v1 vs v2 on CPU (existing `perturbations.py`); write `__docs__/reports/stage3/stage3_h1_v2.md`. | User runs score/evaluate; agent verifies hashes and writes the report |
| **WP4 Paper** | 6 – 21 Jan | Draft per §6. Figures: method/decomposition, risk–coverage, one qualitative case. Main table. Internal review. Freeze 21 Jan. | Authors; agent drafts and formats |
| **WP5 Submit** | 21 – 24 Jan | Final checks against the EMBC template; submit. | User |
| (opt.) ISBI abstract | 24 Jan – 1 Feb | 1 page from the EMBC results, only if Q1 = yes and dual submission is allowed. | Authors |

**Decision point 10 Jan:** if held-out inference is not complete, invoke the §3.3 scope fallback. No extension past this date.

---

## 6. Paper layout (EMBC 4+1)

| Page | Content |
|---|---|
| 1 | Clinical hook: fragment count and configuration drive pelvic-fracture surgical planning (cite). Problem: voxel uncertainty misses merge/split. Gap: structural-uncertainty work does not separate presence from partition. Contributions C1–C3 in three bullets. |
| 2 | Backend in one paragraph (cite ABBC; C4 as a sentence + link). Decomposition: `d_FA` equation, the `log N` argument, v2 definition. Figure 1: CT → ensemble → presence/partition → scores. |
| 3 | Protocol: units, risks, held-out set, pre-registration hash. Table 1: scores × {merge+split, 1−F1} AURC and controlled ρ, validation vs held-out, with CIs for the C2 differences. Figure 2: risk–coverage on held-out. |
| 4 | C3 conditions as a compact table with numbers. Figure 3: one low-entropy / high-partition failure. Limitations: single dataset, M = 3, IoU rule, D4. Conclusion. |
| 5 | ≤ 20 references; acknowledgements; data/code statement. |

Vocabulary rule: no "active learning", "acquisition", or "query" anywhere in the paper.

---

## 7. Rules that still apply

- `__docs__/rule.md` is unchanged. Agent runs only smoke checks; every training, inference, scoring and evaluation run is user-typed and listed as such in the reports.
- No method change after seeing held-out results. The §3.3 fallback changes reporting scope only.
- Every number in the paper traces to a hashed output file; the v2 config hash and the held-out case list are published.
- Stage reports keep the `plan.md` §7 (ISBI) minimal format: Status / Configuration / Result / Gateway / Blocking issue, plus the two command lists.

---

## 8. Decisions needed from the user

| # | Question | Recommendation |
|---|---|---|
| Q1 | Also submit the ISBI 1-page abstract (1 Feb)? | Only if EMBC guidelines permit; low priority. |
| Q2 | Fix `VI / H(P,Q)` a priori as the v2 normalisation (V1)? | Yes. |
| Q3 | Per-bone units chosen by scan-type metadata (V2)? | Yes. |
| Q4 | Members' own per-bone risk as primary, Stage 2 as secondary (V3)? | Yes. |
| Q5 | Held-out = 100 pool cases, seeded, stratified 50/50 (V5)? | Yes (all 210 does not fit the timeline). |
| Q6 | Confirm G2 and record H1 v1 = FAIL? | Yes; required to close WP0. |
| Q7 | Who runs WP2 inference and on which GPUs? | User to assign; see `__docs__/reports/stage3/member_01_handover.md` for the second-user route. |

---

## 9. Commands

**Agent-typed (run while preparing this plan, 2026-11-17):** `git ls-files`; `git log`; `git show --stat`; reads of `README.md`, `__docs__/*.md`, Stage 3 reports, `configs/stage3_h1.json`, `configs/stage2_backend.json`, `outputs/stage3_h1/h1_comparison.csv`, `h1_summary.json`; `python -c` column summary of `uncertainty_scores.csv`; `python -c` read of `splits/pengwin_stage0_split.json`; `grep` of `src/exchange_ml/metrics.py`, `src/exchange_ml/instances.py`, `scripts/evaluate_stage2_backend.py`, `__docs__/data_description.md`. Web lookups of EMBC/ISBI/MIDL dates.

**User-typed:** None prepared yet. WP1–WP3 commands are prepared after the §8 decisions.
