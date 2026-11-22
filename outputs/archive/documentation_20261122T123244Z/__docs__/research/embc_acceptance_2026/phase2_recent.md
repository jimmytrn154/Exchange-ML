# EMBC 2023–2026: recent-paper evidence ledger

**Status: complete for focused source extraction, 22 November 2026.** Scope: bibliography and verification for the requested competitiveness assessment. This document does not infer acceptance probabilities or synthesize a submission verdict.

## Search method and access labels

The search targeted published EMBC full papers from 2023–2025 concerning segmentation uncertainty, failure/quality assessment, annotation inspection and calibration. A 2026 accepted oral was retained separately because it is exceptionally close in topic. Adjacent physiological-signal uncertainty papers are separated from the main segmentation comparators. Inclusion required an EMBC DOI/publication record or an explicitly labeled official acceptance record. ISBI, WACV, journal papers and unverified arXiv venue claims were excluded from the accepted-EMBC comparator set.

Searches ran 21–22 November 2026 using web search, official programs, IEEE Xplore/PubMed and institutional author copies. This is a targeted narrative review with an access constraint, not a complete proceedings census or a PRISMA systematic review. Search-result totals are not stable database counts and were not treated as screening denominators. Deduplication used exact titles and DOIs; no Semantic Scholar ID was fabricated or assumed.

Recorded queries included:

- `EMBC 2024 segmentation uncertainty quality estimation failure detection`
- `EMBC 2023 segmentation uncertainty failure detection quality`
- `EMBC 2025 segmentation uncertainty quality control`
- `"EMBC" "2024" "uncertainty" segmentation`
- `"EMBC" "2023" "uncertainty" segmentation`
- `"EMBC" "2025" "uncertainty" segmentation`
- `"segmentation" "quality control" "EMBC" 2023 2024 2025`
- Exact-title follow-ups for AI2Seg, Damiano's lung-radiomics paper, Hou's retinal-vessel paper, Vavaroutas's augmentation/active-learning paper and Hu's EEG-calibration paper.
- Source-restricted follow-ups: the institutional PDF URL plus `training`, `subjects`, `Results`, `Dice-Plus`, or `confidence`.
- `"Leveraging AI Uncertainty for Targeted 3D Medical Image Segmentation Review" DOI`.

Access labels: **FT-PDF** = full PDF opened and methods/results read; **FT-indexed sections** = institution-hosted full-paper text exposed by the search index, including methods/results/conclusion, while direct PDF retrieval failed; **abstract-only** = no experimental-detail extraction beyond the abstract; **program-only** = accepted presentation verified without full paper/proceedings DOI. Absence of an extracted confidence interval means “not reported in accessed text,” not proof the authors never performed another analysis.

## Main segmentation comparators

### R1. AI2Seg — EMBC 2023, FT-PDF

Schilling, M. P., Klinger, L., Schumacher, U., Schmelzer, S., Bordallo López, M., Nestler, B., & Reischl, M. (2023). *AI²Seg: A method and tool for AI-based annotation inspection of biomedical instance segmentation datasets.* EMBC, 1–6. [DOI](https://doi.org/10.1109/EMBC40787.2023.10341074); [publisher PDF, institutional copy](https://oulurepo.oulu.fi/bitstream/handle/10024/48865/nbnfioulu-202404182835.pdf?isAllowed=y&sequence=1).

**Extraction:** MoNuSeg: 592 crops; LIVECell SkBr3 subset: 2,640 crops, each 256². These are crops, not patients. U-Net predicts distance maps followed by watershed; random 80/20 train–validation split, five training seeds. TTA-based instance confidence is combined with prediction–annotation mismatch and size weights. Nine simulated noise types at three corruption ratios across two datasets yield 54 conditions; 48 exceed the authors' AUC>0.55 threshold. At corruption 0.10, optimized merge AUCs are 0.836/0.761; split: 0.871/0.930. AUC* uses grid-searched optimal weights, not an independently frozen unknown-noise evaluation. Table I gives five-run means without confidence intervals. KaIDA integration provides an inspection GUI; no measured clinician-time saving is reported.

**Locators:** pp. 3–4, Sections III-B/III-C and IV-A/B/C; p. 5, Table I; p. 6, GUI. The method reads existing annotations for inspection: it is not the same GT-free deployed-model failure-detection task. Publication year is verified from the IEEE PDF and [official 2023 full-paper program](https://microsites.arinex.com.au/EMBC/content.html).

### R2. Segmentation variability and radiomics — EMBC 2025, FT-indexed sections

Damiano, R., Merli, A., Lanzarone, E., & Scalco, E. (2025). *Segmentation variability in Bayesian U-Net versus manual annotations: Impact on radiomic reproducibility in lung tumor CT images.* EMBC. [DOI](https://doi.org/10.1109/EMBC58623.2025.11253060); [institutional record](https://aisberg.unibg.it/handle/10446/313827); [publisher PDF](https://aisberg.unibg.it/retrieve/9fe33648-e15f-4aca-aa9e-7aaf19940cb9/Segmentation_Variability_in_Bayesian_U-Net_versus_Manual_Annotations_Impact_on_Radiomic_Reproducibility_in_Lung_Tumor_CT_Images.pdf).

**Extraction:** LUNG1's reported split is 252 training/64 validation/79 internal-test subjects; the collection total is 422. LUNG2 starts with 21 subjects, one excluded; five oncologists each supply manual and initialized contours. Both collections are described as same-center/protocol. Two 2D dropout U-Nets compare BCE-Dice and Dice-Plus, with 50 stochastic passes and manually localized tumor crops. External-test mean Dice±SD is 0.787±0.149 versus 0.777±0.139. Of 105 radiomic features, fewer than half are reproducible at ICC≥0.8 in the best conditions. Conclusions acknowledge partial correspondence with expert variability and limited generalizability. No measured workflow benefit or paired uncertainty interval for a failure-ranking effect is reported in accessed sections.

**Locators:** Section II-A/B/C/D/F/G, Tables I–III and Section IV. “External” here does not establish multicenter validation. Its 422 collection total differs from the 395-subject reported split; the accessed passages do not explain that difference. Full-paper sections were read through the institution-PDF search index; direct PDF fetch timed out. Patient disjointness was not independently verified.

## Adjacent uncertainty-method comparators

### R3. Augmentation-based active learning — EMBC 2023, FT-PDF

Vavaroutas, S., Qendro, L., & Mascolo, C. (2023). *Uncertainty estimation with data augmentation for active learning tasks on health data.* EMBC, 1–4. [DOI](https://doi.org/10.1109/EMBC40787.2023.10340330); [PubMed publication record](https://pubmed.ncbi.nlm.nih.gov/38083574/); [author PDF](https://mobile-systems.cl.cam.ac.uk/papers/embc23-vavaroutas.pdf).

**Extraction:** Two datasets: epileptic-seizure recognition (11,500 signal samples) and heart-disease classification (14 attributes; subject count not given in the inspected paper). CNN for EEG and logistic regression for heart data; train/test splits 70/30 and 75/25. Three controls use classifier uncertainty, margin and entropy sampling. Initial labeled samples are 50/5. For the first ten EEG query batches, mean distance is 2,383 versus 1,706.5/1,549.2/1,669.1 for controls. No independent external cohort, repeated-seed interval or human labeling-time trial is reported. A comparison caveat is explicit in the paper: baseline accuracy uses the original test set, whereas proposed-method accuracy averages original and augmented tests. These are not identical evaluation distributions.

**Locators:** pp. 2–3, Section III Datasets/Baselines/Setup/Results; p. 4, Table I. This is an adjacent accepted uncertainty-method example, not a segmentation or structural-failure comparator. “50% more diverse” is a signal-space distance claim, not 50% less clinical annotation work.

### R4. EEG uncertainty and calibration — EMBC 2024, FT-indexed sections

Hu, J., Ur Rahman, M. M., Al-Naffouri, T., & Laleg-Kirati, T.-M. (2024). *Uncertainty estimation and model calibration in EEG signal classification for epileptic seizures detection.* EMBC, 1–5. [IEEE publication record](https://ieeexplore.ieee.org/document/10782858/); [DOI](https://doi.org/10.1109/EMBC53108.2024.10782858); [institutional accepted-manuscript copy](https://repository.kaust.edu.sa/bitstreams/b3350e28-6d06-4342-aa67-5911bae3e26e/download).

**Extraction:** Temple and Lemon data total 1,593 subjects (1,385 epileptic, 208 healthy), yielding 225,334 ten-second segments. Training/validation uses 1,115 subjects and testing 478, with distinct subjects across sets; validation also fits calibration. GP, Bayesian NN and MC dropout are compared with GCNN. Subject-level AUCs before calibration are 0.918/0.934/0.925 versus 0.862 for GCNN; NLL and Brier score assess uncertainty/calibration. Fifty training epochs are reported. No independent cross-dataset deployment test, repeated-seed interval or clinician-review study was established in retrieved sections. Two source datasets do not themselves prove cross-dataset generalization.

**Locators:** Section III-B/C and Section IV/Table I. Accepted-manuscript methods/results were read through the institutional-PDF search index; direct access returned HTTP403. This is an adjacent calibration comparator, not a structural-segmentation study.

## Verified publication with limited extraction

Hou, J.-M., Lee, C.-K., Lin, Y.-A., & Tseng, P.-H. (2024). *A semi-supervised retinal vessel segmentation method via adaptive uncertainty estimation.* EMBC, 1–5. [DOI](https://doi.org/10.1109/EMBC53108.2024.10782670); [PubMed](https://pubmed.ncbi.nlm.nih.gov/40039152/); [official decision](https://epapers2.org/embc2024/ESR/paper_details.php?paper_id=6792).

**Access: abstract-only.** Official decision is Accept as Lecture. The abstract describes teacher–student adaptive uncertainty thresholds on STARE and comparisons with supervised/semi-supervised controls. Training/test subject counts, effects, seed counts, confidence intervals and external validation are **not extracted**, because no full paper was accessed. It can establish topic fit, not an empirical minimum acceptance bar.

## 2026 accepted-program context, separately labeled

Sousa, F., Costa, C., Nunes, F., Fontes-Carvalho, R., Packet, B., Cattani, L., Williams, H., & Pedrosa, J. *Leveraging AI uncertainty for targeted 3D medical image segmentation review.*

- [Official oral-program paper entry](https://cmsworkshops.com/EMBC2026/view_paper.php?PaperNum=3146&SessionID=1043&bare=1): accepted/presented in Deep Learning Architectures for Medical Image Segmentation, 29 July 2026, 15:30–15:45.
- [Official best-paper-award page](https://embc.embs.org/2026/best-paper-award/) lists this title/authors among multiple entries. This does not justify calling it the sole winner or ranking it above other listed papers.
- **Access: program-only for the conference paper.** No proceedings DOI or conference full text was located in this focused search; no datasets, sample sizes, method details, utility effects or statistics are attributed to that conference paper.
- A related master's-thesis chapter surfaced in search, but it was not silently substituted for the accepted conference paper. That distinction matters because a conference version can have different cohorts, endpoints and results.

## Exclusions and source-quality limits

- *Uncertainty Aware Segmentation Quality Assessment in Medical Images* (2024): ISBI, not EMBC; exclude from the EMBC accepted comparator set.
- *Uncertainty Estimation in Instance Segmentation With Star-Convex Shapes* (2024): WACV, not EMBC; potentially relevant related work, not evidence about EMBC acceptance.
- *Estimating Uncertainty in Neural Networks for Cardiac MRI Segmentation: A Benchmark Study* (2023): IEEE TBME journal, not EMBC; do not conflate it with the 2022 CondenseUNet EMBC paper.
- *Quality Control for Medical Image Segmentation under Domain Shift with Heteroscedastic Regression* (2026): ISBI, not EMBC.
- A lab news item mentions *Uncertainty-Guided Coarse-to-Fine Tumor Segmentation* next to EMBC publications but labels that particular paper ACDSA 2025; exclude rather than infer EMBC publication from the page context.
- Generic IEEE publisher indexing can classify EMBC proceedings as a PubMed “Journal Article”; the conference title and DOI, not that broad field alone, established the venue.
- This corpus deliberately concentrates on uncertainty/segmentation. It does not represent all EMBC tracks or the rejected-submission distribution. Publication records establish acceptance/publication, not why reviewers accepted a paper.
- Dataset totals, image/crop/segment counts and patient counts remain distinct. Reporting standard deviation is not reporting a confidence interval on a paired method difference.
- The author copies acknowledge institutional funding; a complete COI audit was not performed. No unverified claim of “no conflict” or comprehensive retraction clearance is made.

## Retrieval audit

FT-PDF access succeeded for AI2Seg after following the institutional landing-page link with the full `isAllowed=y&sequence=1` query, and for the Cambridge augmentation paper. Initial legacy-link direct retrieval failed. Damiano and Hu had substantive institutional full-paper method/result sections accessible via the search index, with direct fetches failing timeout/403; those are labeled accordingly. Direct Python download attempts first failed restricted DNS and then remote HTTP403, and no downloaded full-paper file was claimed from those attempts. No dataset, model checkpoint, experiment configuration or frozen result was changed.

### Counts for handoff

- Two directly related published segmentation papers: R1–R2 (one full PDF; one indexed full-text section access).
- Two adjacent published uncertainty papers: R3–R4 (one full PDF; one indexed full-text section access).
- One additional verified EMBC 2024 publication with abstract-only access.
- One exceptionally close EMBC 2026 oral/award-listed paper with program-only access and no unverified numerical details.
