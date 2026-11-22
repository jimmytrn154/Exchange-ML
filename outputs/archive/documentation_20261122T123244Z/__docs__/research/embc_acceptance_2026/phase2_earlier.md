# EMBC 2019–2022: verified bibliography and evidence extraction

**Phase 2 evidence record; checked 22 November 2026.** Scope: published full EMBC papers on medical segmentation uncertainty, reliability and uncertainty-guided decisions. This is a targeted narrative search, not an exhaustive systematic review or an estimate of the conference acceptance threshold. No Exchange-ML experiment was executed or changed. Publication is verified separately from the strength of the experimental design.

## Search and screening

Searches combined `EMBC` / the conference's expanded name with `segmentation uncertainty`, `segmentation quality`, `failure detection`, `quality control`, `Bayesian`, `Monte Carlo`, and publication years 2018–2022. Follow-up searches used exact paper titles and DOIs. Source families: publisher/proceedings metadata, PubMed/PMC author manuscripts, institutional repositories and the conference paper host. Search-engine discovery was followed by inspection of primary full-text sections wherever available.

Inclusion: verified EMBC full conference publication, direct relevance to segmentation uncertainty or its downstream use, enough accessible methods/results for extraction. Exclusion: journal papers mistakenly returned for EMBC citations; unrelated classification papers; preprints without independently confirmed EMBC publication. No numerical minimum for datasets, patients, effect size or novelty was chosen retrospectively. Four close published papers are retained rather than filling a quota with less relevant architecture papers. Title/DOI deduplication was used; no Semantic Scholar identifier resolution is claimed.

Access codes: **PDF** = primary full manuscript parsed directly; **indexed full text** = PMC methods/results were returned in search-indexed full text, while direct page opening encountered CAPTCHA. **NR** means not reported in the inspected evidence; it does not prove the information is absent from every version of a paper.

## E1. Vessel uncertainty and uncertainty-aware loss, EMBC 2019

**Reference.** Martin, R., Miró, J., & Duong, L. (2019). Epistemic uncertainty modeling for vessel segmentation. *2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, 5923–5927. [DOI: 10.1109/EMBC.2019.8857785](https://doi.org/10.1109/EMBC.2019.8857785).

**Verification.** The [authors' institutional publication record](https://pure.etsmtl.ca/fr/publications/epistemic-uncertainty-modeling-for-vessel-segmentation/) verifies venue, pages, publisher and DOI. The [accepted-manuscript PDF](https://espace2.etsmtl.ca/id/eprint/19866/1/Duong-L-2019-19866.pdf) explicitly identifies its published EMBC title; its manuscript title differs slightly. Access: PDF, five pages.

**Extraction.** §II-A reports 368 angiography images: 254 training images from 25 patients, 26 validation images, and 88 test images from 10 patients. Only five test patients were new relative to training; therefore the whole test set is not patient-disjoint. The approach modifies FC-DenseNet103 with MC dropout and compares ordinary with uncertainty-weighted cross-entropy. §III-C reports accuracy 94.9% versus 94.7%; Table II contrasts uncertainty for correct and incorrect pixels. The paper studies three artery categories within the local collection, not external-site validation. Inspected results provide descriptive uncertainty maps/distributions rather than risk–coverage or fixed-review-budget experiments; patient-cluster CIs and a measured reader-time study are NR. §IV explicitly leaves stronger evidence linking uncertainty to classification outcome for further work. This is an uncertainty-aware training/application contribution, not an instance-partition failure-ranking study.

**Locators.** PDF pp. 1–2, §II-A/II-B; p. 3, §III-C and Table I; p. 5, Table II and §IV. PDF parser lines 98–121, 304–319, 334–356.

## E2. Cross-center perfusion segmentation and uncertainty, EMBC 2021

**Reference.** Yalcinkaya, D. M., Youssef, K., Heydari, B., Zamudio, L., Dharmakumar, R., & Sharif, B. (2021). Deep learning-based segmentation and uncertainty assessment for automated analysis of myocardial perfusion MRI datasets using patch-level training and advanced data augmentation. *2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society*, 4072–4078. [DOI: 10.1109/EMBC46164.2021.9629581](https://doi.org/10.1109/EMBC46164.2021.9629581).

**Verification.** The [institutional author manuscript](https://scholarworks.indianapolis.iu.edu/server/api/core/bitstreams/14f02a0c-9c04-4dc5-a30c-72982507e601/content) names the final publication and DOI on its first page. The [conference-hosted PDF](https://www.paperhost.org/proceedings/embs/EMBC21/files/2083.pdf) has conference header and printed pages 4072–4078. Access: PDF, seven pages. Do not confuse this DOI with the different adjacent paper ending `9629630`.

**Extraction.** §II-A uses 96 stress/rest perfusion studies for development and 40 independent external-site stress studies, corresponding to 40 patients/120 myocardial slices for testing. The number of unique development patients and exact train/validation partition are NR in the inspected section. Four controlled configurations compare 2D versus 2D+time patches and conventional versus advanced augmentation, keeping the U-Net architecture fixed. §III reports mean test Dice 0.86 and downstream myocardial-blood-flow agreement, including Bland–Altman limits. An uncertainty classifier targets slices whose baseline automatic flow differs from expert-contour flow by over 5%; reported ROC AUC is 0.936, sensitivity 0.826 and specificity 0.918. The operating point is selected from the ROC; a separate threshold-validation cohort and AUC CI are NR. This supplies cross-center/acquisition-protocol and clinical-quantity evidence, but not a measured human correction-time trial.

**Locators.** Printed pp. 4073–4075, §II-A/II-D/II-E; pp. 4077–4078, §III-B/C/D and §IV. PDF parser lines 115–129, 212–225, 243–256, 412–472. The 120 slices must not be reported as 120 patients.

## E3. Bayesian CondenseUNet, EMBC 2022

**Reference.** Hasan, S. M. K., & Linte, C. A. (2022). Joint segmentation and uncertainty estimation of ventricular structures from cardiac MRI using a Bayesian CondenseUNet. *2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society*, 5047–5050. [DOI: 10.1109/EMBC48229.2022.9871780](https://doi.org/10.1109/EMBC48229.2022.9871780).

**Verification.** [PubMed PMID 36085846](https://pubmed.ncbi.nlm.nih.gov/36085846/) and [PMC author manuscript PMC10155706](https://pmc.ncbi.nlm.nih.gov/articles/PMC10155706/) identify the final EMBC publication, DOI and pages. Access: indexed full-text sections including methods, training, results and tables; direct PMC page requests encountered CAPTCHA. This is more evidence than an abstract-only extraction, but no successfully opened PDF is claimed.

**Extraction.** §II-A identifies ACDC with 150 patients, ED/ES cardiac structures and two scanner strengths. §II-E states 70% training, 15% validation and 15% testing with five nonoverlapping folds; it does not justify an exact independently held-out patient count from these percentages alone. The model adds MC dropout (rate 0.1; ten samples) and Brier-based training to CondenseUNet. Table I reports segmentation Dice/HD, e.g. LV ED Dice 96.8%; §III-B reports pixel-level uncertainty/error correlation r=0.94. The displayed quantitative table evaluates the proposed model; an extensive comparator ranking, external validation, case-level failure AURC, paired CI and fixed-budget review advantage are NR in the inspected results. Human-review reduction is discussed as a prospective use. Pixel correlation is not evidence of patient-level review utility or structural merge/split detection.

**Locators.** PMC §II-A, II-D, II-E; Table I and §III-B/Figs. 4–5. Report the source's split description as written rather than silently correcting its ambiguity or assuming all 150 are test patients.

## E4. Image-level uncertainty can be misleading, EMBC 2022

**Reference.** McBee, P., Zulqarnain, F., Syed, S., & Brown, D. E. (2022). Image-level uncertainty in pseudo-label selection for semi-supervised segmentation. *2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society*, 4740–4744. [DOI: 10.1109/EMBC48229.2022.9871359](https://doi.org/10.1109/EMBC48229.2022.9871359).

**Verification.** [PMC10445335](https://pmc.ncbi.nlm.nih.gov/articles/PMC10445335/) provides final publication metadata, author manuscript and complete indexed methods/results. The [2022 proceedings contents](https://www.proceedings.com/content/065/065515webtoc.pdf) independently lists the title and start page. Access: indexed full text; direct PMC page opening encountered CAPTCHA.

**Extraction.** Three histopathology datasets are examined: MoNuSeg 44 biopsies/176 cropped images, CoNSeP 41 source images/164 crops, and EoE 30 patients/514 images. Methods use eight random initializations, 20 MC-dropout predictions and entropy/standard-deviation/confidence summaries. Half the training images are treated as unlabeled; validation controls model selection, followed by a holdout test. Patient-disjoint partitioning and exact holdout sizes are NR. Controls include no pseudo-labels, all pseudo-labels, fixed cutoffs, low-entropy quartile selection and adaptive weighting. Table II uses Welch's t test. Results include reversed uncertainty–quality correlations and detrimental filtering: MoNuSeg Dice is 0.72±0.04 without pseudo-labels versus 0.66±0.07 with the low-entropy quartile. Three datasets are separate tasks, not demonstrated cross-site transfer of one fitted model. The contribution is analysis of when uncertainty-based selection fails, alongside tested interventions; it does not establish clinical review savings.

**Locators.** PMC §II (dataset denominators), §III-A/B/C (20 MC samples, selections, training), §IV and Tables I–II, §V–VI (mixed/negative findings). Quantities are reported using the paper's units; crops are not patients.

## Exclusions and evidence boundaries

- *Estimating Uncertainty in Neural Networks for Cardiac MRI Segmentation: A Benchmark Study*, DOI `10.1109/TBME.2022.3232730`, is a TBME journal paper, not an EMBC accepted-paper example. It can inform broader related work but is excluded from this EMBC sample.
- *Automatic segmentation with detection of local segmentation failures in cardiac MRI*, and *Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation*, are relevant journal work, excluded from the accepted-EMBC comparison for venue mismatch.
- *Calibration of cine MRI segmentation probability for uncertainty estimation using a multi-task cross-task learning architecture*, DOI `10.1117/12.2612269`, is SPIE 2022, not EMBC; it is not counted as an additional accepted EMBC uncertainty paper.
- *QU-net++: Image Quality Detection Framework for Segmentation of Medical 3D Image Stacks* has an arXiv record mentioning IEEE EMBC 2022, but this search did not independently confirm a final EMBC DOI/proceedings entry. It is not counted in the four verified publications.
- A recent program listing titled *Leveraging AI Uncertainty for Targeted 3D Medical Image Segmentation Review* was referred to the recent-literature task for separate verification; this earlier-year record does not treat it as a published full paper.
- No rejected submissions or reviewer scores were available. Accepted-only examples cannot establish an acceptance probability, a minimal patient count or a causal rule for acceptance. Differences across years and subjects limit direct comparability.

**Agent-typed actions:** primary-source web search, full-text/metadata reading and writing this Phase 2 report. PDF screenshot requests returned cache-miss errors; numerical extraction therefore uses the parsed PDF text or indexed full-text sections, with limitations stated above. No training, inference, dataset conversion or experimental result generation occurred.

**User-typed commands:** None.

**Handoff:** four venue-verified evidence records and their reporting limitations for a separate synthesis. This document makes no manuscript acceptance prediction and does not change Exchange-ML's claims.
