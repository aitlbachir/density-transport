Note: this plan was written on 14 Oct 2026, before the RSNA data were loaded, and is kept here as written. It has no public timestamp.

# RSNA screening case study: analysis plan

## Data
RSNA Screening Mammography Breast Cancer Detection, training set (Kaggle competition data, accepted
rules: non-commercial academic use). Images: Theo Viel's 512x512 PNG conversion of the same DICOMs.
Unit of analysis: breast (patient_id, laterality). Label: `cancer` (biopsy-confirmed, breast level).
Density (A-D) exists for site 1 (Emory, USA) only, so all density analyses use site-1 breasts with a
density label.

## Model (fixed before seeing results)
EfficientNet-B0 (ImageNet), 512x512, AMP, AdamW lr 3e-4 cosine, batch 32, 10 epochs, no early stopping.
Each epoch: all positive training images + a fresh random sample of 10x as many negative images.
Augmentation: horizontal/vertical flips, small affine, brightness/contrast. Patient-level
StratifiedGroupKFold (5 folds, seed 42, stratified by patient cancer status). Out-of-fold (OOF)
image scores (average of original and horizontally flipped input); breast score = mean over its images.

## Operating point (fixed)
Primary: per fold, the threshold flagging 10% of that fold's non-cancer breasts (both sites), i.e. a
10% AI false-positive (recall) rate, typical of screening. Secondary: 5%.

## Analyses
- R1. Site-1 breast-level sensitivity and specificity per density A-D with exact 95% CIs; overall and
  per-site AUC.
- R2. Natural population shift by age within site 1 (USPSTF 2024 extended screening to 40-49):
  source = women aged >= 50, target = women aged 40-49. From source density-stratified counts:
  pooled sensitivity and its one-sided 95% Clopper-Pearson lower bound; transported to the target's
  cancer density mix: point, Jeffreys and exact (Sidak) certificates; compared with the observed
  target sensitivity (exact 95% CI). Same for specificity with the target's non-cancer density mix.
  Strata: A+B (non-dense), C, D (A alone is too small). Also the reverse direction (40-49 -> >= 50).
- R3. Coverage simulation calibrated to the real model: E1 of the paper repeated with stratum
  sensitivities set to the R1 estimates (strata A+B, C, D) and the site-1 cancer density mix as source.
Everything else is exploratory and labelled as such.

## Outcome log (added 14 Oct 2026, after the analyses; the plan above is unchanged)
- Run: Kaggle version 358473459, 5 folds completed. Training as planned (AdamW, OneCycle schedule with 10% warm-up
  and cosine annealing, which is the "cosine" schedule above).
- R2 (primary): sensitivity >=50 -> 40-49: transported 0.476, exact certificate 0.362; observed 0.286 [0.132, 0.487]
  on 28 cancers. Specificity: transported 0.853, observed 0.918. Reverse direction: specificity exact certificate
  0.907 > observed 0.862 [0.855, 0.870]. Interpretation: Assumption A (conditional transportability given density)
  is violated by age for this model. Reported in the paper as such.
- Exploratory (not pre-registered, labelled in the paper): CMH tests of age within density; Fisher test of the
  CBIS density-3 change; power of a local check that uses non-cancer examinations only.
