# Params - Train-Only Preprocessing Parameters

## Overview
This folder contains the frozen preprocessing parameters used to
transform all feature tables. The parameters are fitted strictly on
the training split (train ∩ label_valid_H8h domain) and are applied
unchanged to validation and test splits.

## Files
`step3b_preprocess_params_H8h_v2.pkl`
·A serialized Python dictionary containing:
  - impute_median (per feature)
  - clip_p01, clip_p99 (per feature)
  - scale_median, scale_iqr (per feature)
  - feature_cols (ordered feature list for schema validation)
  - binary_cols, count_cols (feature type annotations)
  - fit domain metadata (train ∩ label_valid_H8h)

## Generation
This file is produced by `Step3b_preprocess参数拟合_trainonly_H8h_v2.py`.

### Key design choices
- **Train-only fitting**: All statistics (median, percentiles, IQR) are
  computed exclusively on the training split. No data from validation
  or test sets is used during fitting.
- **Label-valid domain**: Only samples with valid labels (label ≠ NaN)
  are included, ensuring that FAIL-moment and right-censored samples
  do not influence the preprocessing parameters.
- **Single params, multiple H**: The same parameter set (fitted on H=8h)
  is used for all prediction windows (H=4,6,8,12) to avoid
  inconsistent preprocessing across tasks.

## Reproducibility
The SHA256 hash of this file is recorded in `MANIFEST_v2.json`.
Any modification to this file requires re-running the entire
transform pipeline (Step3c onwards).