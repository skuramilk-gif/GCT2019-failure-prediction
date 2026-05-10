# GCT2019 Machine-Level Failure Prediction

**Production-grade data processing pipeline for machine-level failure prediction on the Google Cluster Trace 2019 (GCT2019) dataset.**

## Data Availability

The raw data used in this project is the **Google Borg Cluster Trace 2019**.
Due to its size and license restrictions, it is not included in this repository.

**To obtain the data:**
- Download from Kaggle: [Google Borg Cluster Trace 2019](https://www.kaggle.com/datasets/derrickmwiti/google-borg-cluster-trace-2019)
- Place the downloaded `Borg_Traces_2019.csv` file in the `00_raw/` directory.
- The pipeline will automatically locate and process this file upon execution.

## Pipeline Overview

The pipeline enforces a Core / Gate / Docs architecture:

| Stage | Name | Responsibility |
|:---|:---|:---|
| Stage 1 | Semantic Evidence | Raw data field statistics, label equivalence, time semantics |
| Stage 2 | Clean / Split / Label | Master table, machine-level split, multi-H label, QC |
| Stage 3 | Features / Export | Feature engineering, train-only preprocessing, 48-table export |
| Stage 4 | Freeze / Manifest | Pipeline orchestration, evidence freeze, SHA256 manifest |

## Repository Structure

\```
├── config/             # Global configuration (config.yaml)
├── splits/             # Frozen machine-level train / val / test splits
├── params/             # Train-only preprocessing parameters
├── evidence/           # Audit trail reports (leakage, time order, label QC, etc.)
├── stage1_semantic_evidence/
├── stage2_clean_split_label/
├── stage3_features_export/
├── stage4_pipeline_freeze/
├── requirements.txt    # Python dependencies
├── MANIFEST_v2.json    # Reproducibility fingerprint
└── README.md           # This file
\```

## Reproducibility

All preprocessing parameters are fitted on the training split only (train ∩ label_valid_H8h domain). The SHA256 hash of every key file is recorded in `MANIFEST_v2.json`. The full evidence chain is documented in the `evidence/` folder.

## Citation

If you use this pipeline in your research, please cite our paper:

_\[Your Paper Title, Authors, Journal/Conference, Year]_
## Scope and Target Audience

### Who is this for?
This pipeline is designed for researchers and engineers working on:
- **Cloud failure prediction** — building machine learning models
  to forecast machine-level failures before they occur.
- **Imbalanced tabular data problems** — especially those with
  extreme class imbalance (positive rate < 2%) and weak signals.
- **Production-grade data processing** — anyone who needs to
  construct a reproducible, auditable data pipeline from raw
  industrial traces.

### What types of anomaly detection does this support?
This pipeline transforms raw Google Borg Cluster Trace data into
a machine-slice-level feature matrix suitable for:
- **Failure risk scoring** — predict whether a machine will experience
  a FAIL event within the next H hours (H=4,6,8,12).
- **Failure type clustering** — distinguish between different
  failure modes (e.g., extreme failures vs. common failures).
- **Top-K early warning** — rank machines by risk score and output
  interpretable top-K alerts for operational intervention.

### Key Features
- **Strict leakage prevention**: train-only preprocessing, machine-
  level split, multi-layer audit gates.
- **Informative missingness modeling**: missing indicators are
  explicitly modelled as signals rather than imputed away.
- **Full evidence chain**: every design choice is backed by
  automated Gate scripts and documented in the `evidence/` folder.