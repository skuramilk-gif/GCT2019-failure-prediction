# Splits - Machine-Level Train / Val / Test Partition

## Overview
This folder contains the frozen machine-level splits used throughout
the project. The split is performed at the machine granularity to
prevent intra-machine leakage — all slices belonging to the same
machine reside exclusively in one split.

## Files
| File | Description |
|:---|:---|
| `step2b_train_machines_v2.txt` | List of machine IDs assigned to training |
| `step2b_val_machines_v2.txt` | List of machine IDs assigned to validation |
| `step2b_test_machines_v2.txt` | List of machine IDs assigned to testing |
| `step2b_split_meta_v2.json` | Metadata (seed, ratios, machine counts, SHA256 hash) |

## Generation
These files are produced by `Step2b_训练验证测试机器划分与固化_v2.py`.

### Key design choices
- **Fixed random seed (42)**: Ensures deterministic, reproducible splits.
- **70 / 10 / 20 ratio**: Follows standard ML practice for medium-sized datasets.
- **Machine-level group split**: All slices of the same machine are assigned
  to the same split, eliminating intra-machine leakage.
- **No label stratification**: The split is performed agnostic of label
  distribution to avoid concerns about using label information to engineer
  the split.

## Reproducibility
The generated split files are frozen for the entire pipeline. The
metadata file records the random seed, split ratios, and machine
counts. Any change to the split requires re-running the entire
pipeline from Stage 2 onward.