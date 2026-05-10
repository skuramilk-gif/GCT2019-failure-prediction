# ============================================================
# 文件名：Step3a_派生特征构造_v2.py
# 阶段：Step 3A（v2 修正）
# 作用：
#   1) 在 clean 主表上构造确定性派生特征（不含任何拟合）
#   2) delta 在 gap 大时置 0（不置 NaN），保留 mask
#   3) 仅使用过去信息（shift(1)），不使用未来信息
#   4) 写入 split 标识（train/val/test）
#
# 输入：
#   01_interim/step2a_machine_slice_clean_v2.csv
#   03_splits/step2b_train_machines_v2.txt
#   03_splits/step2b_val_machines_v2.txt
#   03_splits/step2b_test_machines_v2.txt
#
# 输出：
#   01_interim/step3a_features_raw_v2.csv
#论文注意：gap_since_prev 是“观测步长”
# ============================================================

import os
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
IN_CLEAN = os.path.join(ROOT_DIR, "01_interim", "step2a_machine_slice_clean_v2.csv")
SPLIT_DIR = os.path.join(ROOT_DIR, "03_splits")
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

IN_TRAIN = os.path.join(SPLIT_DIR, "step2b_train_machines_v2.txt")
IN_VAL   = os.path.join(SPLIT_DIR, "step2b_val_machines_v2.txt")
IN_TEST  = os.path.join(SPLIT_DIR, "step2b_test_machines_v2.txt")

OUT_FEATURES = os.path.join(INTERIM_DIR, "step3a_features_raw_v2.csv")

MID = "machine_id"
TS  = "time_slice"

def read_machine_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(int(line.strip()) for line in f if line.strip())

print("="*90)
print("Step 3A(v2 fix)：派生特征构造（gap大时 delta=0，保留 mask）")
print("="*90)
print(f"[输入] {IN_CLEAN}")

df = pd.read_csv(IN_CLEAN, low_memory=False)

# split 标识
train_m = read_machine_list(IN_TRAIN)
val_m   = read_machine_list(IN_VAL)
test_m  = read_machine_list(IN_TEST)

df[MID] = pd.to_numeric(df[MID], errors="coerce").astype("int64")
df[TS]  = pd.to_numeric(df[TS], errors="coerce").astype("int64")

df["split"] = "unknown"
df.loc[df[MID].isin(train_m), "split"] = "train"
df.loc[df[MID].isin(val_m), "split"] = "val"
df.loc[df[MID].isin(test_m), "split"] = "test"
unknown_cnt = int((df["split"]=="unknown").sum())
assert unknown_cnt == 0, f"❌ split unknown rows={unknown_cnt}"

df = df.sort_values([MID, TS]).reset_index(drop=True)
# 主键唯一性（防重复 time_slice 造成 gap=0 被误判为连续观测）
dup = df.duplicated([MID, TS]).sum()
assert dup == 0, f"❌ features输入存在重复主键 (machine_id,time_slice) 重复数={dup}"
# gap_since_prev
df["gap_since_prev"] = df.groupby(MID)[TS].diff()
# delta_valid_mask（gap<=2）
df["delta_valid_mask"] = (df["gap_since_prev"].notna() & (df["gap_since_prev"] <= 2)).astype("int64")


# delta（只允许过去：shift(1)），gap大时置0；其余 NaN 也统一置0（避免后续靠运气被填）
for col in ["ms_max_cpu_max", "ms_max_mem_max", "ms_avg_cpu_mean", "ms_avg_mem_mean"]:
    prev = df.groupby(MID)[col].shift(1)
    d = df[col] - prev
    d = d.where(df["delta_valid_mask"] == 1, 0.0).fillna(0.0)
    df[f"delta_{col}"] = d

# gap 特征
df["cpu_gap"] = df["ms_avg_cpu_mean"] - df["ms_req_cpu_mean"]
df["mem_gap"] = df["ms_avg_mem_mean"] - df["ms_req_mem_mean"]

# 控制台摘要
print(f"[数据] rows={len(df)}, machines={df[MID].nunique()}")
print("[派生] delta_valid_mask rate:", float(df["delta_valid_mask"].mean()))
print("[预览] head(10):")
print(df[[MID, TS, "gap_since_prev", "delta_valid_mask", "cpu_gap", "mem_gap",
          "delta_ms_max_cpu_max", "delta_ms_max_mem_max"]].head(10).to_string(index=False))

df.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")
print("\n[输出保存]", OUT_FEATURES)
print("="*90)
print("Step 3A(v2 修正版) 完成")