# ============================================================
# 文件名：Step3f_reports_bundle_v2.py
# 阶段：Stage3（报告打包，v2）
#
# 作用（一次性生成三个“进入训练前必备”的审稿级报告）：
#
# [Part A] 特征字典模板 + derived_core 完整列清单（回应：特征维度来源不透明）
#   - 输入：02_processed/step3c_model_input_derived_H8h_train_v2.csv（只读表头）
#   - 输出：
#       05_reports/step3f_feature_list_derived_v2.txt
#       05_reports/step3f_feature_dictionary_template_derived_v2.csv
#   - 为什么要做：
#       二区审稿硬要求：给出完整特征列表（列名、来源、聚合、类型、预处理说明）。
#
# [Part B] CPU分布缺失指示是否与 label 强相关（回应：missing indicator 是否携带强信号）
#   - 输入：
#       01_interim/step3a_features_raw_v2.csv（含 cpu_dist_missing/tail_cpu_dist_missing）
#       01_interim/step2c_machine_slice_labeled_FAILonly_H8h_v2.csv（label_fail_H8h）
#   - 输出：
#       05_reports/step3f_cpu_dist_missing_vs_label_H8h_trainval_v2.csv
#   - 为什么要做：
#       parse_fail ≈4.4% 不是泄漏，但可能与故障状态相关，需显式报告以便审稿人理解。
#
# [Part C] assigned_memory/page_cache_memory 的零值语义检查（回应：0 是否可能是缺失填充）
#   - 输入：01_interim/step2a_machine_slice_clean_v2.csv
#   - 输出：
#       05_reports/step3f_memory_zero_semantics_check_v2.csv
#   - 为什么要做：
#       0 可能是真实值，也可能是缺失/异常编码；需要在 FAIL slice vs non-FAIL slice 对照其零值率。
#
# 运行要求：
#   - 不修改任何数据表，只生成报告文件
#   - 路径与文件名按 v2 规范落盘到 05_reports
# ============================================================
# ============================================================


import os
import pandas as pd
import numpy as np

# =========================
# 0) 路径与常量
# =========================
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
PROC_DIR = os.path.join(ROOT_DIR, "02_processed")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

MID = "machine_id"
TS = "time_slice"
SPLIT = "split"

print("=" * 90)
print("Step3F(v2)：报告打包生成（特征字典 + missing相关性 + 零值语义）")
print("=" * 90)

# ============================================================
# Part A) derived 特征列清单 + 特征字典模板
# ============================================================
print("\n" + "-" * 90)
print("[Part A] 导出 derived 完整列清单 + 特征字典模板")

IN_SCHEMA = os.path.join(PROC_DIR, "step3c_model_input_derived_H8h_train_v2.csv")
OUT_LIST = os.path.join(REPORT_DIR, "step3f_feature_list_derived_v2.txt")
OUT_DICT = os.path.join(REPORT_DIR, "step3f_feature_dictionary_template_derived_v2.csv")

assert os.path.exists(IN_SCHEMA), f"❌ 缺少输入文件: {IN_SCHEMA}"

# 只读表头
df0 = pd.read_csv(IN_SCHEMA, nrows=0)
cols = df0.columns.tolist()

NON_FEAT = {"meta_machine_id", "meta_time_slice", "label"}
feat_cols = [c for c in cols if c not in NON_FEAT]

def infer_type_hint(c: str) -> str:
    c_low = c.lower()
    # mask/flag/missing/frac 特征
    if ("mask" in c_low) or c_low.endswith("_missing") or c_low.endswith("_flag") or c_low.startswith("sched_frac_"):
        return "binary_or_frac"
    # 计数
    if c_low.endswith("_count") or c_low == "record_count":
        return "count"
    return "continuous"

with open(OUT_LIST, "w", encoding="utf-8") as f:
    for c in feat_cols:
        f.write(c + "\n")

dict_df = pd.DataFrame({
    "feature": feat_cols,
    "type_hint": [infer_type_hint(c) for c in feat_cols],
    "source_field": "",
    "aggregation": "",
    "preprocess": "impute+clip+robust_scale(train-only)",
    "notes": "",
})
dict_df.to_csv(OUT_DICT, index=False, encoding="utf-8-sig")

print(f"  输入(schema) : {IN_SCHEMA}")
print(f"  输出(list)   : {OUT_LIST}")
print(f"  输出(dict)   : {OUT_DICT}")
print(f"  n_features(derived, excl meta/label) = {len(feat_cols)}")

# ============================================================
# Part B) cpu_dist_missing 与 label 的相关性统计（label-valid域，train/val-only）
# ============================================================
print("\n" + "-" * 90)
print("[Part B] CPU分布 missing 指示 vs label（H=8h, label-valid, train/val-only）")

IN_FEAT = os.path.join(INTERIM_DIR, "step3a_features_raw_v2.csv")
IN_LAB8 = os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv")
OUT_MISS = os.path.join(REPORT_DIR, "step3f_cpu_dist_missing_vs_label_H8h_trainval_v2.csv")

LABEL_COL = "label_fail_H8h"

assert os.path.exists(IN_FEAT), f"❌ 缺少输入文件: {IN_FEAT}"
assert os.path.exists(IN_LAB8), f"❌ 缺少输入文件: {IN_LAB8}"

need_feat_cols = [MID, TS, SPLIT, "cpu_dist_missing", "tail_cpu_dist_missing"]
feat = pd.read_csv(IN_FEAT, usecols=need_feat_cols, low_memory=False)
lab = pd.read_csv(IN_LAB8, usecols=[MID, TS, LABEL_COL], low_memory=False)

df = feat.merge(lab, on=[MID, TS], how="inner", validate="one_to_one")

# 只在 label-valid 域统计（与你训练/评估样本域一致）
df = df[df[LABEL_COL].notna()].copy()
df["label"] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)

# Gate：missing 指示必须为 0/1 且不允许 NaN
assert df["cpu_dist_missing"].isna().sum() == 0, "❌ cpu_dist_missing 存在 NaN"
assert df["tail_cpu_dist_missing"].isna().sum() == 0, "❌ tail_cpu_dist_missing 存在 NaN"
assert set(df["cpu_dist_missing"].unique()).issubset({0, 1}), "❌ cpu_dist_missing 非 0/1"
assert set(df["tail_cpu_dist_missing"].unique()).issubset({0, 1}), "❌ tail_cpu_dist_missing 非 0/1"

# train/val-only（evidence chain）
df = df[df[SPLIT].isin(["train", "val"])].copy()

rows = []
for sp in ["train", "val"]:
    sub = df[df[SPLIT] == sp]
    for y in [0, 1]:
        g = sub[sub["label"] == y]
        rows.append({
            "split": sp,
            "label": y,
            "n": int(len(g)),
            "cpu_dist_missing_rate": float(g["cpu_dist_missing"].mean()) if len(g) else np.nan,
            "tail_cpu_dist_missing_rate": float(g["tail_cpu_dist_missing"].mean()) if len(g) else np.nan,
        })

miss_df = pd.DataFrame(rows)

# effect size：每个 split 内 y=1 vs y=0 的差值/比值
eff_rows = []
for sp in ["train", "val"]:
    a = miss_df[(miss_df["split"] == sp) & (miss_df["label"] == 0)]
    b = miss_df[(miss_df["split"] == sp) & (miss_df["label"] == 1)]
    if len(a) and len(b):
        cpu0 = float(a["cpu_dist_missing_rate"].iloc[0])
        cpu1 = float(b["cpu_dist_missing_rate"].iloc[0])
        tail0 = float(a["tail_cpu_dist_missing_rate"].iloc[0])
        tail1 = float(b["tail_cpu_dist_missing_rate"].iloc[0])

        eff_rows.append({
            "split": sp,
            "cpu_missing_rate_diff(y1-y0)": cpu1 - cpu0,
            "cpu_missing_rate_ratio(y1/y0)": (cpu1 / cpu0) if cpu0 > 0 else np.nan,
            "tail_missing_rate_diff(y1-y0)": tail1 - tail0,
            "tail_missing_rate_ratio(y1/y0)": (tail1 / tail0) if tail0 > 0 else np.nan,
        })

eff_df = pd.DataFrame(eff_rows)

# 合并输出（两段表拼在一起，便于一眼读完）
out_df = miss_df.copy()
out_df.to_csv(OUT_MISS, index=False, encoding="utf-8-sig")

print(f"  输入(features): {IN_FEAT}")
print(f"  输入(label8h) : {IN_LAB8}")
print(f"  输出         : {OUT_MISS}")
print("  missing rate by split/label：")
print(miss_df.to_string(index=False))
print("\n  effect size within split (y=1 vs y=0):")
print(eff_df.to_string(index=False))

# ============================================================
# Part C) memory 相关特征零值语义检查（FAIL slice vs non-FAIL slice）
# ============================================================
print("\n" + "-" * 90)
print("[Part C] assigned_memory/page_cache_memory 零值语义检查（FAIL slice vs non-FAIL slice）")

IN_CLEAN = os.path.join(INTERIM_DIR, "step2a_machine_slice_clean_v2.csv")
OUT_ZERO = os.path.join(REPORT_DIR, "step3f_memory_zero_semantics_check_v2.csv")

FAIL_CNT = "failed_count"
ASSIGNED = "ms_assigned_memory_mean"
PAGECACHE = "ms_page_cache_memory_mean"

assert os.path.exists(IN_CLEAN), f"❌ 缺少输入文件: {IN_CLEAN}"

usecols = [FAIL_CNT, ASSIGNED, PAGECACHE]
clean = pd.read_csv(IN_CLEAN, usecols=usecols, low_memory=False)

for c in usecols:
    assert c in clean.columns, f"❌ clean 缺少列 {c}"

clean[FAIL_CNT] = pd.to_numeric(clean[FAIL_CNT], errors="coerce").fillna(0).astype(int)
clean[ASSIGNED] = pd.to_numeric(clean[ASSIGNED], errors="coerce")
clean[PAGECACHE] = pd.to_numeric(clean[PAGECACHE], errors="coerce")

fail = clean[clean[FAIL_CNT] > 0]
nonfail = clean[clean[FAIL_CNT] == 0]

def group_row(name, sub):
    def q(s, p):
        s2 = s.dropna()
        return float(s2.quantile(p)) if len(s2) else np.nan

    return {
        "group": name,
        "n": int(len(sub)),

        "assigned_zero_rate": float((sub[ASSIGNED] == 0).mean()),
        "page_cache_zero_rate": float((sub[PAGECACHE] == 0).mean()),

        "assigned_missing_rate": float(sub[ASSIGNED].isna().mean()),
        "page_cache_missing_rate": float(sub[PAGECACHE].isna().mean()),

        "assigned_mean": float(sub[ASSIGNED].mean()) if len(sub[ASSIGNED].dropna()) else np.nan,
        "assigned_median": q(sub[ASSIGNED], 0.5),
        "assigned_p99": q(sub[ASSIGNED], 0.99),

        "page_cache_mean": float(sub[PAGECACHE].mean()) if len(sub[PAGECACHE].dropna()) else np.nan,
        "page_cache_median": q(sub[PAGECACHE], 0.5),
        "page_cache_p99": q(sub[PAGECACHE], 0.99),
    }

zero_df = pd.DataFrame([group_row("fail_slice", fail), group_row("nonfail_slice", nonfail)])
zero_df.to_csv(OUT_ZERO, index=False, encoding="utf-8-sig")

print(f"  输入(clean) : {IN_CLEAN}")
print(f"  输出       : {OUT_ZERO}")
print("  预览：")
print(zero_df.to_string(index=False))

print("\n" + "=" * 90)
print("Step3F(v2) 完成：三份报告已全部落盘到 05_reports/")
print("=" * 90)