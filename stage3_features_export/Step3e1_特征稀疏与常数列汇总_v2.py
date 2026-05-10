# ============================================================
# 文件名：Step3e1_特征稀疏与常数列汇总_v2.py
# 阶段：Step 3E-1（v2，Docs）
# 作用：
#   基于 Step3D 的列级统计表，汇总：
#     1) 常数列（std==0 或 nunique<=1）
#     2) 近常数列（std 很小）
#     3) 低离散列（nunique<=2，常见于二值/极低多样性特征）
#
# 注意（关键修正）：
#   processed 表经过 impute+clip+robust_scale 后，zero_rate 的“0”不再等价于原始零值，
#   它更接近“等于训练集中位数”的堆积。因此 zero_rate 不用于 sparse 判定，避免误导。
#
# 输入：
#   05_reports/step3d_input_health_collevel_v2.csv
#
# 输出：
#   05_reports/step3e1_sparse_constant_summary_v2.csv
# ============================================================

import os
import pandas as pd
import numpy as np

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")

IN_COL = os.path.join(REPORT_DIR, "step3d_input_health_collevel_v2.csv")
OUT_SUM = os.path.join(REPORT_DIR, "step3e1_sparse_constant_summary_v2.csv")

assert os.path.exists(IN_COL), f"❌ 缺少输入文件: {IN_COL}"

df = pd.read_csv(IN_COL, low_memory=False)

need_cols = {"file", "col", "std", "nunique", "zero_rate"}
miss = need_cols - set(df.columns)
assert len(miss) == 0, f"❌ collevel 表缺少列: {miss}"

# 只关心特征列（排除 meta/label）
EXCLUDE = {"meta_machine_id", "meta_time_slice", "label"}
df = df[~df["col"].isin(EXCLUDE)].copy()

# 常数列：std==0 或 nunique<=1
df["is_constant"] = (df["std"] == 0) | (df["nunique"] <= 1)

# 近常数：std 很小（经验阈值；processed 空间下仍可用）
df["is_near_constant"] = (df["std"] < 1e-6)

# 低离散（常见于二值/极低多样性列）；注意排除 constant 避免重复统计
df["is_low_discrete"] = (~df["is_constant"]) & (df["nunique"] <= 2)

# 汇总到 col 级：跨多少文件被判为常数/近常数/低离散
g = df.groupby("col").agg(
    files=("file", "nunique"),
    constant_files=("is_constant", "sum"),
    near_constant_files=("is_near_constant", "sum"),
    low_discrete_files=("is_low_discrete", "sum"),

    # 这些是参考性统计（用于解释“中位数堆积/低多样性”现象），不用于稀疏判定
    zero_rate_mean=("zero_rate", "mean"),
    nunique_mean=("nunique", "mean"),
    std_mean=("std", "mean"),
).reset_index()

den = g["files"].replace(0, np.nan)
g["constant_ratio"] = g["constant_files"] / den
g["near_constant_ratio"] = g["near_constant_files"] / den
g["low_discrete_ratio"] = g["low_discrete_files"] / den

# 排序：优先把“到处是常数/近常数/低离散”的列排前
g = g.sort_values(
    ["constant_ratio", "near_constant_ratio", "low_discrete_ratio", "zero_rate_mean"],
    ascending=False
)

g.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

print("=" * 90)
print("Step3E-1：常数/近常数/低离散列汇总（Top 50）")
print("=" * 90)
print(g.head(50).to_string(index=False))
print("\n[输出保存]", OUT_SUM)
print("=" * 90)