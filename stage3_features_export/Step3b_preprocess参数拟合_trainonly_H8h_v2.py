# ============================================================
# 文件名：Step3b_preprocess参数拟合_trainonly_H8h_v2.py
# 阶段：Step 3B（v2，H=8h 主任务）
# 作用：
#   1) 仅使用真实 Train + label有效样本（H=8h）拟合预处理参数：
#        - impute: median（基于观测值 dropna）
#        - clip:
#            * binary: [0,1]
#            * count : [min,max]（基于观测值 dropna）
#            * cont  : [p01,p99]（基于观测值 dropna）
#        - scale: robust (x - median) / IQR（基于观测值 dropna；IQR过小则置1）
#   2) 固化特征列 schema（feature_cols）并记录类型（binary/count/continuous）
#   3) 保存 params 供 Step3C transform 使用
#   4)查看是否有机器id跨train/test的情况，并且做test和train的正率占比，一般pos_rate_ratio数字在0.8以下是合格，0.5以下是安全。
# 输入：
#   01_interim/step3a_features_raw_v2.csv
#   01_interim/step2c_machine_slice_labeled_FAILonly_H8h_v2.csv
#
# 输出：
#   04_params/step3b_preprocess_params_H8h_v2.pkl
#   05_reports/step3b_preprocess_fit_summary_H8h_v2.txt
#论文影响和约束：控制台的delta_valid_mask rate说明短期动态信息严重不足，在论文中就不要硬讲，
#             并且将delta拆分成core和delta两套，这个之后可以进行实验进一步证明delta是否有用（我大概率认为是没用的）
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

# ----------------------------
# 路径与常量
# ----------------------------
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
PARAM_DIR = os.path.join(ROOT_DIR, "04_params")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(PARAM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

IN_FEATURES = os.path.join(INTERIM_DIR, "step3a_features_raw_v2.csv")
IN_LABEL8H  = os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv")

OUT_PARAMS = os.path.join(PARAM_DIR, "step3b_preprocess_params_H8h_v2.pkl")
OUT_REPORT = os.path.join(REPORT_DIR, "step3b_preprocess_fit_summary_H8h_v2.txt")

MID = "machine_id"
TS  = "time_slice"
LABEL_COL = "label_fail_H8h"
EPS = 1e-12

print("=" * 90)
print("Step 3B(v2)：预处理参数拟合（train-only, H=8h 有效样本）")
print("=" * 90)
print("[输入]", IN_FEATURES)
print("[输入]", IN_LABEL8H)

# ----------------------------
# 读入与合并
# ----------------------------
feat = pd.read_csv(IN_FEATURES, low_memory=False)
lab  = pd.read_csv(IN_LABEL8H, usecols=[MID, TS, LABEL_COL], low_memory=False)

# 强一致性：one-to-one
df = feat.merge(lab, on=[MID, TS], how="left", validate="one_to_one")

assert "split" in df.columns, "❌ features_raw 缺少 split 列（Step3a 应写入 split）"
assert LABEL_COL in df.columns, f"❌ 合并后缺少 {LABEL_COL}"

train_df = df[(df["split"] == "train") & df[LABEL_COL].notna()].copy()
pos = int(train_df[LABEL_COL].sum())
print(f"[拟合数据] rows={len(train_df)}  pos={pos}")

# ----------------------------
# drop：索引/时间锚点/split/label/泄漏列
# ----------------------------
DROP = {
    MID, TS, "slice_last_time_us", "split",
    LABEL_COL,
    "failed_count",  # 绝不作为特征
}

feature_cols = [c for c in feat.columns if c not in DROP]
feature_cols = [c for c in feature_cols if c != LABEL_COL]
feature_cols = [c for c in feature_cols if c != "kill_count"]


assert len(feature_cols) > 0, "❌ feature_cols 为空：检查 DROP 列表是否误删"

# ----------------------------
# 特征类型识别（修复 abnormal_nonfail_flag 被当 binary 的错误）
# ----------------------------
binary_cols, count_cols, continuous_cols = [], [], []

BINARY_WHITELIST = {"delta_valid_mask", "ms_cpi_missing", "ms_mpi_missing", "abnormal_nonfail_flag"}
COUNT_WHITELIST = {"record_count"}  # 只保留真正的计数特征

# 排除 train 中完全为0的计数特征（clip后为常数，无信息量）
ZERO_FEATURE_DROP = {"kill_count"}  # train中p99=0且max=0，完全失效


for c in feature_cols:
    # 明确将 sched_frac_* 列视为 binary/frac（clip=[0,1]）
    if (c in BINARY_WHITELIST) or c.endswith("_missing") or c.startswith("sched_frac_"):
        binary_cols.append(c)
    elif (c in COUNT_WHITELIST) or c.endswith("_count"):
        count_cols.append(c)
    else:
        continuous_cols.append(c)

print(f"[特征列] total={len(feature_cols)}  binary={len(binary_cols)}  count={len(count_cols)}  cont={len(continuous_cols)}")

# ----------------------------
# 参数容器
# ----------------------------
params = {
    "feature_cols": feature_cols,
    "binary_cols": binary_cols,
    "count_cols": count_cols,
    "continuous_cols": continuous_cols,
    "impute_median": {},
    "clip_p01": {},
    "clip_p99": {},
    "scale_median": {},
    "scale_iqr": {},
    "near_constant_cols": [],
}

def median_obs(x: pd.Series) -> float:
    x_obs = x.dropna()
    return float(x_obs.median()) if len(x_obs) else 0.0

# ----------------------------
# 拟合每列参数（clip/scale 基于观测值 dropna）
# ----------------------------
for c in feature_cols:
    x = pd.to_numeric(train_df[c], errors="coerce")
    x_obs = x.dropna()

    med = median_obs(x)

    # clip bounds：基于观测值（不使用 fillna 后的堆尖分布）
    if c in binary_cols:
        lo, hi = 0.0, 1.0
    elif c in count_cols:
        if len(x_obs):
            lo = float(x_obs.min())
            if c == "record_count":
                # 特判：record_count 取值极小（<=3），保留稀有 >1 的信息
                hi = float(x_obs.max())
            else:
                # 其它 count 仍建议用分位数剪裁（更稳）
                hi = float(x_obs.quantile(0.99))
        else:
            lo, hi = 0.0, 0.0
    else:
        if len(x_obs):
            lo, hi = float(x_obs.quantile(0.01)), float(x_obs.quantile(0.99))
        else:
            lo, hi = med, med

    if hi < lo:
        lo, hi = hi, lo

    # robust scale：同样基于观测值
    if len(x_obs):
        q25, q75 = float(x_obs.quantile(0.25)), float(x_obs.quantile(0.75))
    else:
        q25, q75 = med, med
    iqr = q75 - q25
    if abs(iqr) < EPS:
        iqr = 1.0
        params["near_constant_cols"].append(c)

    params["impute_median"][c] = med
    params["clip_p01"][c] = lo
    params["clip_p99"][c] = hi
    params["scale_median"][c] = med
    params["scale_iqr"][c] = iqr

# ----------------------------
# 保存 params
# ----------------------------
with open(OUT_PARAMS, "wb") as f:
    pickle.dump(params, f)

# ----------------------------
# 报告
# ----------------------------
lines = []
lines.append("=== Step3B(v2) preprocess fit summary (train-only, H=8h) ===")
lines.append(f"train_rows_used={len(train_df)}  pos={pos}")
# gate1: split 互斥（按 machine）
m_train = set(df.loc[df["split"]=="train", MID].unique())
m_val   = set(df.loc[df["split"]=="val", MID].unique())
m_test  = set(df.loc[df["split"]=="test", MID].unique())
assert len(m_train & m_val)==0 and len(m_train & m_test)==0 and len(m_val & m_test)==0, "❌ split machine overlap"

# gate2: H=8 label-valid train 行数对齐（可选：与 Step2c/Step2d 报告一致）
lines.append(f"feature_cols_total={len(feature_cols)}")
lines.append(f"binary_cols_n={len(binary_cols)}  count_cols_n={len(count_cols)}  cont_cols_n={len(continuous_cols)}")
lines.append("binary_cols=" + str(binary_cols))
lines.append("count_cols=" + str(count_cols))
lines.append("near_constant_cols_n=" + str(len(params["near_constant_cols"])))
lines.append("near_constant_cols_preview=" + str(params["near_constant_cols"][:80]))
lines.append("NOTE: clip(binary)=[0,1]; clip(count)=[min,max] on observed; clip(cont)=[p01,p99] on observed.")
lines.append("NOTE: scale uses median/IQR on observed; all params fitted ONLY on train & label-notna (H=8h).")

lines.append("\n[count features clip bounds — 检查 p99=0 的稀疏特征是否被废掉]")
for c in count_cols:
    lines.append(f"  {c}: lo={params['clip_p01'][c]}, hi={params['clip_p99'][c]}, "
                 f"median={params['impute_median'][c]}")
# 标记可疑特征（hi <= 1 的 count 特征，clip 后几乎无信息量）
suspicious = [c for c in count_cols if params["clip_p99"][c] <= 1]
if suspicious:
    lines.append(f"[WARN] 以下 count 特征 p99<=1，clip 后可能信息量极低: {suspicious}")

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# 控制台关键摘要
print("\n[关键检查] abnormal_nonfail_count clip bounds (count 特征，合理应 >=1):")
c = "abnormal_nonfail_count"
if c in params["clip_p99"]:
    print("  abnormal_nonfail_count:",
          "lo=", params["clip_p01"][c],
          "hi=", params["clip_p99"][c],
          "type=", ("count" if c in count_cols else "NOT_COUNT"))

print("\n[输出保存]", OUT_PARAMS)
print("[输出保存]", OUT_REPORT)
print("=" * 90)
print("Step 3B(v2) 完成")