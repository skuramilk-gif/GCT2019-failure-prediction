# ============================================================
# 文件名：Step3c_transform导出模型输入表_multiH_v2.py
# 阶段：Step 3C（v2，方案A修正版，主 params=H8h）
# 作用：
#   1) 使用 Step3B 拟合的 preprocess params（H=8h train-only）对数据做 transform
#   2) 对 H=4h/6h/8h/12h 分别导出 train/val/test 的模型输入表（分层特征组）
#   3) 【方案A】“derived = workload + gap_since_prev + delta_valid_mask（不包含其它 delta_rolling 特征）”
#   4) 每个输出表做 NaN/Inf 快速检查（控制台摘要）
#
# 输入：
#   01_interim/step3a_features_raw_v2.csv
#   04_params/step3b_preprocess_params_H8h_v2.pkl
#   01_interim/step2c_machine_slice_labeled_FAILonly_H{H}h_v2.csv  (H=4,6,8,12)
#
# 输出（02_processed/）：
#   step3c_model_input_{set}_H{H}h_{split}_v2.csv
#     set ∈ {base_real, extended_real, workload, derived}=set ∈ {基础资源统计, 扩展资源统计+分布特征, 事件计数+调度属性, 派生特征}
#
# 报告（05_reports/）：
#   step3c_export_summary_multiH_v2.csv
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
PARAM_DIR = os.path.join(ROOT_DIR, "04_params")
OUT_DIR = os.path.join(ROOT_DIR, "02_processed")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

INCLUDE_TEST_IN_SUMMARY = False#开关

IN_FEATURES = os.path.join(INTERIM_DIR, "step3a_features_raw_v2.csv")
IN_PARAMS = os.path.join(PARAM_DIR, "step3b_preprocess_params_H8h_v2.pkl")

LABEL_FILES = {
    4:  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H4h_v2.csv"),
    6:  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H6h_v2.csv"),
    8:  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv"),
    12: os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H12h_v2.csv"),
}

MID = "machine_id"
TS  = "time_slice"
SPLIT = "split"
#====================================================================
def quick_nan_inf_report(df: pd.DataFrame, name: str):
    num = df.select_dtypes(include=[np.number])
    nan_cnt = int(num.isna().sum().sum())
    inf_cnt = int(np.isinf(num.to_numpy()).sum())
    print(f"    [QC] {name}: shape={df.shape}, nan={nan_cnt}, inf={inf_cnt}")

def transform_df(df: pd.DataFrame, params: dict, feature_cols: list) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        med = params["impute_median"][c]
        x = x.fillna(med)

        lo = params["clip_p01"][c]
        hi = params["clip_p99"][c]
        x = x.clip(lo, hi)

        cen = params["scale_median"][c]
        iqr = params["scale_iqr"][c]
        x = (x - cen) / iqr
        X[c] = x
    return X

print("=" * 90)
print("Step 3C(v2)：transform + 导出模型输入表（multi-H=4/6/8/12, no-delta, params=H8h）")
print("=" * 90)
print("[输入]", IN_FEATURES)
print("[输入]", IN_PARAMS)

feat = pd.read_csv(IN_FEATURES, low_memory=False)
with open(IN_PARAMS, "rb") as f:
    params = pickle.load(f)

# 关键列检查
for c in [MID, TS, SPLIT]:
    assert c in feat.columns, f"❌ features_raw 缺少列 {c}"

all_feature_cols = set(params["feature_cols"])

# -----------------------------
# 分层特征组（方案A：只到 derived_core，不导出 delta_*）
# -----------------------------
base_real = [
    "ms_avg_cpu_mean", "ms_avg_mem_mean", "ms_max_cpu_max", "ms_max_mem_max",
    "ms_cpi_mean", "ms_mpi_mean", "ms_cpi_missing", "ms_mpi_missing",
    "record_count",
]
extended_real = base_real + [
    "ms_req_cpu_mean", "ms_req_mem_mean",
    "ms_assigned_memory_mean", "ms_page_cache_memory_mean",
    "ms_cpi_coverage", "ms_mpi_coverage",
    "cpu_gap", "mem_gap",
]

# CPU 分布向量（11+9）+ missing 指示（2）
cpu_dist_cols = [f"cpu_dist_{k}" for k in range(11)]
tail_dist_cols = [f"tail_cpu_dist_{k}" for k in range(9)]
dist_ind_cols = ["cpu_dist_missing", "tail_cpu_dist_missing"]

# sched/priority（如果你在 aux 脚本里启用了 ADD_SCHED_PRIORITY）
# scheduling_class 比例向量 + priority 三统计

# 只导出3列，去掉一个参考类（例如 sched_frac_3）以避免完美线性相关
sched_cols = [f"sched_frac_{k}" for k in [0, 1, 2]]
prio_cols = ["prio_min", "prio_max", "prio_mean"]
extended_real = extended_real + cpu_dist_cols + tail_dist_cols + dist_ind_cols + sched_cols + prio_cols
event_set = extended_real + [
    "evict_count", "lost_count",
    "abnormal_nonfail_count", "abnormal_nonfail_flag",
]
derived_core = event_set + [
    "gap_since_prev", "delta_valid_mask",
]

feature_sets = {
    "base_real": base_real,
    "extended_real": extended_real,
    "workload": event_set,
    "derived": derived_core,
}
# ========== 新增：特征集尺寸断言（防止后续代码误改导致维度漂移）==========
assert len(base_real) == 9, f"base_real length changed: {len(base_real)}"
assert len(extended_real) == 45, f"extended_real length changed: {len(extended_real)}"
assert len(event_set) == 49, f"event_set length changed: {len(event_set)}"
assert len(derived_core) == 51, f"derived_core length changed: {len(derived_core)}"

# ====================================================================
# 不允许静默丢列：每个 set 的列必须同时存在于 features_raw 且存在于 params.feature_cols
for set_name, cols in feature_sets.items():
    miss_feat = [c for c in cols if c not in feat.columns]
    miss_param = [c for c in cols if c not in all_feature_cols]
    if miss_feat:
        raise AssertionError(f"❌ feature set {set_name} 缺少列(不在 features_raw): {miss_feat}")
    if miss_param:
        raise AssertionError(f"❌ feature set {set_name} 列未出现在 params.feature_cols: {miss_param}")

print("[特征组列数]")
for k, cols in feature_sets.items():
    print(f"  {k}: {len(cols)}")

export_summary = []

for H, lab_path in LABEL_FILES.items():
    label_col = f"label_fail_H{H}h"
    print("\n" + "-" * 90)
    print(f"[H={H}h] 读取标签文件: {lab_path}")

    lab = pd.read_csv(lab_path, usecols=[MID, TS, label_col], low_memory=False)
    df = feat.merge(lab, on=[MID, TS], how="left", validate="one_to_one")

    # 仅保留 label-valid
    df = df[df[label_col].notna()].copy()
    df["label"] = df[label_col].astype(int)
    df.drop(columns=[label_col], inplace=True)

    # split 导出
    for split_name in ["train", "val", "test"]:
        sub = df[df[SPLIT] == split_name].copy()
        y = sub["label"].to_numpy(dtype=np.int64)

        # meta：保留对齐信息，但不要以 machine_id/time_slice 名字出现，降低误入模风险
        meta = sub[[MID, TS]].rename(columns={MID: "meta_machine_id", TS: "meta_time_slice"}).reset_index(drop=True)

        for set_name, cols in feature_sets.items():
            X = transform_df(sub, params, cols)

            out = pd.concat(
                [meta, X.reset_index(drop=True), pd.Series(y, name="label")],
                axis=1
            )

            out_path = os.path.join(OUT_DIR, f"step3c_model_input_{set_name}_H{H}h_{split_name}_v2.csv")
            out.to_csv(out_path, index=False, encoding="utf-8-sig")

            # 快速 QC：打印主任务 H=8 的 train 两个集合
            if split_name == "train" and H == 8 and set_name in ["base_real", "derived"]:
                quick_nan_inf_report(out, f"{set_name}_H{H}h_{split_name}")

        # summary 默认只记录 train/val；test 的统计属于 after-freeze 阶段
        if (split_name in ["train", "val"]) or INCLUDE_TEST_IN_SUMMARY:
            export_summary.append({
                "H_hours": H,
                "split": split_name,
                "rows": int(len(sub)),
                "pos": int(y.sum()) if len(y) else 0,
                "pos_rate": float(y.mean()) if len(y) else np.nan,
            })

    print(f"[完成] H={H}h 导出完成（train/val/test, sets={list(feature_sets.keys())})")
INCLUDE_TEST_IN_SUMMARY = False  # 默认 False：summary/print 不包含 test（避免证据链污染）

SUM_PATH_TRAINVAL = os.path.join(REPORT_DIR, "step3c_export_summary_multiH_trainval_v2.csv")
SUM_PATH_FULL = os.path.join(REPORT_DIR, "step3c_export_summary_multiH_full_after_freeze_v2.csv")


summary_df = pd.DataFrame(export_summary)

# 永远落盘 train/val-only 版本（证据链）
summary_df.to_csv(SUM_PATH_TRAINVAL, index=False, encoding="utf-8-sig")

# 若需要 full（含 test），才额外落盘（after-freeze）
if INCLUDE_TEST_IN_SUMMARY:
    summary_df.to_csv(SUM_PATH_FULL, index=False, encoding="utf-8-sig")

print("\n" + "=" * 90)
print("Step 3C(v2) 完成")
print("=" * 90)
print("[输出目录]", OUT_DIR)
print("[汇总报告(train/val only)]", SUM_PATH_TRAINVAL)
if INCLUDE_TEST_IN_SUMMARY:
    print("[汇总报告(full after-freeze)]", SUM_PATH_FULL)

print(summary_df.groupby(["H_hours", "split"])[["rows", "pos", "pos_rate"]].first().to_string())
print("=" * 90)
with open(os.path.join(REPORT_DIR, "step3c_export_summary_note_v2.txt"), "w", encoding="utf-8") as f:
    f.write(f"INCLUDE_TEST_IN_SUMMARY={INCLUDE_TEST_IN_SUMMARY}\n")
    f.write(f"params_file={IN_PARAMS}\n")
    f.write(f"features_file={IN_FEATURES}\n")