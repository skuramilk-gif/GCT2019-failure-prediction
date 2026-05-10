# ============================================================
# 文件名：Step2c_预警标签生成_FAILonly_多窗口_v2.py
# 阶段：Step 2C（v2，正式）
# 作用：
#   1) 在 machine-slice clean 表上生成 FAIL-only 预警标签（多窗口）
#   2) 丢弃故障当刻 slice（failed_count>0 -> label=NaN）
#   3) 右删失处理：t+H 超过该机器最后观测时刻 -> label=NaN
#   4) 输出 H=4h/6h/8/12h 三份 labeled 表 + 每份统计报告 + 汇总表
#
# 输入：
#   01_interim/step2a_machine_slice_clean_v2.csv
#   03_splits/step2b_train_machines_v2.txt
#   03_splits/step2b_val_machines_v2.txt
#   03_splits/step2b_test_machines_v2.txt
#
# 输出：
#   01_interim/step2c_machine_slice_labeled_FAILonly_H{H}h_v2.csv
#   05_reports/step2c_label_stats_FAILonly_H{H}h_v2.txt
#   05_reports/step2c_label_summary_FAILonly_multiH_v2.csv
#论文约束：必须在实验设置或者附录放出horizon sweep 表，解释为什么主 H=6h、短窗 4h、稳健 12h。否则审稿人会说你拍脑袋选窗口。
#K步扫描只是作为一个探索脚本它不适合进入论文主线。
#论文强调：由于观察时间稀少（中位数间隔~110小时），只有~5.6%的故障机器在故障前8小时内至少有一次观测。因此，模型的回忆上界受观测密度限制，而非模型容量。
#该强调是堵住审稿人对1.8%正例率会直接质疑任务的可行性
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# 0) 路径与参数
# ============================================================
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

HOURS_LIST = [4, 6, 8, 12]  # 你已确认：短窗4h，主任务8h，稳健性12h
MICRO_PER_HOUR = 60 * 60 * 1_000_000

MID = "machine_id"
T = "slice_last_time_us"
FAIL_CNT = "failed_count"
TS="time_slice"

OUT_SUMMARY = os.path.join(REPORT_DIR, "step2c_label_summary_FAILonly_multiH_v2.csv")

# ============================================================
# 1) 工具函数
# ============================================================
def read_machine_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(int(line.strip()) for line in f if line.strip())

def write_txt(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def stats_for_split(df_labeled, label_col, split_name):
    sub = df_labeled[df_labeled["split"] == split_name]
    sub_valid = sub[sub[label_col].notna()]
    if len(sub) == 0:
        return {"rows": 0, "valid_rows": 0, "valid_rate": np.nan, "pos_count": 0, "pos_rate": np.nan}
    return {
        "rows": int(len(sub)),
        "valid_rows": int(len(sub_valid)),
        "valid_rate": float(len(sub_valid) / len(sub)),
        "pos_count": int(sub_valid[label_col].sum()) if len(sub_valid) else 0,
        "pos_rate": float(sub_valid[label_col].mean()) if len(sub_valid) else 0.0,
    }

# ============================================================
# 2) 读取 clean + split 标注
# ============================================================
print("=" * 90)
print("Step 2C(v2)：FAIL-only 多窗口标签生成（4h/6h/12h）")
print("=" * 90)
print(f"[输入] {IN_CLEAN}")

df = pd.read_csv(IN_CLEAN, low_memory=False)

assert MID in df.columns and T in df.columns and FAIL_CNT in df.columns, "❌ clean表缺少关键列"
df[MID] = pd.to_numeric(df[MID], errors="coerce").astype("int64")
df[T] = pd.to_numeric(df[T], errors="coerce")
df[FAIL_CNT] = pd.to_numeric(df[FAIL_CNT], errors="coerce").fillna(0).astype("int64")

train_m = read_machine_list(IN_TRAIN)
val_m   = read_machine_list(IN_VAL)
test_m  = read_machine_list(IN_TEST)

df["split"] = "unknown"
df.loc[df[MID].isin(train_m), "split"] = "train"
df.loc[df[MID].isin(val_m), "split"] = "val"
df.loc[df[MID].isin(test_m), "split"] = "test"

unknown_cnt = int((df["split"] == "unknown").sum())
assert unknown_cnt == 0, f"❌ 存在 split==unknown 的行数={unknown_cnt}"

# 排序
df = df.sort_values([MID, T]).reset_index(drop=True)

# 主键列必须存在：time_slice 缺失属于上游 clean 表结构错误，不能静默跳过
assert TS in df.columns, f"❌ clean表缺少列 {TS}，无法做主键重复检查"
dup = df.duplicated([MID, TS]).sum()
assert dup == 0, f"❌ clean表存在重复主键 (machine_id,{TS}) 重复数={dup}"
assert (df[T].notna().all() and (df[T] > 0).all()), "❌ slice_last_time_us 存在 NaN 或 非正数"

print(f"[数据] rows={len(df)}, machines={df[MID].nunique()}")
print(f"[数据] train/val/test rows = "
      f"{(df['split']=='train').sum()}/"
      f"{(df['split']=='val').sum()}/"
      f"{(df['split']=='test').sum()}")

# ============================================================
# ============================================================
# 3) 计算每台机器的 tmax 与 首次 FAIL 时刻 t_first_fail_us（方案A核心）
# ============================================================
# 每台机器最大观测时刻
tmax_map = df.groupby(MID)[T].max()

# 每台机器首次 FAIL 时刻（没有 FAIL 的机器 -> NaN）
first_fail_map = df.loc[df[FAIL_CNT] > 0].groupby(MID)[T].min()

df["machine_tmax_us"] = df[MID].map(tmax_map).astype("int64")
df["t_first_fail_us"] = df[MID].map(first_fail_map)   # float (NaN allowed)

# 诊断：有 FAIL 的机器比例
print(f"[诊断] machines_with_fail={df.loc[df['t_first_fail_us'].notna(), MID].nunique()} / {df[MID].nunique()}")

# ============================================================
# 4) 对每个 H 生成 label 并输出文件 + 报告
# ============================================================
summary_rows = []

# 计算不依赖 H 的公共变量（移到循环外）
t = df[T].astype("int64")
tmax = df["machine_tmax_us"].astype("int64")
tfail = df["t_first_fail_us"]
has_fail_machine = tfail.notna()
pre_first = has_fail_machine & (t < tfail)
no_fail_machine = ~has_fail_machine

for H in HOURS_LIST:
    H_us = int(H * MICRO_PER_HOUR)
    label_col = f"label_fail_H{H}h"

    y = pd.Series(0.0, index=df.index, dtype="float32")

    # A) 有首次 FAIL：仅保留首次 FAIL 之前的 slice
    dt_to_first = (tfail - t).astype("float64")
    # 标签逻辑正确性论证：
    # 对于有 FAIL 的机器，first_fail = min(slice_last_time_us where failed_count>0)
    # 若 slice 时间 t 满足 tfail - t <= H，则首个 FAIL 在 H 小时内 → label=1
    # 若 tfail - t > H，则首个 FAIL 在 H 小时外，且所有后续 FAIL 更远 → label=0
    # 因此 "首次 FAIL 在 H 内" 等价于 "至少一次 FAIL 在 H 内"
    y.loc[pre_first] = ((dt_to_first.loc[pre_first] > 0) & (dt_to_first.loc[pre_first] <= H_us)).astype("float32")
    y.loc[has_fail_machine & (~pre_first)] = np.nan   # 含 fail 当刻 + post-fail

    # B) 无 FAIL：窗口不完整 => 右删失 NaN；完整 => 0
    censored = no_fail_machine & (t + H_us > tmax)
    y.loc[censored] = np.nan

    df_out = df.copy()
    df_out[label_col] = y

    # 输出 CSV（请保留你原脚本中的 keep_cols 列表，不要变化）增加了"ms_cpi_coverage", "ms_mpi_coverage",
    keep_cols = [
        MID, "time_slice", T, "record_count",
        FAIL_CNT, "evict_count", "lost_count", "kill_count",
        "ms_req_cpu_mean", "ms_req_mem_mean",
        "ms_avg_cpu_mean", "ms_avg_mem_mean",
        "ms_max_cpu_max", "ms_max_mem_max",
        "ms_assigned_memory_mean", "ms_page_cache_memory_mean",
        "ms_cpi_mean", "ms_mpi_mean",
        "ms_cpi_missing", "ms_mpi_missing",
        "ms_cpi_coverage", "ms_mpi_coverage",
        "abnormal_nonfail_count", "abnormal_nonfail_flag",
        "split", label_col
    ]

    keep_cols = [c for c in keep_cols if c in df_out.columns]

    out_csv = os.path.join(INTERIM_DIR, f"step2c_machine_slice_labeled_FAILonly_H{H}h_v2.csv")
    # 注意：输出 CSV 包含 train/val/test 全部行的标签。
    # test 标签在此生成是为了最终评估，但 Step2c_aux 的 horizon scan
    # 严格只读 train/val，不窥 test 统计。

    df_out[keep_cols].to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ----- 统计报告：只计算 train 和 val，test 不参与 -----
    st_train = stats_for_split(df_out, label_col, "train")
    st_val   = stats_for_split(df_out, label_col, "val")
    # test 统计全部填 NaN（避免任何泄漏嫌疑）
    st_test  = {"valid_rows": np.nan, "pos_count": np.nan, "pos_rate": np.nan}

    valid_rate_all = float(df_out[label_col].notna().mean())
    pos_rate_all = float(df_out.loc[df_out[label_col].notna(), label_col].mean()) if df_out[label_col].notna().any() else np.nan
    pos_count_all = int(df_out.loc[df_out[label_col].notna(), label_col].sum()) if df_out[label_col].notna().any() else 0

    # time-to-fail 分布（正例样本）
    pos_mask = (df_out[label_col] == 1)
    if pos_mask.any():
        pos_dt = dt_to_first.loc[pos_mask].dropna()
        pos_dt_hours = pos_dt / MICRO_PER_HOUR
    else:
        pos_dt_hours = pd.Series(dtype=float)

    if len(pos_dt_hours) > 0:
        q = pos_dt_hours.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
        # 诊断：正例的 time-to-fail 应全部在 (0, H] 小时内
        assert pos_dt_hours.min() > 0, f"❌ H={H}h: 存在 time_to_fail <= 0 的正例"
        assert pos_dt_hours.max() <= H + 0.001, f"❌ H={H}h: 存在 time_to_fail > H 的正例"

    else:
        q = {}

    # 生成文本报告（报告中不出现 test 统计）
    report_lines = []
    report_lines.append(f"=== Step2C(v2) Label Stats: FAIL-only, H={H}h ===")
    report_lines.append(f"rows_total={len(df_out)}")
    report_lines.append(f"valid_rate_all={valid_rate_all:.6f}")
    report_lines.append(f"pos_count_all(valid)={pos_count_all}")
    report_lines.append(f"pos_rate_all(valid)={pos_rate_all:.6f}")
    report_lines.append("")
    report_lines.append("[Split stats]")
    report_lines.append(f"train: {st_train}")
    report_lines.append(f"val  : {st_val}")
    # 不列出 test 行，避免任何信息泄露
    report_lines.append("")
    report_lines.append("[Pos time-to-next-fail (hours) quantiles]")
    if q:
        for k, v in q.items():
            report_lines.append(f"q{k}: {v:.4f} h")
    else:
        report_lines.append("no positive samples")
    report_lines.append("")
    report_lines.append("[Preview head(10)]")
    report_lines.append(df_out[[MID, "time_slice", T, FAIL_CNT, "split", label_col]].head(10).to_string(index=False))

    out_txt = os.path.join(REPORT_DIR, f"step2c_label_stats_FAILonly_H{H}h_v2.txt")
    write_txt(out_txt, report_lines)

    # 汇总行：test 相关字段全部填 NaN
    summary_rows.append({
        "H_hours": H,
        "all_valid_rate": valid_rate_all,
        "all_pos_count": pos_count_all,
        "all_pos_rate": pos_rate_all,
        "train_valid_rows": st_train["valid_rows"],
        "train_pos_count": st_train["pos_count"],
        "train_pos_rate": st_train["pos_rate"],
        "val_valid_rows": st_val["valid_rows"],
        "val_pos_count": st_val["pos_count"],
        "val_pos_rate": st_val["pos_rate"],
        "out_csv": out_csv,
        "out_report": out_txt,
    })



    print("\n" + "-" * 90)
    print(f"H={H}h  输出: {out_csv}")
    print(f"valid_rate_all={valid_rate_all:.4f}  pos_rate_all(valid)={pos_rate_all:.6f}  pos_count_all={pos_count_all}")
    print("train/val pos_count:", st_train["pos_count"], st_val["pos_count"])

# 保存汇总表
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

print("\n" + "=" * 90)
print("Step 2C(v2) 多窗口标签生成完成")
print("=" * 90)
print(f"[汇总] {OUT_SUMMARY}")
# 打印汇总表前几行（避免重复打印全部）
print(summary_df[["H_hours","train_pos_count","val_pos_count","train_pos_rate","val_pos_rate",]].to_string(index=False))
print("=" * 90)