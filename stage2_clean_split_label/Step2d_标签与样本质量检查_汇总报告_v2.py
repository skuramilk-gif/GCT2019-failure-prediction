# ============================================================
# 文件名：Step2d_标签与样本质量检查_汇总报告_v2.py  (严格版：默认不输出test + NaN原因分解)
# 作用：
#   1) 对 labeled 表进行 QC 汇总
#   2) 强 sanity check（必须通过）：
#      - failed_count>0 的行 label 必须为 NaN
#      - 方案A：t >= t_first_fail 的行 label 必须为 NaN
#      - 方案A：仅 no-fail 机器尾部 (t+H>tmax) 的行 label 必须为 NaN
#      - pre-first 行不应出现 NaN
#      - label 只能取 {0,1,NaN}
# 3) 输出 NaN 原因分解（审稿必问），一个样本可能因为多种样本被设为NaN，如1因为“首次 FAIL 之后”而剔除（post_first_fail）→ 这是任务定义要求的，合理（nan_post_first_fail）。
# 2因为“从未 FAIL 且右删失”而剔除（no_fail_censored）→ 也是合理的。3如果还有其他原因（比如解析错误、时间异常等），那就说明数据处理有 bug（nan_other）。
# 输出：
#   05_reports/step2d_label_qc_summary_trainval_v2.csv   (默认：不含test)
#   05_reports/step2d_label_qc_report_trainval_v2.txt
#   （可选）若 INCLUDE_TEST=True：
#       05_reports/step2d_label_qc_summary_full_v2.csv  (含test)
#       05_reports/step2d_label_qc_report_full_v2.txt
# ============================================================

import os
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

INCLUDE_TEST = False   # ★默认 False：不计算/不输出 test；最终一次性评测后可手动改 True 再跑


FILES = [
    (4,  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H4h_v2.csv")),
    (6,  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H6h_v2.csv")),
    (8,  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv")),
    (12, os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H12h_v2.csv")),
]

OUT_SUM_TRAINVAL = os.path.join(REPORT_DIR, "step2d_label_qc_summary_trainval_v2.csv")
OUT_TXT_TRAINVAL = os.path.join(REPORT_DIR, "step2d_label_qc_report_trainval_v2.txt")

OUT_SUM_FULL = os.path.join(REPORT_DIR, "step2d_label_qc_summary_full_v2.csv")
OUT_TXT_FULL = os.path.join(REPORT_DIR, "step2d_label_qc_report_full_v2.txt")

MID = "machine_id"
T = "slice_last_time_us"
FAIL_CNT = "failed_count"
SPLIT = "split"

MICRO_PER_HOUR = 60 * 60 * 1_000_000

def split_stats(df, label_col, split_name):
    sub = df[df[SPLIT] == split_name]
    sub_valid = sub[sub[label_col].notna()]
    if len(sub) == 0:
        return dict(rows=0, valid_rows=0, valid_rate=np.nan, pos_count=0, pos_rate=np.nan)
    if len(sub_valid) == 0:
        return dict(rows=int(len(sub)), valid_rows=0, valid_rate=0.0, pos_count=0, pos_rate=0.0)

    y = pd.to_numeric(sub_valid[label_col], errors="coerce")
    return dict(
        rows=int(len(sub)),
        valid_rows=int(len(sub_valid)),
        valid_rate=float(len(sub_valid) / len(sub)),
        pos_count=int((y == 1).sum()),
        pos_rate=float((y == 1).mean())
    )

all_summary = []
lines = []
lines.append("=== Step2D(v2) Label QC Report (scheme-A: first-fail only) ===")
lines.append(f"INCLUDE_TEST={INCLUDE_TEST}")
lines.append("NOTE: By default, test split statistics are excluded to avoid any test-set info entering the evidence chain.")
lines.append("")

for H, path in FILES:
    assert os.path.exists(path), f"❌ labeled file missing: {path}"
    label_col = f"label_fail_H{H}h"
    df = pd.read_csv(path, low_memory=False)

    need_cols = [MID, T, FAIL_CNT, SPLIT, label_col]
    for c in need_cols:
        assert c in df.columns, f"❌ {path} 缺少列 {c}"

    df[MID] = pd.to_numeric(df[MID], errors="coerce").astype("int64")
    df[T] = pd.to_numeric(df[T], errors="coerce").astype("int64")
    df[FAIL_CNT] = pd.to_numeric(df[FAIL_CNT], errors="coerce").fillna(0).astype("int64")
    df[SPLIT] = df[SPLIT].astype(str)

    # label value check
    lv = df[label_col].dropna().unique().tolist()
    bad = [x for x in lv if x not in [0, 1, 0.0, 1.0]]
    assert len(bad) == 0, f"❌ {label_col} 出现非法取值 {bad}"

    H_us = int(H * MICRO_PER_HOUR)
    t = df[T]
    tmax = df.groupby(MID)[T].transform("max").astype("int64")

    # 每台机器首次 FAIL 时刻；无 FAIL -> NaN
    first_fail_map = df.loc[df[FAIL_CNT] > 0].groupby(MID)[T].min()
    tfail = df[MID].map(first_fail_map)  # float with NaN

    has_fail_machine = tfail.notna()
    no_fail_machine = ~has_fail_machine

    fail_slice = df[FAIL_CNT] > 0
    post_first = has_fail_machine & (t >= tfail)          # 方案A：含 fail 当刻 + post-fail
    pre_first  = has_fail_machine & (t < tfail)
    censored   = no_fail_machine & (t + H_us > tmax)      # 方案A：仅 no-fail 机器尾部

    # ---- Gate checks ----
    assert df.loc[fail_slice, label_col].notna().sum() == 0, f"❌ {label_col}: failed_count>0 未全部 NaN"
    assert df.loc[post_first, label_col].notna().sum() == 0, f"❌ {label_col}: post-first 未全部 NaN"
    assert df.loc[censored, label_col].notna().sum() == 0, f"❌ {label_col}: no-fail censored 未全部 NaN"
    assert df.loc[pre_first, label_col].isna().sum() == 0, f"❌ {label_col}: pre-first 出现 NaN"
    # 正向验证：label=1 的行，tfail - t 必须在 (0, H_us] 内
    pos_rows = df.loc[df[label_col] == 1]
    if len(pos_rows) > 0:
        dt_check = (tfail.loc[pos_rows.index] - t.loc[pos_rows.index]).astype("float64")
        assert (dt_check > 0).all(), f"❌ {label_col}: 存在 label=1 但 dt_to_fail <= 0"
        assert (dt_check <= H_us).all(), f"❌ {label_col}: 存在 label=1 但 dt_to_fail > H_us"

    # 正向验证：label=0 的 pre-first 行，tfail - t 必须 > H_us
    neg_pre = df.loc[(df[label_col] == 0) & pre_first]
    if len(neg_pre) > 0:
        dt_neg = (tfail.loc[neg_pre.index] - t.loc[neg_pre.index]).astype("float64")
        assert (dt_neg > H_us).all(), f"❌ {label_col}: 存在 label=0 的 pre-first 行但 dt_to_fail <= H_us"

    # ---- NaN reason decomposition (审稿必问) ----
    nan_total = int(df[label_col].isna().sum())
    nan_post_first = int(post_first.sum())                 # 这些按定义应为 NaN
    nan_no_fail_censored = int(censored.sum())             # 这些按定义应为 NaN
    nan_other = nan_total - nan_post_first - nan_no_fail_censored
    assert nan_other == 0, f"❌ {label_col}: nan_other={nan_other} (说明存在未解释的NaN来源)"

    valid_mask = df[label_col].notna()
    valid_df = df[valid_mask].copy()
    y_valid = pd.to_numeric(valid_df[label_col], errors="coerce")

    overall = dict(
        H_hours=H,
        rows_total=int(len(df)),
        valid_rows=int(valid_mask.sum()),
        valid_rate=float(valid_mask.mean()),
        pos_count=int((y_valid == 1).sum()) if len(valid_df) else 0,
        pos_rate=float((y_valid == 1).mean()) if len(valid_df) else np.nan,

        nan_total=nan_total,
        nan_post_first_fail=nan_post_first,
        nan_no_fail_censored=nan_no_fail_censored,
        nan_other=nan_other,
        nan_total_rate=float(nan_total / len(df)) if len(df) else np.nan,
    )

    st_train = split_stats(df, label_col, "train")
    st_val = split_stats(df, label_col, "val")
    overall.update({
        "train_valid_rows": st_train["valid_rows"], "train_pos_count": st_train["pos_count"], "train_pos_rate": st_train["pos_rate"],
        "val_valid_rows": st_val["valid_rows"],     "val_pos_count": st_val["pos_count"],     "val_pos_rate": st_val["pos_rate"],
    })

    if INCLUDE_TEST:
        st_test = split_stats(df, label_col, "test")
        overall.update({
            "test_valid_rows": st_test["valid_rows"], "test_pos_count": st_test["pos_count"], "test_pos_rate": st_test["pos_rate"],
        })

    all_summary.append(overall)

    # ---- Text report (train/val only by default) ----
    lines.append("\n" + "-" * 90)
    lines.append(f"[H={H}h] file={path}")
    lines.append("overall=" + str({k: overall[k] for k in overall.keys() if not k.startswith("test_")}))
    lines.append(f"NaN decomposition: total={nan_total}, post_first={nan_post_first}, no_fail_censored={nan_no_fail_censored}, other={nan_other}")
    lines.append("split stats:")
    lines.append("  train=" + str(st_train))
    lines.append("  val  =" + str(st_val))
    if INCLUDE_TEST:
        lines.append("  test =" + str(st_test))

# ---- Save outputs ----
sum_df = pd.DataFrame(all_summary)

# train/val-only artifacts (default)
trainval_cols = [c for c in sum_df.columns if not c.startswith("test_")]
sum_df[trainval_cols].to_csv(OUT_SUM_TRAINVAL, index=False, encoding="utf-8-sig")
with open(OUT_TXT_TRAINVAL, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# full artifacts (optional)
if INCLUDE_TEST:
    sum_df.to_csv(OUT_SUM_FULL, index=False, encoding="utf-8-sig")
    with open(OUT_TXT_FULL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("=" * 90)
print("Step2D(v2) 完成：QC汇总已保存")
print("=" * 90)
print("[输出]", OUT_SUM_TRAINVAL)
print("[输出]", OUT_TXT_TRAINVAL)
if INCLUDE_TEST:
    print("[输出]", OUT_SUM_FULL)
    print("[输出]", OUT_TXT_FULL)

preview_cols = ["H_hours", "train_pos_count", "val_pos_count", "train_pos_rate", "val_pos_rate",
                "valid_rate", "nan_total_rate", "nan_post_first_fail", "nan_no_fail_censored", "nan_other"]
print("\n[预览] summary (train/val only):")
print(sum_df[preview_cols].to_string(index=False))