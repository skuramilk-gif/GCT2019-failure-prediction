# ============================================================
# 文件名：Step2f_post_fail_observation_report_v2.py
# 作用：
#   统计发生 FAIL 的机器中，有多少机器在“首次 FAIL 之后”仍然有后续 slice 记录。
#   post_fail_rate_all==机器在首次FALL之后还有一个后续slice。
#   n_slices_after_first_fail quantiles={0.5: 1.0, 0.9: 3.0, 0.99: 4.0}，P50的机器1个slice（中位数）后面以此类推
# 背景：
#   这用于解释观测机制：如果大多数机器 FAIL 后仍继续上报，则我们的右删失（仅对从未 FAIL 的机器做尾部删失）是合理的
# 输入：
#   01_interim/step2a_machine_slice_clean_v2.csv
#   03_splits/step2b_{train,val,test}_machines_v2.txt
# 输出：
#   05_reports/step2f_post_fail_observation_machinelevel_v2.csv
#   05_reports/step2f_post_fail_observation_summary_v2.txt
# ============================================================

import os
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
IN_CLEAN = os.path.join(ROOT_DIR, "01_interim", "step2a_machine_slice_clean_v2.csv")
SPLIT_DIR = os.path.join(ROOT_DIR, "03_splits")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

IN_TRAIN = os.path.join(SPLIT_DIR, "step2b_train_machines_v2.txt")
IN_VAL   = os.path.join(SPLIT_DIR, "step2b_val_machines_v2.txt")
IN_TEST  = os.path.join(SPLIT_DIR, "step2b_test_machines_v2.txt")

OUT_MACH = os.path.join(REPORT_DIR, "step2f_post_fail_observation_machinelevel_v2.csv")
OUT_TXT  = os.path.join(REPORT_DIR, "step2f_post_fail_observation_summary_v2.txt")

# ========== 开关：是否在报告中包含 test 统计（默认不包含，避免设定阶段泄漏）==========
INCLUDE_TEST = False          # 设为 True 时才会输出 test 信息

MID = "machine_id"
T = "slice_last_time_us"
FAIL_CNT = "failed_count"


def read_list(p):
    with open(p, "r", encoding="utf-8") as f:
        return set(int(x.strip()) for x in f if x.strip())

print("=" * 90)
print("Step2F(v2)：FAIL 后是否仍有记录（post-fail observation）统计")
print("=" * 90)
print("[输入]", IN_CLEAN)

df = pd.read_csv(IN_CLEAN, usecols=[MID, T, FAIL_CNT], low_memory=False)
df[MID] = pd.to_numeric(df[MID], errors="coerce").astype("int64")
df[T] = pd.to_numeric(df[T], errors="coerce").astype("int64")
df[FAIL_CNT] = pd.to_numeric(df[FAIL_CNT], errors="coerce").fillna(0).astype("int64")

# split 标注（机器级）
train_m = read_list(IN_TRAIN)
val_m   = read_list(IN_VAL)
test_m  = read_list(IN_TEST)

def split_of(mid: int) -> str:
    if mid in train_m: return "train"
    if mid in val_m:   return "val"
    if mid in test_m:  return "test"
    return "unknown"

# 每台机器的 tmax
tmax = df.groupby(MID)[T].max().rename("tmax_us")

# 每台机器的首次 FAIL 时间（若无 FAIL 则为 NaN）
fail_df = df[df[FAIL_CNT] > 0].copy()
first_fail = fail_df.groupby(MID)[T].min().rename("first_fail_us")
n_fail_slices = fail_df.groupby(MID).size().rename("n_fail_slices")

# 合并成机器级表
mach = pd.concat([tmax, first_fail, n_fail_slices], axis=1).reset_index()
mach["has_fail"] = mach["first_fail_us"].notna().astype(int)

# 只看发生 FAIL 的机器
mach_fail = mach[mach["has_fail"] == 1].copy()
mach_fail["split"] = mach_fail[MID].apply(split_of)
# 断言：没有 unknown split 的机器
assert (mach_fail["split"] != "unknown").all(), "❌ 存在 split unknown 的机器"

# 统计 FAIL 后仍有记录：tmax > first_fail
mach_fail["has_post_fail_record"] = (mach_fail["tmax_us"] > mach_fail["first_fail_us"]).astype(int)

# FAIL 后还有多少 slice（严格大于 first_fail 的 slice 数）
# 注：这一步需要回到 slice 表按机器计数，成本可控（341k 行）
df2 = df[[MID, T]].copy()
# merge first_fail 到每行
df2 = df2.merge(first_fail.reset_index(), on=MID, how="left", validate="many_to_one")
df2 = df2[df2["first_fail_us"].notna()]
df2["is_after_first_fail"] = (df2[T] > df2["first_fail_us"]).astype(int)
n_after = df2.groupby(MID)["is_after_first_fail"].sum().rename("n_slices_after_first_fail")

mach_fail = mach_fail.merge(n_after.reset_index(), on=MID, how="left", validate="one_to_one")
mach_fail["n_slices_after_first_fail"] = mach_fail["n_slices_after_first_fail"].fillna(0).astype(int)

mach_fail.to_csv(OUT_MACH, index=False, encoding="utf-8-sig")

# 汇总
lines = []
lines.append("=== Step2F(v2) Post-fail observation summary ===")
lines.append(f"fail_machines_total={len(mach_fail)}")
lines.append(f"post_fail_rate_all={mach_fail['has_post_fail_record'].mean():.6f}")
lines.append(f"n_slices_after_first_fail quantiles="
             f"{mach_fail['n_slices_after_first_fail'].quantile([0.5,0.9,0.99]).to_dict()}")

# 选择要输出的 splits 不输出test集。
splits = ["train", "val"] if not INCLUDE_TEST else ["train", "val", "test"]
for sp in splits:
    sub = mach_fail[mach_fail["split"] == sp]
    if len(sub) == 0:
        continue
    lines.append("")
    lines.append(f"[{sp}] fail_machines={len(sub)} post_fail_rate={sub['has_post_fail_record'].mean():.6f} "
                 f"after_fail_slices_q={sub['n_slices_after_first_fail'].quantile([0.5,0.9,0.99]).to_dict()}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n[输出]", OUT_MACH)
print("[输出]", OUT_TXT)
print("\n".join(lines))