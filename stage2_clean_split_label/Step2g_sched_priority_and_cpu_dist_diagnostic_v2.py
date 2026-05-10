# ============================================================
# 文件名：Step2g_sched_priority_and_cpu_dist_diagnostic_v2.py
# 作用：（这是特征选择诊断脚本，就对不能看test）
#   A) 诊断：scheduling_class / priority 是否与 FAIL 风险显著相关（slice级别，连接 H=8 label-valid）
#      - 不修改现有 clean/features/export，仅输出报告，用于决定是否纳入特征与是否重跑管线
#   B) 统计：cpu_usage_distribution / tail_cpu_usage_distribution 的 raw 缺失率与解析失败率
# ============================================================
#Length 守恒： both_ok + cpu_only + tail_only + both_fail = cpu_total，且断言通过，输出一致性检查通过
# ============================================================
# 输入：
#   00_raw/borg_traces_data.csv
#   01_interim/step2c_machine_slice_labeled_FAILonly_H8h_v2.csv   (用于 label-valid 样本域)
#
# 输出（05_reports）：
#   step2g_sched_priority_slice_agg_v2.csv          # slice级聚合后的 sched/priority（可选，便于复核）
#   step2g_sched_priority_assoc_summary_v2.csv      # 关联统计表（pos_rate by group/bin）
#   step2g_cpu_dist_missing_parse_summary_v2.csv    # cpu分布列缺失/解析失败统计
#   step2g_diagnostic_report_v2.txt                 # 总结文字
# ============================================================

import os
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

IN_LAB8 = os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv")

OUT_AGG = os.path.join(REPORT_DIR, "step2g_sched_priority_slice_agg_v2.csv")
OUT_SUM = os.path.join(REPORT_DIR, "step2g_sched_priority_assoc_summary_v2.csv")
OUT_CPU = os.path.join(REPORT_DIR, "step2g_cpu_dist_missing_parse_summary_v2.csv")
OUT_TXT = os.path.join(REPORT_DIR, "step2g_diagnostic_report_v2.txt")

MID="machine_id"
TIME="time"
TS="time_slice"
SPLIT="split"
LABEL="label_fail_H8h"

SLICE_US = 300_000_000
INT64_MAX = 9223372036854775807
CHUNKSIZE = 200_000

print("=" * 90)
print("Step2G(v2)：sched/priority 关联诊断 + CPU分布列缺失/解析统计")
print("=" * 90)
print("[输入 raw]", RAW_PATH)
print("[输入 labeled H8]", IN_LAB8)

# ------------------------------------------------------------
# 1) 读取 H=8 label-valid 样本域（只在此域上做关联统计）
# ------------------------------------------------------------
lab = pd.read_csv(IN_LAB8, usecols=[MID, TS, SPLIT, LABEL], low_memory=False)
lab[MID] = pd.to_numeric(lab[MID], errors="coerce").astype("int64")
lab[TS]  = pd.to_numeric(lab[TS], errors="coerce").astype("int64")
lab[LABEL] = pd.to_numeric(lab[LABEL], errors="coerce")

labv = lab[lab[LABEL].notna()].copy()
labv["label"] = labv[LABEL].astype(int)
labv.drop(columns=[LABEL], inplace=True)

# ========== 特征选择诊断只使用 train/val，禁止 test ==========
DIAG_SPLITS = ["train", "val"]
labv = labv[labv[SPLIT].isin(DIAG_SPLITS)]

keys_n = len(labv)
print(f"[label-valid H=8] rows={keys_n} (train/val/test="
      f"{(labv[SPLIT]=='train').sum()}/"
      f"{(labv[SPLIT]=='val').sum()}/"
      f"{(labv[SPLIT]=='test').sum()})")

# ------------------------------------------------------------
# 2) 扫 raw：聚合 scheduling_class/priority 到 slice 级（与 Step2a 的 time_slice 定义一致）
#    同时统计 cpu_distribution 两列的缺失/解析失败
# ------------------------------------------------------------

usecols = [MID, TIME, "scheduling_class", "priority",
           "cpu_usage_distribution", "tail_cpu_usage_distribution"]
# 用于 CPU 分布列统计
cpu_total = 0
cpu_na = 0
cpu_parse_fail = 0
tail_total = 0
tail_na = 0
tail_parse_fail = 0
#新增：全量累加变量（保证进行全局统计chunk）
cpu_len11_total = 0
tail_len9_total = 0

import re

def parse_np_array_string(val):
    """
    解析形如：
      "[0.00314331 0.00381088 ... 0.01194763]"
    或带逗号/多空格的变体。
    返回：np.ndarray 或 None
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if inner == "":
        return None
    # 将逗号与多空格统一成单空格，适配不同写法
    inner = re.sub(r"[\s,]+", " ", inner)
    arr = np.fromstring(inner, sep=" ")
    if arr.size == 0:
        return None
    return arr

def can_parse_and_len(val):
    arr = parse_np_array_string(val)
    if arr is None:
        return False, 0
    return True, int(arr.size)

parts = []
chunk_idx = 0
# ---- CPU/tail 解析一致性交叉检查（必须与 cpu_total 同口径）----
cross_total_valid_rows = 0
both_ok = 0
cpu_only = 0
tail_only = 0
both_fail = 0

# 把 fail 拆成两类：parse_fail vs len_mismatch
cpu_parse_fail_lenient = 0     # 完全无法 parse
tail_parse_fail_lenient = 0
cpu_len_mismatch = 0           # parse 成功但长度 != 11
tail_len_mismatch = 0          # parse 成功但长度 != 9

for chunk in pd.read_csv(RAW_PATH, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
    chunk_idx += 1
    # ---- 清洗 machine/time（与 Step2a 一致）----
    mid = pd.to_numeric(chunk[MID], errors="coerce")
    t = pd.to_numeric(chunk[TIME], errors="coerce")

    valid = mid.notna() & (mid >= 0) & t.notna() & (t > 0) & (t < INT64_MAX)
    chunk = chunk.loc[valid].copy()
    cross_total_valid_rows += len(chunk)

    cpu_col = chunk["cpu_usage_distribution"]
    tail_col = chunk["tail_cpu_usage_distribution"]

    # 对非空做解析：返回 (ok_parse, length)
    cpu_ok_len = cpu_col.apply(can_parse_and_len)
    tail_ok_len = tail_col.apply(can_parse_and_len)

    cpu_ok_parse = cpu_ok_len.apply(lambda x: x[0])
    cpu_len = cpu_ok_len.apply(lambda x: x[1])

    tail_ok_parse = tail_ok_len.apply(lambda x: x[0])
    tail_len = tail_ok_len.apply(lambda x: x[1])

    # “严格 OK”：既 parse 成功又长度正确
    cpu_ok11 = cpu_ok_parse & (cpu_len == 11)
    tail_ok9 = tail_ok_parse & (tail_len == 9)

    # 四象限计数（严格）
    both_ok += int((cpu_ok11 & tail_ok9).sum())
    cpu_only += int((cpu_ok11 & ~tail_ok9).sum())
    tail_only += int((~cpu_ok11 & tail_ok9).sum())
    both_fail += int((~cpu_ok11 & ~tail_ok9).sum())

    # 额外分解：lenient parse fail vs length mismatch（用于解释上面差异）
    cpu_parse_fail_lenient += int((~cpu_ok_parse).sum())
    tail_parse_fail_lenient += int((~tail_ok_parse).sum())
    cpu_len_mismatch += int((cpu_ok_parse & (cpu_len != 11)).sum())
    tail_len_mismatch += int((tail_ok_parse & (tail_len != 9)).sum())

    if len(chunk) == 0:
        continue

    chunk[MID] = pd.to_numeric(chunk[MID], errors="coerce").astype("int64")
    chunk["time_us"] = pd.to_numeric(chunk[TIME], errors="coerce").astype("int64")
    chunk[TS] = (chunk["time_us"] // SLICE_US).astype("int64")

    # ---- CPU 分布列缺失/解析统计（raw 级别）----
    # cpu_usage_distribution
    cpu_total += len(chunk)
    # 删除：cpu_len11 = 0 （改为累加外部变量）
    cpu_col = chunk["cpu_usage_distribution"]
    cpu_na += int(cpu_col.isna().sum())
    cpu_non_na = cpu_col.dropna()
    if len(cpu_non_na):
        ok_len = cpu_non_na.apply(lambda v: can_parse_and_len(v))
        ok = ok_len.apply(lambda x: x[0])
        lens = ok_len.apply(lambda x: x[1])
        cpu_parse_fail += int((~ok).sum())
        cpu_len11_total += int((lens == 11).sum())   # 改为累加外部变量

    tail_total += len(chunk)
    # 删除：tail_len9=0 （改为累加外部变量）
    tail_col = chunk["tail_cpu_usage_distribution"]
    tail_na += int(tail_col.isna().sum())
    tail_non_na = tail_col.dropna()
    if len(tail_non_na):
        ok_len = tail_non_na.apply(lambda v: can_parse_and_len(v))
        ok = ok_len.apply(lambda x: x[0])
        lens = ok_len.apply(lambda x: x[1])
        tail_parse_fail += int((~ok).sum())
        tail_len9_total += int((lens == 9).sum())   # 改为累加外部变量

    # ---- sched/priority 转数值 ----
    chunk["scheduling_class"] = pd.to_numeric(chunk["scheduling_class"], errors="coerce")
    chunk["priority"] = pd.to_numeric(chunk["priority"], errors="coerce")

    # ---- slice 级聚合：min/max/mean（不做 mode，避免复杂流式统计）----
    g = chunk.groupby([MID, TS], sort=False, observed=False)
    part = pd.DataFrame({
        "sched_min": g["scheduling_class"].min(),
        "sched_max": g["scheduling_class"].max(),
        "sched_mean": g["scheduling_class"].mean(),
        "prio_min": g["priority"].min(),
        "prio_max": g["priority"].max(),
        "prio_mean": g["priority"].mean(),
        "n_raw_rows": g.size(),
    }).reset_index()
    parts.append(part)

    print(f"  chunk{chunk_idx}: valid_rows={len(chunk)} slice_rows={len(part)}")

# 合并 chunk 聚合（同 key 再聚合一次）
agg = pd.concat(parts, ignore_index=True)
g2 = agg.groupby([MID, TS], sort=False, observed=False)
agg2 = pd.DataFrame({
    "sched_min": g2["sched_min"].min(),
    "sched_max": g2["sched_max"].max(),
    "prio_min": g2["prio_min"].min(),
    "prio_max": g2["prio_max"].max(),
    "prio_mean": g2["prio_mean"].mean(),   # 必须保留，用于分桶诊断
    "n_raw_rows": g2["n_raw_rows"].sum(),
}).reset_index()

# 保存聚合表（便于复核；行数约等于 clean 的 machine-slice 数量级）
agg2.to_csv(OUT_AGG, index=False, encoding="utf-8-sig")

# 将交叉检查结果追加到 lines 中（但 lines 在后面才初始化，所以可以先存到变量，后面再追加）
cross_summary = (f"both_ok={both_ok}, cpu_only={cpu_only}, tail_only={tail_only}, both_fail={both_fail}")

# ------------------------------------------------------------
# 3) 连接 label-valid(H=8) 样本域，做关联统计（pos_rate by sched / priority bins）
# ------------------------------------------------------------
df = labv.merge(agg2, on=[MID, TS], how="left", validate="one_to_one")

# 连接成功率：如果低，说明 raw 的 sched/priority 在某些 slice 缺失严重
match_rate = float(df["sched_max"].notna().mean())

# scheduling_class 分组：优先用 sched_max（若一 slice 内有多个 class，max 代表“最不敏感”一侧）
# 这里仅作诊断，不宣称因果
df["sched_max_int"] = df["sched_max"].round().astype("Int64")  # 允许 NA

# priority 分桶：边界只用训练集样本确定（避免 val/test 影响）
train_prio = df[(df[SPLIT] == "train") & df["prio_mean"].notna()]["prio_mean"]
if len(train_prio) >= 5:
    # 训练集分位数边界（例如 0,20%,40%,60%,80%,100%）
    qs = train_prio.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
    qs = np.unique(qs)  # 去除重复边界
    if len(qs) >= 3:
        df["prio_bin"] = pd.cut(df["prio_mean"], bins=qs, include_lowest=True, duplicates="drop")
    else:
        df["prio_bin"] = pd.NA
else:
    df["prio_bin"] = pd.NA

# 汇总：按 split 输出 sched/priority 的 pos_rate
sum_rows = []

def add_group_stats(group_col, group_name):
    for sp in DIAG_SPLITS:
        sub = df[df[SPLIT] == sp]
        # 只看 group_col 非空
        sub = sub[sub[group_col].notna()]
        if len(sub) == 0:
            continue
        tab = (sub.groupby(group_col, observed=False)["label"]
               .agg(n="count", pos="sum", pos_rate="mean")
               .reset_index()
               .sort_values("n", ascending=False))
        tab["split"] = sp
        tab["group"] = group_name
        tab.rename(columns={group_col: "bucket"}, inplace=True)
        sum_rows.append(tab)

add_group_stats("sched_max_int", "scheduling_class_by_sched_max")
add_group_stats("prio_bin", "priority_by_cut_quantiles_train")

summary = pd.concat(sum_rows, ignore_index=True) if sum_rows else pd.DataFrame()
summary.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 4) CPU 分布列缺失/解析失败汇总（raw级别）
# ------------------------------------------------------------
cpu_report = pd.DataFrame([{
    "field": "cpu_usage_distribution",
    "total_valid_rows_after_time_machine_filter": cpu_total,
    "na_count": cpu_na,
    "na_rate": (cpu_na / cpu_total) if cpu_total else np.nan,
    "parse_fail_count_on_non_na": cpu_parse_fail,
    "parse_fail_rate_on_non_na": (cpu_parse_fail / max(cpu_total - cpu_na, 1)),
    "len_11_count": cpu_len11_total,                               # 改为累加总变量
    "len_11_rate": (cpu_len11_total / max(cpu_total - cpu_na, 1)),  # 改为累加总变量
}, {
    "field": "tail_cpu_usage_distribution",
    "total_valid_rows_after_time_machine_filter": tail_total,
    "na_count": tail_na,
    "na_rate": (tail_na / tail_total) if tail_total else np.nan,
    "parse_fail_count_on_non_na": tail_parse_fail,
    "parse_fail_rate_on_non_na": (tail_parse_fail / max(tail_total - tail_na, 1)),
    "len_9_count": tail_len9_total,                                # 改为累加总变量
    "len_9_rate": (tail_len9_total / max(tail_total - tail_na, 1)), # 改为累加总变量
}])
cpu_report.to_csv(OUT_CPU, index=False, encoding="utf-8-sig")
# ---- Gate: 计数必须守恒（否则证据链无效）----
assert cpu_total == cross_total_valid_rows, f"❌ cpu_total({cpu_total}) != cross_total_valid_rows({cross_total_valid_rows})"
assert cpu_total == (both_ok + cpu_only + tail_only + both_fail), \
    f"❌ cross quadrant does not sum to total_valid_rows: {both_ok}+{cpu_only}+{tail_only}+{both_fail} != {cpu_total}"

# ---- 新增：长度不匹配检查（若数据中确实没有长度错误，可断言为0）----
assert cpu_len_mismatch == 0, f"❌ cpu_len_mismatch={cpu_len_mismatch} > 0"
assert tail_len_mismatch == 0, f"❌ tail_len_mismatch={tail_len_mismatch} > 0"

# ------------------------------------------------------------
# 5) 总结报告（审稿人可读）
# ------------------------------------------------------------
lines = []
lines.append("=== Step2G(v2) Diagnostic Report: sched/priority + cpu distributions ===")
lines.append(f"label_valid_rows(H=8)={len(labv)}")
lines.append(f"slice_join_match_rate(sched_max_notna)={match_rate:.6f}")
lines.append("")
lines.append("[CPU distribution missing/parse]")
lines.append(cpu_report.to_string(index=False))
lines.append("")
lines.append("")
lines.append("[CPU/tail parse consistency]")
lines.append(cross_summary)
# ========== 新增：细粒度解析失败分解 ==========
lines.append("")
lines.append("[CPU/tail parse failure breakdown]")
lines.append(f"cpu_parse_fail_lenient (cannot parse) = {cpu_parse_fail_lenient}")
lines.append(f"cpu_len_mismatch (parsed but length != 11) = {cpu_len_mismatch}")
lines.append(f"tail_parse_fail_lenient = {tail_parse_fail_lenient}")
lines.append(f"tail_len_mismatch (parsed but length != 9) = {tail_len_mismatch}")
lines.append(f"note: both_ok = {both_ok}, total rows = {cross_total_valid_rows}")

# ========== 门禁守恒检查（落盘 + 断言） ==========
lines.append("")
lines.append("[Gate: consistency check]")
total_quadrant = both_ok + cpu_only + tail_only + both_fail
lines.append(f"total_valid_rows (cpu_total) = {cpu_total}")
lines.append(f"both_ok + cpu_only + tail_only + both_fail = {total_quadrant}")
if cpu_total == total_quadrant:
    lines.append("✓ Consistency check passed.")
else:
    lines.append("❌ Consistency check FAILED!")

# 硬性断言（阻止 pipeline 继续运行）
assert cpu_total == total_quadrant, \
    f"Gate failed: cpu_total={cpu_total}, quadrant sum={total_quadrant}"
# =============================================

lines.append("[Outputs]")
lines.append(f"slice_agg_csv={OUT_AGG}")
lines.append(f"assoc_summary_csv={OUT_SUM}")
lines.append(f"cpu_missing_parse_csv={OUT_CPU}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n[输出]", OUT_AGG)
print("[输出]", OUT_SUM)
print("[输出]", OUT_CPU)
print("[输出]", OUT_TXT)
print("\n".join(lines[:25]))
