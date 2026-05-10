# ============================================================
# 文件名：Step3a_aux_raw_cpu_dist_vector_merge_v2.py
# 版本：v2（方案1：sched_frac_0..3 + prio_min/max/mean）
#
# 作用：
#   从 raw 表按 (machine_id, time_slice) 聚合并注入到 Step3a features_raw：
#
#   (1) cpu_usage_distribution  -> 11维向量 cpu_dist_0..cpu_dist_10
#   (2) tail_cpu_usage_distribution -> 9维向量 tail_cpu_dist_0..tail_cpu_dist_8
#   (3) 缺失/解析失败指示：
#         cpu_dist_missing, tail_cpu_dist_missing
#
#   (4) scheduling_class（离散类别0/1/2/3）：
#         输出比例向量 sched_frac_0..sched_frac_3
#       语义：在同一 slice 内，属于类别k的记录占 scheduling_class 有效记录的比例。
#
#   (5) priority（数值型）：
#         输出 prio_min/prio_max/prio_mean
#       语义：slice 内任务优先级的范围与均值，用于描述工作负载混合属性。
#
# 聚合语义（必须写进论文/附录）：
#   - 对于 CPU 分布向量：同一 slice 内若出现多条记录，按“逐维均值”聚合为 slice 级向量。
#   - 对于 scheduling_class：按类别计数并归一化为比例向量（非均值）。
#   - 对于 priority：min/max/mean。
#
# 重要约束：
#   - 不改 Stage2 标签、不改 split
#   - time_slice 定义与 Step2a 一致：time_us // 300s，且 time_us>0 且 <INT64_MAX
#
# 输入：
#   00_raw/borg_traces_data.csv
#   01_interim/step3a_features_raw_v2.csv
#
# 输出（覆盖写回）：
#   01_interim/step3a_features_raw_v2.csv
# 报告：
#   05_reports/step3a_aux_cpu_dist_merge_report_v2.txt
# ============================================================

import os
import re
import numpy as np
import pandas as pd

# -----------------------------
# 0) 路径与常量
# -----------------------------
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")

IN_FEATURES = os.path.join(ROOT_DIR, "01_interim", "step3a_features_raw_v2.csv")
OUT_FEATURES = IN_FEATURES  # 覆盖写回

REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)
OUT_REPORT = os.path.join(REPORT_DIR, "step3a_aux_cpu_dist_merge_report_v2.txt")

MID = "machine_id"
TIME = "time"
TS = "time_slice"

SLICE_US = 300_000_000
INT64_MAX = 9223372036854775807
CHUNKSIZE = 200_000

# 是否同时注入 sched/priority（方案1要求 True）
ADD_SCHED_PRIORITY = True

CPU_LEN = 11
TAIL_LEN = 9

# -----------------------------
# 1) 解析函数：解析 numpy 打印式数组字符串
# -----------------------------
def parse_np_array_string(val, expected_len: int):
    """
    解析形如：
      "[0.00314331 0.00381088 ... 0.01194763]"
    或带逗号/多空格的变体，返回固定长度向量 np.ndarray；否则返回 None。
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if inner == "":
        return None
    inner = re.sub(r"[\s,]+", " ", inner)
    arr = np.fromstring(inner, sep=" ")

    if arr.size != expected_len:
        return None
    return arr


print("=" * 90)
print("Step3A-aux(v2)：raw CPU分布向量 + sched/priority 聚合并 merge 覆盖 features_raw")
print("=" * 90)
print("[输入 raw]     ", RAW_PATH)
print("[输入 features]", IN_FEATURES)

# -----------------------------
# 2) 读取 features_raw（主表），并做主键断言
# -----------------------------
feat = pd.read_csv(IN_FEATURES, low_memory=False)

assert MID in feat.columns and TS in feat.columns, "❌ features_raw 缺少 machine_id/time_slice"
dup_feat = int(feat.duplicated([MID, TS]).sum())
assert dup_feat == 0, f"❌ features_raw 主键重复: {dup_feat}"

feat[MID] = pd.to_numeric(feat[MID], errors="coerce").astype("int64")
feat[TS] = pd.to_numeric(feat[TS], errors="coerce").astype("int64")

# -----------------------------
# 3) 分块读取 raw 并在块内聚合到 (MID,TS)
# -----------------------------
usecols = [MID, TIME, "cpu_usage_distribution", "tail_cpu_usage_distribution"]
if ADD_SCHED_PRIORITY:
    usecols += ["scheduling_class", "priority"]

parts = []

# 统计解析情况（raw有效行级）
cpu_total = cpu_ok = cpu_fail = 0
tail_total = tail_ok = tail_fail = 0

chunk_id = 0
for chunk in pd.read_csv(RAW_PATH, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
    chunk_id += 1

    mid = pd.to_numeric(chunk[MID], errors="coerce")
    t = pd.to_numeric(chunk[TIME], errors="coerce")

    # 与 Step2a 一致的有效性过滤：machine_id>=0 且 time 在 (0, INT64_MAX)
    valid = mid.notna() & (mid >= 0) & t.notna() & (t > 0) & (t < INT64_MAX)
    chunk = chunk.loc[valid].copy()
    if len(chunk) == 0:
        continue

    chunk[MID] = pd.to_numeric(chunk[MID], errors="coerce").astype("int64")
    chunk["time_us"] = pd.to_numeric(chunk[TIME], errors="coerce").astype("int64")
    chunk[TS] = (chunk["time_us"] // SLICE_US).astype("int64")

    n = len(chunk)

    # ---------- 3.1 解析 CPU 11维向量 ----------
    cpu_mat = np.full((n, CPU_LEN), np.nan, dtype=np.float64)
    cpu_ok_flag = np.zeros(n, dtype=np.int64)

    for i, v in enumerate(chunk["cpu_usage_distribution"].to_numpy()):
        cpu_total += 1
        arr = parse_np_array_string(v, expected_len=CPU_LEN)
        if arr is None:
            cpu_fail += 1
        else:
            cpu_ok += 1
            cpu_mat[i, :] = arr
            cpu_ok_flag[i] = 1

    # ---------- 3.2 解析 tail CPU 9维向量 ----------
    tail_mat = np.full((n, TAIL_LEN), np.nan, dtype=np.float64)
    tail_ok_flag = np.zeros(n, dtype=np.int64)

    for i, v in enumerate(chunk["tail_cpu_usage_distribution"].to_numpy()):
        tail_total += 1
        arr = parse_np_array_string(v, expected_len=TAIL_LEN)
        if arr is None:
            tail_fail += 1
        else:
            tail_ok += 1
            tail_mat[i, :] = arr
            tail_ok_flag[i] = 1

    # 将向量展开为列（便于 groupby sum）
    for k in range(CPU_LEN):
        chunk[f"cpu_dist_{k}"] = cpu_mat[:, k]
    for k in range(TAIL_LEN):
        chunk[f"tail_cpu_dist_{k}"] = tail_mat[:, k]

    chunk["cpu_ok"] = cpu_ok_flag
    chunk["tail_ok"] = tail_ok_flag

    # ---------- 3.3 sched/priority 预处理（方案1） ----------
    if ADD_SCHED_PRIORITY:
        # scheduling_class：离散类别0/1/2/3 -> 比例向量
        chunk["scheduling_class"] = pd.to_numeric(chunk["scheduling_class"], errors="coerce")
        for k in [0, 1, 2, 3]:
            chunk[f"sched_is_{k}"] = (chunk["scheduling_class"] == k).astype("int64")
        valid_sched = chunk["scheduling_class"].isin([0, 1, 2, 3])
        chunk["sched_obs"] = valid_sched.astype("int64")
        chunk["sched_unknown"] = (chunk["scheduling_class"].notna() & (~valid_sched)).astype("int64")

        # priority：数值，slice内取 min/max/mean
        chunk["priority"] = pd.to_numeric(chunk["priority"], errors="coerce")

    # ---------- 3.4 块内聚合到 (MID,TS) ----------
    g = chunk.groupby([MID, TS], sort=False)

    agg = {
        # CPU向量：逐维 sum + ok计数（后续 mean = sum/cnt）
        **{f"cpu_dist_{k}_sum": g[f"cpu_dist_{k}"].sum(min_count=1) for k in range(CPU_LEN)},
        "cpu_cnt": g["cpu_ok"].sum(),

        **{f"tail_cpu_dist_{k}_sum": g[f"tail_cpu_dist_{k}"].sum(min_count=1) for k in range(TAIL_LEN)},
        "tail_cnt": g["tail_ok"].sum(),
    }

    if ADD_SCHED_PRIORITY:
        # sched 计数：sum
        for k in [0, 1, 2, 3]:
            agg[f"sched_is_{k}_sum"] = g[f"sched_is_{k}"].sum()
        agg["sched_obs_sum"] = g["sched_obs"].sum()
        agg["sched_unknown_sum"] = g["sched_unknown"].sum()

        # priority：min/max/mean
        agg["prio_min"] = g["priority"].min()
        agg["prio_max"] = g["priority"].max()
        agg["prio_sum"] = g["priority"].sum(min_count=1)
        agg["prio_cnt"] = g["priority"].count()

    part = pd.DataFrame(agg).reset_index()
    parts.append(part)

    print(f"  chunk{chunk_id}: valid_rows={n}, slice_rows={len(part)}")
# -----------------------------
# 3.x 解析计数守恒 Gate（必须在所有 chunk 处理完后执行一次）
# -----------------------------
assert cpu_total == tail_total, f"❌ cpu_total({cpu_total}) != tail_total({tail_total})"
assert cpu_total == cpu_ok + cpu_fail, \
    f"❌ cpu_total({cpu_total}) != cpu_ok({cpu_ok}) + cpu_fail({cpu_fail})"
assert tail_total == tail_ok + tail_fail, \
    f"❌ tail_total({tail_total}) != tail_ok({tail_ok}) + tail_fail({tail_fail})"
cpu_fail_rate = cpu_fail / max(cpu_total, 1)
tail_fail_rate = tail_fail / max(tail_total, 1)
assert abs(cpu_fail_rate - tail_fail_rate) < 1e-9, "❌ cpu/tail fail rate mismatch (unexpected)"


# -----------------------------
# 4) 二次聚合（跨 chunk 同 key 合并）
# -----------------------------
agg_all = pd.concat(parts, ignore_index=True)
g2 = agg_all.groupby([MID, TS], sort=False)

agg2 = {
    **{f"cpu_dist_{k}_sum": g2[f"cpu_dist_{k}_sum"].sum(min_count=1) for k in range(CPU_LEN)},
    "cpu_cnt": g2["cpu_cnt"].sum(),

    **{f"tail_cpu_dist_{k}_sum": g2[f"tail_cpu_dist_{k}_sum"].sum(min_count=1) for k in range(TAIL_LEN)},
    "tail_cnt": g2["tail_cnt"].sum(),
}

if ADD_SCHED_PRIORITY:
    for k in [0, 1, 2, 3]:
        agg2[f"sched_is_{k}_sum"] = g2[f"sched_is_{k}_sum"].sum()
    agg2["sched_obs_sum"] = g2["sched_obs_sum"].sum()
    agg2["sched_unknown_sum"] = g2["sched_unknown_sum"].sum()

    agg2["prio_min"] = g2["prio_min"].min()
    agg2["prio_max"] = g2["prio_max"].max()
    agg2["prio_sum"] = g2["prio_sum"].sum(min_count=1)
    agg2["prio_cnt"] = g2["prio_cnt"].sum()

extra = pd.DataFrame(agg2).reset_index()  # ★关键：reset_index 保证 MID/TS 是列而不是索引

# 主键唯一性（必须）
dup_extra = int(extra.duplicated([MID, TS]).sum())
assert dup_extra == 0, f"❌ extra 主键重复: {dup_extra}"

# -----------------------------
# 5) 计算 slice 级均值向量 + missing 指示；计算 sched_frac
# -----------------------------
# 5.1 CPU/Tail 均值向量（逐维 mean = sum/cnt）
for k in range(CPU_LEN):
    extra[f"cpu_dist_{k}"] = np.where(
        extra["cpu_cnt"] > 0,
        extra[f"cpu_dist_{k}_sum"] / extra["cpu_cnt"],
        np.nan
    )
for k in range(TAIL_LEN):
    extra[f"tail_cpu_dist_{k}"] = np.where(
        extra["tail_cnt"] > 0,
        extra[f"tail_cpu_dist_{k}_sum"] / extra["tail_cnt"],
        np.nan
    )

# missing：cnt==0 视为该 slice 没有成功解析到固定长度向量（解析失败/格式异常等）
extra["cpu_dist_missing"] = (extra["cpu_cnt"] == 0).astype("int64")
extra["tail_cpu_dist_missing"] = (extra["tail_cnt"] == 0).astype("int64")

# 5.2 scheduling_class 比例向量
if ADD_SCHED_PRIORITY:
    denom = extra["sched_obs_sum"].to_numpy(dtype=np.float64)
    denom_safe = np.where(denom > 0, denom, np.nan)

    for k in [0, 1, 2, 3]:
        extra[f"sched_frac_{k}"] = (extra[f"sched_is_{k}_sum"].to_numpy(dtype=np.float64) / denom_safe)
        # 若该 slice 没有 sched 观测，则比例定义为 0（并不引入额外维度）
        extra[f"sched_frac_{k}"] = np.nan_to_num(extra[f"sched_frac_{k}"], nan=0.0)

    # ---- sched_frac simplex gate（必须放在所有比例计算完成后，且只执行一次）----
    sched_obs_sum = extra["sched_obs_sum"].to_numpy(dtype=np.int64)
    sched_unknown_sum = extra["sched_unknown_sum"].to_numpy(dtype=np.int64)
    unknown_cnt = int((sched_unknown_sum > 0).sum())
    frac_sum = (extra["sched_frac_0"] + extra["sched_frac_1"] + extra["sched_frac_2"] + extra["sched_frac_3"]).to_numpy(dtype=np.float64)
    mask_obs = sched_obs_sum > 0
    max_abs_err = float(np.max(np.abs(frac_sum[mask_obs] - 1.0))) if mask_obs.any() else 0.0
    viol_cnt = int((np.abs(frac_sum[mask_obs] - 1.0) > 1e-6).sum()) if mask_obs.any() else 0
    mask_no = sched_obs_sum == 0
    no_obs_viol = int((np.abs(frac_sum[mask_no] - 0.0) > 1e-9).sum()) if mask_no.any() else 0

    assert unknown_cnt == 0, f"❌ sched_unknown exists: slices_with_unknown={unknown_cnt}"
    assert viol_cnt == 0, f"❌ sched_frac simplex violated: viol_cnt={viol_cnt}, max_abs_err={max_abs_err}"
    assert no_obs_viol == 0, f"❌ sched_frac sum not zero when no obs: count={no_obs_viol}"

# 5.3 计算 prio_mean（slice 级加权平均）并添加一致性门禁
if ADD_SCHED_PRIORITY:
    extra["prio_mean"] = np.where(
        extra["prio_cnt"] > 0,
        extra["prio_sum"] / extra["prio_cnt"],
        np.nan
    )
    bad_prio = int(((extra["prio_min"].notna()) & (extra["prio_mean"].isna())).sum())
    assert bad_prio == 0, f"❌ prio_mean NaN but prio_min notna: {bad_prio}"

# -----------------------------
# 6) 清理中间列（sum/cnt/计数）
# -----------------------------
drop_cols = []
drop_cols += [f"cpu_dist_{k}_sum" for k in range(CPU_LEN)]
drop_cols += [f"tail_cpu_dist_{k}_sum" for k in range(TAIL_LEN)]
drop_cols += ["cpu_cnt", "tail_cnt"]

if ADD_SCHED_PRIORITY:
    drop_cols += [f"sched_is_{k}_sum" for k in [0, 1, 2, 3]]
    drop_cols += ["sched_obs_sum"]
    drop_cols += ["sched_unknown_sum"]
    drop_cols += ["prio_sum", "prio_cnt"]

extra.drop(columns=[c for c in drop_cols if c in extra.columns], inplace=True)

# -----------------------------
# 7) merge 回 features_raw（one_to_one 强约束）
# -----------------------------

merged_feat = feat.merge(extra, on=[MID, TS], how="left", validate="one_to_one")

# join 质量检查：cpu_dist_0 非空比例（成功解析并聚合的比例）应接近 Step2H 的可解析率
join_nonnull_rate = float(merged_feat["cpu_dist_0"].notna().mean())

# === 新增：合并覆盖率门禁（必须 100% 匹配）===
assert merged_feat["cpu_dist_missing"].isna().sum() == 0, \
    f"❌ merge key mismatch: {merged_feat['cpu_dist_missing'].isna().sum()} rows have NaN in cpu_dist_missing"
assert merged_feat["tail_cpu_dist_missing"].isna().sum() == 0, \
    f"❌ merge key mismatch: {merged_feat['tail_cpu_dist_missing'].isna().sum()} rows have NaN in tail_cpu_dist_missing"
# 注意：cpu_dist_missing 本身是 int64，不应该为 NaN；若为 NaN 说明 key 不在 extra 中
# Gate: 验证 Step2A 新增特征在 merge 后存活
for col_check in ["ms_cpi_coverage", "ms_mpi_coverage"]:
    assert col_check in merged_feat.columns, f"❌ merge 后丢失列 {col_check}"
    assert merged_feat[col_check].notna().sum() == feat[col_check].notna().sum(), \
        f"❌ {col_check} 在 merge 后非空数减少"

# slice-level missing 率：把 join 失败视为 missing（fillna(1)）
slice_missing_rate = float(merged_feat["cpu_dist_missing"].fillna(1).mean())
tail_slice_missing_rate = float(merged_feat["tail_cpu_dist_missing"].fillna(1).mean())

# priority/sched 的 notna 检查（join 成功率）
prio_notna = float(merged_feat["prio_mean"].notna().mean()) if "prio_mean" in merged_feat.columns else np.nan
sched_frac0_notna = float(merged_feat["sched_frac_0"].notna().mean()) if "sched_frac_0" in merged_feat.columns else np.nan

# -----------------------------
# 8) 写报告 + 覆盖写回 features_raw
# -----------------------------
lines = []
lines.append("=== Step3A-aux(v2) CPU dist + sched/priority merge report ===")
lines.append("")
lines.append("[Raw parsing stats on valid raw rows]")
lines.append(f"cpu_total={cpu_total} cpu_ok={cpu_ok} cpu_fail={cpu_fail} cpu_fail_rate={cpu_fail/max(cpu_total,1):.6f}")
lines.append(f"tail_total={tail_total} tail_ok={tail_ok} tail_fail={tail_fail} tail_fail_rate={tail_fail/max(tail_total,1):.6f}")
lines.append("")
lines.append("[Merge stats]")
lines.append(f"features_rows={len(feat)} extra_rows={len(extra)} merged_rows={len(merged_feat)}")
lines.append(f"join_nonnull_rate(cpu_dist_0)={join_nonnull_rate:.6f}")
lines.append(f"slice_level_cpu_missing_rate(cpu_dist_missing)={slice_missing_rate:.6f}")
lines.append(f"slice_level_tail_missing_rate(tail_cpu_dist_missing)={tail_slice_missing_rate:.6f}")
if ADD_SCHED_PRIORITY:
    lines.append(f"prio_mean_notna_rate={prio_notna:.6f}")
    lines.append(f"sched_frac_0_notna_rate={sched_frac0_notna:.6f}")
lines.append("")
lines.append("[Columns added]")
lines.append("cpu_dist_0..10, tail_cpu_dist_0..8, cpu_dist_missing, tail_cpu_dist_missing")
if ADD_SCHED_PRIORITY:
    lines.append("sched_frac_0..3, prio_min, prio_max, prio_mean")
    # 添加 simplex gate 结果到报告
    lines.append(f"[sched_frac simplex] unknown_cnt={unknown_cnt}, viol_cnt={viol_cnt}, no_obs_viol={no_obs_viol}")

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# 覆盖写回（你要求干净覆盖）
merged_feat.to_csv(OUT_FEATURES, index=False, encoding="utf-8-sig")

print("\n[输出覆盖]", OUT_FEATURES)
print("[报告]", OUT_REPORT)
print("\n".join(lines[:25]))
print("=" * 90)
print("Step3A-aux(v2) 完成")