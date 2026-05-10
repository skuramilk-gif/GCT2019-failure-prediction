# ============================================================
# 文件名：Step2e_time_order_qc_v2.py
# 阶段：Step 2E（v2，证据链补强）
# 作用（审稿可 defend 的“硬证据”）：
#   1) 验证 clean 表在 machine 内的时间轴是否自洽、是否存在回跳/乱序风险
#      - 你的标签生成 next-fail/searchsorted 依赖 machine 内时间序（至少可被排序成非降序）
#   2) 验证 time_slice 与 slice_last_time_us 的映射一致性：
#      - 理论上 slice_last_time_us // SLICE_US 应当等于 time_slice（来自同一 time_us）
#      - 若不一致，说明 Step2a 聚合或 time_slice 计算存在 bug，会直接污染标签与特征
#   3) 输出 machine 级/行级统计，落盘到 05_reports，用于论文实验设置或附录
# ============================================================
#时间轴自洽完成（gate)：dup_keys=0、ts_mismatch_cnt=0、neg_dt_cnt=0、zero_dt_cnt=0，并包含 delta_end 分布，断言已通过。
# ============================================================
# 输入：
#   01_interim/step2a_machine_slice_clean_v2.csv
#
# 输出（05_reports/）：
#   step2e_time_order_qc_summary_v2.txt        # 总览结论（审稿人最关心）
#   step2e_time_order_qc_machinelevel_v2.csv   # 每台机器的回跳/异常统计（便于复核）
#   step2e_time_order_qc_examples_v2.csv       # 异常样例（若存在）-存在要继续进行后续检测
#审稿人可能会问：
#1为什么pos_dt_count（有效间隔数）只有 245,344，而总行数是 341,291？--两个相减是95947这就是机器数，并且每台机器数的第一个slice没有前向间隔
#2pos_dt 的分布是全局的，但训练/验证/测试集的稀疏性是否一致？--可以在论文附录中补充按train/val/test分别统计的gap分布，或者说明由于spilt是按机器随机划分的，稀疏性在不同数据集是一致的
#3. 时间轴完美是否有可能是过滤过于严格导致？==time=0 的 FAIL 不管其比例我们都不能使用，因为在谷歌白皮书中规定这部分FALL其实是前一段时间的FALL因为谷歌系统的快速回照缘故才记录下来
# ============================================================
#关键解释：delta_end分布对于slice_last_time_us要解释成往往不是接近300s的slice末端，而更像slice内最后一次被观测到的时间点，中位数距离末端还有2.5分钟
#论文中要写成：预测时间定义为该slice内最后一次可观测记录的时间（slice_last_time_us），它位于固定300s sslice的内部；我们报告了与固定末端的偏移分布，偏移上界小于300s，相对与H是可以忽略的。
# ============================================================

import os
import numpy as np
import pandas as pd
#输入文件名称
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
IN_CLEAN = os.path.join(ROOT_DIR, "01_interim", "step2a_machine_slice_clean_v2.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)
#输出文件名称
OUT_TXT = os.path.join(REPORT_DIR, "step2e_time_order_qc_summary_v2.txt")
OUT_MACH = os.path.join(REPORT_DIR, "step2e_time_order_qc_machinelevel_v2.csv")
OUT_EX = os.path.join(REPORT_DIR, "step2e_time_order_qc_examples_v2.csv")
#缩写定义，防止调用报错
MID = "machine_id"
TS = "time_slice"
T = "slice_last_time_us"

# Step2a 的 slice 定义：300s = 300,000,000 us
SLICE_US = 300_000_000

print("=" * 90)  #控制台大部分宽度为80-120字符，90刚好美观作用。
print("Step2E(v2)：时间轴单调性/一致性 QC（不改数据，仅出证据报告）")
print("=" * 90)
print("[输入]", IN_CLEAN)

# 只读必要列：避免内存开销，也避免误用其它列
usecols = [MID, TS, T]
df = pd.read_csv(IN_CLEAN, usecols=usecols, low_memory=False)

# -------- 基本类型/合法性 --------
df[MID] = pd.to_numeric(df[MID], errors="coerce")
df[TS] = pd.to_numeric(df[TS], errors="coerce")
df[T] = pd.to_numeric(df[T], errors="coerce")

# 任何关键列缺失都不应继续：这是上游 clean 表结构错误
assert df[MID].notna().all(), "❌ machine_id 存在 NaN"
assert df[TS].notna().all(), "❌ time_slice 存在 NaN"
assert df[T].notna().all(), "❌ slice_last_time_us 存在 NaN"
assert (df[MID] >= 0).all(), "❌ machine_id 出现负值（哨兵未清理）"
assert (df[TS] >= 0).all(), "❌ time_slice 出现负值"
assert (df[T] > 0).all(), "❌ slice_last_time_us 出现非正数（time<=0 未完全清理）"

df[MID] = df[MID].astype("int64")
df[TS] = df[TS].astype("int64")
df[T] = df[T].astype("int64")

# -------- 主键唯一性（必须）--------
dup_keys = int(df.duplicated([MID, TS]).sum())
# 这里不直接 assert fail，因为你已在 Step2c 断言过；我们在报告里也输出，便于证据链一致
# 若 dup_keys>0，说明 clean 地基有问题，应回滚 Step2a 修复
# -------------------------------------------------------------

# -------- 映射一致性：slice_last_time_us 与 time_slice 是否同源 --------
# 理论上：time_slice = floor(time_us / SLICE_US)，slice_last_time_us 是 slice 内 time_us 的 max
# 因此 slice_last_time_us // SLICE_US 应当等于 time_slice
derived_ts_from_t = (df[T] // SLICE_US).astype("int64")
ts_mismatch = (derived_ts_from_t != df[TS])
ts_mismatch_cnt = int(ts_mismatch.sum())

# -------- machine 内单调性：按 time_slice 排序后 slice_last_time_us 不应回跳 --------
# 注意：我们按 time_slice 排序检查是最关键的，因为模型/标签逻辑通常沿 slice 序推进
df_sorted = df.sort_values([MID, TS]).reset_index(drop=True)

# 同机内的 time_slice diff（不应 <=0，因为主键唯一且排序）
d_ts = df_sorted.groupby(MID)[TS].diff()
nonpos_ts_cnt = int((d_ts.notna() & (d_ts <= 0)).sum())

# 同机内的 slice_last_time_us diff（不应 <0；允许 =0 理论上不该出现，但不作为硬错）
d_t = df_sorted.groupby(MID)[T].diff()
neg_dt_cnt = int((d_t.notna() & (d_t < 0)).sum())
zero_dt_cnt = int((d_t.notna() & (d_t == 0)).sum())



# machine 级别：是否存在回跳/异常
mach_neg_dt = df_sorted.assign(neg_dt=(d_t < 0)).groupby(MID)["neg_dt"].any()
mach_neg_dt_cnt = int(mach_neg_dt.sum())
mach_total = int(df_sorted[MID].nunique())


# diff 分布（用于论文描述：观测稀疏/跨度）
# 用 slice_last_time_us 的正差分（排除 NaN 和非正）
pos_dt = d_t[(d_t.notna()) & (d_t > 0)]
pos_dt_hours = (pos_dt / (60 * 60 * 1_000_000)).astype(float)

# -------- machine-level 汇总表（便于复核）--------
mach_tbl = (df_sorted
            .groupby(MID)
            .agg(
                n_slices=(TS, "count"),
                ts_min=(TS, "min"),
                ts_max=(TS, "max"),
                t_min=(T, "min"),
                t_max=(T, "max"),
            )
            .reset_index())

# 统计每台机器内部，时间戳是否出现倒退（负差值）或重复（0差值）.df_sorted 已经按 machine_id 和 time_slice 排序好。
#d_t 是同一机器内，每行与上一行的 slice_last_time_us 的差值（即时间间隔）。第一行是 NaN。
tmp = df_sorted[[MID, TS, T]].copy()  #复制到所需列到新表tmp-避免后续修改时影响原始数据
tmp["d_t"] = d_t.values               #将d_t赋给tmp的新列d_t上
tmp["neg_dt"] = (tmp["d_t"].notna() & (tmp["d_t"] < 0)).astype(int) #满足差值不是NaN且小于0为回跳行，标记为1否则0.
tmp["zero_dt"] = (tmp["d_t"].notna() & (tmp["d_t"] == 0)).astype(int)#.astype(int) 把布尔值 True/False 转成 1/0。
mach_dt_stat = tmp.groupby(MID).agg(
    neg_dt_count=("neg_dt", "sum"),    #统计该机器内部发生回跳的总次数
    zero_dt_count=("zero_dt", "sum"),  #统计不变的次数
).reset_index()
#将机器级时间差统计表（mach_dt_stat）合并到机器汇总表，并且按照MID左连接并且两边连接都要1对1
mach_tbl = mach_tbl.merge(mach_dt_stat, on=MID, how="left", validate="one_to_one")
mach_tbl["has_neg_dt"] = (mach_tbl["neg_dt_count"] > 0).astype(int)  #neg_dt时间回跳行
mach_tbl["has_zero_dt"] = (mach_tbl["zero_dt_count"] > 0).astype(int)  #zero_dt时间0行
#将最后的机器级汇总表保存
mach_tbl.to_csv(OUT_MACH, index=False, encoding="utf-8-sig")
# 统计存在零差值的机器数
mach_zero_dt_cnt = int(mach_tbl["has_zero_dt"].sum())

# -------- 异常样例输出（若存在）--------
examples = []
if ts_mismatch_cnt > 0: #处理时间片映射不一定
    ex1 = df_sorted.loc[ts_mismatch, [MID, TS, T]].head(200)#筛选出映射不一致的行，只取前200.
    ex1 = ex1.assign(issue="time_slice_mismatch_with_slice_last_time_us")#新增一列'issue'，内容为异常类型描述。
    examples.append(ex1)  #加入列表

if neg_dt_cnt > 0:  #处理时间回跳（负差值）异常
    ex2 = tmp.loc[tmp["neg_dt"] == 1, [MID, TS, T, "d_t"]].head(200)
    ex2 = ex2.assign(issue="negative_dt(slice_last_time_us_backwards)")
    examples.append(ex2)

if nonpos_ts_cnt > 0:  #处理时间片差非正异常
    ex3 = tmp.loc[(d_ts.notna() & (d_ts <= 0)).values, [MID, TS, T]].head(200)#d_ts.notna()=条件：d_ts不为NaN
    ex3 = ex3.assign(issue="non_positive_time_slice_diff")
    examples.append(ex3)

if examples:
    ex_all = pd.concat(examples, ignore_index=True)
    ex_all.to_csv(OUT_EX, index=False, encoding="utf-8-sig")
else:
    # 若无异常也可不生成文件；这里生成一个空文件便于流水线对齐
    pd.DataFrame(columns=[MID, TS, T, "d_t", "issue"]).to_csv(OUT_EX, index=False, encoding="utf-8-sig")

# -------- 总结报告（最重要）--------
lines = []
lines.append("=== Step2E(v2) Time Order QC Summary ===")
lines.append(f"rows={len(df_sorted)}, machines={mach_total}")
lines.append("")
lines.append("[Key integrity]")
lines.append(f"duplicate_keys(machine_id,time_slice)={dup_keys}")
lines.append(f"time_slice_mismatch( slice_last_time_us//SLICE_US != time_slice )={ts_mismatch_cnt}")
lines.append("")
lines.append("[Monotonicity checks within machine (sorted by time_slice)]")
lines.append(f"non_positive_diff_time_slice_count={nonpos_ts_cnt}  (should be 0 if key unique & sorted)")
lines.append(f"negative_diff_slice_last_time_us_count={neg_dt_cnt}  (should be 0; otherwise label risk)")
lines.append(f"zero_diff_slice_last_time_us_count={zero_dt_cnt}     (ideally 0; ties may indicate duplicates)")
lines.append(f"machines_with_any_negative_dt={mach_neg_dt_cnt} / {mach_total} = {mach_neg_dt_cnt/mach_total:.6f}")
lines.append(f"machines_with_any_zero_dt={mach_zero_dt_cnt} / {mach_total} = {mach_zero_dt_cnt/mach_total:.6f}")
lines.append("")
lines.append("[Positive dt (slice_last_time_us) gap distribution (hours) — reflects observation sparsity]")
if len(pos_dt_hours):
    q = pos_dt_hours.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    lines.append(f"pos_dt_count={len(pos_dt_hours)}")
    lines.append("quantiles_hours=" + str({k: float(v) for k, v in q.items()}))
    lines.append(f"mean_hours={float(pos_dt_hours.mean()):.4f}")
    lines.append(f"std_hours={float(pos_dt_hours.std()):.4f}")
    lines.append(f"min_hours={float(pos_dt_hours.min()):.4f}")
    lines.append(f"max_hours={float(pos_dt_hours.max()):.4f}")

else:
    lines.append("pos_dt_count=0 (unexpected)")

# ========== 补充：slice_last_time_us 与固定末端偏差分布 ==========
df_sorted["slice_end_fixed_us"] = (df_sorted[TS] + 1) * SLICE_US
df_sorted["delta_end_us"] = df_sorted["slice_end_fixed_us"] - df_sorted[T]
delta_end_sec = df_sorted["delta_end_us"] / 1_000_000

neg_delta = (delta_end_sec < 0).sum()
assert neg_delta == 0, f"❌ delta_end < 0 出现 {neg_delta} 行，slice_last_time_us 超出 slice 边界"

delta_end_positive = delta_end_sec[delta_end_sec >= 0]
if len(delta_end_positive) > 0:
    q_delta = delta_end_positive.quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_dict()
else:
    q_delta = {}

lines.append("")
lines.append("[Delta end: fixed_slice_end - slice_last_time_us (seconds)]")
lines.append(f"delta_end_negative_count={neg_delta}")
if q_delta:
    lines.append("quantiles_seconds=" + str(q_delta))
else:
    lines.append("quantiles_seconds=no valid data")

lines.append("")
lines.append("[Outputs]")
lines.append(f"machine_level_csv={OUT_MACH}")
lines.append(f"examples_csv={OUT_EX}")
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n[输出保存]", OUT_TXT)
print("[输出保存]", OUT_MACH)
print("[输出保存]", OUT_EX)
print("\n".join(lines))       # 打印完整报告
print("=" * 90)

# -------- 建议的硬阈值（审稿一票否决类）--------
# 1) duplicate_keys > 0 或 time_slice_mismatch > 0：说明 clean 构建逻辑有 bug，必须回滚 Step2a 修复
# 2) negative_dt_count > 0：说明时间轴出现回跳，next-fail 的 searchsorted 前提被破坏，需要明确处理策略
# ------------------------------------------------