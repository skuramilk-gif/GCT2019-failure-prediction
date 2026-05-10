# ============================================================
# 文件名：Step2a_machine_slice主表构建_clean_v2.py
# 阶段：Step 2A（v2 修正版）
# 作用：
#   1) 从原始 borg_traces_data.csv 构建 machine-slice clean 主表（machine_id + time_slice）
#   2) time_slice = floor(clean_time / 300s)，其中 clean_time 使用 raw 的 time
#   3) 【关键修正】过滤 time_us <= 0（特别是 time==0 的 FAIL 边界效应）谷歌白皮书该情况为快速回照，也就是系统开始记录时会快速记录前一段时间的发生的FALL
#   4) 过滤 machine_id < 0（-1 哨兵）
#   5) 聚合得到 slice 级资源统计（request/average/maximum）
#   6) 聚合得到 CPI/MPI（保留缺失：count=0 -> mean=NaN，并输出 missing 指示列）
#   7) 聚合事件计数：
#        - failed_count: event==FAIL
#        - evict/lost/kill_count: non-FAIL 异常事件计数（用于 event 特征层，不作为标签）
#   8) 输出 slice_last_time_us = 该 slice 内 time 的 max（用于后续 H 标签与 censoring）
#指标解释：raw_rows_total-原始 CSV 文件中有效行数（初步清洗后，排除完全无效的行）
#rows_dropped_invalid_machine-因为 machine_id 无效（如 NaN 或负数）而被丢弃的行数。你之前发现 machine_id=-1 的 19 行，这里就是它们。
#output_rows (machine-slice)-经过聚合后，最终输出的 machine-slice 级别的行数。
#time_slice_max-有 slice 中最大的 time_slice 索引。因为每个 slice 是 300 秒，最大索引 8929 对应的时间跨度约为 8929×300 秒 ≈ 31 天
#
# 输入：
#   00_raw/borg_traces_data.csv
#
# 输出：
#   01_interim/step2a_machine_slice_clean_v2.csv #后面大部分质检、构建都基于这个表。
#
# 报告输出（05_reports/）：
#   step2a_build_log_v2.txt
#   step2a_time_slice_outliers_v2.csv（若存在异常 time_slice）
#
# 说明：
#   本脚本只做 clean 主表构建（可复现、无泄漏），不做任何阈值拟合/分位数/标准化/填充。
#论文注意
# ============================================================

import os
import pandas as pd
import numpy as np
import ast
from collections import defaultdict

# ============================================================
# 0) 路径与常量
# ============================================================
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")

INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

OUT_CLEAN = os.path.join(INTERIM_DIR, "step2a_machine_slice_clean_v2.csv")
OUT_LOG = os.path.join(REPORT_DIR, "step2a_build_log_v2.txt")
OUT_OUTLIERS = os.path.join(REPORT_DIR, "step2a_time_slice_outliers_v2.csv")

INT64_MAX = 9223372036854775807
SLICE_US = 300_000_000  # 300s
CHUNKSIZE = 200_000

# ============================================================
# 1) 工具函数
# ============================================================
parse_fail_counter = defaultdict(int)
parse_total_counter = defaultdict(int)
#定义两个 defaultdict，用于统计每个 JSON 字段的解析总次数和失败次数。键是字段名，值默认为整数 0

def parse_usage_dict(val, field_name: str):
    parse_total_counter[field_name] += 1
    try:
        d = ast.literal_eval(str(val))
        if not isinstance(d, dict):
            parse_fail_counter[field_name] += 1
            return np.nan, np.nan
        cpu = d.get("cpus", np.nan)
        mem = d.get("memory", np.nan)
        if cpu is None: cpu = np.nan
        if mem is None: mem = np.nan
        return cpu, mem
    except Exception:
        parse_fail_counter[field_name] += 1
        return np.nan, np.nan

def safe_mean(sum_s, cnt_s):
    return np.where(cnt_s > 0, sum_s / cnt_s, np.nan)

# ============================================================
# 2) 分块读取与块内聚合
# ============================================================
print("=" * 90)
print("Step 2A(v2 修正版)：machine-slice clean 主表构建（过滤 time<=0 & machine_id<0）")
print("=" * 90)
print(f"[输入] {RAW_PATH}")
print(f"[输出] {OUT_CLEAN}")

usecols = [
    "machine_id", "time",
    "average_usage", "maximum_usage", "resource_request",
    "assigned_memory", "page_cache_memory",
    "cycles_per_instruction", "memory_accesses_per_instruction",
    "event",
]

agg_parts = []
chunk_count = 0
rows_raw_total = 0
#agg_parts 列表用于存储每个数据块聚合后的局部结果。chunk_count 和 rows_raw_total 用于进度追踪。

# 统计剔除原因（用于 build log）
rows_drop_invalid_machine = 0 #machine_id 无效（NaN 或负数）丢弃
rows_drop_time_nonpos = 0 #time 小于等于 0。丢弃
rows_drop_time_ge_intmax = 0 #time 大于等于 INT64_MAX（哨兵值）。丢弃
rows_drop_time_nan = 0 #time 为 NaN。丢弃
rows_drop_time_other = 0 #其他异常时间（理论上不会出现，仅作保险）。丢弃
rows_drop_time_nonpos_FAIL = 0  # 关键：time<=0 & FAIL丢弃，
#time<=0 & FAIL这一步是解决分布式系统的状态快照逻辑（brog会定期对集群所有活跃机器做一次全量快照）

for chunk in pd.read_csv(RAW_PATH, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
    chunk_count += 1
    rows_raw_total += len(chunk)
    print(f"  处理第{chunk_count}块（{len(chunk)}行），累计 {rows_raw_total} 行...")#更新计数器并打印进度。

    machine_id = pd.to_numeric(chunk["machine_id"], errors="coerce")
    time_us = pd.to_numeric(chunk["time"], errors="coerce")
    evt = chunk["event"].astype(str) #将 machine_id 和 time 转为数值类型，无法转换的变为 NaN。event 统一转为字符串，防止意外类型。

    valid_machine = machine_id.notna() & (machine_id >= 0) #定义机器 ID 有效的布尔条件：非空且大于等于 0。-1 是哨兵值（专用机器），因此被排除。

    # ---- time 细分剔除原因 ----
    time_nan = time_us.isna()
    time_nonpos = time_us.notna() & (time_us <= 0)              # ★关键修正：<=0 全剔除
    time_ge_max = time_us.notna() & (time_us >= INT64_MAX)      #无穷达大的也要去掉
    time_other_invalid = time_us.notna() & (~time_nonpos) & (~time_ge_max) & (time_us < 0)  # 其他负数情况（理论上已被time_nonpos覆盖），留作保险

    rows_drop_invalid_machine += int((~valid_machine).sum())
    # 只在 machine 有效时统计 time 剔除（更直观）
    rows_drop_time_nan += int((valid_machine & time_nan).sum())
    rows_drop_time_nonpos += int((valid_machine & time_nonpos).sum())
    rows_drop_time_ge_intmax += int((valid_machine & time_ge_max).sum())
    rows_drop_time_other += int((valid_machine & time_other_invalid).sum())
    #这里四个对应前四个的剔除，也就是统计时间剔除在机器id前提下因时间原因被丢弃的行数
    rows_drop_time_nonpos_FAIL += int((valid_machine & time_nonpos & (evt == "FAIL")).sum())
    #专门统计时间小于等于 0 且为 FAIL 事件的行数，

    valid_time = time_us.notna() & (time_us > 0) & (time_us < INT64_MAX)
    valid = valid_machine & valid_time#最终有效条件：机器 ID 有效，且时间戳严格大于 0 且小于 INT64_MAX。

    chunk = chunk.loc[valid].copy()
    chunk["machine_id"] = machine_id.loc[valid].astype("int64")
    chunk["time_us"] = time_us.loc[valid].astype("int64")
    #保留有效行，并将 machine_id 和 time_us 赋值为已清洗的版本，转为 64 位整数。

    # time_slice
    chunk["time_slice"] = (chunk["time_us"] // SLICE_US).astype("int64")
    #用整除计算时间片索引。SLICE_US 是 300,000,000 微秒（300 秒），因此 time_slice 代表从时间零点开始的第几个完整 5 分钟窗口。

    # JSON 解析
    avg = chunk["average_usage"].apply(lambda x: parse_usage_dict(x, "average_usage"))
    mx  = chunk["maximum_usage"].apply(lambda x: parse_usage_dict(x, "maximum_usage"))
    req = chunk["resource_request"].apply(lambda x: parse_usage_dict(x, "resource_request"))

    chunk["avg_cpu"] = [p[0] for p in avg]
    chunk["avg_mem"] = [p[1] for p in avg]
    chunk["max_cpu"] = [p[0] for p in mx]
    chunk["max_mem"] = [p[1] for p in mx]
    chunk["req_cpu"] = [p[0] for p in req]
    chunk["req_mem"] = [p[1] for p in req]

    # 数值字段
    chunk["assigned_memory"] = pd.to_numeric(chunk["assigned_memory"], errors="coerce")
    chunk["page_cache_memory"] = pd.to_numeric(chunk["page_cache_memory"], errors="coerce")
    chunk["cpi"] = pd.to_numeric(chunk["cycles_per_instruction"], errors="coerce")
    chunk["mpi"] = pd.to_numeric(chunk["memory_accesses_per_instruction"], errors="coerce")
    #将这四个列统一转为数值类型，无效值变为 NaN。

    # 事件
    evt2 = chunk["event"].astype(str)
    chunk["is_fail"] = (evt2 == "FAIL").astype("int64")
    chunk["is_evict"] = (evt2 == "EVICT").astype("int64")
    chunk["is_lost"] = (evt2 == "LOST").astype("int64")
    chunk["is_kill"] = (evt2 == "KILL").astype("int64")
    #根据 event 字符串生成四个 0/1 指示列，方便后续聚合计数。

    g = chunk.groupby(["machine_id", "time_slice"], sort=False)
    #按 machine_id 和 time_slice 分组，同一个时间片内同一台机器上的所有任务记录将被归入一组。

    part = pd.DataFrame({
        "slice_last_time_us": g["time_us"].max(),

        "req_cpu_sum": g["req_cpu"].sum(min_count=1),
        "req_cpu_cnt": g["req_cpu"].count(),
        "req_mem_sum": g["req_mem"].sum(min_count=1),
        "req_mem_cnt": g["req_mem"].count(),

        "avg_cpu_sum": g["avg_cpu"].sum(min_count=1),
        "avg_cpu_cnt": g["avg_cpu"].count(),
        "avg_mem_sum": g["avg_mem"].sum(min_count=1),
        "avg_mem_cnt": g["avg_mem"].count(),

        "max_cpu_max": g["max_cpu"].max(),
        "max_mem_max": g["max_mem"].max(),

        "assigned_mem_sum": g["assigned_memory"].sum(min_count=1),
        "assigned_mem_cnt": g["assigned_memory"].count(),
        "page_cache_sum": g["page_cache_memory"].sum(min_count=1),
        "page_cache_cnt": g["page_cache_memory"].count(),

        "cpi_sum": g["cpi"].sum(min_count=1),
        "cpi_cnt": g["cpi"].count(),
        "mpi_sum": g["mpi"].sum(min_count=1),
        "mpi_cnt": g["mpi"].count(),

        "record_count": g.size(),
        "failed_count": g["is_fail"].sum(),
        "evict_count": g["is_evict"].sum(),
        "lost_count": g["is_lost"].sum(),
        "kill_count": g["is_kill"].sum(),
    }).reset_index()

    agg_parts.append(part)#将本块的聚合结果添加到列表中，待所有块处理完毕后统一合并。

# ============================================================
# 3) 合并块级聚合结果（同 key 再聚合一次）
# ============================================================
print("\n" + "=" * 90)
print("Step 2A(v2)：合并块级聚合结果")
print("=" * 90)

agg_all = pd.concat(agg_parts, ignore_index=True)
g2 = agg_all.groupby(["machine_id", "time_slice"], sort=False)
#对拼接后的全量数据，再次按 machine_id 和 time_slice 分组。sort=False 表示不排序，以提高性能。这次分组的目的是将同一个时间片被拆开的多个局部结果重新合并。

merged = pd.DataFrame({
    "slice_last_time_us": g2["slice_last_time_us"].max(),
#对于 slice_last_time_us，取组内的最大值。因为一个时间片内的所有记录，其最晚采集时间必定是其中最大的那个。用 max 可以正确还原这个时间片的结束时刻。
    "req_cpu_sum": g2["req_cpu_sum"].sum(min_count=1),
    "req_cpu_cnt": g2["req_cpu_cnt"].sum(),
    "req_mem_sum": g2["req_mem_sum"].sum(min_count=1),
    "req_mem_cnt": g2["req_mem_cnt"].sum(),
#对于 CPU 和内存请求量的总和与计数
    "avg_cpu_sum": g2["avg_cpu_sum"].sum(min_count=1),
    "avg_cpu_cnt": g2["avg_cpu_cnt"].sum(),
    "avg_mem_sum": g2["avg_mem_sum"].sum(min_count=1),
    "avg_mem_cnt": g2["avg_mem_cnt"].sum(),

    "max_cpu_max": g2["max_cpu_max"].max(),
    "max_mem_max": g2["max_mem_max"].max(),
#取组内最大值，每个局部结果记录该局部范围内的最大值，合并只要局部最大值的最大值，就可以得到所有任务的峰值。
    "assigned_mem_sum": g2["assigned_mem_sum"].sum(min_count=1),
    "assigned_mem_cnt": g2["assigned_mem_cnt"].sum(),
    "page_cache_sum": g2["page_cache_sum"].sum(min_count=1),
    "page_cache_cnt": g2["page_cache_cnt"].sum(),

    "cpi_sum": g2["cpi_sum"].sum(min_count=1),
    "cpi_cnt": g2["cpi_cnt"].sum(),
    "mpi_sum": g2["mpi_sum"].sum(min_count=1),
    "mpi_cnt": g2["mpi_cnt"].sum(),

    "record_count": g2["record_count"].sum(),
    "failed_count": g2["failed_count"].sum(),
    "evict_count": g2["evict_count"].sum(),
    "lost_count": g2["lost_count"].sum(),
    "kill_count": g2["kill_count"].sum(),
}).reset_index()

# ============================================================
# 4) 计算 mean + missing flags + abnormal_nonfail(聚合操作）
# ============================================================
merged["ms_req_cpu_mean"] = safe_mean(merged["req_cpu_sum"], merged["req_cpu_cnt"])#计算该机器时间片内所有任务的平均 CPU 申请量
merged["ms_req_mem_mean"] = safe_mean(merged["req_mem_sum"], merged["req_mem_cnt"])
merged["ms_avg_cpu_mean"] = safe_mean(merged["avg_cpu_sum"], merged["avg_cpu_cnt"])
merged["ms_avg_mem_mean"] = safe_mean(merged["avg_mem_sum"], merged["avg_mem_cnt"])
#重点是safe_mean函数，他对[]中进行一下逻辑：1计数>0，返回总和/计数；等于0返回NaN
merged["ms_max_cpu_max"] = merged["max_cpu_max"]
merged["ms_max_mem_max"] = merged["max_mem_max"] #重命名

merged["ms_assigned_memory_mean"] = safe_mean(merged["assigned_mem_sum"], merged["assigned_mem_cnt"])
merged["ms_page_cache_memory_mean"] = safe_mean(merged["page_cache_sum"], merged["page_cache_cnt"])

merged["ms_cpi_mean"] = safe_mean(merged["cpi_sum"], merged["cpi_cnt"])
merged["ms_mpi_mean"] = safe_mean(merged["mpi_sum"], merged["mpi_cnt"])

merged["ms_cpi_missing"] = (merged["cpi_cnt"] == 0).astype("int64")#缺失 → 1；有数据 → 0。这是标准的 missing indicator。
merged["ms_cpi_coverage"] = merged["cpi_cnt"] / merged["record_count"]#计算该 slice 内 CPI 有效记录数 占 总原始记录数 的比例
merged["ms_mpi_missing"] = (merged["mpi_cnt"] == 0).astype("int64")
merged["ms_mpi_coverage"] = merged["mpi_cnt"] / merged["record_count"]
#关键决策：构造缺失指标特征的构造，我们在S1A中分析知道CPI缺失是随机的，他与时间类型、任务时长都有关联（缺失本身就可能有信息），这一步做显式特征

merged["ms_cpi_coverage"] = np.where(
    merged["record_count"] > 0,
    merged["cpi_cnt"] / merged["record_count"],
    np.nan
)
merged["ms_mpi_coverage"] = np.where(
    merged["record_count"] > 0,
    merged["mpi_cnt"] / merged["record_count"],
    np.nan
)

merged["abnormal_nonfail_count"] = merged["evict_count"] + merged["lost_count"] + merged["kill_count"]
# 关键决策;得到一个新字段abnormal_nonfail_coun,我们将系统异常与应用失败的语义分离这个字段象征系统层面的干扰信号，作为模型输入。
merged["abnormal_nonfail_flag"] = (merged["abnormal_nonfail_count"] > 0).astype("int64")
#基于上述计数生成一个二值标记，供模型需要时快速判断

# total 特征（机器级负载总量，区别于 per-instance mean）
merged["ms_req_cpu_total"] = merged["req_cpu_sum"]
merged["ms_avg_cpu_total"] = merged["avg_cpu_sum"]
merged["ms_req_mem_total"] = merged["req_mem_sum"]
merged["ms_avg_mem_total"] = merged["avg_mem_sum"]

# ============================================================
# 5) 清理中间列 + 排序 + outlier
# ============================================================
drop_cols = [
    "req_cpu_sum","req_cpu_cnt","req_mem_sum","req_mem_cnt",
    "avg_cpu_sum","avg_cpu_cnt","avg_mem_sum","avg_mem_cnt",
    "max_cpu_max","max_mem_max",
    "assigned_mem_sum","assigned_mem_cnt","page_cache_sum","page_cache_cnt",
    "cpi_sum","cpi_cnt","mpi_sum","mpi_cnt",
] #删除中间列
merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)

merged = merged.sort_values(["machine_id", "time_slice"]).reset_index(drop=True)
#按machine_id 升序、time_slice 升序对数据进行排序。后面处理数据都依赖于同一机器内事连续且有序的
# 硬门禁：(machine_id, time_slice) 必须唯一
dup_slices = int(merged.duplicated(subset=["machine_id", "time_slice"]).sum())
assert dup_slices == 0, f"❌ (machine_id, time_slice) 存在 {dup_slices} 行重复"
print(f"[门禁] (machine_id, time_slice) 唯一性检查通过，重复行数={dup_slices}")

# outlier 检查
outlier_mask = merged["time_slice"] > 20000 #创建一个布尔掩码，筛选出 time_slice 大于 20000 的行。数据最大时间片是8900左右
outliers = merged.loc[outlier_mask, ["machine_id","time_slice","record_count","failed_count"]].copy()
if len(outliers) > 0:
    outliers.to_csv(OUT_OUTLIERS, index=False, encoding="utf-8-sig")
#如果有离群点，也就是>20000的，建立csv文件，实际运行日志中显示无离群点。

# 保存
merged.to_csv(OUT_CLEAN, index=False, encoding="utf-8-sig")#清洗后的机器时间片主表保存为csv文件
#并且utf-8-sig 编码确保在 Excel 中打开时中文不乱码。这是后续所有建模工作的唯一数据源。

# build log
parse_report = []
for k in sorted(parse_total_counter.keys()):
    total = parse_total_counter[k]
    fail = parse_fail_counter[k]
    parse_report.append(f"{k}: total={total}, fail={fail}, fail_rate={fail/total if total else 0:.6f}")
#遍历在解析 JSON 字段时累积的计数器，生成每个字段的解析总数、失败数和失败率，证明我们的解析过程可靠

log_lines = []
log_lines.append("=== Step2A(v2) build log (time>0) ===")
log_lines.append(f"raw_rows_total={rows_raw_total}")
log_lines.append(f"rows_dropped_invalid_machine(machine_id<0 or NaN)={rows_drop_invalid_machine}")
log_lines.append(f"rows_dropped_time_nan(valid_machine)={rows_drop_time_nan}")
log_lines.append(f"rows_dropped_time_nonpos(time<=0, valid_machine)={rows_drop_time_nonpos}")
log_lines.append(f"rows_dropped_time_nonpos_FAIL(time<=0 & FAIL, valid_machine)={rows_drop_time_nonpos_FAIL}")
log_lines.append(f"rows_dropped_time_ge_intmax(valid_machine)={rows_drop_time_ge_intmax}")
log_lines.append(f"rows_dropped_time_other(valid_machine)={rows_drop_time_other}")
log_lines.append(f"output_rows(machine-slice)={len(merged)}")
log_lines.append(f"machines={int(merged['machine_id'].nunique())}")
log_lines.append(f"time_slice_nunique={int(merged['time_slice'].nunique())}")
log_lines.append(f"time_slice_max={int(merged['time_slice'].max())}")
log_lines.append(f"cpi_missing_rate(slice-level)={merged['ms_cpi_missing'].mean():.6f}")
log_lines.append(f"mpi_missing_rate(slice-level)={merged['ms_mpi_missing'].mean():.6f}")
log_lines.append(f"cpi_coverage_mean(slice-level)={merged['ms_cpi_coverage'].mean():.6f}")
log_lines.append(f"mpi_coverage_mean(slice-level)={merged['ms_mpi_coverage'].mean():.6f}")
log_lines.append(f"duplicate_(machine_id,time_slice)={dup_slices}")
log_lines.append("json_parse_fail_rates:")
log_lines.extend(["  " + s for s in parse_report])
log_lines.append(f"time_slice_outliers(>20000)={len(outliers)}" + (f" saved_to={OUT_OUTLIERS}" if len(outliers)>0 else ""))

with open(OUT_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("\n" + "=" * 90)
print("Step 2A(v2 修正版：time>0) 控制台摘要")
print("=" * 90)
print(f"raw_rows_total={rows_raw_total}")
print(f"rows_dropped_invalid_machine={rows_drop_invalid_machine}")
print(f"rows_dropped_time_nonpos(time<=0)={rows_drop_time_nonpos}  (FAIL={rows_drop_time_nonpos_FAIL})")
print(f"output_rows(machine-slice)={len(merged)}")
print(f"machines={int(merged['machine_id'].nunique())}")
print(f"time_slice_max={int(merged['time_slice'].max())}")
print(f"slice_last_time_us==0 rows={(merged['slice_last_time_us']==0).sum()}  (should be 0 now)")
print("\n[输出文件]")
print(" ", OUT_CLEAN)
print(" ", OUT_LOG)
print("=" * 90)
print("Step 2A(v2) 完成")

#结果解读：
#rows_dropped_invalid_machine=19，这部分是由于machine_id无效而丢弃的，根据S1B的检测应该是machine_id=-1的记录，数量极少对我们有利，
#rows_dropped_time_nonpos(time<=0)=56452  (FAIL=37082)这是最关键，验证了之前检查的边界效应，time=0存在大量历史快照信息，其中FALL
#事件占比65.7%，比例极高，这一步我们保证主表的时序纯净度。
#output_rows(machine-slice)=341291是后续建模的样本总量；machines=95947最终主表中出现的不同机器数量。S1A这数字是96174