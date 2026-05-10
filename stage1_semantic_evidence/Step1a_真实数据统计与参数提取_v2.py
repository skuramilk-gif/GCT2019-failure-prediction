# ============================================================
# 文件名：Step1a_真实数据统计与参数提取_v2.py
# 阶段：Step 1A（v2，Docs-only，修法A：双域计数）
#
# 作用：
#   1) 解析 raw 的 JSON/结构化字段，输出关键字段分布统计
#   2) 输出 event/failed 一致性证据（failed 与 event==FAIL 的等价性）
#   3) 输出 CPI/MPI 缺失模式（按 event / failed 分组）
#   4) 输出 JSON 字段 missing / parse_fail（区分缺失与解析失败）
#   5) 同时输出 RAW_ALL 与 VALID_LIKE_STEP2A 两套口径：
#        VALID_LIKE_STEP2A 过滤与 Step2a 一致：machine_id>=0 且 0<time<INT64_MAX
#
# 输入：
#   00_raw/borg_traces_data.csv
#
# 输出（05_reports/）：
#   RAW_ALL:
#     step1a_basic_stats_v2.csv
#     step1a_event_failed_summary_v2.csv
#     step1a_missing_by_event_failed_v2.csv
#     step1a_parse_fail_report_v2.csv
#
#   VALID_LIKE_STEP2A（对齐训练域口径）：
#     step1a_basic_stats_valid_v2.csv
#     step1a_event_failed_summary_valid_v2.csv
#     step1a_missing_by_event_failed_valid_v2.csv
#     step1a_parse_fail_report_valid_v2.csv
#
#   总览：
#     step1a_summary_report_v2.txt
#
# 关键修正（修法A）：
#   - JSON missing/parse_fail 统计在 chunk 内同时维护两套 domain 计数器，
#     因此 valid 域的 total/missing/parse_fail 是真实的（不会再错误等于 RAW_ALL）。
# ============================================================

import os
import pandas as pd
import numpy as np
import ast
from collections import defaultdict

# ============================================================
# 0) 路径设置
# ============================================================
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

OUT_BASIC_STATS_ALL   = os.path.join(REPORT_DIR, "step1a_basic_stats_v2.csv")
OUT_EVENT_FAILED_ALL  = os.path.join(REPORT_DIR, "step1a_event_failed_summary_v2.csv")
OUT_MISSING_GROUP_ALL = os.path.join(REPORT_DIR, "step1a_missing_by_event_failed_v2.csv")
OUT_PARSE_FAIL_ALL    = os.path.join(REPORT_DIR, "step1a_parse_fail_report_v2.csv")

OUT_BASIC_STATS_VALID   = os.path.join(REPORT_DIR, "step1a_basic_stats_valid_v2.csv")
OUT_EVENT_FAILED_VALID  = os.path.join(REPORT_DIR, "step1a_event_failed_summary_valid_v2.csv")
OUT_MISSING_GROUP_VALID = os.path.join(REPORT_DIR, "step1a_missing_by_event_failed_valid_v2.csv")
OUT_PARSE_FAIL_VALID    = os.path.join(REPORT_DIR, "step1a_parse_fail_report_valid_v2.csv")

OUT_SUMMARY_TXT = os.path.join(REPORT_DIR, "step1a_summary_report_v2.txt")

INT64_MAX = 9223372036854775807
CHUNKSIZE = 200_000

# ============================================================
# 1) JSON 解析：双域统计计数器（RAW_ALL vs VALID_LIKE_STEP2A）
# ============================================================
DOMAINS = ["RAW_ALL", "VALID_LIKE_STEP2A"]
parse_total = {d: defaultdict(int) for d in DOMAINS}
parse_missing = {d: defaultdict(int) for d in DOMAINS}
parse_fail = {d: defaultdict(int) for d in DOMAINS}

def parse_dict_field(val, field_name: str, is_valid: bool):
    """
    解析形如 {'cpus': 0.01, 'memory': 0.02} 的字段。
    双域计数：
      - RAW_ALL：每行都计数
      - VALID_LIKE_STEP2A：仅当该行满足 valid_like_step2a 时计数
    """
    # RAW_ALL 一定计数
    parse_total["RAW_ALL"][field_name] += 1
    if is_valid:
        parse_total["VALID_LIKE_STEP2A"][field_name] += 1

    # missing（不计为 parse_fail）
    if pd.isna(val) or str(val).strip().lower() in {"", "nan", "none"}:
        parse_missing["RAW_ALL"][field_name] += 1
        if is_valid:
            parse_missing["VALID_LIKE_STEP2A"][field_name] += 1
        return np.nan, np.nan

    try:
        d = ast.literal_eval(str(val))
        if not isinstance(d, dict):
            parse_fail["RAW_ALL"][field_name] += 1
            if is_valid:
                parse_fail["VALID_LIKE_STEP2A"][field_name] += 1
            return np.nan, np.nan
        cpu = d.get("cpus", np.nan)
        mem = d.get("memory", np.nan)
        if cpu is None: cpu = np.nan
        if mem is None: mem = np.nan
        return cpu, mem
    except Exception:
        parse_fail["RAW_ALL"][field_name] += 1
        if is_valid:
            parse_fail["VALID_LIKE_STEP2A"][field_name] += 1
        return np.nan, np.nan

def save_csv_and_preview(df: pd.DataFrame, path: str, name: str, head_n: int = 10):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[输出保存] {name}: {path}")
    print(f"[预览] {name} shape={df.shape}")
    print(df.head(head_n).to_string(index=False))

def basic_stats_table(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    rows = []
    for col in indicators:
        x = pd.to_numeric(df[col], errors="coerce")
        miss_rate = float(x.isna().mean())
        vals = x.dropna()
        if len(vals) == 0:
            rows.append({
                "col": col, "count": 0, "missing_rate": miss_rate,
                "mean": np.nan, "std": np.nan, "min": np.nan,
                "p25": np.nan, "p50": np.nan, "p75": np.nan,
                "p90": np.nan, "p95": np.nan, "p99": np.nan,
                "max": np.nan, "zero_pct": np.nan
            })
            continue

        rows.append({
            "col": col,
            "count": int(len(vals)),
            "missing_rate": miss_rate,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "p25": float(vals.quantile(0.25)),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
            "p95": float(vals.quantile(0.95)),
            "p99": float(vals.quantile(0.99)),
            "max": float(vals.max()),
            "zero_pct": float((vals == 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("col").reset_index(drop=True)

def event_failed_summary(df: pd.DataFrame):
    """
    在 failed∈{0,1} 子域上输出 event x failed 的计数与 failed_rate，并返回 same_rate。
    """
    df = df.copy()
    df["event"] = df["event"].astype(str)
    df["failed01"] = pd.to_numeric(df["failed"], errors="coerce").fillna(-1).astype(int)

    mask01 = df["failed01"].isin([0, 1])
    df01 = df.loc[mask01, ["event", "failed01"]].copy()
    n01 = int(len(df01))
    if n01 == 0:
        return pd.DataFrame(), np.nan, 0

    same_rate = float((df01["failed01"] == (df01["event"] == "FAIL").astype(int)).mean())

    ct = pd.crosstab(df01["event"], df01["failed01"], dropna=False).reset_index()
    ct = ct.rename(columns={"event": "event_name", 0: "failed0_count", 1: "failed1_count"})
    if "failed0_count" not in ct.columns: ct["failed0_count"] = 0
    if "failed1_count" not in ct.columns: ct["failed1_count"] = 0

    ct["event_count"] = ct["failed0_count"] + ct["failed1_count"]
    ct["event_pct"] = ct["event_count"] / ct["event_count"].sum()
    ct["failed_rate_in_event"] = ct["failed1_count"] / ct["event_count"].replace(0, np.nan)

    summary = pd.DataFrame([{
        "event_name": "__GLOBAL__",
        "failed0_count": int((df01["failed01"] == 0).sum()),
        "failed1_count": int((df01["failed01"] == 1).sum()),
        "event_count": int(len(df01)),
        "event_pct": 1.0,
        "failed_rate_in_event": float(df01["failed01"].mean()),
        "same_rate_failed_equals_eventFAIL": same_rate,
    }])

    out = pd.concat([summary, ct], ignore_index=True)
    return out, same_rate, n01

def missing_by_event_failed(df: pd.DataFrame) -> pd.DataFrame:
    """
    只在 failed∈{0,1} 子域上输出 CPI/MPI 缺失率（按 event/failed 分组）。
    """
    tmp = df.copy()
    tmp["event"] = tmp["event"].astype(str)
    tmp["failed01"] = pd.to_numeric(tmp["failed"], errors="coerce").fillna(-1).astype(int)

    mask01 = tmp["failed01"].isin([0, 1])
    tmp = tmp.loc[mask01].copy()

    tmp["cpi"] = pd.to_numeric(tmp["cpi"], errors="coerce")
    tmp["mpi"] = pd.to_numeric(tmp["mpi"], errors="coerce")
    tmp["cpi_missing"] = tmp["cpi"].isna().astype(int)
    tmp["mpi_missing"] = tmp["mpi"].isna().astype(int)

    grp = (tmp.groupby(["event", "failed01"], dropna=False)
           .agg(n=("event", "count"),
                cpi_missing_rate=("cpi_missing", "mean"),
                mpi_missing_rate=("mpi_missing", "mean"))
           .reset_index()
           .sort_values("n", ascending=False))
    return grp

def build_parse_report(domain: str) -> pd.DataFrame:
    rows = []
    # 固定字段顺序
    fields = ["average_usage", "maximum_usage", "random_sample_usage", "resource_request"]
    for field in fields:
        total = int(parse_total[domain][field])
        miss = int(parse_missing[domain][field])
        fail = int(parse_fail[domain][field])
        ok = max(total - miss - fail, 0)
        denom = max(total - miss, 1)
        rows.append({
            "domain": domain,
            "field": field,
            "total": total,
            "missing": miss,
            "parse_fail": fail,
            "ok": ok,
            "missing_rate": miss / max(total, 1),
            "parse_fail_rate_on_non_missing": fail / denom,
        })
    return pd.DataFrame(rows)

# ============================================================
# 2) 读取与解析（分块）
# ============================================================
print("=" * 90)
print("Step 1A(v2)：读取并解析原始数据（Docs-only，修法A双域计数）")
print("=" * 90)
print(f"[输入] {RAW_PATH}")
print(f"[输出目录] {REPORT_DIR}")

usecols = [
    "machine_id", "time", "start_time", "end_time",
    "average_usage", "maximum_usage", "random_sample_usage", "resource_request",
    "assigned_memory", "page_cache_memory",
    "cycles_per_instruction", "memory_accesses_per_instruction",
    "sample_rate",
    "event", "failed", "instance_events_type",
    "priority", "scheduling_class", "cluster",
]

all_parts = []
chunk_count = 0
total_rows = 0
valid_rows_counter = 0  # 用于核对 VALID_LIKE_STEP2A 行数是否等于 349420

for chunk in pd.read_csv(RAW_PATH, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False):
    chunk_count += 1
    total_rows += len(chunk)
    print(f"  处理第{chunk_count}块（{len(chunk)}行），累计 {total_rows} 行...")

    # 先算 valid mask（与 Step2a 一致）
    mid = pd.to_numeric(chunk["machine_id"], errors="coerce")
    t = pd.to_numeric(chunk["time"], errors="coerce")
    valid_mask = mid.notna() & (mid >= 0) & t.notna() & (t > 0) & (t < INT64_MAX)
    valid_arr = valid_mask.to_numpy(dtype=bool)
    valid_rows_counter += int(valid_mask.sum())

    rec = pd.DataFrame()
    rec["machine_id"] = chunk["machine_id"]
    rec["time"] = pd.to_numeric(chunk["time"], errors="coerce")
    rec["start_time"] = pd.to_numeric(chunk["start_time"], errors="coerce")
    rec["end_time"] = pd.to_numeric(chunk["end_time"], errors="coerce")

    # JSON: average_usage（双域计数）
    cpu_list, mem_list = [], []
    for v, isv in zip(chunk["average_usage"].to_list(), valid_arr):
        c, m = parse_dict_field(v, "average_usage", isv)
        cpu_list.append(c); mem_list.append(m)
    rec["cpu_avg"] = cpu_list
    rec["mem_avg"] = mem_list

    # JSON: maximum_usage
    cpu_list, mem_list = [], []
    for v, isv in zip(chunk["maximum_usage"].to_list(), valid_arr):
        c, m = parse_dict_field(v, "maximum_usage", isv)
        cpu_list.append(c); mem_list.append(m)
    rec["cpu_max"] = cpu_list
    rec["mem_max"] = mem_list

    # JSON: random_sample_usage
    cpu_list, mem_list = [], []
    for v, isv in zip(chunk["random_sample_usage"].to_list(), valid_arr):
        c, m = parse_dict_field(v, "random_sample_usage", isv)
        cpu_list.append(c); mem_list.append(m)
    rec["cpu_random"] = cpu_list
    rec["mem_random"] = mem_list

    # JSON: resource_request
    cpu_list, mem_list = [], []
    for v, isv in zip(chunk["resource_request"].to_list(), valid_arr):
        c, m = parse_dict_field(v, "resource_request", isv)
        cpu_list.append(c); mem_list.append(m)
    rec["cpu_request"] = cpu_list
    rec["mem_request"] = mem_list

    # 数值字段
    rec["mem_assigned"] = pd.to_numeric(chunk["assigned_memory"], errors="coerce")
    rec["mem_page_cache"] = pd.to_numeric(chunk["page_cache_memory"], errors="coerce")
    rec["cpi"] = pd.to_numeric(chunk["cycles_per_instruction"], errors="coerce")
    rec["mpi"] = pd.to_numeric(chunk["memory_accesses_per_instruction"], errors="coerce")
    rec["sample_rate"] = pd.to_numeric(chunk["sample_rate"], errors="coerce")

    # 分类/标签相关字段
    rec["event"] = chunk["event"].astype(str)
    rec["failed"] = pd.to_numeric(chunk["failed"], errors="coerce")
    rec["instance_events_type"] = pd.to_numeric(chunk["instance_events_type"], errors="coerce")
    rec["priority"] = pd.to_numeric(chunk["priority"], errors="coerce")
    rec["scheduling_class"] = pd.to_numeric(chunk["scheduling_class"], errors="coerce")
    rec["cluster"] = pd.to_numeric(chunk["cluster"], errors="coerce")

    all_parts.append(rec)

df = pd.concat(all_parts, ignore_index=True)
print(f"\n[完成] 解析完成，总行数：{len(df)}")
print(f"[完成] 机器数（unique machine_id）：{df['machine_id'].nunique()}")

# ============================================================
# 2.1 域对齐（raw_all vs valid_like_step2a）
# ============================================================
df_all = df.copy()
mid_num = pd.to_numeric(df_all["machine_id"], errors="coerce")
t_num = pd.to_numeric(df_all["time"], errors="coerce")
evt = df_all["event"].astype(str)

mask_valid = (
    mid_num.notna() & (mid_num >= 0) &
    t_num.notna() & (t_num > 0) & (t_num < INT64_MAX)
)
df_valid = df_all.loc[mask_valid].copy()

# Gate：valid 行数应与 chunk 内计数一致
assert int(mask_valid.sum()) == valid_rows_counter, \
    f"❌ valid rows mismatch: post_concat={int(mask_valid.sum())} chunk_counter={valid_rows_counter}"

cnt_invalid_machine = int((mid_num.isna() | (mid_num < 0)).sum())
cnt_time_nan = int(t_num.isna().sum())
cnt_time_nonpos = int((t_num.notna() & (t_num <= 0)).sum())
cnt_time_ge_intmax = int((t_num.notna() & (t_num >= INT64_MAX)).sum())
cnt_time_nonpos_fail = int((t_num.notna() & (t_num <= 0) & (evt == "FAIL")).sum())

filter_evidence = {
    "raw_rows_total": int(len(df_all)),
    "rows_invalid_machine(machine_id<0 or NaN)": cnt_invalid_machine,
    "rows_time_nan": cnt_time_nan,
    "rows_time_nonpos(time<=0)": cnt_time_nonpos,
    "rows_time_ge_INT64_MAX": cnt_time_ge_intmax,
    "rows_time_nonpos_and_FAIL": cnt_time_nonpos_fail,
    "valid_like_step2a_rows": int(len(df_valid)),
    "valid_like_step2a_rate": float(len(df_valid) / max(len(df_all), 1)),
}

print(f"[域对齐] raw_all_rows={len(df_all)}  valid_like_step2a_rows={len(df_valid)}  "
      f"valid_rate={len(df_valid)/max(len(df_all),1):.6f}")
# ============================================================
# 2.2) (machine_id, time) 唯一性检查
# ============================================================
dup_count = int(df_valid.duplicated(subset=["machine_id", "time"]).sum())
dup_rate = dup_count / max(len(df_valid), 1)
print(f"\n[唯一性] valid 域 (machine_id, time) 重复行数={dup_count}  重复率={dup_rate:.6f}")
if dup_count > 0:
    print("  ⚠️ 存在重复，后续 slice 聚合可能受污染，需检查 Step2A 的 groupby 逻辑")

# ============================================================
# 2.3) 时间范围与跨度
# ============================================================
time_valid = df_valid["time"].dropna()
time_min_us = float(time_valid.min())
time_max_us = float(time_valid.max())
span_hours = (time_max_us - time_min_us) / 1e6 / 3600.0
print(f"\n[时间范围] time_min={time_min_us:.0f}  time_max={time_max_us:.0f}  span={span_hours:.2f} hours")

start_valid = pd.to_numeric(df_valid["start_time"], errors="coerce").dropna()
end_valid = pd.to_numeric(df_valid["end_time"], errors="coerce").dropna()
if len(start_valid) > 0 and len(end_valid) > 0:
    print(f"[时间范围] start_time min={start_valid.min():.0f} max={start_valid.max():.0f}")
    print(f"[时间范围] end_time   min={end_valid.min():.0f} max={end_valid.max():.0f}")

# ============================================================
# 2.4) failed 字段值域审计
# ============================================================
failed_raw = pd.to_numeric(df_valid["failed"], errors="coerce")
failed_dist = failed_raw.value_counts(dropna=False).reset_index()
failed_dist.columns = ["failed_value", "count"]
failed_dist["pct"] = failed_dist["count"] / len(df_valid)
print(f"\n[标签审计] failed 值域分布（VALID 域）:")
print(failed_dist.to_string(index=False))

n_failed_valid = int(failed_raw.isin([0, 1]).sum())
n_failed_other = int(len(df_valid) - n_failed_valid)
print(f"  failed∈{{0,1}} 行数={n_failed_valid}  非 {{0,1}} 行数={n_failed_other}")

# ============================================================
# 2.5) cluster 字段审计
# ============================================================
cluster_valid = df_valid["cluster"].value_counts(dropna=False).reset_index()
cluster_valid.columns = ["cluster_value", "count"]
print(f"\n[cluster 审计] VALID 域 cluster 分布:")
print(cluster_valid.to_string(index=False))


# ============================================================
# 3) 输出：basic stats（RAW_ALL + VALID）
# ============================================================
print("\n" + "=" * 90)
print("Step 1A.1：关键数值字段分布统计（RAW_ALL & VALID_LIKE_STEP2A）")
print("=" * 90)

indicators = [
    "cpu_avg", "cpu_max", "cpu_random",
    "mem_avg", "mem_max", "mem_random",
    "cpu_request", "mem_request",
    "mem_assigned", "mem_page_cache",
    "cpi", "mpi",
    "sample_rate", "priority"
]

basic_all = basic_stats_table(df_all, indicators)
save_csv_and_preview(basic_all, OUT_BASIC_STATS_ALL, "step1a_basic_stats_v2.csv (RAW_ALL)", head_n=20)

basic_valid = basic_stats_table(df_valid, indicators)
save_csv_and_preview(basic_valid, OUT_BASIC_STATS_VALID, "step1a_basic_stats_valid_v2.csv (VALID_LIKE_STEP2A)", head_n=20)

# CPI/MPI 缺失率摘要（两域）
cpi_miss_all = float(pd.to_numeric(df_all["cpi"], errors="coerce").isna().mean())
mpi_miss_all = float(pd.to_numeric(df_all["mpi"], errors="coerce").isna().mean())
cpi_miss_valid = float(pd.to_numeric(df_valid["cpi"], errors="coerce").isna().mean())
mpi_miss_valid = float(pd.to_numeric(df_valid["mpi"], errors="coerce").isna().mean())

print("\n[摘要] CPI/MPI missing_rate:")
print("  RAW_ALL:  cpi=", round(cpi_miss_all, 6), " mpi=", round(mpi_miss_all, 6))
print("  VALID  :  cpi=", round(cpi_miss_valid, 6), " mpi=", round(mpi_miss_valid, 6))

# ============================================================
# 4) 输出：event/failed 一致性与分布（failed01 子域，RAW_ALL + VALID）
# ============================================================
print("\n" + "=" * 90)
print("Step 1A.2：event/failed 分布与一致性证据（condition on failed in {0,1}）")
print("=" * 90)

event_failed_all, same_rate_all, n01_all = event_failed_summary(df_all)
save_csv_and_preview(event_failed_all, OUT_EVENT_FAILED_ALL, "step1a_event_failed_summary_v2.csv (RAW_ALL)", head_n=15)
print(f"[摘要] RAW_ALL same_rate failed == (event=='FAIL') on failed∈{{0,1}} = {same_rate_all:.6f}, n01={n01_all}")

event_failed_valid, same_rate_valid, n01_valid = event_failed_summary(df_valid)
save_csv_and_preview(event_failed_valid, OUT_EVENT_FAILED_VALID, "step1a_event_failed_summary_valid_v2.csv (VALID)", head_n=15)
print(f"[摘要] VALID same_rate failed == (event=='FAIL') on failed∈{{0,1}} = {same_rate_valid:.6f}, n01={n01_valid}")

# ============================================================
# 5) 输出：CPI/MPI 缺失模式（按 event/failed 分组，failed01 子域）
# ============================================================
print("\n" + "=" * 90)
print("Step 1A.3：缺失模式（按 event / failed 分组；failed∈{0,1}）")
print("=" * 90)

missing_all = missing_by_event_failed(df_all)
save_csv_and_preview(missing_all, OUT_MISSING_GROUP_ALL, "step1a_missing_by_event_failed_v2.csv (RAW_ALL)", head_n=20)

missing_valid = missing_by_event_failed(df_valid)
save_csv_and_preview(missing_valid, OUT_MISSING_GROUP_VALID, "step1a_missing_by_event_failed_valid_v2.csv (VALID)", head_n=20)

# ============================================================
# 6) 输出：JSON missing/parse_fail 报告（两域分别落盘，计数真实对齐）
# ============================================================
print("\n" + "=" * 90)
print("Step 1A.4：JSON missing/parse_fail 报告（修法A：双域真实计数）")
print("=" * 90)

pf_all = build_parse_report("RAW_ALL")
save_csv_and_preview(pf_all, OUT_PARSE_FAIL_ALL, "step1a_parse_fail_report_v2.csv (RAW_ALL)", head_n=10)

pf_valid = build_parse_report("VALID_LIKE_STEP2A")
save_csv_and_preview(pf_valid, OUT_PARSE_FAIL_VALID, "step1a_parse_fail_report_valid_v2.csv (VALID)", head_n=10)

# Gate：VALID 域 parse_report total 应等于 valid 行数（每行每字段计一次）
# 这里用 resource_request 的 total 来核对（四个字段 total 都应一致）
valid_total_rr = int(pf_valid.loc[pf_valid["field"] == "resource_request", "total"].iloc[0])
assert valid_total_rr == len(df_valid), \
    f"❌ VALID parse_report total != valid rows: total={valid_total_rr}, valid_rows={len(df_valid)}"

# ============================================================
# 7) 写总览 summary txt（便于论文引用）
# ============================================================
lines = []
lines.append("=== Step1A(v2) Summary Report (RAW_ALL vs VALID_LIKE_STEP2A; dual-domain parse counters) ===")
lines.append("")
lines.append("[Filter evidence (raw -> valid_like_step2a)]")
for k, v in filter_evidence.items():
    lines.append(f"{k}={v}")
lines.append("")
lines.append("[CPI/MPI missing rate]")
lines.append(f"RAW_ALL:  cpi_missing_rate={cpi_miss_all:.6f} mpi_missing_rate={mpi_miss_all:.6f}")
lines.append(f"VALID  :  cpi_missing_rate={cpi_miss_valid:.6f} mpi_missing_rate={mpi_miss_valid:.6f}")
lines.append("")
lines.append("[failed equivalence evidence (condition on failed in {0,1})]")
lines.append(f"RAW_ALL: same_rate_failed_equals_eventFAIL={same_rate_all:.6f}  n01={n01_all}")
lines.append(f"VALID  : same_rate_failed_equals_eventFAIL={same_rate_valid:.6f}  n01={n01_valid}")
lines.append("")
lines.append("[JSON missing/parse_fail]")
lines.append("RAW_ALL parse report: " + OUT_PARSE_FAIL_ALL)
lines.append("VALID  parse report: " + OUT_PARSE_FAIL_VALID)
lines.append("")
lines.append("[(machine_id, time) uniqueness]")
lines.append(f"duplicate_rows={dup_count}  duplicate_rate={dup_rate:.6f}")
lines.append("")
lines.append("[Time range]")
lines.append(f"time_min_us={time_min_us:.0f}  time_max_us={time_max_us:.0f}  span_hours={span_hours:.2f}")
lines.append("")
lines.append("[Failed field audit (VALID domain)]")
lines.append(f"n_failed_in_01={n_failed_valid}  n_failed_other={n_failed_other}")
lines.append("")
lines.append("[Cluster field]")
lines.append(f"unique_clusters={int(df_valid['cluster'].nunique())}")
lines.append("[Outputs]")
lines.append(OUT_BASIC_STATS_ALL)
lines.append(OUT_BASIC_STATS_VALID)
lines.append(OUT_EVENT_FAILED_ALL)
lines.append(OUT_EVENT_FAILED_VALID)
lines.append(OUT_MISSING_GROUP_ALL)
lines.append(OUT_MISSING_GROUP_VALID)
lines.append(OUT_PARSE_FAIL_ALL)
lines.append(OUT_PARSE_FAIL_VALID)
lines.append(OUT_SUMMARY_TXT)

with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n" + "=" * 90)
print("Step 1A(v2) 完成（控制台摘要）")
print("=" * 90)
print(f"raw_all_rows={len(df_all)} valid_like_step2a_rows={len(df_valid)}")
print(f"RAW_ALL same_rate_failed_equals_eventFAIL(on01)={same_rate_all:.6f}  VALID={same_rate_valid:.6f}")
print("[输出 summary]", OUT_SUMMARY_TXT)
print("=" * 90)