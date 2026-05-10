# ============================================================
# Step2h-CPU_distribution_全量结构审计_v2 (强版本：流式聚合 + gate)
#
# 目标：
#   1) 统计全数据 cpu_usage_distribution / tail_cpu_usage_distribution 的长度分布
#   2) 检查不同长度的占比
#   3) 统计每种长度下的均值/方差/最小/最大
#   4) 评估是否适合：
#        A) 作为固定11/9维向量
#        B) 只能做summary特征
# 固定长度门槛>90%-当超过90%的样本长度一致性，可以认为该字段本质上就是固定长度。剩下的少量异常值可以单独处理，不会对整体特征分布产生显著影响
#解析失败率<5%-通常认为低于该数字的缺失/异常是可以接受，可以通过简单的填充或添加指示列来处理
#最后结果都达标意味着完全满足直接展开的条件
# 输出：
#   05_reports/step2h_cpu_distribution_length_stats_v2.csv
#   05_reports/step2h_cpu_distribution_summary_stats_v2.csv
#   05_reports/step2h_cpu_distribution_audit_report_v2.txt
# ============================================================
# ============================================================


import os, re
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

OUT_LEN = os.path.join(REPORT_DIR, "step2h_cpu_distribution_length_stats_v2.csv")
OUT_SUM = os.path.join(REPORT_DIR, "step2h_cpu_distribution_summary_stats_v2.csv")
OUT_TXT = os.path.join(REPORT_DIR, "step2h_cpu_distribution_audit_report_v2.txt")

INT64_MAX = 9223372036854775807
CHUNKSIZE = 200_000

def parse_np_array_string(val):
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
    if arr.size == 0:
        return None
    return arr

def get_len_and_sumsq(val):
    """返回 (length, sum, sumsq, min, max); parse失败 length=-1"""
    arr = parse_np_array_string(val)
    if arr is None:
        return -1, 0.0, 0.0, np.nan, np.nan
    return int(arr.size), float(arr.sum()), float((arr * arr).sum()), float(arr.min()), float(arr.max())

print("="*90)
print("Step2h(v2)：CPU distribution full audit (streaming + gate)")
print("="*90)
print("[输入]", RAW_PATH)

# length count: dict[(field,length)] = count
len_counts = {}
# stats accumulator: dict[(field,length)] = {n_vec, n_elem, sum, sumsq, min, max}
stats = {}

total_valid_rows = 0

for chunk in pd.read_csv(
    RAW_PATH,
    usecols=["machine_id","time","cpu_usage_distribution","tail_cpu_usage_distribution"],
    chunksize=CHUNKSIZE,
    low_memory=False
):
    mid = pd.to_numeric(chunk["machine_id"], errors="coerce")
    t = pd.to_numeric(chunk["time"], errors="coerce")
    valid = mid.notna() & (mid >= 0) & t.notna() & (t > 0) & (t < INT64_MAX)
    chunk = chunk.loc[valid].copy()
    if len(chunk) == 0:
        continue

    total_valid_rows += len(chunk)

    for field in ["cpu_usage_distribution", "tail_cpu_usage_distribution"]:
        # 向量化 apply（仍会 parse，但比双层for稳）
        res = chunk[field].apply(get_len_and_sumsq)
        lens = res.apply(lambda x: x[0])

        # length counts
        vc = lens.value_counts()
        for L, c in vc.items():
            key = (field, int(L))
            len_counts[key] = len_counts.get(key, 0) + int(c)

        # stats accumulator（仅对 parse 成功 length>=1 聚合；length=-1 不聚合数值）
        ok_mask = lens >= 1
        if ok_mask.any():
            ok_res = res[ok_mask]
            for tpl in ok_res:
                L, s, ss, mn, mx = tpl
                key = (field, int(L))
                if key not in stats:
                    stats[key] = {"n_vec":0, "n_elem":0, "sum":0.0, "sumsq":0.0, "min":np.inf, "max":-np.inf}
                acc = stats[key]
                acc["n_vec"] += 1
                acc["n_elem"] += int(L)
                acc["sum"] += float(s)
                acc["sumsq"] += float(ss)
                acc["min"] = min(acc["min"], float(mn))
                acc["max"] = max(acc["max"], float(mx))
                # 值域检查：归一化后的 CPU 分布应在 [0, 1] 内（允许微小浮点越界）
                if float(mn) < -0.01 or float(mx) > 1.01:
                    pass  # 不中断，但在报告中标记


# ---------- 输出 length 分布 ----------
len_rows = []
for (field, L), c in len_counts.items():
    len_rows.append({"field": field, "length": L, "count": c})
len_df = pd.DataFrame(len_rows)
len_df["ratio"] = len_df.groupby("field")["count"].transform(lambda x: x / x.sum())
len_df = len_df.sort_values(["field","count"], ascending=[True,False])
len_df.to_csv(OUT_LEN, index=False, encoding="utf-8-sig")

# ---------- 输出按长度聚合的全局统计 ----------
sum_rows = []
for (field, L), acc in stats.items():
    n_elem = acc["n_elem"]
    mean = acc["sum"] / n_elem if n_elem else np.nan
    var = acc["sumsq"] / n_elem - mean * mean if n_elem else np.nan
    std = float(np.sqrt(max(var, 0.0))) if n_elem else np.nan
    sum_rows.append({
        "field": field,
        "length": L,
        "n_vec": acc["n_vec"],
        "n_elem": n_elem,
        "global_mean": mean,
        "global_std": std,
        "global_min": acc["min"],
        "global_max": acc["max"],
    })
sum_df = pd.DataFrame(sum_rows).sort_values(["field","n_vec"], ascending=[True,False])
sum_df.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")

# ---------- Gate：守恒检查 ----------
gate_lines = []
gate_lines.append("[Gate checks]")
for field in ["cpu_usage_distribution", "tail_cpu_usage_distribution"]:
    total_field = int(len_df[len_df["field"]==field]["count"].sum())
    gate_lines.append(f"{field}: total_count_by_lengths={total_field}")
    assert total_field == total_valid_rows, f"❌ {field} length counts do not sum to total_valid_rows ({total_field} != {total_valid_rows})"
#进行gate# Gate: 固定长度占比 > 90%
    field_len = len_df[len_df["field"] == field].copy()
    expected_len = 11 if field == "cpu_usage_distribution" else 9
    expected_count = int(field_len.loc[field_len["length"] == expected_len, "count"].sum())
    expected_ratio = expected_count / total_valid_rows
    parse_fail_count = int(field_len.loc[field_len["length"] == -1, "count"].sum())
    parse_fail_ratio = parse_fail_count / total_valid_rows

    gate_lines.append(f"  {field}: expected_length={expected_len}, ratio={expected_ratio:.6f} (threshold>0.90)")
    gate_lines.append(f"  {field}: parse_fail_count={parse_fail_count}, ratio={parse_fail_ratio:.6f} (threshold<0.05)")

    assert expected_ratio > 0.90, \
        f"❌ {field} 固定长度({expected_len})占比={expected_ratio:.4f} < 0.90，不满足展开条件"
    assert parse_fail_ratio < 0.05, \
        f"❌ {field} 解析失败率={parse_fail_ratio:.4f} >= 0.05，不满足展开条件"

# 报告
lines = []
lines.append("=== Step2H CPU Distribution Full Audit (streaming) ===")
lines.append(f"total_valid_rows_after_time_machine_filter={total_valid_rows}")
lines.append("")
lines.extend(gate_lines)
lines.append("")
lines.append("Length distribution (top 20):")
lines.append(len_df.head(20).to_string(index=False))
lines.append("")
lines.append("Global stats by length (NOTE: mean/std are across ALL positions within vectors, not per-position):")
lines.append("")
lines.append("[Value range check (should be in [0, 1] for normalized CPU distribution)]:")
for field in ["cpu_usage_distribution", "tail_cpu_usage_distribution"]:
    expected_len = 11 if field == "cpu_usage_distribution" else 9
    key = (field, expected_len)
    if key in stats:
        acc = stats[key]
        in_range = acc["min"] >= -0.01 and acc["max"] <= 1.01
        lines.append(f"  {field}(len={expected_len}): min={acc['min']:.6f}, max={acc['max']:.6f}, in_range={in_range}")

lines.append(sum_df.head(20).to_string(index=False))

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nAudit completed.")
print("[输出]", OUT_LEN)
print("[输出]", OUT_SUM)
print("[输出]", OUT_TXT)
print("="*90)
print("\n".join(lines))