# ============================================================
# 文件名：Step3k_leakage_audit_processed_v2.py
# 阶段：Stage 3K（v2，审计/防火墙，GATE）
#
# 作用：
#   全量扫描 02_processed 下所有 step3c_model_input_*_v2.csv（应为48个文件），
#   确认：
#     1) 文件数=48
#     2) 不存在禁用列（答案列/未来信息/中间标签列）
#     3) processed 仅允许 meta_* 索引列与 label；禁止 machine_id/time_slice/split 等
#目的：
#     以“硬审计报告”回应审稿式质疑：泄漏风险是否彻底封死。# 失败条件（直接抛错）：
#   - 文件数不是48
#   - 任意文件表头含禁用列
# ============================================================

import os
import glob
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
IN_DIR = os.path.join(ROOT_DIR, "02_processed")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

OUT = os.path.join(REPORT_DIR, "step3k_leakage_audit_v2.txt")

pattern = os.path.join(IN_DIR, "step3c_model_input_*_v2.csv")
files = sorted(glob.glob(pattern))
assert len(files) == 48, f"❌ 期望48个输入表，实际找到 {len(files)} 个：{pattern}"

# 绝对禁用列（出现即一票否决）
FORBIDDEN_EXACT = {
    # 标签/答案相关（raw/clean字段）
    "failed_count", "failed", "event", "instance_events_type",
    # 时间锚点/未来信息中间列
    "slice_last_time_us", "dt_to_next_fail_us", "dt_to_first_fail_us",
    "machine_tmax_us", "t_first_fail_us", "first_fail_us",
    # 不应出现在 processed 的字段
    "split", "machine_id", "time_slice",
}

# 禁用子串：任何 label_fail_H*h 这类中间列不允许进入（最终只允许 'label'）
FORBIDDEN_SUBSTR = ["label_fail_"]

# meta 结构约束（强建议写死）
REQUIRED_META = {"meta_machine_id", "meta_time_slice"}
REQUIRED_LABEL = {"label"}

# 白名单：每个 (H, set) 的期望列 = meta + label + 该 set 的特征列
# 从 Step3c 的 feature_sets 定义中硬编码（与 Step3c 保持同步）
EXPECTED_FEATURE_SETS = {
    "base_real": [
        "ms_avg_cpu_mean", "ms_avg_mem_mean", "ms_max_cpu_max", "ms_max_mem_max",
        "ms_cpi_mean", "ms_mpi_mean", "ms_cpi_missing", "ms_mpi_missing",
        "record_count",
    ],
    "extended_real": [
        "ms_avg_cpu_mean", "ms_avg_mem_mean", "ms_max_cpu_max", "ms_max_mem_max",
        "ms_cpi_mean", "ms_mpi_mean", "ms_cpi_missing", "ms_mpi_missing",
        "record_count",
        "ms_req_cpu_mean", "ms_req_mem_mean",
        "ms_assigned_memory_mean", "ms_page_cache_memory_mean",
        "ms_cpi_coverage", "ms_mpi_coverage",
        "cpu_gap", "mem_gap",
        "cpu_dist_0", "cpu_dist_1", "cpu_dist_2", "cpu_dist_3", "cpu_dist_4",
        "cpu_dist_5", "cpu_dist_6", "cpu_dist_7", "cpu_dist_8", "cpu_dist_9", "cpu_dist_10",
        "tail_cpu_dist_0", "tail_cpu_dist_1", "tail_cpu_dist_2", "tail_cpu_dist_3",
        "tail_cpu_dist_4", "tail_cpu_dist_5", "tail_cpu_dist_6", "tail_cpu_dist_7", "tail_cpu_dist_8",
        "cpu_dist_missing", "tail_cpu_dist_missing",
        "sched_frac_0", "sched_frac_1", "sched_frac_2",
        "prio_min", "prio_max", "prio_mean",
    ],
}
# workload = extended_real + event features
EXPECTED_FEATURE_SETS["workload"] = EXPECTED_FEATURE_SETS["extended_real"] + [
    "evict_count", "lost_count",
    "abnormal_nonfail_count", "abnormal_nonfail_flag",
]
# derived = workload + gap features
EXPECTED_FEATURE_SETS["derived"] = EXPECTED_FEATURE_SETS["workload"] + [
    "gap_since_prev", "delta_valid_mask",
]

META_COLS = ["meta_machine_id", "meta_time_slice", "label"]

bad = []
for fp in files:
    cols = pd.read_csv(fp, nrows=0).columns.tolist()
    colset = set(cols)

    # required columns
    miss_req = sorted(list((REQUIRED_META | REQUIRED_LABEL) - colset))
    if miss_req:
        bad.append((os.path.basename(fp), "missing_required", miss_req))

    # forbidden exact hits
    hit_exact = sorted(list(colset & FORBIDDEN_EXACT))

    # forbidden substr hits (excluding final label)
    hit_sub = [c for c in cols if any(s in c for s in FORBIDDEN_SUBSTR) and c != "label"]

    # forbid any unexpected meta_ columns (防止 meta_* 被误当成特征扩散)
    unexpected_meta = sorted([c for c in cols if c.startswith("meta_") and c not in REQUIRED_META])

    if hit_exact or hit_sub or unexpected_meta:
        bad.append((os.path.basename(fp), "forbidden",
                    {"exact": hit_exact, "substr": hit_sub, "unexpected_meta": unexpected_meta}))

    # 白名单检查：列集合必须精确匹配期望
    fn = os.path.basename(fp)
    set_name = None
    for s in EXPECTED_FEATURE_SETS:
        if f"_{s}_" in fn:
            set_name = s
            break
    if set_name:
        expected_cols = set(EXPECTED_FEATURE_SETS[set_name] + META_COLS)
        actual_cols = set(cols)
        extra = sorted(actual_cols - expected_cols)
        missing_wl = sorted(expected_cols - actual_cols)
        if extra or missing_wl:
            bad.append((os.path.basename(fp), "whitelist_violation",
                        {"extra_cols": extra[:20], "missing_cols": missing_wl[:20]}))

lines = []
lines.append("=== Step3K Leakage Audit (processed tables, GATE) ===")
lines.append(f"pattern={pattern}")
lines.append(f"files_scanned={len(files)}")
# 文件数量明细
from collections import Counter
set_counts = Counter()
for fp in files:
    fn = os.path.basename(fp)
    for s in ["base_real", "extended_real", "workload", "derived"]:
        if f"_{s}_" in fn:
            set_counts[s] += 1
            break
lines.append(f"files_per_set={dict(set_counts)}")
# 期望：每个 set 12 个文件（4H × 3split）
for s in EXPECTED_FEATURE_SETS:
    expected_n = 12  # 4 H values × 3 splits
    actual_n = set_counts.get(s, 0)
    assert actual_n == expected_n, f"❌ set={s}: 期望{expected_n}个文件，实际{actual_n}个"

lines.append(f"forbidden_exact={sorted(list(FORBIDDEN_EXACT))}")
lines.append(f"forbidden_substrings={FORBIDDEN_SUBSTR}")
lines.append(f"required_meta={sorted(list(REQUIRED_META))} required_label={sorted(list(REQUIRED_LABEL))}")
lines.append("")

if bad:
    lines.append("❌ FOUND VIOLATIONS:")
    for item in bad[:80]:
        lines.append(str(item))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    raise AssertionError("❌ Step3K FAILED: 发现禁用列/缺失必需列，processed 输入表不可用于训练。")
else:
    lines.append("✅ PASS: no forbidden columns; required meta/label present in all processed tables.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("\n".join(lines))
print("[输出]", OUT)