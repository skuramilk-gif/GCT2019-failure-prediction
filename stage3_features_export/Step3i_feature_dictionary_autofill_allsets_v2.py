# ============================================================
# 文件名：Step3i_feature_dictionary_autofill_allsets_v2.py
# 阶段：Stage 3I（v2，特征字典模板/自动填充版）
#
# 作用：
#   生成四套 feature set（base_real/extended_real/workload/derived）的“特征字典模板”，
#   自动填充每个特征的：
#     - type（粗粒度：continuous/binary/count/fraction）
#     - source_field（来自 raw/clean/derived 的哪个字段）
#     - aggregation（slice内聚合方式或派生方式）
#     - preprocess（统一写成 train-only 的预处理说明）
#     - notes（简要语义说明）
#
#   该脚本定位：
#     - 用于快速生成“可读的特征表模板”，便于我/导师/审稿人核对特征来源与语义；
#     - 不包含 clip/scale 数值参数（定稿版请用 Step3i_feature_dictionary_final_allsets_v2.py）。
#
# 输入：
#   02_processed/step3c_model_input_{set}_H8h_train_v2.csv
#     - 只读取表头（nrows=0），以实际导出表为准，防止字典与数据不一致
#     - set ∈ {base_real, extended_real, workload, derived}
#
# 输出（05_reports/）：
#   1) step3i_feature_dictionary_autofill_allsets_v2.csv
#      - 四个 set 的特征字典合并表（每行一个特征）
#   2) step3i_feature_list_{set}_v2.txt
#      - 每个 set 一份“列清单”（保持顺序，便于人工复核）
#
# 关键约束：
#   - NON_FEAT（meta_machine_id/meta_time_slice/label）不进入特征字典
#   - 若出现 UNKNOWN 特征，说明 meta_for_feature 未覆盖，需要你补充映射规则
# ============================================================

import os
import re
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
PROC_DIR = os.path.join(ROOT_DIR, "02_processed")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

H = 8
SETS = ["base_real", "extended_real", "workload", "derived"]
SPLIT = "train"

NON_FEAT = {"meta_machine_id", "meta_time_slice", "label"}

OUT_DICT = os.path.join(REPORT_DIR, "step3i_feature_dictionary_autofill_allsets_v2.csv")

CPU_VEC_RE = re.compile(r"^cpu_dist_(\d+)$")
TAIL_VEC_RE = re.compile(r"^tail_cpu_dist_(\d+)$")

def infer_type(feature: str) -> str:
    f = feature.lower()
    if f.startswith("sched_frac_"):
        return "fraction_[0,1]"
    if ("mask" in f) or f.endswith("_missing") or f.endswith("_flag"):
        return "binary_{0,1}"
    if f.endswith("_count") or f == "record_count":
        return "count"
    return "continuous"

def meta_for_feature(f: str):
    """
    返回 (source_field, aggregation, notes)
    """
    # ======= missing 指示（必须放在 prefix 匹配之前，避免 cpu_dist_missing 被误判成向量）=======
    if f in ["cpu_dist_missing", "tail_cpu_dist_missing"]:
        return ("raw:cpu_*_distribution", "解析失败/无固定长度向量→missing指示", "1表示该slice无有效固定长度向量")

    # ======= CPU 分布向量：只匹配 cpu_dist_0..10 / tail_cpu_dist_0..8 =======
    m = CPU_VEC_RE.match(f)
    if m:
        return ("raw:cpu_usage_distribution[k]", "slice内逐维均值(mean over records)", "CPU分布向量展开(11维)")
    m = TAIL_VEC_RE.match(f)
    if m:
        return ("raw:tail_cpu_usage_distribution[k]", "slice内逐维均值(mean over records)", "CPU尾部分布向量展开(9维)")

    # ======= scheduling_class（比例向量；processed 仅导出 3 列，参考类在导出阶段移除）=======
    if f.startswith("sched_frac_"):
        return ("raw:scheduling_class", "slice内类别占比(count/obs)", "原始为4维占比向量；导出时移除参考类避免共线性")

    # ======= priority（min/max/mean；prio_mean 为 slice 内加权均值）=======
    if f in ["prio_min", "prio_max", "prio_mean"]:
        if f == "prio_mean":
            return ("raw:priority", "slice内加权均值(sum/cnt)", "优先级混合属性：均值（跨chunk加权）")
        return ("raw:priority", "slice内min/max", "优先级混合属性：范围")

    # ======= 事件计数（来自 clean）=======
    if f in ["evict_count", "lost_count", "kill_count"]:
        return ("clean:event", "slice内计数(sum)", "non-FAIL异常事件计数")
    if f == "abnormal_nonfail_count":
        return ("clean:event", "evict+lost+kill", "异常事件总计数")
    if f == "abnormal_nonfail_flag":
        return ("clean:event", "1(abnormal_nonfail_count>0)", "二值异常指示")

    # ======= gap/mask（来自 Step3a 派生）=======
    if f == "gap_since_prev":
        return ("derived:time_slice", "groupby(machine) diff(time_slice)", "观测间隔（单位=300秒slice数）")
    if f == "delta_valid_mask":
        return ("derived:gap_since_prev", "1(gap<=2)", "连续观测指示")

    # ======= clean/derived：资源/使用/计数器 =======
    clean_map = {
        "ms_avg_cpu_mean": ("clean:average_usage.cpus", "slice均值", "CPU平均使用"),
        "ms_avg_mem_mean": ("clean:average_usage.memory", "slice均值", "Mem平均使用"),
        "ms_max_cpu_max": ("clean:maximum_usage.cpus", "slice最大", "CPU最大使用"),
        "ms_max_mem_max": ("clean:maximum_usage.memory", "slice最大", "Mem最大使用"),
        "ms_req_cpu_mean": ("clean:resource_request.cpus", "slice均值", "CPU请求"),
        "ms_req_mem_mean": ("clean:resource_request.memory", "slice均值", "Mem请求"),
        "ms_cpi_mean": ("clean:cycles_per_instruction", "slice均值", "CPI"),
        "ms_mpi_mean": ("clean:memory_accesses_per_instruction", "slice均值", "MPI"),
        "ms_cpi_missing": ("clean:cpi", "1(cnt==0)", "CPI缺失指示"),
        "ms_mpi_missing": ("clean:mpi", "1(cnt==0)", "MPI缺失指示"),
        "ms_cpi_coverage": ("clean:cpi", "slice内CPI有效记录数/总记录数", "CPI覆盖率(0~1)，反映性能计数器可用程度"),
        "ms_mpi_coverage": ("clean:mpi", "slice内MPI有效记录数/总记录数", "MPI覆盖率(0~1)，反映性能计数器可用程度"),
        "ms_assigned_memory_mean": ("clean:assigned_memory", "slice均值", "分配内存"),
        "ms_page_cache_memory_mean": ("clean:page_cache_memory", "slice均值", "页缓存内存"),
        "record_count": ("clean", "slice内raw记录数", "该slice聚合的记录数"),
        "cpu_gap": ("derived", "ms_avg_cpu_mean - ms_req_cpu_mean", "使用-请求差"),
        "mem_gap": ("derived", "ms_avg_mem_mean - ms_req_mem_mean", "使用-请求差"),
    }
    if f in clean_map:
        return clean_map[f]

    return ("UNKNOWN", "UNKNOWN", "UNKNOWN")

rows = []
unknown = []

print("=" * 90)
print("Step3I(v2)：feature dictionary autofill (all sets)")
print("=" * 90)

for set_name in SETS:
    in_file = os.path.join(PROC_DIR, f"step3c_model_input_{set_name}_H{H}h_{SPLIT}_v2.csv")
    assert os.path.exists(in_file), f"❌ 缺少输入文件: {in_file}"

    df0 = pd.read_csv(in_file, nrows=0)
    cols = [c for c in df0.columns.tolist() if c not in NON_FEAT]

    # 输出列清单（保持顺序）
    out_list = os.path.join(REPORT_DIR, f"step3i_feature_list_{set_name}_v2.txt")
    with open(out_list, "w", encoding="utf-8") as f:
        for c in cols:
            f.write(c + "\n")

    for idx, c in enumerate(cols):
        src, agg, notes = meta_for_feature(c)
        if src == "UNKNOWN":
            unknown.append((set_name, c))
        rows.append({
            "set": set_name,
            "feature": c,
            "order_in_set": idx,
            "type": infer_type(c),
            "source_field": src,
            "aggregation": agg,
            "preprocess": "train-only: median_impute + clip + robust_scale(IQR)",
            "notes": notes,
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_DICT, index=False, encoding="utf-8-sig")

print("[输出]", OUT_DICT)
if unknown:
    print("\n[警告] 存在未自动识别的特征（需你在字典中补充来源/聚合/备注）:")
    for s, c in unknown[:50]:
        print(f"  set={s} feature={c}")
    if len(unknown) > 50:
        print(f"  ... 还有 {len(unknown)-50} 个未显示")
else:
    print("\n✅ 所有特征均已自动识别并填充元信息")