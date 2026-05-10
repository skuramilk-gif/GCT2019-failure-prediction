# ============================================================
# 文件名：Step3i_feature_dictionary_final_allsets_v2.py  //最终特征审稿字典
# 阶段：Stage 3I（v2，定稿文档交付）
#之前版本的特征又缺少，这个是在哪个基础上进行补充，以这个为准。
# 作用：
#   生成“最终可审稿的特征字典”（四套 feature set 全覆盖），包含：
#     - 列名、所属set、在set内顺序
#     - 类型（binary/count/continuous/fraction）
#     - 来源字段（raw/clean/derived）
#     - slice内聚合方式（mean/max/min/frac/逐维均值等）
#     - 预处理参数（clip_lo/clip_hi/scale_median/scale_iqr），从 Step3B params 自动读取
#
#   目的：
#     关闭审稿人/Claude 的致命问题：完整特征列表缺失、聚合语义不透明、clip/scale未明确。
#
# 输入：
#   1) 02_processed/step3c_model_input_{set}_H8h_train_v2.csv
#      其中 set ∈ {base_real, extended_real, workload, derived}
#      （只读表头以获得真实导出schema）
#   2) 04_params/step3b_preprocess_params_H8h_v2.pkl
#
# 输出（05_reports/）：
#   1) step3i_feature_dictionary_final_allsets_v2.csv   # 主字典（论文附录/复现用）
#   2) step3i_feature_list_{set}_v2.txt                # 每个set一份列清单（便于人工核对）
#
# 关键约束：
#   - 字典以“实际导出的 processed 表头”为准（防止文档与数据不一致）
#   - 每个导出特征必须在 params 中存在，否则直接报错（防止fit/transform schema 漂移）
# ============================================================

import os
import re
import pickle
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
PROC_DIR = os.path.join(ROOT_DIR, "02_processed")
PARAM_PATH = os.path.join(ROOT_DIR, "04_params", "step3b_preprocess_params_H8h_v2.pkl")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

H = 8
SPLIT = "train"
SETS = ["base_real", "extended_real", "workload", "derived"]

NON_FEAT = {"meta_machine_id", "meta_time_slice", "label"}

OUT_DICT = os.path.join(REPORT_DIR, "step3i_feature_dictionary_final_allsets_v2.csv")

CPU_VEC_RE = re.compile(r"^cpu_dist_(\d+)$")
TAIL_VEC_RE = re.compile(r"^tail_cpu_dist_(\d+)$")

def infer_type(ftr: str) -> str:
    f = ftr.lower()
    if f.startswith("sched_frac_"):
        return "fraction_[0,1]"
    if ("mask" in f) or f.endswith("_missing") or f.endswith("_flag"):
        return "binary_{0,1}"
    if f.endswith("_count") or f == "record_count":
        return "count"
    return "continuous"

def meta_for_feature(f: str):
    # missing 指示（必须优先）
    if f in ["cpu_dist_missing", "tail_cpu_dist_missing"]:
        return ("raw:cpu_*_distribution", "解析失败/无固定长度向量→missing指示", "1表示该slice无有效固定长度向量")

    # CPU向量列：只匹配 cpu_dist_0..10 / tail_cpu_dist_0..8
    if CPU_VEC_RE.match(f):
        return ("raw:cpu_usage_distribution[k]", "slice内逐维均值(mean over records)", "CPU分布向量展开(11维)")
    if TAIL_VEC_RE.match(f):
        return ("raw:tail_cpu_usage_distribution[k]", "slice内逐维均值(mean over records)", "CPU尾部分布向量展开(9维)")

    # sched_frac：processed 仅导出3列（参考类移除）
    if f.startswith("sched_frac_"):
        return ("raw:scheduling_class", "slice内类别占比(count/obs)", "原始为4维占比向量；导出时移除参考类避免共线性")

    # priority：prio_mean 为 slice 内加权均值
    if f in ["prio_min", "prio_max", "prio_mean"]:
        if f == "prio_mean":
            return ("raw:priority", "slice内加权均值(sum/cnt)", "优先级混合属性：均值（跨chunk加权）")
        return ("raw:priority", "slice内min/max", "优先级混合属性：范围")

    # 事件计数
    if f in ["evict_count", "lost_count", "kill_count"]:
        return ("clean:event", "slice内计数(sum)", "non-FAIL异常事件计数")
    if f == "abnormal_nonfail_count":
        return ("clean:event", "evict+lost+kill", "异常事件总计数")
    if f == "abnormal_nonfail_flag":
        return ("clean:event", "1(abnormal_nonfail_count>0)", "二值异常指示")

    # 派生：gap/mask
    if f == "gap_since_prev":
        return ("derived:time_slice", "groupby(machine) diff(time_slice)", "观测间隔（单位=300秒slice数）")
    if f == "delta_valid_mask":
        return ("derived:gap_since_prev", "1(gap<=2)", "连续观测指示")

    # clean/derived 映射
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

# 读取 Step3B params
assert os.path.exists(PARAM_PATH), f"❌ 缺少 params: {PARAM_PATH}"
with open(PARAM_PATH, "rb") as f:
    params = pickle.load(f)

clip_lo = params["clip_p01"]
clip_hi = params["clip_p99"]
scale_med = params["scale_median"]
scale_iqr = params["scale_iqr"]

rows = []
unknown = []

print("=" * 90)
print("Step3I(v2)：final feature dictionary (all sets)")
print("=" * 90)
print("[输入 params]", PARAM_PATH)

for set_name in SETS:
    in_file = os.path.join(PROC_DIR, f"step3c_model_input_{set_name}_H{H}h_{SPLIT}_v2.csv")
    assert os.path.exists(in_file), f"❌ 缺少输入文件: {in_file}"

    cols = pd.read_csv(in_file, nrows=0).columns.tolist()
    feat_cols = [c for c in cols if c not in NON_FEAT]

    out_list = os.path.join(REPORT_DIR, f"step3i_feature_list_{set_name}_v2.txt")
    with open(out_list, "w", encoding="utf-8") as fw:
        fw.write("\n".join(feat_cols))

    for order, ftr in enumerate(feat_cols):
        src, agg, note = meta_for_feature(ftr)
        if src == "UNKNOWN":
            unknown.append((set_name, ftr))

        # Gate：processed 导出特征必须在 params 中存在
        assert ftr in clip_lo, f"❌ {ftr} 不在 Step3B params 中（set={set_name}）"

        rows.append({
            "set": set_name,
            "feature": ftr,
            "order_in_set": order,
            "type": infer_type(ftr),
            "source_field": src,
            "aggregation": agg,
            "clip_lo": clip_lo[ftr],
            "clip_hi": clip_hi[ftr],
            "scale_median": scale_med[ftr],
            "scale_iqr": scale_iqr[ftr],
            "scale_method": "robust=(x-median)/IQR",
            "notes": note,
        })

out = pd.DataFrame(rows)
out.to_csv(OUT_DICT, index=False, encoding="utf-8-sig")

print("[输出字典]", OUT_DICT)
if unknown:
    print("\n[需人工补充的UNKNOWN特征]（仅显示前50）")
    for s, ftr in unknown[:50]:
        print("  set=", s, "feature=", ftr)
else:
    print("\n✅ 所有特征均已自动识别并填充元信息")