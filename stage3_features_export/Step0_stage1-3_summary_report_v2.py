# ============================================================
# 文件名：Step0_stage1_3_summary_report_v2.py
# 阶段：Stage 0（v2，总结报告/证据链整合）
#
# 作用：
#   将 Stage1–3 的关键证据与统计从 05_reports 汇总成 1–2页文本：
#     - 数据规模、split规模
#     - multi-H标签统计、pos_machines
#     - 时间轴一致性QC（Step2E）
#     - CPU分布结构审计（Step2H）
#     - 导出健康检查（Step3D）
#   并写出“证据 -> 建模策略”的固定文本，可直接写进论文 Experimental Protocol。
#
# 输入（若存在则读取）：
#   05_reports/step2c_label_summary_FAILonly_multiH_v2.csv
#   05_reports/step2d_pos_machine_stats_multiH_v2.csv
#   05_reports/step2e_time_order_qc_summary_v2.txt
#   05_reports/step2h_cpu_distribution_audit_report_v2.txt
#   05_reports/step3d_schema_check_v2.txt
#
# 输出：
#   05_reports/step0_stage1_3_summary_v2.txt
# ============================================================

import os
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
OUT = os.path.join(REPORT_DIR, "step0_stage1_3_summary_v2.txt")

# 关键证据链文件（以你当前 v2 产物为准）
PATHS = {
# -------- Stage1 docs evidence --------
    "step1a_summary": os.path.join(REPORT_DIR, "step1a_summary_report_v2.txt"),
    "step1b_semantics": os.path.join(REPORT_DIR, "step1b_time_semantics_report_v2.txt"),
    # Stage2: core evidence
    "label_summary": os.path.join(REPORT_DIR, "step2c_label_summary_FAILonly_multiH_v2.csv"),
    "label_qc_trainval": os.path.join(REPORT_DIR, "step2d_label_qc_summary_trainval_v2.csv"),
    "pos_machines": os.path.join(REPORT_DIR, "step2d_pos_machine_stats_multiH_v2.csv"),

    # Stage2: gates/audits/variants
    "time_qc": os.path.join(REPORT_DIR, "step2e_time_order_qc_summary_v2.txt"),
    "cpu_audit": os.path.join(REPORT_DIR, "step2h_cpu_distribution_audit_report_v2.txt"),
    "cluster_holdout": os.path.join(REPORT_DIR, "step2h2_clusterholdout_diagnostics_v2.txt"),

    # Stage3: evidence & gates
    "preprocess_fit": os.path.join(REPORT_DIR, "step3b_preprocess_fit_summary_H8h_v2.txt"),
    "export_summary_trainval": os.path.join(REPORT_DIR, "step3c_export_summary_multiH_trainval_v2.csv"),
    "schema_qc": os.path.join(REPORT_DIR, "step3d_schema_check_v2.txt"),
    "sched_simplex_gate": os.path.join(REPORT_DIR, "step3j_sched_frac_simplex_qc_v2.txt"),
    "leakage_gate": os.path.join(REPORT_DIR, "step3k_leakage_audit_v2.txt"),
    # (可选) 未来你加的 fit-domain audit（若不存在会显示 missing）
    "fit_domain_audit": os.path.join(REPORT_DIR, "step3k2_fit_domain_audit_v2.txt"),
# -------- Stage3 paper appendix artifacts (summarize only) --------
    "bundle_missing_vs_label": os.path.join(REPORT_DIR, "step3f_cpu_dist_missing_vs_label_H8h_trainval_v2.csv"),
    "feature_dict_final": os.path.join(REPORT_DIR, "step3i_feature_dictionary_final_allsets_v2.csv"),

    # -------- Step4 freeze evidence --------
    "log_step4b_latest": os.path.join(REPORT_DIR, "step4b_run_data_pipeline_stage1_3_v2_latest.log"),
    "log_step4c_latest": os.path.join(REPORT_DIR, "step4c_run_docs_variants_freeze_v2_latest.log"),
    "manifest": os.path.join(REPORT_DIR, "step4a_MANIFEST_v2.json"),
}

lines = []
lines.append("=== Stage1–3 Summary (v2, final; train/val-only evidence chain) ===")
lines.append("")

def add_file_text(title, path, head_lines=80, tail_lines=25):
    lines.append(f"[{title}]")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        tlines = txt.splitlines()
        if len(tlines) > head_lines + tail_lines:
            txt = "\n".join(tlines[:head_lines]) + f"\n... (truncated, total_lines={len(tlines)})\n" + "\n".join(tlines[-tail_lines:])
        lines.append(txt)
    else:
        lines.append(f"(missing) {path}")
    lines.append("")

# ------------------------------------------------------------
# A) Multi-H label summary（强制 train/val-only 输出）
# ------------------------------------------------------------
p = PATHS["label_summary"]
lines.append("[Multi-H label summary (train/val only)]")
if os.path.exists(p):
    df = pd.read_csv(p)
    keep = [c for c in ["H_hours", "train_pos_count", "val_pos_count", "train_pos_rate", "val_pos_rate"] if c in df.columns]
    # 防止历史版本带 test 列：不输出 test
    lines.append(df[keep].to_string(index=False))
else:
    lines.append(f"(missing) {p}")
lines.append("")

# ------------------------------------------------------------
# B) Label QC summary（含 NaN 分解）
# ------------------------------------------------------------
p = PATHS["label_qc_trainval"]
lines.append("[Label QC summary (scheme-A; includes NaN decomposition; train/val)]")
if os.path.exists(p):
    df = pd.read_csv(p)
    keep = [c for c in [
        "H_hours", "valid_rate", "nan_total_rate",
        "nan_post_first_fail", "nan_no_fail_censored", "nan_other",
        "train_pos_count", "val_pos_count", "train_pos_rate", "val_pos_rate"
    ] if c in df.columns]
    lines.append(df[keep].to_string(index=False))
else:
    lines.append(f"(missing) {p}")
lines.append("")

# ------------------------------------------------------------
# C) pos_machines（train/val-only 输出）
# ------------------------------------------------------------
p = PATHS["pos_machines"]
lines.append("[Pos machines summary (for CI credibility; train/val)]")
if os.path.exists(p):
    df = pd.read_csv(p)
    if "split" in df.columns:
        df = df[df["split"].isin(["train", "val"])].copy()
    keep = [c for c in ["H_hours","split","valid_rows","pos_rows","valid_machines","pos_machines","pos_rate"] if c in df.columns]
    if keep:
        lines.append(df[keep].sort_values(["H_hours", "split"]).to_string(index=False))
    else:
        lines.append("(unexpected schema) " + p)
else:
    lines.append(f"(missing) {p}")
lines.append("")

add_file_text("Raw stats summary (Step1A, Docs)", PATHS["step1a_summary"])
add_file_text("Time semantics report (Step1B, Docs)", PATHS["step1b_semantics"])

# ------------------------------------------------------------
# D) Stage2/Stage3 Gate txts
# ------------------------------------------------------------
add_file_text("Time order QC (Step2E, GATE; includes delta_end evidence)", PATHS["time_qc"])
add_file_text("CPU distribution audit (Step2H, GATE)", PATHS["cpu_audit"])
add_file_text("Cluster holdout diagnostics (Step2H2, Variant)", PATHS["cluster_holdout"])

add_file_text("Preprocess fit summary (Step3B, train-only)", PATHS["preprocess_fit"])
add_file_text("Processed schema check (Step3D, GATE)", PATHS["schema_qc"])
add_file_text("sched_frac simplex QC (Step3J, GATE)", PATHS["sched_simplex_gate"])
add_file_text("Processed leakage audit (Step3K, GATE)", PATHS["leakage_gate"])
add_file_text("Fit-domain audit (Step3K2, optional gate)", PATHS["fit_domain_audit"])
# ------------------------------------------------------------
# Step4 freeze evidence (orchestrator logs + manifest)
# ------------------------------------------------------------
add_file_text("Step4A MANIFEST (JSON fingerprint)", PATHS["manifest"])
add_file_text("Step4B core freeze log (latest)", PATHS["log_step4b_latest"])
if os.path.exists(PATHS["log_step4c_latest"]):
    add_file_text("Step4C docs/variants freeze log (latest)", PATHS["log_step4c_latest"])

# ------------------------------------------------------------
# E) Export summary（train/val-only）
# ------------------------------------------------------------
p = PATHS["export_summary_trainval"]
lines.append("[Processed export summary (train/val only)]")
if os.path.exists(p):
    df = pd.read_csv(p)
    keep = [c for c in ["H_hours","split","rows","pos","pos_rate"] if c in df.columns]
    lines.append(df[keep].sort_values(["H_hours","split"]).to_string(index=False))
else:
    lines.append(f"(missing) {p}")
lines.append("")
# ------------------------------------------------------------
# G) Stage3 paper appendix artifacts (summary only)
# ------------------------------------------------------------
lines.append("[Final feature dictionary (Step3I, for appendix)]")
p = PATHS["feature_dict_final"]
if os.path.exists(p):
    df = pd.read_csv(p)
    lines.append(f"path = {p}")
    lines.append(f"rows = {len(df)}")
    lines.append(f"columns = {list(df.columns)}")
else:
    lines.append(f"(missing) {p}")
lines.append("")

lines.append("[CPU distribution missing vs label (train/val) from Step3F]")
p = PATHS["bundle_missing_vs_label"]
if os.path.exists(p):
    df = pd.read_csv(p)
    lines.append(f"path = {p}")
    lines.append(f"shape = {df.shape}")
    lines.append(df.head(10).to_string(index=False))
else:
    lines.append(f"(missing) {p}")
lines.append("")

# ------------------------------------------------------------
# F) Evidence -> Protocol（修正为方案A，加入预测锚点措辞）
# ------------------------------------------------------------
lines.append("[Evidence -> Modeling Protocol (frozen)]")
lines.append("1) 时间轴硬QC通过（dup/mismatch/回跳=0），time_slice 与 slice_last_time_us 映射一致。预测锚点定义为 slice 内最后一次可观测时间 slice_last_time_us；与固定 slice_end 的偏移 <300s。")
lines.append("2) 观测极度稀疏（gap 分位数极大）：任务定位为小时级 risk scoring，而非密集时间序列预测。")
lines.append("3) 方案A标签：仅预测首次 FAIL；对每台机器 t>=t_first_fail 的样本全部剔除；无FAIL机器仅尾部窗口右删失。NaN分解 nan_other=0。")
lines.append("4) 预处理参数（impute/clip/scale）仅在 H=8 train label-valid 域拟合；val/test 仅 transform，避免全局统计泄漏。")
lines.append("5) processed 输入门禁：48文件齐全、NaN/Inf=0、schema一致；sched_frac simplex gate 通过；processed 禁用列泄漏审计通过。")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("[输出]", OUT)