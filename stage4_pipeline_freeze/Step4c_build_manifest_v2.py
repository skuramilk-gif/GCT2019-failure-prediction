# ============================================================
# 文件名：Step4c_build_manifest_v2.py
# 阶段：Step 4C（Pipeline层｜复现指纹｜v2｜定稿）
#
# 【定位/职责边界（非常重要）】
#   本脚本是数据管线 v2 的"复现指纹 MANIFEST"生成器（JSON），用于绑定：
#     1) raw 数据版本（sha256）
#     2) splits 版本（sha256）
#     3) 拟合产物 params（sha256）
#     4) 最终 processed（48 files sha256）
#     5) 关键 gate/evidence 报告（sha256）
#     6) 关键代码脚本（sha256，用于 git_commit=null 时的代码指纹）
#     7) Step4A/Step4B pipeline log（stable alias），证明本版本确实跑通
#
# 【执行顺序（非常重要）】
#   本脚本是 Stage 4 的最后一步，必须在 Step4A 和 Step4B 完成后运行：
#     Step4A（主线冻结）→ Step4B（文档冻结）→ Step4C（本脚本，生成 manifest）
#   编号与执行顺序一致。本脚本不创建 pipeline log 的 stable alias——
#   alias 由 Step4A/Step4B 在各自所有步骤完成后创建，确保内容完整、SHA256 稳定。
#
# 【输入（只读）】
#   00_raw/borg_traces_data.csv
#   03_splits/*.txt
#   04_params/*.pkl
#   02_processed/step3c_model_input_*_v2.csv（应为48个）
#   05_reports/ 下关键证据链文件（REQUIRED_EVIDENCE）
#   05_reports/ 下 Step4A/Step4B stable alias（由 orchestrator 创建）
#
# 【输出】
#   05_reports/step4c_MANIFEST_v2.json
#
# 【Gate（失败即抛错）】
#   - processed 文件数必须为48
#   - REQUIRED_EVIDENCE 必须全部存在
#   - CODE_FILES 必须全部存在
#   - Step4A/Step4B stable alias 必须存在（证明 pipeline 已跑通）
#
# 【SHA256 自洽性说明】
#   本脚本只读取已存在的 stable alias，不创建 alias。
#   Step4A 在所有核心步骤完成后创建 step4a_run_core_freeze_v2_latest.log；
#   Step4B 在所有文档步骤完成后创建 step4b_run_docs_variants_freeze_v2_latest.log。
#   因此本脚本读取的 alias 内容是完整的，SHA256 自洽。
# ============================================================

import os
import sys
import json
import glob
import hashlib
import platform
import datetime

import pandas as pd
import numpy as np

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")
SPLIT_DIR = os.path.join(ROOT_DIR, "03_splits")
PARAM_DIR = os.path.join(ROOT_DIR, "04_params")
PROC_DIR = os.path.join(ROOT_DIR, "02_processed")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

OUT = os.path.join(REPORT_DIR, "step4c_MANIFEST_v2.json")

PROC_PATTERN = os.path.join(PROC_DIR, "step3c_model_input_*_v2.csv")
PROC_EXPECTED_N = 48

# ------------------------------------------------------------
# Stable alias 路径（由 Step4A/Step4B orchestrator 创建，本脚本只读取）
# ------------------------------------------------------------
ALIAS_STEP4A = os.path.join(REPORT_DIR, "step4a_run_core_freeze_v2_latest.log")
ALIAS_STEP4B = os.path.join(REPORT_DIR, "step4b_run_docs_variants_freeze_v2_latest.log")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: str) -> dict:
    st = os.stat(path)
    return {
        "path": os.path.basename(path),
        "full_path": path,
        "size_bytes": int(st.st_size),
        "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),
        "sha256": sha256_file(path),
    }


def try_git_commit(root: str):
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None

# ------------------------------------------------------------
# 1) 关键 gate/evidence 报告（必须存在）
#
#    选取原则：
#    (a) 一票否决 gate 的输出（证明 pipeline 通过）-+
#    (b) 论文实验设置的关键统计（Step0 summary 直接读取的文件）
#    (c) pipeline log 的 stable alias（证明 pipeline 已跑通）
# ------------------------------------------------------------
REQUIRED_EVIDENCE = [
    # ── pipeline logs（stable alias，由 orchestrator 创建）──
    ALIAS_STEP4A,
    ALIAS_STEP4B,

    # ── 总结报告 ──
    os.path.join(REPORT_DIR, "step0_stage1_3_summary_v2.txt"),

    # ── Stage 2 gate outputs ──
    os.path.join(REPORT_DIR, "step2d_label_qc_summary_trainval_v2.csv"),
    os.path.join(REPORT_DIR, "step2e_time_order_qc_summary_v2.txt"),

    # ── Stage 2 key statistics（论文实验设置表格来源）──
    os.path.join(REPORT_DIR, "step2c_label_summary_FAILonly_multiH_v2.csv"),
    os.path.join(REPORT_DIR, "step2h_cpu_distribution_audit_report_v2.txt"),

    # ── Stage 3 gate outputs ──
    os.path.join(REPORT_DIR, "step3d_schema_check_v2.txt"),
    os.path.join(REPORT_DIR, "step3j_sched_frac_simplex_qc_v2.txt"),
    os.path.join(REPORT_DIR, "step3k_leakage_audit_v2.txt"),
    os.path.join(REPORT_DIR, "step3k2_fit_domain_audit_v2.txt"),

    # ── Stage 3 key statistics（证明 fit domain 和导出正确）──
    os.path.join(REPORT_DIR, "step3b_preprocess_fit_summary_H8h_v2.txt"),
    os.path.join(REPORT_DIR, "step3c_export_summary_multiH_v2.csv"),
]

# ------------------------------------------------------------
# 2) 关键代码指纹（必须存在）
#
#    选取原则：
#    (a) 影响数据定义/标签/特征/预处理的 Core 脚本
#    (b) 一票否决 Gate 脚本
#    (c) 含 assert gate 的 Docs 脚本（Step3i final）
#    (d) pipeline orchestrator（决定执行顺序）
#    不纳入：纯 Docs/诊断脚本（Step1a/1b/2f/2g/2h/2h2/3e1/3f 等）
#    这些脚本不影响训练数据，其 hash 由 Step4B log 间接覆盖。
# ------------------------------------------------------------
CODE_FILES = [
    # ── Stage 2 core + gates ──
    os.path.join(ROOT_DIR, "Step2a_machine_slice主表构建_clean_v2.py"),
    os.path.join(ROOT_DIR, "Step2b_训练验证测试机器划分与固化_v2.py"),
    os.path.join(ROOT_DIR, "Step2c_预警标签生成_FAILonly_多窗口_v2.py"),
    os.path.join(ROOT_DIR, "Step2d_标签与样本质量检查_汇总报告_v2.py"),
    os.path.join(ROOT_DIR, "Step2e_time_order_qc_v2.py"),

    # ── Stage 3 core + gates ──
    os.path.join(ROOT_DIR, "Step3a_派生特征构造_v2.py"),
    os.path.join(ROOT_DIR, "Step3a_aux_raw_cpu_dist_vector_merge_v2.py"),
    os.path.join(ROOT_DIR, "Step3b_preprocess参数拟合_trainonly_H8h_v2.py"),
    os.path.join(ROOT_DIR, "Step3c_transform导出模型输入表_multiH_v2.py"),
    os.path.join(ROOT_DIR, "Step3d_输入表健康检查与稀疏性报告_v2.py"),
    os.path.join(ROOT_DIR, "step3j_sched_frac_simplex_qc_v2.py"),
    os.path.join(ROOT_DIR, "Step3k_leakage_audit_processed_v2.py"),
    os.path.join(ROOT_DIR, "Step3k2_fit_domain_audit_v2.py"),

    # ── 含 assert gate 的 Docs 脚本 ──
    os.path.join(ROOT_DIR, "Step3i_feature_dictionary_final_allsets_v2.py"),

    # ── pipeline orchestrators ──
    os.path.join(ROOT_DIR, "Step0_stage1-3_summary_report_v2.py"),
    os.path.join(ROOT_DIR, "Step4a_run_core_freeze_v2.py"),
    os.path.join(ROOT_DIR, "Step4b_run_docs_variants_freeze_v2.py"),
    os.path.join(ROOT_DIR, "Step4c_build_manifest_v2.py"),
]


# ------------------------------------------------------------
# Build manifest
# ------------------------------------------------------------
manifest = {
    "generated_at": datetime.datetime.now().isoformat(),
    "root_dir": ROOT_DIR,
    "pipeline_version": "v2",
    "execution_order": [
        "Step4a_run_core_freeze_v2.py (core freeze: Stage2 clean/split/label + Stage3 features/params/processed + gates)",
        "Step4b_run_docs_variants_freeze_v2.py (docs freeze: Stage1-3 diagnostics/audit/feature dict)",
        "Step4c_build_manifest_v2.py (this script: generate reproducibility fingerprint)",
    ],
    "environment": {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    },
    "git_commit": try_git_commit(ROOT_DIR),

    "inputs": {
        "raw": {},
        "splits": {"count": 0, "files": []},
    },
    "fit_artifacts": {
        "params": {"count": 0, "files": []},
    },
    "final_artifacts": {
        "processed_pattern": PROC_PATTERN,
        "processed_expected_n": PROC_EXPECTED_N,
        "processed_count": 0,
        "processed_files": [],
    },
    "evidence_reports": {
        "required_files": [os.path.basename(p) for p in REQUIRED_EVIDENCE],
        "files": [],
        "count": 0,
    },
    "code_fingerprint": {
        "required_files": [os.path.basename(p) for p in CODE_FILES],
        "files": [],
        "count": 0,
    },
    "pipeline_logs": {
        "step4a_core_freeze_alias": ALIAS_STEP4A,
        "step4a_core_freeze_exists": False,
        "step4b_docs_freeze_alias": ALIAS_STEP4B,
        "step4b_docs_freeze_exists": False,
    },
}

# ── raw ──
if os.path.exists(RAW_PATH):
    manifest["inputs"]["raw"] = file_info(RAW_PATH)
else:
    manifest["inputs"]["raw"] = {"missing": RAW_PATH}

# ── splits ──
split_files = sorted(glob.glob(os.path.join(SPLIT_DIR, "*.txt")))
manifest["inputs"]["splits"]["files"] = [file_info(p) for p in split_files]
manifest["inputs"]["splits"]["count"] = len(split_files)

# ── params ──
param_files = sorted(glob.glob(os.path.join(PARAM_DIR, "*.pkl")))
manifest["fit_artifacts"]["params"]["files"] = [file_info(p) for p in param_files]
manifest["fit_artifacts"]["params"]["count"] = len(param_files)

# ── processed ──
proc_files = sorted(glob.glob(PROC_PATTERN))
manifest["final_artifacts"]["processed_files"] = [file_info(p) for p in proc_files]
manifest["final_artifacts"]["processed_count"] = len(proc_files)

# Gate: processed must be 48
assert manifest["final_artifacts"]["processed_count"] == PROC_EXPECTED_N, \
    f"❌ processed file count != {PROC_EXPECTED_N}, " \
    f"got {manifest['final_artifacts']['processed_count']}"

# ── evidence reports (Gate: must exist) ──
missing_evi = [p for p in REQUIRED_EVIDENCE if not os.path.exists(p)]
assert len(missing_evi) == 0, \
    f"❌ missing required evidence reports ({len(missing_evi)}):\n" + \
    "\n".join(f"  - {p}" for p in missing_evi)

manifest["evidence_reports"]["files"] = [file_info(p) for p in REQUIRED_EVIDENCE]
manifest["evidence_reports"]["count"] = len(REQUIRED_EVIDENCE)

# ── code fingerprint (Gate: must exist) ──
missing_code = [p for p in CODE_FILES if not os.path.exists(p)]
assert len(missing_code) == 0, \
    f"❌ missing required code files ({len(missing_code)}):\n" + \
    "\n".join(f"  - {p}" for p in missing_code)

manifest["code_fingerprint"]["files"] = [file_info(p) for p in CODE_FILES]
manifest["code_fingerprint"]["count"] = len(CODE_FILES)

# ── pipeline logs ──
manifest["pipeline_logs"]["step4a_core_freeze_exists"] = os.path.exists(ALIAS_STEP4A)
manifest["pipeline_logs"]["step4b_docs_freeze_exists"] = os.path.exists(ALIAS_STEP4B)

# ------------------------------------------------------------
# Write
# ------------------------------------------------------------
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------
print("=" * 90)
print("Step4C(v2) MANIFEST generated")
print("=" * 90)
print(f"[输出] {OUT}")
print(f"  processed_files = {manifest['final_artifacts']['processed_count']}")
print(f"  split_files     = {manifest['inputs']['splits']['count']}")
print(f"  params_files    = {manifest['fit_artifacts']['params']['count']}")
print(f"  evidence_files  = {manifest['evidence_reports']['count']}")
print(f"  code_files      = {manifest['code_fingerprint']['count']}")
print(f"  git_commit      = {manifest['git_commit']}")
print(f"  step4a_log      = {manifest['pipeline_logs']['step4a_core_freeze_exists']}")
print(f"  step4b_log      = {manifest['pipeline_logs']['step4b_docs_freeze_exists']}")
print("=" * 90)
