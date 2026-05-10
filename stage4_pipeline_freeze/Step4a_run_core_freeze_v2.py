# ============================================================
# 文件名：Step4a_run_core_freeze_v2.py
# 阶段：Step 4A（Pipeline层｜Docs+Gate｜主线数据冻结入口｜v2｜定稿||一共13个脚本）
#
# 【定位/职责边界（非常重要）】
#   本脚本是"主线数据定义链条"的统一入口（core freeze orchestrator），用于冻结：
#     - clean（Step2a）
#     - main split（Step2b）
#     - labels（Step2c，方案A：first-fail only）
#     - features_raw + aux 注入（Step3a + Step3a_aux）
#     - sched_frac simplex gate（Step3j：尽早验证，fail-fast）
#     - train-only preprocess params（Step3b）
#     - processed 输入表（Step3c：48个CSV）
#   并运行所有"一票否决 GATE"（失败即停止）：
#     - label QC（Step2d：nan_other==0）
#     - time QC（Step2e：dup/mismatch/回跳=0）
#     - processed 健康检查（Step3d：NaN/Inf=0 + schema一致 + 48 files）
#     - processed 泄漏列审计 gate（Step3k）
#     - train-only fit 域审计 gate（Step3k2）
#   最后生成：
#     - Stage0 summary（Step0）
#     - stable log alias（供 Step4c manifest 引用）
#
# 【执行顺序说明】
#   本脚本是 Stage 4 的第一步（4A = 主线冻结），执行完后运行 Step4b（文档冻结），
#   最后由 Step4c 生成 manifest。编号与执行顺序一致。
#
# 【本脚本不做什么（防止证据链污染）】
#   - 不运行探索/诊断/画图脚本（这些放到 Step4b）
#   - 不输出 test 的标签分布统计用于"设定选择"（主线证据链默认 train/val-only）
#   - 不运行 Step4c manifest（manifest 需要读取本脚本的完整 log，
#     必须在本脚本全部完成后由 Step4b 调用，或手动运行）
#
# 【输入】
#   - 项目根目录下的各 Step 脚本（见 STEPS）
#   - 每个 Step 自己读取 raw/interim/processed 等文件
#
# 【输出】
#   - 05_reports/step4a_run_core_freeze_v2_YYYYMMDD_HHMMSS.log（带时间戳的完整日志）
#   - 05_reports/step4a_run_core_freeze_v2_latest.log（稳定别名，供 manifest 引用）
#
# 【Gate 失败条件】
#   - 任一步 returncode != 0 => pipeline 失败（fail-fast）
#   - 任一步脚本文件不存在 => 失败
#
# 【工程设计要点】
#   - 支持"候选文件名自动解析"：同一步脚本可能存在文件名微差版本，
#     只要在 candidates 列表里列出即可，避免因为文件名微差卡住。
#   - stable alias 在所有步骤完成后创建，确保 alias 内容完整、SHA256 稳定。
# ============================================================

import os
import sys
import shutil
import subprocess
import datetime
import platform

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PY = sys.executable

# 带时间戳的 log（每次运行独立）
LOG_PATH = os.path.join(
    REPORT_DIR,
    f"step4a_run_core_freeze_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
# 稳定别名（manifest 引用此路径，内容与带时间戳的 log 完全一致）
LATEST_ALIAS = os.path.join(REPORT_DIR, "step4a_run_core_freeze_v2_latest.log")


def resolve_script(candidates):
    """
    candidates: List[str]，候选脚本相对路径（相对于 ROOT_DIR）
    返回：第一个存在的脚本路径（相对路径）
    """
    for rel in candidates:
        p = os.path.join(ROOT_DIR, rel)
        if os.path.exists(p):
            return rel
    searched = "\n".join(f"  - {os.path.join(ROOT_DIR, r)}" for r in candidates)
    raise FileNotFoundError(f"❌ none of candidates exists:\n{searched}")


def run_step(script_relpath: str):
    """运行单个步骤，stdout/stderr 写入主 log，检查 returncode。"""
    script_path = os.path.join(ROOT_DIR, script_relpath)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"❌ script missing: {script_path}")

    cmd = [PY, script_path]
    start = datetime.datetime.now()

    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 90 + "\n")
        log.write(f"[RUN] {script_relpath}\n")
        log.write(f"start={start.isoformat()}\n")
        log.write(f"cmd={' '.join(cmd)}\n")
        log.flush()
        p = subprocess.run(cmd, cwd=ROOT_DIR, stdout=log, stderr=log)
        end = datetime.datetime.now()
        log.write(f"\nend={end.isoformat()}\n")
        log.write(f"elapsed_seconds={(end - start).total_seconds():.3f}\n")
        log.write(f"returncode={p.returncode}\n")

    if p.returncode != 0:
        raise RuntimeError(
            f"❌ step failed: {script_relpath}\n"
            f"   returncode={p.returncode}\n"
            f"   see log: {LOG_PATH}"
        )


# ------------------------------------------------------------
# 主线冻结步骤（Core + GATE + Summary）
# 注意：每个元素是"候选文件名列表"，resolve 后得到真实脚本
#
# 执行顺序设计原则：
#   1) Stage2 core（clean → split → label）
#   2) Stage2 gates（标签 QC + 时间轴 QC）
#   3) Stage3 core（特征 → 注入 → simplex 验证 → 拟合 → 导出）
#      - Step3j 紧跟 Step3a_aux：在投入 Step3b/3c 计算前尽早验证 sched_frac
#   4) Stage3 gates（针对 processed 表的健康检查 + 泄漏审计 + fit 域审计）
#   5) Summary（汇总所有证据链）
#stag2 未纳入脚本的理由：
#Step2c_aux==不产出任何训练表，只产出 H 选择的统计证据||Step2d_pos（正例机器统计）==只统计 pos_machines/valid_machines，不改变标签
#Step2f（post-fail 观测报告）==只统计 FAIL 后是否有后续 slice，纯描述性||Step2g（sched/prio 诊断）==只做特征价值诊断，不产出训练特征
#Step2h（CPU 分布审计）==只审计 CPU 向量长度稳定性，不产出训练特征||Step2h2（cluster holdout）==只产出额外的 split 列表，不替代主 split
#stag3 未纳入脚本的理由：
#Step3e1（稀疏/常数列汇总）==纯报告脚本，基于 Step3d 的列级统计做汇总||Step3f（报告打包）==汇总特征清单、missing vs label 统计等
#Step3i autofill（过渡版）==不含 clip/scale 参数的快速核对版||Step3i final（定稿字典）==生成论文附录用的特征字典
# ------------------------------------------------------------
STEPS = [
    # ── Stage 2 core ──
    ["Step2a_machine_slice主表构建_clean_v2.py"],
    ["Step2b_训练验证测试机器划分与固化_v2.py"],
    ["Step2c_预警标签生成_FAILonly_多窗口_v2.py"],

    # ── Stage 2 gates ──
    ["Step2d_标签与样本质量检查_汇总报告_v2.py"],
    ["Step2e_time_order_qc_v2.py"],

    # ── Stage 3 core ──
    ["Step3a_派生特征构造_v2.py"],
    ["Step3a_aux_raw_cpu_dist_vector_merge_v2.py"],
    ["step3j_sched_frac_simplex_qc_v2.py"],       # 紧跟 aux：尽早验证比例向量
    ["Step3b_preprocess参数拟合_trainonly_H8h_v2.py"],
    ["Step3c_transform导出模型输入表_multiH_v2.py"],

    # ── Stage 3 gates（processed 表检查）──
    ["Step3d_输入表健康检查与稀疏性报告_v2.py"],
    ["Step3k_leakage_audit_processed_v2.py"],
    ["Step3k2_fit_domain_audit_v2.py"],

    # ── Summary ──
    ["Step0_stage1-3_summary_report_v2.py"],       # 注意：下划线，非连字符
]

# ------------------------------------------------------------
# 执行
# ------------------------------------------------------------
print("=" * 90)
print("Step4A(v2) Core freeze orchestrator (Stage2-3 + gates + summary)")
print("[log]", LOG_PATH)
print("[alias]", LATEST_ALIAS)
print(f"[total steps] {len(STEPS)}")
print("=" * 90)

# 写 log 头部（计划步骤清单，便于审阅者了解预期流程）
with open(LOG_PATH, "w", encoding="utf-8") as log:
    log.write("=== Step4A core freeze ===\n")
    log.write(f"root_dir={ROOT_DIR}\n")
    log.write(f"python={PY}\n")
    log.write(f"platform={platform.platform()}\n")
    log.write(f"start_time={datetime.datetime.now().isoformat()}\n")
    log.write(f"total_steps={len(STEPS)}\n")
    log.write("\n[Planned steps]\n")
    for i, cand in enumerate(STEPS):
        log.write(f"  {i + 1:2d}. {cand[0]}\n")
    log.write("\n")

# 逐步执行
for i, cand in enumerate(STEPS):
    script = resolve_script(cand)
    print(f"\n[{i + 1}/{len(STEPS)}] {script}")
    run_step(script)

# ------------------------------------------------------------
# 所有核心步骤完成后：写入 log 终结标记
# ------------------------------------------------------------
with open(LOG_PATH, "a", encoding="utf-8") as log:
    log.write("\n" + "=" * 90 + "\n")
    log.write("ALL CORE STEPS PASSED\n")
    log.write(f"end_time={datetime.datetime.now().isoformat()}\n")

# ------------------------------------------------------------
# 创建 stable alias（此时 log 已完整，SHA256 稳定）
# Step4c manifest 将引用此 alias 文件计算 hash
# ------------------------------------------------------------
shutil.copy2(LOG_PATH, LATEST_ALIAS)

# ------------------------------------------------------------
# 控制台输出（不写 log，避免修改已完成的 log 内容）
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("✅ Step4A: ALL CORE FREEZE STEPS PASSED")
print(f"[log]     {LOG_PATH}")
print(f"[alias]   {LATEST_ALIAS}")
print(f"[next]    运行 Step4b（docs/variant freeze），")
print(f"          最后由 Step4c 生成 manifest（引用此 alias 的 SHA256）")
print("=" * 90)
