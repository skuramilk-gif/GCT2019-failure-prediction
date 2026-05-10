# ============================================================
# 文件名：Step4b_run_docs_variants_freeze_v2.py
# 阶段：Step 4B（Pipeline层｜Docs + Variant 冻结入口｜v2｜定稿||一共12个脚本）
#
# 【定位/职责边界（非常重要）】
#   本脚本用于冻结"论文/附录要引用的证据链报告（Docs）"与"同数据集更强泛化设定（Variant）"。
#   它与 Step4A 的区别：
#     - Step4A：冻结主线数据定义链条，产出最终训练输入 processed(48 files) + 一票否决 gate
#     - Step4B（本脚本）：冻结写作证据与扩展设定，不改变主线数据定义
#     - Step4C：在 Step4A 和 Step4B 都完成后，生成 manifest 复现指纹
#
# 【为什么需要 Step4B】
#   - Docs/diagnostics 脚本很多，不适合塞进主线 freeze（会慢、会漂移、可能引入 test 描述）
#   - 但最终投稿前需要一次性把"证据链报告"跑全，并纳入 manifest hash
#
# 【运行前置（强约束）】
#   - 必须先运行 Step4A，确保：
#       02_processed/ 已存在（Step3f/Step3i 需要读取 processed 表头）
#       主线 gate 全 PASS
#       step4a_run_core_freeze_v2_latest.log 已创建
#
# 【执行顺序】
#   Step4A（主线冻结）→ Step4B（本脚本，文档冻结）→ Step4C（生成 manifest）
#   编号与执行顺序一致。
#
# 【本脚本包含的典型内容】
#   Stage1 docs：raw 语义证据（Step1a/1b）
#   Stage2 docs：horizon scan、pos_machines、post-fail observation、sched/priority诊断、CPU结构审计等
#   Stage2 variant：cluster holdout split（用于泛化设定扩展）
#   Stage3 docs：特征稀疏/常数列、报告 bundle、特征字典 autofill/final
#   最后：再跑一次 Step0 summary（把 docs/variant 产物也纳入总报告）
#
# 【输出】
#   - 05_reports/step4b_run_docs_variants_freeze_v2_YYYYMMDD_HHMMSS.log（带时间戳）
#   - 05_reports/step4b_run_docs_variants_freeze_v2_latest.log（稳定别名，供 manifest 引用）
#
# 【Gate 失败条件】
#   - 任一步 returncode != 0 => fail-fast
#   - 任一步脚本文件不存在 => 失败
#
# 【SHA256 自洽性说明】
#   本脚本在所有步骤完成后、写入终结标记后，才创建 stable alias。
#   Step4C manifest 读取此 alias 时内容完整，SHA256 自洽。
#   本脚本不调用 Step4C manifest——manifest 由用户或外部流程单独运行。
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
    f"step4b_run_docs_variants_freeze_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
# 稳定别名（manifest 引用此路径，内容与带时间戳的 log 完全一致）
LATEST_ALIAS = os.path.join(REPORT_DIR, "step4b_run_docs_variants_freeze_v2_latest.log")


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
# Docs + Variant 冻结步骤
# 注意：每个元素是候选文件名列表，resolve 后得到真实脚本
#
# 执行顺序设计原则：
#   1) Stage1 docs（raw 语义证据，不依赖 processed 表）
#   2) Stage2 docs/diagnostics（标签统计、观测机制、特征诊断）
#   3) Stage2 variant（cluster holdout）
#   4) Stage3 docs（稀疏/常数列、报告 bundle、特征字典）
#   5) Summary（汇总所有证据链，包括本次 docs 产出）
# ------------------------------------------------------------
STEPS = [
    # ── Stage 1 docs ──
    ["Step1a_真实数据统计与参数提取_v2.py"],
    ["Step1b真实数据字段语义检测-v2.py"],

    # ── Stage 2 docs/diagnostics（train/val-only 版本为准）──
    ["Step2c_aux_H窗口扫描_v2.py"],
    ["Step2d_pos_machine_stats_multiH_v2.py"],
    ["Step2f_post_fail_observation_report_v2.py"],
    ["Step2g_sched_priority_and_cpu_dist_diagnostic_v2.py"],

    # ── Stage 2 audit + variant split ──
    ["Step2h_CPU_distribution_全量结构审计_v2.py",
     "Step2h-CPU_distribution_全量结构审计-v2.py"],
    ["Step2h2_cluster_holdout_split_v2.py"],

    # ── Stage 3 docs ──
    ["Step3e1_特征稀疏与常数列汇总_v2.py"],
    ["Step3f_reports_bundle_v2.py"],
    # Step3i autofill：文件名可能带 Step3j 前缀（历史遗留），列出候选
    ["Step3i_feature_dictionary_autofill_allsets_v2.py",
     "Step3j_feature_dictionary_autofill_allsets_v2.py"],
    ["Step3i_feature_dictionary_final_allsets_v2.py"],

    # ── Summary（把 docs/variant 产物也纳入总报告）──
    ["Step0_stage1-3_summary_report_v2.py"],       # 注意：下划线，非连字符
]

# ------------------------------------------------------------
# 执行
# ------------------------------------------------------------
print("=" * 90)
print("Step4B(v2) Docs+Variants freeze orchestrator")
print("[log]", LOG_PATH)
print("[alias]", LATEST_ALIAS)
print(f"[total steps] {len(STEPS)}")
print(f"[prerequisite] Step4A must have been run (processed tables + gates)")
print("=" * 90)

# 写 log 头部（计划步骤清单）
with open(LOG_PATH, "w", encoding="utf-8") as log:
    log.write("=== Step4B docs+variants freeze ===\n")
    log.write(f"root_dir={ROOT_DIR}\n")
    log.write(f"python={PY}\n")
    log.write(f"platform={platform.platform()}\n")
    log.write(f"start_time={datetime.datetime.now().isoformat()}\n")
    log.write(f"total_steps={len(STEPS)}\n")
    log.write(f"prerequisite=Step4A (step4a_run_core_freeze_v2_latest.log must exist)\n")
    log.write("\n[Planned steps]\n")
    for i, cand in enumerate(STEPS):
        log.write(f"  {i + 1:2d}. {cand[0]}\n")
    log.write("\n")

# 前置检查：Step4A 的 stable alias 必须存在
alias_step4a = os.path.join(REPORT_DIR, "step4a_run_core_freeze_v2_latest.log")
if not os.path.exists(alias_step4a):
    raise FileNotFoundError(
        f"❌ Step4A 的 stable alias 不存在：{alias_step4a}\n"
        f"   请先运行 Step4A（主线冻结），再运行本脚本。"
    )

# 逐步执行
for i, cand in enumerate(STEPS):
    script = resolve_script(cand)
    print(f"\n[{i + 1}/{len(STEPS)}] {script}")
    run_step(script)

# ------------------------------------------------------------
# 所有文档步骤完成后：写入 log 终结标记
# ------------------------------------------------------------
with open(LOG_PATH, "a", encoding="utf-8") as log:
    log.write("\n" + "=" * 90 + "\n")
    log.write("ALL DOCS/VARIANTS STEPS PASSED\n")
    log.write(f"end_time={datetime.datetime.now().isoformat()}\n")

# ------------------------------------------------------------
# 创建 stable alias（此时 log 已完整，SHA256 稳定）
# Step4C manifest 将引用此 alias 文件计算 hash
# ------------------------------------------------------------
shutil.copy2(LOG_PATH, LATEST_ALIAS)

# ------------------------------------------------------------
# 控制台输出（不写 log，避免修改已完成的 log 内容）
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("✅ Step4B: ALL DOCS/VARIANTS STEPS PASSED")
print(f"[log]     {LOG_PATH}")
print(f"[alias]   {LATEST_ALIAS}")
print(f"[next]    运行 Step4C（生成 manifest），")
print(f"          引用此 alias 和 Step4A alias 的 SHA256")
print("=" * 90)
