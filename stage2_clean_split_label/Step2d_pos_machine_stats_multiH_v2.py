# ============================================================
# 文件名：Step2d_pos_machine_stats_multiH_v2.py
# 作用：
#   基于 Step2c 输出的 labeled 表，统计每个 H × split：
#     - valid_rows / pos_rows / pos_rate
#     - valid_machines / pos_machines
#     - 每台正例机器的正例样本数分布（p50/p90/p99/max）
#      -统计正例绝对数量决定了当前结论的可信度上限
# 用途：
#   为后续 machine bootstrap CI 的“有效分母”提供证据（审稿必问点）。
# 输入：
#   01_interim/step2c_machine_slice_labeled_FAILonly_H{4.6.8.12}h_v2.csv
# 输出：
#   05_reports/step2d_pos_machine_stats_multiH_v2.csv
#   05_reports/step2d_pos_machine_stats_multiH_v2.txt
#注释：split-数据集划分：train（训练集）、val（验证集）、test（测试集）.
#valid_rows-标签有效（非 NaN）的样本总数（即剔除故障当刻 slice 和右删失样本后，有明确 0/1 标签的行数）。
#pos_rows-正例样本数，即 label=1（未来 H 小时内会 FAIL）的样本数量。
#valid_machines-至少包含一个有效样本的机器数量（机器 ID 去重后计数）。
#pos_machines-至少包含一个正例样本的机器数量（即发生过 FAIL 事件的机器数，可能包含多个正例 slice）。
# ============================================================

import os
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

INCLUDE_TEST = False   # ★默认 False：不输出 test

FILES = [
    (4,  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H4h_v2.csv")),
    (6,  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H6h_v2.csv")),
    (8,  os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv")),
    (12, os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H12h_v2.csv")),
]

OUT_CSV = os.path.join(REPORT_DIR, "step2d_pos_machine_stats_multiH_v2.csv")
OUT_TXT = os.path.join(REPORT_DIR, "step2d_pos_machine_stats_multiH_v2.txt")

MID = "machine_id"
SPLIT = "split"

splits = ["train", "val"] + (["test"] if INCLUDE_TEST else [])

rows = []
lines = []
lines.append("=== Step2D-aux(v2) pos machine statistics ===")
lines.append(f"INCLUDE_TEST={INCLUDE_TEST}")
lines.append("NOTE: Test split is excluded by default to avoid test information entering early-stage diagnostics.")
lines.append("")

for H, path in FILES:
    label_col = f"label_fail_H{H}h"
    assert os.path.exists(path), f"❌ missing labeled file: {path}"

    df = pd.read_csv(path, usecols=[MID, SPLIT, label_col], low_memory=False)

    # 只统计 label-valid 子集
    dfv = df[df[label_col].notna()].copy()
    dfv[MID] = pd.to_numeric(dfv[MID], errors="coerce").astype("int64")
    dfv[label_col] = pd.to_numeric(dfv[label_col], errors="coerce").round().astype(int)

    dfv[SPLIT] = dfv[SPLIT].astype(str)

    for sp in splits:
        sub = dfv[dfv[SPLIT] == sp]
        if len(sub) == 0:
            rows.append(dict(H_hours=H, split=sp))
            continue

        pos = sub[sub[label_col] == 1]

        valid_rows = int(len(sub))
        pos_rows = int(len(pos))
        pos_rate = float(pos_rows / valid_rows) if valid_rows else np.nan

        valid_machines = int(sub[MID].nunique())
        pos_machines = int(pos[MID].nunique())
        pos_machine_rate = float(pos_machines / valid_machines) if valid_machines else np.nan


        pos_per_machine = pos.groupby(MID).size()
        if len(pos_per_machine):
            p50 = float(pos_per_machine.quantile(0.5))
            p90 = float(pos_per_machine.quantile(0.9))
            p99 = float(pos_per_machine.quantile(0.99))
            mx = int(pos_per_machine.max())
            mean = float(pos_per_machine.mean())
        else:
            p50 = p90 = p99 = mean = np.nan
            mx = 0

        rows.append({
            "H_hours": H,
            "split": sp,
            "valid_rows": valid_rows,
            "pos_rows": pos_rows,
            "pos_rate": pos_rate,
            "valid_machines": valid_machines,
            "pos_machines": pos_machines,
            "pos_per_pos_machine_mean": mean,
            "pos_per_pos_machine_p50": p50,
            "pos_per_pos_machine_p90": p90,
            "pos_per_pos_machine_p99": p99,
            "pos_per_pos_machine_max": mx,
            "pos_machine_rate": pos_machine_rate,

        })

        lines.append(f"[H={H}h split={sp}] valid_rows={valid_rows} pos_rows={pos_rows} "
                     f"valid_machines={valid_machines} pos_machines={pos_machines} "
                     f"pos_per_pos_machine(p50/p90/p99/max)=({p50},{p90},{p99},{mx})")

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("=" * 90)
print("Step2D-aux 完成：pos_machines 统计已保存")
print("=" * 90)
print("[输出]", OUT_CSV)
print("[输出]", OUT_TXT)
print("\n[预览]")
preview_cols = ["H_hours","split","valid_rows","pos_rows","valid_machines","pos_machines","pos_rate"]
print(out.sort_values(["H_hours","split"])[preview_cols].to_string(index=False))