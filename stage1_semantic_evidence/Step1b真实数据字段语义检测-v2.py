# ============================================================
# 文件名：Step1b真实数据字段语义检测-v2.py
# 阶段：Step 1B（v2，Docs-only）
#
# 作用（审稿可 defend 的语义证据）：
#   1) 时间字段语义验证：time/start_time/end_time 的关系是否自洽
#   2) duration 截断证据：end_time-start_time 是否大量等于 300s（用微秒整型判定，避免浮点误差）
#   3) failed 与 event、instance_events_type 的等价性证据（防泄漏）
#   4) CPU average_usage 与 request 的比值范围（粗检查）
#   5) （补充）机器层面：EVICT 与 FAIL 的相关性（仅证据，不做因果结论）
#
# 关键修正（v2 强化）：
#   - same_rate 严格在 failed∈{0,1} 子域统计（避免被无效 failed 稀释）
#   - duration==300 使用 dur_us == 300_000_000 判定（不用浮点秒等号）
#   - 同时输出两个域的统计：
#       RAW_ALL：原始读取后，仅处理 INT64_MAX 哨兵为 NaN
#       VALID_LIKE_STEP2A：与 Step2a 训练域一致的过滤（machine_id>=0 且 0<time<INT64_MAX）
#
# 输入：
#   00_raw/borg_traces_data.csv
#
# 输出（05_reports/）：
#   step1b_time_semantics_report_v2.txt
#   step1b_event_failed_instance_crosstab_v2.csv
#   step1b_duration_quantiles_v2.csv
#
# 注意：
#   - 本脚本仅输出证据报告，不生成训练数据、不写 interim/processed。
# ============================================================

import os
import pandas as pd
import numpy as np
import ast

# ============================================================
# 0) 路径设置（v2 根目录）
# ============================================================
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

OUT_TXT = os.path.join(REPORT_DIR, "step1b_time_semantics_report_v2.txt")
OUT_CT  = os.path.join(REPORT_DIR, "step1b_event_failed_instance_crosstab_v2.csv")
OUT_DUR = os.path.join(REPORT_DIR, "step1b_duration_quantiles_v2.csv")

INT64_MAX = 9223372036854775807
NROWS = None  # None=全量

# ============================================================
# 1) 工具函数
# ============================================================
def save_csv_and_preview(df: pd.DataFrame, path: str, name: str, head_n: int = 15):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[输出保存] {name}: {path}")
    print(f"[预览] {name} shape={df.shape}")
    print(df.head(head_n).to_string(index=False))

def parse_dict(val):
    try:
        d = ast.literal_eval(str(val))
        if not isinstance(d, dict):
            return np.nan, np.nan
        cpu = d.get("cpus", np.nan)
        mem = d.get("memory", np.nan)
        if cpu is None: cpu = np.nan
        if mem is None: mem = np.nan
        return cpu, mem
    except Exception:
        return np.nan, np.nan

def compute_core_stats(df0: pd.DataFrame, domain_name: str) -> dict:
    """
    返回该 domain 的关键证据统计（用于写入 txt）
    """
    out = {"domain": domain_name, "rows": int(len(df0))}
    if len(df0) == 0:
        return out

    # 时间字段有效行
    valid_time = df0[["time", "start_time", "end_time"]].notna().all(axis=1)
    out["valid_time_rows"] = int(valid_time.sum())

    if out["valid_time_rows"] > 0:
        tv = df0.loc[valid_time, "time"]
        stv = df0.loc[valid_time, "start_time"]
        etv = df0.loc[valid_time, "end_time"]

        out["time_in_window_rate"] = float(((tv >= stv) & (tv <= etv)).mean())
        out["time_before_start_rate"] = float((tv < stv).mean())
        out["time_after_end_rate"] = float((tv > etv).mean())

        dur_us = (etv - stv).astype("float64")
        out["neg_duration_rate"] = float((dur_us < 0).mean())
        out["dur_300s_rate"] = float((dur_us == 300_000_000).mean())

        # duration quantiles（秒）——只用于输出分布，不用于 gate
        dur_sec = (dur_us / 1e6)
        q = pd.Series(dur_sec).dropna().quantile([0, 0.5, 0.9, 0.95, 0.99, 1.0]).to_dict()
        out["duration_sec_quantiles"] = {float(k): float(v) for k, v in q.items()}
    else:
        out["time_in_window_rate"] = np.nan
        out["time_before_start_rate"] = np.nan
        out["time_after_end_rate"] = np.nan
        out["neg_duration_rate"] = np.nan
        out["dur_300s_rate"] = np.nan
        out["duration_sec_quantiles"] = {}

    # failed 与 event==FAIL 等价性（只在 failed∈{0,1} 子域）
    mask01 = df0["failed01"].isin([0, 1])
    out["failed01_rows"] = int(mask01.sum())
    if mask01.any():
        out["same_rate_failed_equals_eventFAIL"] = float(
            (df0.loc[mask01, "failed01"] == (df0.loc[mask01, "event"] == "FAIL").astype(int)).mean()
        )
    else:
        out["same_rate_failed_equals_eventFAIL"] = np.nan

    # CPU avg/request ratio quantiles
    avg = df0["average_usage"].apply(parse_dict)
    req = df0["resource_request"].apply(parse_dict)
    cpu_avg = pd.Series([p[0] for p in avg])
    cpu_req = pd.Series([p[0] for p in req])
    ratio = (cpu_avg / cpu_req.replace(0, np.nan)).dropna()

    if len(ratio):
        rq = ratio.quantile([0, 0.5, 0.9, 0.95, 0.99, 1.0]).to_dict()
        out["cpu_avg_over_req_quantiles"] = {float(k): float(v) for k, v in rq.items()}
    else:
        out["cpu_avg_over_req_quantiles"] = {}

    # machine-level evict vs fail corr（raw-level evidence only）
    mp = (df0.groupby("machine_id")
          .agg(
              n=("machine_id", "count"),
              evict_rate=("event", lambda s: (s == "EVICT").mean()),
              fail_rate=("event", lambda s: (s == "FAIL").mean()),
          )
          .reset_index())
    mp2 = mp[mp["n"] >= 2].copy()
    if len(mp2) > 10:
        out["machines_n_ge_2"] = int(len(mp2))
        out["corr_evict_rate_fail_rate"] = float(mp2[["evict_rate", "fail_rate"]].corr().iloc[0, 1])
    else:
        out["machines_n_ge_2"] = int(len(mp2))
        out["corr_evict_rate_fail_rate"] = np.nan

    return out

# ============================================================
# 2) 读取必要列
# ============================================================
print("=" * 90)
print("Step 1B(v2)：真实数据字段语义检测（RAW_ALL vs VALID_LIKE_STEP2A）")
print("=" * 90)
print(f"[输入] {RAW_PATH}")
print(f"[输出目录] {REPORT_DIR}")

usecols = [
    "machine_id", "time", "start_time", "end_time",
    "average_usage", "resource_request",
    "event", "failed", "instance_events_type",
]

df = pd.read_csv(RAW_PATH, usecols=usecols, nrows=NROWS, low_memory=False)

# 数值化 + 清洗哨兵（仅用于语义判断；不改变原始文件）
# 记录 INT64_MAX 哨兵率（作为证据）
sentinel_rates = {}
for c in ["time", "start_time", "end_time"]:
    x = pd.to_numeric(df[c], errors="coerce")
    sentinel_rates[c] = float((x >= INT64_MAX).mean())
    x.loc[x >= INT64_MAX] = np.nan
    df[c] = x

df["machine_id"] = pd.to_numeric(df["machine_id"], errors="coerce")
df["event"] = df["event"].astype(str)
df["failed01"] = pd.to_numeric(df["failed"], errors="coerce").fillna(-1).astype(int)
df["instance_events_type"] = pd.to_numeric(df["instance_events_type"], errors="coerce")

rows_total = int(len(df))
miss_time = float(df["time"].isna().mean())
miss_start = float(df["start_time"].isna().mean())
miss_end = float(df["end_time"].isna().mean())

# VALID_LIKE_STEP2A 域（与后续训练域一致）
valid_like_step2a = (
    df["machine_id"].notna() & (df["machine_id"] >= 0) &
    df["time"].notna() & (df["time"] > 0) & (df["time"] < INT64_MAX)
)
df_valid = df.loc[valid_like_step2a].copy()

print(f"[数据] RAW_ALL rows={rows_total}")
print(f"[数据] VALID_LIKE_STEP2A rows={len(df_valid)} rate={len(df_valid)/max(rows_total,1):.6f}")

# ============================================================
# 3) duration quantiles 输出（使用 RAW_ALL valid_time 子集；这是全局证据）
# ============================================================
valid_time = df[["time", "start_time", "end_time"]].notna().all(axis=1)
if valid_time.any():
    dur_us = (df.loc[valid_time, "end_time"] - df.loc[valid_time, "start_time"]).astype("float64")
    dur_sec = dur_us / 1e6
    dur_q = pd.Series(dur_sec).dropna().quantile([0, 0.5, 0.9, 0.95, 0.99, 1.0]).reset_index()
    dur_q.columns = ["quantile", "duration_sec"]
else:
    dur_q = pd.DataFrame({"quantile": [0,0.5,0.9,0.95,0.99,1.0], "duration_sec": [np.nan]*6})

save_csv_and_preview(dur_q, OUT_DUR, "step1b_duration_quantiles_v2.csv", head_n=10)

# ============================================================
# 4) 交叉表证据（RAW_ALL，failed01 不限定；但后续 same_rate 是条件一致率）
# ============================================================
ct_event = pd.crosstab(df["event"], df["failed01"], dropna=False).reset_index()
ct_event["key_type"] = "event"
ct_event = ct_event.rename(columns={"event": "key_value"})

ct_type = pd.crosstab(df["instance_events_type"], df["failed01"], dropna=False).reset_index()
ct_type["key_type"] = "instance_events_type"
ct_type = ct_type.rename(columns={"instance_events_type": "key_value"})

ct_all = pd.concat([ct_event, ct_type], ignore_index=True)

for col in [0, 1]:
    if col not in ct_all.columns:
        ct_all[col] = 0

ct_all["count_01"] = ct_all[0] + ct_all[1]
ct_all["failed_rate_on01"] = ct_all[1] / ct_all["count_01"].replace(0, np.nan)

ct_all = ct_all.sort_values(["key_type", "count_01"], ascending=[True, False]).reset_index(drop=True)
save_csv_and_preview(ct_all, OUT_CT, "step1b_event_failed_instance_crosstab_v2.csv", head_n=25)

# ============================================================
# 5) 计算两域核心统计并写报告
# ============================================================
stats_raw = compute_core_stats(df, "RAW_ALL")
stats_valid = compute_core_stats(df_valid, "VALID_LIKE_STEP2A")

report_lines = []
report_lines.append("=== Step1B(v2) 时间语义与等价性报告（RAW_ALL vs VALID_LIKE_STEP2A） ===")
report_lines.append("")
report_lines.append("[Input summary]")
report_lines.append(f"rows_raw_all={rows_total}")
report_lines.append(f"rows_valid_like_step2a={len(df_valid)}  rate={len(df_valid)/max(rows_total,1):.6f}")
report_lines.append("")
report_lines.append("[Missing/sentinel evidence on time fields (RAW_ALL)]")
report_lines.append(f"time_missing_rate={miss_time:.6f} start_time_missing_rate={miss_start:.6f} end_time_missing_rate={miss_end:.6f}")
report_lines.append(f"time_sentinel(INT64_MAX)_rate={sentinel_rates['time']:.6f} "
                    f"start_sentinel_rate={sentinel_rates['start_time']:.6f} end_sentinel_rate={sentinel_rates['end_time']:.6f}")
report_lines.append("")

def dump_domain(d: dict):
    report_lines.append(f"[{d['domain']}]")
    report_lines.append(f"rows={d.get('rows',np.nan)} valid_time_rows={d.get('valid_time_rows',np.nan)} failed01_rows={d.get('failed01_rows',np.nan)}")
    report_lines.append("[Time semantics]")
    report_lines.append(f"time_in_[start,end]_rate={d.get('time_in_window_rate',np.nan):.6f}")
    report_lines.append(f"time<start_rate={d.get('time_before_start_rate',np.nan):.6f}")
    report_lines.append(f"time>end_rate={d.get('time_after_end_rate',np.nan):.6f}")
    report_lines.append("[Duration]")
    report_lines.append(f"neg_duration_rate={d.get('neg_duration_rate',np.nan):.6f}")
    report_lines.append(f"duration==300s_rate={d.get('dur_300s_rate',np.nan):.6f}  (判定: dur_us==300_000_000)")
    report_lines.append(f"duration_sec_quantiles={d.get('duration_sec_quantiles',{})}")
    report_lines.append("[Leakage equivalence evidence]")
    report_lines.append(f"same_rate_failed_equals_eventFAIL(condition on failed in {{0,1}})={d.get('same_rate_failed_equals_eventFAIL',np.nan):.6f}")
    report_lines.append("[CPU avg/request ratio quantiles]")
    report_lines.append(str(d.get("cpu_avg_over_req_quantiles", {})))
    report_lines.append("[Machine-level evict vs fail correlation]")
    report_lines.append(f"machines(n>=2)={d.get('machines_n_ge_2',np.nan)} corr={d.get('corr_evict_rate_fail_rate',np.nan)}")
    report_lines.append("")

dump_domain(stats_raw)
dump_domain(stats_valid)

report_lines.append("[Outputs]")
report_lines.append(f"time_semantics_txt={OUT_TXT}")
report_lines.append(f"crosstab_csv={OUT_CT}")
report_lines.append(f"duration_quantiles_csv={OUT_DUR}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# 控制台摘要（关键结论）
print("\n" + "=" * 90)
print("Step 1B(v2) 控制台摘要（关键结论）")
print("=" * 90)
print("RAW_ALL:     time_in_window_rate=", f"{stats_raw.get('time_in_window_rate',np.nan):.6f}",
      "dur300_rate=", f"{stats_raw.get('dur_300s_rate',np.nan):.6f}",
      "same_rate=", f"{stats_raw.get('same_rate_failed_equals_eventFAIL',np.nan):.6f}")
print("VALID_LIKE:  time_in_window_rate=", f"{stats_valid.get('time_in_window_rate',np.nan):.6f}",
      "dur300_rate=", f"{stats_valid.get('dur_300s_rate',np.nan):.6f}",
      "same_rate=", f"{stats_valid.get('same_rate_failed_equals_eventFAIL',np.nan):.6f}")

print("\n[输出文件]")
print(" ", OUT_TXT)
print(" ", OUT_CT)
print(" ", OUT_DUR)
print("=" * 90)
print("Step1B(v2) 完成")