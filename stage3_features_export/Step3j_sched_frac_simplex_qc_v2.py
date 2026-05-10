# ============================================================
# 文件名：step3j_sched_frac_simplex_qc_v2.py
# 作用：
#   Gate：验证 step3a_features_raw_v2.csv 中 sched_frac_0..3 为合法比例向量：
#     1) 不允许 NaN
#     2) 每分量在[0,1]
#     3) sum≈1（有观测）或 sum≈0（无观测→全0填充）
#   若失败：直接抛错（不允许进入训练）
# ============================================================
# 作用：
#   验证 sched_frac_0..3 是否符合比例向量的性质：
#     1) 每个分量在 [0,1] 区间
#     2) 总和应 ≈1（有观测时）或 ≈0（无观测时，填充0）
#   若不符合，说明聚合或填充逻辑存在问题，需回溯 Step3a_aux 脚本。
# ============================================================

import os
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
IN_FEAT = os.path.join(ROOT_DIR, "01_interim", "step3a_features_raw_v2.csv")
OUT = os.path.join(ROOT_DIR, "05_reports", "step3j_sched_frac_simplex_qc_v2.txt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

cols = ["sched_frac_0","sched_frac_1","sched_frac_2","sched_frac_3"]
assert os.path.exists(IN_FEAT), f"❌ 缺少输入文件: {IN_FEAT}"

df = pd.read_csv(IN_FEAT, usecols=cols, low_memory=False)

EPS_RANGE = 1e-6
EPS_SUM1 = 1e-4
EPS_SUM0 = 1e-9

# 不允许 NaN（否则说明 merge/聚合没覆盖）
nan_cnt = int(df[cols].isna().sum().sum())
assert nan_cnt == 0, f"❌ sched_frac 存在 NaN，总数={nan_cnt}"

X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

# 范围检查
in_range = np.all((X >= -EPS_RANGE) & (X <= 1 + EPS_RANGE), axis=1)

# 和检查：≈1 或 ≈0
sched_sum = X.sum(axis=1)
sum_close_1 = np.isclose(sched_sum, 1.0, atol=EPS_SUM1)
sum_close_0 = np.isclose(sched_sum, 0.0, atol=EPS_SUM0)

bad = (~in_range) | (~(sum_close_1 | sum_close_0))
bad_count = int(bad.sum())

q = pd.Series(sched_sum).quantile([0,0.01,0.5,0.99,1.0]).to_dict()

lines = []
lines.append("=== Step3J sched_frac simplex QC (GATE) ===")
lines.append(f"input={IN_FEAT}")
lines.append(f"rows={len(df)}")
lines.append("")
lines.append("[Rates]")
lines.append(f"in_[0,1]_rate={float(in_range.mean()):.6f}")
lines.append(f"sum_close_1_rate={float(sum_close_1.mean()):.6f}")
lines.append(f"sum_close_0_rate={float(sum_close_0.mean()):.6f}")
lines.append(f"bad_rate={float(bad.mean()):.6f}  bad_count={bad_count}")
lines.append("")
lines.append("[sched_sum quantiles]")
lines.append(str({k: float(v) for k, v in q.items()}))

# 输出坏样例（最多20行）
if bad_count > 0:
    bad_idx = np.where(bad)[0][:20]
    lines.append("")
    lines.append("[Bad examples head(20)]")
    for i in bad_idx:
        lines.append(f"row={int(i)} vals={X[i,:].tolist()} sum={float(sched_sum[i])}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print("[输出]", OUT)

# Gate：必须 0 个 bad
assert bad_count == 0, f"❌ Step3J failed: bad_count={bad_count}"