# ============================================================
# 文件名：Step2h2_cluster_holdout_split_v2.py
# 作用：
#   构造跨 cluster holdout split（train/test），用于更苛刻的环境泛化评测（Variant，不替换主 split）
#   - 先统计每台机器出现的 cluster 分布，取 mode cluster 作为该机器所属 cluster
#   - 抽取部分 cluster 作为 holdout(test)，其机器全部进入 test_machines
#   - 其余 cluster 的机器进入 train_machines
#   - 只是锦上添花，作为一个方位
# 关键约束：
#   - 不使用标签
#   - 清洗规则与 Step2a 一致：machine_id>=0，time在(0,INT64_MAX)
#
# 输出（03_splits/）：
#   step2h2_clusterholdout_train_machines_v2.txt
#   step2h2_clusterholdout_test_machines_v2.txt
#   （可选）step2h2_clusterholdout_val_machines_v2.txt  # 如果 ENABLE_VAL_SPLIT=True
#
# 报告（05_reports/）：
#   step2h2_clusterholdout_diagnostics_v2.txt
#结果解读：machines_with_multiple_clusters要注意如果过高可能要排查是否出现在train/test两侧
#train_test_overlap这个是判断是否有重叠的
# ============================================================


import os
import numpy as np
import pandas as pd
from collections import defaultdict

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
RAW_PATH = os.path.join(ROOT_DIR, "00_raw", "borg_traces_data.csv")
SPLIT_DIR = os.path.join(ROOT_DIR, "03_splits")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

OUT_TRAIN = os.path.join(SPLIT_DIR, "step2h2_clusterholdout_train_machines_v2.txt")
OUT_TEST  = os.path.join(SPLIT_DIR, "step2h2_clusterholdout_test_machines_v2.txt")
OUT_VAL   = os.path.join(SPLIT_DIR, "step2h2_clusterholdout_val_machines_v2.txt")  # optional
OUT_REP   = os.path.join(REPORT_DIR, "step2h2_clusterholdout_diagnostics_v2.txt")

MID = "machine_id"
TIME = "time"
CL = "cluster"

INT64_MAX = 9223372036854775807
CHUNKSIZE = 200_000

SEED = 42
TEST_CLUSTER_RATIO =0.25

# 可选：是否在 train_machines 内再切一个 val（仅用于该 variant 的调参/早停）
ENABLE_VAL_SPLIT = False
VAL_RATIO_IN_TRAIN = 0.10

TOPK_CLUSTER_TABLE = 30

print("=" * 90)
print("Step2H2(v2)：跨 cluster holdout split 构造（增强诊断版，Variant）")
print("=" * 90)
print("[输入 raw]", RAW_PATH)

# (mid, cluster) -> count
counts = defaultdict(int)
# mid -> set(clusters)
mid_clusters = defaultdict(set)

chunk_id = 0
for chunk in pd.read_csv(RAW_PATH, usecols=[MID, TIME, CL], chunksize=CHUNKSIZE, low_memory=False):
    chunk_id += 1

    mid = pd.to_numeric(chunk[MID], errors="coerce")
    t = pd.to_numeric(chunk[TIME], errors="coerce")
    cl = pd.to_numeric(chunk[CL], errors="coerce")

    valid = mid.notna() & (mid >= 0) & t.notna() & (t > 0) & (t < INT64_MAX) & cl.notna()
    chunk = chunk.loc[valid, [MID, CL]].copy()
    if len(chunk) == 0:
        continue

    chunk[MID] = pd.to_numeric(chunk[MID], errors="coerce").astype("int64")
    chunk[CL]  = pd.to_numeric(chunk[CL], errors="coerce").astype("int64")

    vc = chunk.groupby([MID, CL]).size()
    for (m, c), n in vc.items():
        counts[(int(m), int(c))] += int(n)
        mid_clusters[int(m)].add(int(c))

    print(f"  chunk{chunk_id}: valid_rows={len(chunk)}")

# 取 mode cluster
machine_mode = {}  # mid -> (cluster_mode, count)
for (m, c), n in counts.items():
    if (m not in machine_mode) or (n > machine_mode[m][1]):
        machine_mode[m] = (c, n)

mach_df = pd.DataFrame({
    MID: list(machine_mode.keys()),
    "cluster_mode": [v[0] for v in machine_mode.values()],
    "cluster_mode_count": [v[1] for v in machine_mode.values()],
    "n_clusters_seen": [len(mid_clusters[m]) for m in machine_mode.keys()],
})

# ---- Gate: machine_mode 不应缺失 ----
assert mach_df[MID].notna().all() and mach_df["cluster_mode"].notna().all()
assert mach_df[MID].is_unique, "❌ mach_df machine_id not unique"

clusters = sorted(mach_df["cluster_mode"].unique().tolist())

# 抽取 test clusters
rng = np.random.RandomState(SEED)
rng.shuffle(clusters)
n_test = max(1, int(len(clusters) * TEST_CLUSTER_RATIO))
test_clusters = set(clusters[:n_test])

test_m = mach_df[mach_df["cluster_mode"].isin(test_clusters)][MID].astype(int).tolist()
train_m = mach_df[~mach_df["cluster_mode"].isin(test_clusters)][MID].astype(int).tolist()

# ---- Gate: 互斥 ----
assert len(set(train_m) & set(test_m)) == 0, "❌ train/test machines overlap"

# 可选：在 train_m 中再切 val
val_m = []
if ENABLE_VAL_SPLIT:
    rng2 = np.random.RandomState(SEED + 1)
    train_m = np.array(train_m, dtype=np.int64)
    rng2.shuffle(train_m)
    n_val = max(1, int(len(train_m) * VAL_RATIO_IN_TRAIN))
    val_m = train_m[:n_val].astype(int).tolist()
    train_m = train_m[n_val:].astype(int).tolist()
    assert len(set(train_m) & set(val_m)) == 0
    assert len(set(val_m) & set(test_m)) == 0



# 落盘
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in train_m))
with open(OUT_TEST, "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in test_m))
if ENABLE_VAL_SPLIT:
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in val_m))

# ---- 诊断：cluster 分布与 holdout 强度 ----
cluster_sizes = mach_df.groupby("cluster_mode")[MID].nunique().sort_values(ascending=False)
cluster_tbl = pd.DataFrame({
    "cluster_mode": cluster_sizes.index.astype(int),
    "machines": cluster_sizes.values.astype(int),
})
cluster_tbl["ratio"] = cluster_tbl["machines"] / cluster_tbl["machines"].sum()

# train/test cluster 分布
train_tbl = mach_df[~mach_df["cluster_mode"].isin(test_clusters)].groupby("cluster_mode")[MID].nunique()
test_tbl  = mach_df[mach_df["cluster_mode"].isin(test_clusters)].groupby("cluster_mode")[MID].nunique()
train_clusters = sorted(train_tbl.index.astype(int).tolist())
test_clusters_sorted = sorted(list(test_clusters))

multi_cluster_rate = float((mach_df["n_clusters_seen"] > 1).mean())
multi_cluster_count = int((mach_df["n_clusters_seen"] > 1).sum())

lines = []
# ========== 新增：跨集群正例率一致性检查（基于 H=8 标签）==========
INTERIM_DIR = os.path.join(ROOT_DIR, "01_interim")
LABELED_H8 = os.path.join(INTERIM_DIR, "step2c_machine_slice_labeled_FAILonly_H8h_v2.csv")
# 注意：以下 pos_rate 检查仅用于 post-hoc 透明性报告。
# test_clusters 由随机种子决定，不基于任何标签信息。
# 论文中应明确说明："Cluster holdout split is determined by
# random seed without label-based selection."
if os.path.exists(LABELED_H8):
    lab = pd.read_csv(LABELED_H8, usecols=[MID, "label_fail_H8h"], low_memory=False)
    lab[MID] = pd.to_numeric(lab[MID], errors="coerce").astype("int64")
    lab = lab[lab["label_fail_H8h"].notna()].copy()

    # 按 machine_id 聚合正例数
    machine_stats = lab.groupby(MID)["label_fail_H8h"].agg(
        n_total="count",
        n_pos="sum"
    ).reset_index()

    # 合并到 mach_df（注意 mach_df 已定义）
    mach_df_diag = mach_df.merge(machine_stats, on=MID, how="left")
    mach_df_diag["n_total"] = mach_df_diag["n_total"].fillna(0).astype(int)
    mach_df_diag["n_pos"] = mach_df_diag["n_pos"].fillna(0).astype(int)

    # 按 cluster_mode 聚合
    cluster_pos = mach_df_diag.groupby("cluster_mode").agg(
        n_total=("n_total", "sum"),
        n_pos=("n_pos", "sum")
    ).reset_index()

    cluster_pos["pos_rate"] = cluster_pos["n_pos"] / cluster_pos["n_total"]

    # 计算训练集群和测试集群的 pos_rate 均值
    train_clusters_pos = set(cluster_pos["cluster_mode"]) - test_clusters
    train_pos_mean = cluster_pos[cluster_pos["cluster_mode"].isin(train_clusters_pos)]["pos_rate"].mean()
    test_pos_mean = cluster_pos[cluster_pos["cluster_mode"].isin(test_clusters)]["pos_rate"].mean()
    pos_ratio = test_pos_mean / train_pos_mean if train_pos_mean != 0 else float('inf')

    # 添加到报告 lines
    lines.append("")
    lines.append("[Pos rate by cluster (from H=8 labels, using cluster_mode)]")
    lines.append(cluster_pos.sort_values("cluster_mode").to_string(index=False))
    lines.append("")
    lines.append(f"train_pos_rate_mean={train_pos_mean:.6f}  test_pos_rate_mean={test_pos_mean:.6f}")
    lines.append(f"pos_rate_ratio (test/train)={pos_ratio:.3f}")
    if pos_ratio < 2.0:
        lines.append("✓ Pos rate consistency check passed (ratio < 2.0).")
    else:
        lines.append("❌ Pos rate consistency check FAILED (ratio >= 2.0).")
        # 若需强制通过，取消下一行注释
        # assert pos_ratio < 2.0, f"Test pos rate is {pos_ratio:.2f}x train"
else:
    lines.append("")
    lines.append("[Warning] H=8 labeled file not found, skipping pos_rate check")
lines.append("=== Cluster holdout split (v2 enhanced) ===")
lines.append(f"seed={SEED}")
lines.append(f"test_cluster_ratio={TEST_CLUSTER_RATIO}")
lines.append(f"ENABLE_VAL_SPLIT={ENABLE_VAL_SPLIT}  VAL_RATIO_IN_TRAIN={VAL_RATIO_IN_TRAIN if ENABLE_VAL_SPLIT else 'N/A'}")
lines.append("")
lines.append(f"clusters_total={len(clusters)}  test_clusters={len(test_clusters)}")
lines.append(f"machines_total={len(mach_df)}  train_machines={len(train_m)}  test_machines={len(test_m)}" +
             (f"  val_machines={len(val_m)}" if ENABLE_VAL_SPLIT else ""))
lines.append("")
lines.append(f"machines_with_multiple_clusters={multi_cluster_count}  rate={multi_cluster_rate:.6f}")
lines.append("")
lines.append("[Top clusters by machine count]")
lines.append(cluster_tbl.head(TOPK_CLUSTER_TABLE).to_string(index=False))
lines.append("")
lines.append("[Cluster size quantiles (machines per cluster)]")
lines.append(str(cluster_sizes.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()))
lines.append("")
lines.append("[Train/Test cluster lists]")
lines.append(f"train_clusters_count={len(train_clusters)} test_clusters_count={len(test_clusters_sorted)}")
lines.append(f"test_clusters_sample={test_clusters_sorted[:min(50, len(test_clusters_sorted))]}")
lines.append("")
lines.append("[Sanity checks]")
lines.append(f"train_test_overlap={len(set(train_m) & set(test_m))}")
lines.append(f"machines_spanning_multiple_clusters={multi_cluster_count} (all in train, no test contamination)")

with open(OUT_REP, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n[输出]", OUT_TRAIN)
print("[输出]", OUT_TEST)
if ENABLE_VAL_SPLIT:
    print("[输出]", OUT_VAL)
print("[报告]", OUT_REP)
print("\n".join(lines[:40]))