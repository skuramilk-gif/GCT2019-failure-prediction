# ============================================================
# 文件名：Step2b_训练验证测试机器划分与固化_v2.py
# 阶段：Step 2B（v2）
# 作用：
#   1) 基于 clean 主表 machine_slice_clean_v2 对 machine_id 做 Train/Val/Test 划分（互斥）
#   2) 固化机器列表到 03_splits，保证后续所有步骤可复现
#   3) 输出 split 诊断报告（每集合机器数/行数/每机slice数分布/是否有FAIL机器占比等）
#
# 输入：
#   01_interim/step2a_machine_slice_clean_v2.csv
#
# 输出（03_splits/）：
#   step2b_train_machines_v2.txt
#   step2b_val_machines_v2.txt
#   step2b_test_machines_v2.txt
#   step2b_split_meta_v2.json
#
# 报告输出（05_reports/）：
#   step2b_split_diagnostics_v2.txt
#
# 说明：
#   - 划分按 machine_id，避免同机泄漏
#   - 不基于标签分层（最干净），但会输出事后诊断用于检查分布是否失衡
#
#论文影响（有利点）-该阶段结果写入Experimental Protocol：
#1-Train/Val/Test 按 machine_id 互斥：防止同机泄漏（这是 v1 训练脚本的致命问题之一）
#2-Seed 固化 + 列表落盘：可复现；3-不按标签分层：最干净，审稿人挑不出“你用标签设计 split”的刺
#
#约束（后面谁违反谁就是在够泄漏）-后续所有步骤都必须遵守：
#1Step2C 打标签、Step3 预处理拟合、Step4 仿真参数拟合、Step7 训练都只能按 step2b_*_machines_v2.txt 这三份列表切数据
#2任何 fit（imputer/scaler/clip/阈值/分位数/仿真参数）只能用真实 Train machines
#Val 只用于：超参、仿真配置选择；Test 只用于最终一次评估
# ============================================================

import os
import json
import numpy as np
import pandas as pd

# ============================================================
# 0) 路径与参数
# ============================================================
ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"

IN_CLEAN = os.path.join(ROOT_DIR, "01_interim", "step2a_machine_slice_clean_v2.csv")

SPLIT_DIR = os.path.join(ROOT_DIR, "03_splits")
REPORT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

OUT_TRAIN = os.path.join(SPLIT_DIR, "step2b_train_machines_v2.txt")
OUT_VAL   = os.path.join(SPLIT_DIR, "step2b_val_machines_v2.txt")
OUT_TEST  = os.path.join(SPLIT_DIR, "step2b_test_machines_v2.txt")
OUT_META  = os.path.join(SPLIT_DIR, "step2b_split_meta_v2.json")

OUT_DIAG  = os.path.join(REPORT_DIR, "step2b_split_diagnostics_v2.txt")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
#随机种子保证可复现性，并且规定训练集、验证集、测试集。
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "比例之和必须为1"

# ============================================================
# 1) 读取 clean 表，提取机器集合与机器级统计
# ============================================================
print("=" * 90)
print("Step 2B(v2)：训练/验证/测试 机器划分与固化")
print("=" * 90)
print(f"[输入] {IN_CLEAN}")

df = pd.read_csv(IN_CLEAN, usecols=["machine_id", "time_slice", "failed_count", "record_count"], low_memory=False)
#IN_CLEAN 是输入文件路径，low_memory=False让Pandas进行一次性推断列类型，保证数据干净防止前面和后面数据类型不一致这种情况。
machines = df["machine_id"].dropna().astype(int).unique()
#.dropna() 移除可能存在的缺失值（NaN）。.unique() 返回去重后的机器 ID 数组
machines = np.array(machines, dtype=np.int64)

print(f"[数据] machines={len(machines)}, rows={len(df)}")

# 机器级统计：slice数、是否出现过FAIL（用于诊断，不用于分层）
machine_stats = (df.groupby("machine_id")
                   .agg(   #agg 是 aggregate 的缩写，允许同时对不同列应用不同的聚合函数。
                       slice_count=("time_slice", "count"), #count统计每个机器组内的t_s的时间片数量
                       has_fail=("failed_count", lambda s: int((s > 0).any()))#判断每台机器在它整个生命周期中是否出现过一次FALL事件
                   )
                   .reset_index())
#s 代表当前这台机器的 failed_count 列（一个时间片一个值，是一个小序列）。.any() 检查整个序列里是否有至少一个 True。有就返回 True，没有就返回 False。
#int(...) 把 True 转成 1，False 转成 0，这个字段没有参与划分，只是划分完成后用来进行事后体验，保证划分平衡，因为可能出现train全是故障，test全是正常
machine_time_range = (df.groupby("machine_id")
                        .agg(
                            time_slice_min=("time_slice", "min"),
                            time_slice_max=("time_slice", "max"),
                        )
                        .reset_index())
machine_stats = machine_stats.merge(machine_time_range, on="machine_id", how="left")
machine_stats["time_span_slices"] = machine_stats["time_slice_max"] - machine_stats["time_slice_min"]

# ============================================================
# 2) 随机划分（按 machine_id）
# ============================================================
rng = np.random.RandomState(RANDOM_SEED)#创建一个 NumPy 随机状态对象 rng，并用之前定义的固定随机种子 RANDOM_SEED（42）初始化。
rng.shuffle(machines)#调用rng.shuffle将机器id数组（(machines)随机打乱

n_total = len(machines) #参与划分的机器总数
n_train = int(n_total * TRAIN_RATIO) #训练集的机器总数
n_val   = int(n_total * VAL_RATIO)
# 剩余全部给 test，避免浮点取整误差
n_test  = n_total - n_train - n_val

train_m = machines[:n_train] #从0到n_train-1都归训练集
val_m   = machines[n_train:n_train+n_val]
test_m  = machines[n_train+n_val:] #做划分，从n_train+n_val:开始它以后的都归测试集了

assert len(set(train_m) & set(val_m)) == 0
assert len(set(train_m) & set(test_m)) == 0
assert len(set(val_m) & set(test_m)) == 0
assert len(train_m) + len(val_m) + len(test_m) == n_total #保证数据不泄漏

# ============================================================
# 3) 保存机器列表
# ============================================================
def save_list(path, arr):
    #定义一个函数 save_list，接收两个参数：path 表示要保存的文件路径，arr 表示要写入的机器 ID 数组。
    with open(path, "w", encoding="utf-8") as f:
        for x in arr:
            f.write(str(int(x)) + "\n")

save_list(OUT_TRAIN, train_m)
save_list(OUT_VAL, val_m)
save_list(OUT_TEST, test_m)
#分别调用 save_list 函数，将训练集、验证集、测试集的机器 ID 数组保存到对应的 .txt 文件中

meta = {
    "random_seed": RANDOM_SEED,
    "train_ratio": TRAIN_RATIO,
    "val_ratio": VAL_RATIO,
    "test_ratio": TEST_RATIO,
    "machines_total": int(n_total),
    "machines_train": int(len(train_m)),
    "machines_val": int(len(val_m)),
    "machines_test": int(len(test_m)),
} #创建一个字典 meta，存储本次划分的元数据，machines_train、machines_val、machines_test：实际分到各集合的机器数量（由于取整可能与理论比例略有偏差）

with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
#这一步保证复现实验，meta JSON提供规则与参数，并且三个txt文件直接给出每个集合是那些机器id，join回答怎么分，TXT回答分出来的结果。

# ============================================================
# 4) split 诊断报告（事后检查）
# ============================================================
    #df["machine_id"].isin(machine_ids) 对主表每一行判断其机器 ID 是否属于当前集合，返回布尔序列
    def subset_diag(name, machine_ids):
        ms = machine_stats[machine_stats["machine_id"].isin(machine_ids)]
        rows = int(df["machine_id"].isin(machine_ids).sum())
        return {
            "name": name,
            "machines": int(len(machine_ids)),
            "rows": rows,
            "slice_count_desc": ms["slice_count"].describe().to_dict(),
            "fail_machine_rate": float(ms["has_fail"].mean()),
            "fail_machine_count": int(ms["has_fail"].sum()),
            "slices_per_machine_median": float(ms["slice_count"].median()),
            "time_span_slices_median": float(ms["time_span_slices"].median()),
            "time_span_slices_p25": float(ms["time_span_slices"].quantile(0.25)),
            "time_slice_min_median": float(ms["time_slice_min"].median()),
            "time_slice_max_median": float(ms["time_slice_max"].median()),
        }


diag = {
    "train": subset_diag("train", set(train_m)),
    "val": subset_diag("val", set(val_m)),
    "test": subset_diag("test", set(test_m)), #分别调用 subset_diag 函数，将三个返回的字典嵌套在一个总字典 diag 中，
}

lines = []
lines.append("=== Step2B(v2) split diagnostics ===")
lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
lines.append("")

for k in ["train", "val", "test"]:
    d = diag[k] #获取当前集合的诊断信息字典。
    desc = d["slice_count_desc"] #获取该集合的时间片数量描述性统计字典。
    lines.append(f"[{k}] machines={d['machines']}, rows={d['rows']}, "
                 f"fail_machines={d['fail_machine_count']}, fail_machine_rate={d['fail_machine_rate']:.4f}, "
                 f"slices_per_machine_median={d['slices_per_machine_median']:.0f}")
    lines.append(f"  time_coverage: slice_min_median={d['time_slice_min_median']:.0f}, "
                 f"slice_max_median={d['time_slice_max_median']:.0f}, "
                 f"span_slices_median={d['time_span_slices_median']:.0f}, "
                 f"span_slices_p25={d['time_span_slices_p25']:.0f}")

    lines.append("  slice_count describe:")
    lines.append(f"    mean={desc['mean']:.4f}, std={desc['std']:.4f}, min={desc['min']:.0f}, p50={desc['50%']:.0f},  max={desc['max']:.0f}")
    #输出均值、标准差、最小值、中位数、最大值

    # 直接把 describe 完整字典也写进去（更可复现）
    lines.append("    full=" + str(desc)) #保留了生成的所有分位数信息，提供了更细粒度的数据。
    lines.append("")  #增加空行，不重要。

with open(OUT_DIAG, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))#文件保存在 05_reports 目录下，文件名为 step2b_split_diagnostics_v2.txt

# ============================================================
# 5) 控制台摘要 + 预览
# ============================================================
print("\n" + "=" * 90)
print("Step 2B(v2) 控制台摘要")
print("=" * 90)
print(meta)

print("\n[输出文件]")
print(" ", OUT_TRAIN)
print(" ", OUT_VAL)
print(" ", OUT_TEST)
print(" ", OUT_META)
print(" ", OUT_DIAG)

print("\n[预览] train_machines head(10):")
print("\n".join([str(int(x)) for x in train_m[:10]]))

print("\n[预览] val_machines head(10):")
print("\n".join([str(int(x)) for x in val_m[:10]]))

print("\n[预览] test_machines head(10):")
print("\n".join([str(int(x)) for x in test_m[:10]]))

print("=" * 90)
print("Step 2B(v2) 完成")

#问题：
#1：为什么划分要按 machine_id 而不是按 time_slice 或直接随机行划分？
# 2：代码中 assert len(set(train_m) & set(val_m)) == 0 这行是干什么的？如果删掉它，程序还能跑吗？删掉会有什么潜在风险
# 3：machine_stats 里的 has_fail 列是怎么生成的？它用来做什么？它有没有参与划分决策？
# 4：如果我想把划分比例改成 60% 训练、20% 验证、20% 测试，需要改代码的哪几行？
# 5：meta JSON 文件和三个 .txt 机器列表文件分别有什么作用？为什么不能只保留其中一种？
# 6：切片操作 machines[n_train:n_train+n_val] 中的 n_train+n_val 是左闭还是右开？取到的最后一个元素的索引是多少？
#问题5的详细解读：只给你一个 meta JSON 文件，不给你三个 .txt 文件就等于我知道种子是 42，比例是 70/10/20。但我怎么知道你真的按这个跑了
#只给你 .txt 文件，不给你 JSON 参数呢？就等于这三个文件是怎么分出来的？你是按机器随机分的，还是按时间分的，还是按失败率分层等等
#总结--JSON是方法声明，TXT是执行结果。