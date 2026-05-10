# ============================================================
# 文件名：Step3d_输入表健康检查与稀疏性报告_v2.py
# 阶段：Step 3D（v2，第三阶段收官）
# 作用：
#   1) 全量扫描 02_processed 下 Step3C 导出的所有输入表
#   2) 文件级检查：NaN/Inf 是否为 0，行列数、列集合
#   3) 列级报告：missing率、inf计数、0占比、nunique、std（用于识别常数/稀疏/被抹掉的列）
#   4) schema一致性：同一个(H,set)的 train/val/test 列集合与顺序必须一致
#总的来讲就是防止污染数据进入模型，也就是我们之前检测的NAN/INF这种哨兵数据。
# 输入：
#   02_processed/step3c_model_input_*_H*h_*_v2.csv
#
# 输出（05_reports）：
#   step3d_input_health_filelevel_v2.csv
#   step3d_input_health_collevel_v2.csv
#   step3d_schema_check_v2.txt
# ============================================================

import os, re, glob
import numpy as np
import pandas as pd

ROOT_DIR = r"D:\pycharmcode\GCT数据集-v2处理"
IN_DIR = os.path.join(ROOT_DIR, "02_processed")
OUT_DIR = os.path.join(ROOT_DIR, "05_reports")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILELEVEL = os.path.join(OUT_DIR, "step3d_input_health_filelevel_v2.csv")
OUT_COLLEVEL  = os.path.join(OUT_DIR, "step3d_input_health_collevel_v2.csv")
OUT_SCHEMA_TXT = os.path.join(OUT_DIR, "step3d_schema_check_v2.txt")

pattern = os.path.join(IN_DIR, "step3c_model_input_*_H*h_*_v2.csv")
files = sorted(glob.glob(pattern))
assert len(files) > 0, f"❌ 未找到输入表：{pattern}"

# 解析文件名：set / H / split
rx = re.compile(r"step3c_model_input_(?P<set>.+)_H(?P<H>\d+)h_(?P<split>train|val|test)_v2\.csv")
#从文件名提取SET特征组名。H窗口。Spilt训练、验证、测试，
file_rows = []
col_rows = []
schema_map = {}  # key=(H,set,split)->cols(list)

print("="*90)
print("Step3D(v2)：输入表健康检查（全量扫描）")
print("="*90)
print(f"[扫描文件数] {len(files)}")
print(f"[输入目录] {IN_DIR}")

bad_nan_inf = []
bad_schema = []

for fp in files:
    fn = os.path.basename(fp)
    m = rx.match(fn)
    if not m:
        continue
    set_name = m.group("set")
    H = int(m.group("H"))
    split = m.group("split")

    # 先读表头拿列
    cols = pd.read_csv(fp, nrows=0).columns.tolist()
    schema_map[(H, set_name, split)] = cols

    df = pd.read_csv(fp, low_memory=False)
    nrow, ncol = df.shape

    num = df.select_dtypes(include=[np.number])
    nan_cnt = int(num.isna().sum().sum())
    inf_cnt = int(np.isinf(num.to_numpy()).sum())

    file_rows.append({
        "file": fn, "H_hours": H, "set": set_name, "split": split,
        "rows": nrow, "cols": ncol,
        "nan_total": nan_cnt, "inf_total": inf_cnt
    })

    if nan_cnt > 0 or inf_cnt > 0:
        bad_nan_inf.append((fn, nan_cnt, inf_cnt))

    # 列级统计（missing_rate, zero_rate, nunique, std以及排除 index 列也可以，但这里全部统计，）
    for c in num.columns:
        x = num[c]
        miss = float(x.isna().mean())
        infc = int(np.isinf(x.to_numpy()).sum())
        # zero_rate：仅对数值列
        zero_rate = float((x == 0).mean())
        nunique = int(x.nunique(dropna=True))
        std = float(x.std(skipna=True))
        col_rows.append({
            "file": fn, "H_hours": H, "set": set_name, "split": split,
            "col": c,
            "missing_rate": miss,
            "inf_count": infc,
            "zero_rate": zero_rate,
            "nunique": nunique,
            "std": std
        })

# schema 一致性检查：对每个(H,set)要求 train/val/test列完全一致且顺序一致
lines = []
lines.append("=== Step3D(v2) schema check ===")
keys = sorted({(H, s) for (H, s, sp) in schema_map.keys()})

for H, s in keys:
    cols_train = schema_map.get((H, s, "train"))
    cols_val   = schema_map.get((H, s, "val"))
    cols_test  = schema_map.get((H, s, "test"))

    ok = (cols_train == cols_val == cols_test)
    lines.append(f"[H={H}h set={s}] schema_equal={ok}")

    if not ok:
        bad_schema.append((H, s))
        # 输出差异摘要
        def _diff(a, b):
            return list(sorted(set(a) ^ set(b)))
        if cols_train and cols_val:
            lines.append(f"  diff(train,val) cols_symdiff={_diff(cols_train, cols_val)[:50]}")
        if cols_train and cols_test:
            lines.append(f"  diff(train,test) cols_symdiff={_diff(cols_train, cols_test)[:50]}")
        # 顺序不同也算不一致
        lines.append("  NOTE: order mismatch also triggers schema_equal=False")

# 保存输出
pd.DataFrame(file_rows).to_csv(OUT_FILELEVEL, index=False, encoding="utf-8-sig")
pd.DataFrame(col_rows).to_csv(OUT_COLLEVEL, index=False, encoding="utf-8-sig")
with open(OUT_SCHEMA_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n[输出保存]", OUT_FILELEVEL)
print("[输出保存]", OUT_COLLEVEL)
print("[输出保存]", OUT_SCHEMA_TXT)

# 控制台摘要 + 强约束
file_df = pd.DataFrame(file_rows)
print("\n[摘要] 文件级 nan/inf 最大值：")
print(file_df[["nan_total","inf_total"]].max().to_string())

if bad_nan_inf:
    raise AssertionError(f"❌ 发现 NaN/Inf 不为0的文件：{bad_nan_inf[:5]}（仅显示前5）")

if bad_schema:
    raise AssertionError(f"❌ 发现 schema 不一致的 (H,set)：{bad_schema}")

print("\n✅ Step3D 检查通过：全文件 NaN/Inf=0 且 schema 一致")
print("="*90)