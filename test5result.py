import os
import glob
import pandas as pd


def merge_high_low_results(op_path: str, mode: str):
    """
    mode: 'ew' or 'vw'
    合并 *_ew_result_s.csv / *_vw_result_s.csv
    输出列: factor, high_low, mean, t, p
    """
    assert mode in {"ew", "vw"}

    suffix = f"_{mode}_result_s.csv"
    files = glob.glob(os.path.join(op_path, f"*{suffix}"))

    rows = []

    for fp in files:
        factor = os.path.basename(fp).replace(suffix, "")

        try:
            df = pd.read_csv(fp, index_col=0)
        except Exception as e:
            print(f"[跳过] 无法读取: {fp} | {e}")
            continue

        # index 清洗
        df.index = df.index.astype(str).str.strip()

        if "high_low" not in df.columns:
            print(f"[跳过] {factor} 缺少 high_low 列")
            continue

        rows.append({
            "factor": factor,
            "high_low": "high_low",
            "mean": df.loc["mean", "high_low"] if "mean" in df.index else float("nan"),
            "t": df.loc["t", "high_low"] if "t" in df.index else float("nan"),
            "p": df.loc["p", "high_low"] if "p" in df.index else float("nan"),
        })

    out = pd.DataFrame(rows)

    out_path = os.path.join(op_path, f"{mode}_high_low_merged_s.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ {mode.upper()} 合并完成：{out_path}（{len(out)} 个因子）")
    return out


# ======= 用法示例 =======
op_path = "test5output"

ew_df = merge_high_low_results(op_path, mode="ew")
vw_df = merge_high_low_results(op_path, mode="vw")

