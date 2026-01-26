import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

def say_hello():
    print("Hello World")

#初次读取数据，从spss上读为csv模式并且保存
def read_spss_to_csv(spss_path, csv_path,*,chunk_size=100000) :
    # 创建一个迭代器，而不是一次性读完
    reader = pd.read_sas(spss_path, encoding='utf-8', chunksize=chunk_size)

    first_chunk = True

    for chunk in reader:
        # 这里可以加入你的数据处理逻辑，比如：
        # chunk = chunk[chunk['value'] > 0] 

        # 写入模式：如果是第一块则 'w' (写)，后续块则 'a' (追加)
        mode = 'w' if first_chunk else 'a'
        header = first_chunk  # 只有第一块需要表头
        
        chunk.to_csv(csv_path, mode=mode, index=False, header=header)
        
        first_chunk = False
        print(f"已处理一块 ({chunk_size} 行)...")

    print("转换完成！")

#两种读取数据的方法
def read_csv(csv_path, chunk_size=100000):
    try:
        # 方法 A: 标准读取 (默认逗号分隔)
        #df_csv = pd.read_csv('CHN24/all_predictors.csv')
        # 强制将 'STKCD' 列读取为字符串，保留 000002
        df_csv = pd.read_csv(csv_path, dtype={'STKCD': str})

        # 检查一下
        print(df_csv['STKCD'].head())
    except UnicodeDecodeError:
        # 方法 B: 如果是中文 CSV (特别是 Excel 导出的)，通常需要 gbk 或 gb18030 编码
        print("默认编码失败，尝试 GBK...")
        df_csv = pd.read_csv('CHN24/all_predictors.csv', encoding='gbk')

    # 预览 CSV 数据
    print("\nCSV 数据预览：")
    print(df_csv.head())
    return df_csv

def read_sas(sas_path):
    try:
        # 方法 A: 尝试默认读取 (UTF-8)
        df_sas = pd.read_sas('CHN24/all_predictors.sas7bdat', encoding='utf-8')
    except:
        # 方法 B: 如果报错或乱码，尝试使用 latin1 编码 (常见于旧版 SAS)
        print("UTF-8 读取失败，尝试 latin1...")
        df_sas = pd.read_sas('CHN24/all_predictors.sas7bdat', encoding='latin1')

    # 预览 SAS 数据
    print("SAS 数据预览：")
    print(df_sas.head())
    return df_sas

def cleanBlank(df, sort1, sort2):
    # 1️⃣ 先保存一份 df
    df = df.copy()

    # 2️⃣ 排序
    df = df.sort_values(by=[sort1, sort2])

    # 3️⃣ 记录删除前行数
    n_before = len(df)

    # 4️⃣ 删除含 NaN 的行（只要有一个 NaN 就删）
    df = df.dropna(axis=0)

    # 5️⃣ 记录删除后行数
    n_after = len(df)

    # 6️⃣ 打印删除信息
    print(f"firstSort: 删除了 {n_before - n_after} 行（{n_before} → {n_after}）")

    return df


#======================================================================
# 函数功能：对某一变量进行N分组
'''def GroupN(in_df, sort_var, vars, n_group=10):
    out_df = in_df.copy()
    # 对每一个 sort_var（通常是每个月）
    # 取出这一期所有股票的 vars（因子值）
    # 用 qcut 按分位数切成 n_group 份
    # 并给每只股票贴上 1~n_group 的组号
    out_df[f"{vars}_g{n_group}"] = out_df.groupby(sort_var)[vars].apply(
        lambda x: pd.qcut(x, q=n_group, 
                          labels=[i for i in range(1, n_group+1)]))
    
    out_df[f"{vars}_g{n_group}"] = out_df[f"{vars}_g{n_group}"] .astype(int)

    return out_df'''

import pandas as pd
import numpy as np

def GroupN(in_df, sort_var, factor, n_group=10):
    out_df = in_df.copy()
    gcol = f"{factor}_g{n_group}"

    out_df[gcol] = out_df.groupby(sort_var)[factor].transform(
        lambda x: pd.qcut(
            x,
            q=n_group,
            labels=range(1, n_group + 1)
            #,duplicates="drop"   # 防止某些月重复值太多直接炸
        )
    )

    # qcut 输出是 Categorical/可能含 NaN，先转成数值更稳
    out_df[gcol] = pd.to_numeric(out_df[gcol], errors="coerce")

    return out_df

def safe_groupN_loop(df_merged, factors, *, time_col="TRDMNT", n_group=10):
    """
    对一组因子依次调用 GroupN
    只保留不报错的因子

    Returns
    -------
    ok_factors : list
        成功分组的因子名
    results : dict
        每个成功因子的分组结果 DataFrame
    """
    ok_factors = []
    results = {}

    for fac in factors:
        try:
            df_tmp = GroupN(
                df_merged,time_col,
                fac,
                n_group=n_group
            )
            ok_factors.append(fac)
            results[fac] = df_tmp
            print(f"[OK] {fac}")
        except Exception as e:
            print(f"[SKIP] {fac} → {type(e).__name__}: {e}")

    print("=" * 50)
    print(f"成功因子数: {len(ok_factors)}")
    print("成功因子列表:")
    print(ok_factors)

    return ok_factors, results
  

#======================================================================
# winsorize 函数
#按组（例如按月 TRDMNT）对指定列做 winsorize 或 trim
#  - 支持单列或多列
# perc=1 表示 1% 的缩尾

class Winsorize:
    def __init__(self, in_df, sort_var, vars, perc=1, trim=0) -> None:
        self.in_df = in_df
        self.sort_var = sort_var
        self.vars = vars
        self.perc = perc
        self.trim = trim

    def func_trim(self, in_ser, perc):
        perc_upper = (100 - perc) / 100
        perc_lower = perc / 100

        qt_lower, qt_upper = in_ser.quantile([perc_lower, perc_upper])
        in_ser[in_ser > qt_upper] = np.nan
        in_ser[in_ser < qt_lower] = np.nan
        return in_ser

    def func_winsor(self, in_ser, perc):
        perc_upper = (100 - perc) / 100
        perc_lower = perc / 100
        qt_lower, qt_upper = in_ser.quantile([perc_lower, perc_upper])

        in_ser[in_ser > qt_upper] = qt_upper
        in_ser[in_ser < qt_lower] = qt_lower
        return in_ser

    def get(self, ):
        out_df = self.in_df.copy()
        if self.trim == 1:
            proc_method = self.func_trim
        if self.trim == 0:
            proc_method = self.func_winsor

        out_df[f"{self.vars}"] = out_df.groupby(
            self.sort_var)[self.vars].transform(lambda x: proc_method(x, 1))

        return out_df



#======================================================================
#匹配函数
#接收两个dataframe。进行二重的匹配
#根据时间和股票代码进行匹配，这两个列名可以发生变化，有时也可能扩充到三个、四个。最好是一个列表
def match_df(df1,df2,*,on_list):
    #判断这两个列名是否都存在于两个dataframe中
    for col in on_list:
        if col not in df1.columns:
            raise ValueError(f"列 {col} 不存在于第一个 DataFrame 中。")
        if col not in df2.columns:
            raise ValueError(f"列 {col} 不存在于第二个 DataFrame 中。")
    #进行匹配
    merged_df = pd.merge(df1, df2, how='left', on=on_list)
    return merged_df

def match_df_flex(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    left_on: list[str],
    right_on: list[str],
    how: str = "left",
    validate: str = "many_to_one"
):
    if len(left_on) != len(right_on):
        raise ValueError("left_on 和 right_on 长度必须一致")

    for col in left_on:
        if col not in df1.columns:
            raise ValueError(f"[df1] 缺少列: {col}")

    for col in right_on:
        if col not in df2.columns:
            raise ValueError(f"[df2] 缺少列: {col}")

    # df2 去重保护
    if df2.duplicated(subset=right_on).any():
        raise ValueError("df2 在 right_on 上存在重复键")

    df_merged = pd.merge(
        df1,
        df2,
        how=how,
        left_on=left_on,
        right_on=right_on,
        validate=validate
    )

    return df_merged

#======================================================================
#修改日期格式为月度
import pandas as pd

def to_month_period(df, col, inplace=True):
    """
    将指定列转换为 pandas 的月频 Period[M]

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        年-月列名（如 'TRDMNT'）
    inplace : bool
        是否原地修改 DataFrame

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    if col not in df.columns:
        raise ValueError(f"列 {col} 不存在于 DataFrame 中")

    result = pd.to_datetime(df[col], errors="coerce").dt.to_period("M")

    if inplace:
        df[col] = result
        return df
    else:
        return result
