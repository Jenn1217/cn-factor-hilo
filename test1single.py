import func
import pandas as pd
import numpy as np
import func
import warnings
warnings.filterwarnings("ignore")
import os

input_file = 'CHN24/all_predictors.sas7bdat'
output_file = 'CHN24/all_predictors.csv'
chunk_size = 100000  # 每次处理 10 万行

#定义起止时间
start_month = '2000-01'
end_month = '2024-12'

#定义分组数
ngroup = 10

#读取数据
df_csv=func.read_csv('CHN24/all_predictors.csv')

#统一日期格式
df_csv["TRDMNT"] = df_csv["date"]
df_csv=func.to_month_period(df_csv, "TRDMNT")
print(df_csv["TRDMNT"].dtype)
print(df_csv["TRDMNT"].isna().sum())
print(df_csv[["TRDMNT"]].head())

#日期筛选
df_csv_filtered = df_csv[
    (df_csv['TRDMNT'] >= start_month) &
    (df_csv['TRDMNT'] <= end_month)
].copy()

#日期排序
df_clean=func.cleanBlank(df_csv_filtered, 'TRDMNT','STKCD')
print(df_clean.head())

# 合并 月度return
df_monret=func.read_csv('CHN24/monret.csv')
df_monret["TRDMNT"] = (
    pd.to_datetime(df_monret["TRDMNT"], errors="coerce")
      .dt.to_period("M")
)

#使用固定范围内的df_csv_filtered
df_merged=func.match_df_flex(df_csv_filtered, df_monret, left_on=['STKCD','TRDMNT'], right_on=['STKCD','TRDMNT'], how='left',validate='many_to_one')

#删除列'OPNDT', 'MOPNPRC', 'CLSDT', 'MCLSPRC', 'MNSHRTRD', 'MNVALTRD', 'MSMVOSD', 'MSMVTTL', 'NDAYTRD', 'MARKETTYPE', 'CAPCHGDT', 'AHSHRTRD_M', 'AHVALTRD_M'
cols_to_drop = [
    'OPNDT', 'MOPNPRC', 'CLSDT', 'MCLSPRC', 'MNSHRTRD', 'MNVALTRD',
    'MSMVOSD', 'MSMVTTL', 'NDAYTRD', 'CAPCHGDT',
    'AHSHRTRD_M', 'AHVALTRD_M'
]

df_merged = df_merged.drop(columns=cols_to_drop)

print(df_merged.columns.tolist())

#只要这一行里有任何一个字段是 NaN，这一整行就直接删掉。
df_merged_clean = func.cleanBlank(df_merged, 'TRDMNT','STKCD')

#========================================================
#                   第一步、分组
#========================================================
ok_factors,df_list = func.safe_groupN_loop(df_merged_clean, colist, time_col="TRDMNT", n_group= ngroup)

#========================================================
#                   第二步、缩尾处理
#          对除了keep_cols以外的所有因子进行缩尾处理
#========================================================
#获得全部因子的列表
col=df_merged.columns.tolist()
#keep_cols = {'STKCD', 'date', 'TRDMNT', 'MRETWD', 'MRETND','MARKETTYPE'}
keep_cols = ['STKCD', 'date', 'TRDMNT', 'MRETWD']
factor_list = [c for c in col if c not in keep_cols]
#1%缩尾处理
df_winsor=df_merged_clean.copy()
#缩尾的列
winsor_cols = []
#每一个因子依次缩尾
for factor in factor_list:
    wcol = f"{factor}_w"
    winsor_cols.append(wcol)

    df_winsor[wcol] = df_winsor[factor]

    winsor = func.Winsorize(
        df_winsor,
        "TRDMNT",
        wcol
    )
    df_winsor = winsor.get()
print(df_winsor.head())

winsor_cols = [f"{factor}_w" for factor in factor_list]
df_winsor_only = df_winsor[keep_cols + winsor_cols].copy()

print(df_winsor_only.head())

