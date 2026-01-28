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
#输出内容
op_path='dataouput'

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
df_merged=func.match_df_flex(df_clean, df_monret, left_on=['STKCD','TRDMNT'], right_on=['STKCD','TRDMNT'], how='left',validate='many_to_one')

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

#获得全部因子的列表
col=df_merged.columns.tolist()
#keep_cols = {'STKCD', 'date', 'TRDMNT', 'MRETWD', 'MRETND','MARKETTYPE'}
keep_cols = ['STKCD', 'date', 'TRDMNT', 'MRETWD']
factor_list = [c for c in col if c not in keep_cols]

#========================================================
#                   第一步、分组
#========================================================
ok_factors,df_list = func.safe_groupN_loop(df_merged_clean, factor_list, time_col="TRDMNT", n_group= ngroup)

#========================================================
#                   第二步、缩尾处理
#          对除了keep_cols以外的所有因子进行缩尾处理
#========================================================

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

# 缩尾前数据进行分组
ok_factors,df_list = func.safe_groupN_loop(df_merged_clean, factor_list, time_col="TRDMNT", n_group= ngroup)
# 缩尾后数据进行分组，这个分不出来组的哈哈哈哈
#ok_factors_2,df_list_2 = func.safe_groupN_loop(df_winsor_only, factor_list, time_col="TRDMNT", n_group= ngroup)

#首先单独对size进行缩尾
winsor = func.Winsorize(df_winsor, "TRDMNT",'size')
df_winsor = winsor.get()
#print(df_winsor)
# 第二次缩尾：按（月 × 因子组）对 size 缩尾
df_winsor_list = {}

for fac, df_fac in df_list.items():
    gcol = f"{fac}_g{ngroup}"

    winsor = func.Winsorize(
        df_fac,
        ["TRDMNT", gcol],
        "size",
        perc=1,      # 你原来的 1%
        trim=0       # winsorize，不是 trim
    )
    df_winsor = winsor.get()

    df_winsor_list[fac] = df_winsor
    print(f"[WINSOR OK] {fac} → size by (TRDMNT × {gcol})")

#========================================================
#                   第三步、计算EW VW
#           EW=同一个月 (TRDMNT)
#           同一个因子分组（比如 AM_g10 的第 3 组）
#           把这一组里所有股票的收益率 ret 简单平均。
#           VW=同一个月
#           同一个因子组
#           用上一期市值 size 当权重，对收益率加权平均
#
#       在这里的收益率指的是Mretwd [考虑现金红利再投资的月个股回报率]
#       从这里开始就只有一个了，上面都是对全部的因子进行循环处理
#========================================================

sort_factor=ok_factors[0]
in_ret = df_winsor_list[sort_factor].copy(deep =True)
ew_ret = in_ret.groupby(['TRDMNT', f"{sort_factor}_g{ngroup}"])['MRETWD'].mean()
#vw_ret = in_ret.groupby(['TRDMNT', f"{sort_factor}_g{ngroup}"]).transform(lambda x: np.average(x['MRETWD'], weights=x['size']))
vw_ret = in_ret.groupby(
    ['TRDMNT', f"{sort_factor}_g{ngroup}"]
).apply(
    lambda g: np.average(g['MRETWD'], weights=g['size'])
)

ew_mean = ew_ret.copy(deep = True)
ew_mean.name = 'Ew_ret'
vw_mean = vw_ret.copy(deep = True)
vw_mean.name = 'Vw_ret'
month_count = in_ret.groupby(['TRDMNT', f"{sort_factor}_g{ngroup}"])['MRETWD'].count()
month_count.name = 'Count'
sort_factor_mean = in_ret.groupby(['TRDMNT', f"{sort_factor}_g{ngroup}"])[sort_factor].mean()

month_result = pd.concat([month_count, sort_factor_mean, ew_mean, vw_mean], axis=1, ignore_index=False)

ew_ret = ew_ret.unstack()
vw_ret = vw_ret.unstack()

ew_ret.columns = [f"col_{i+1}" for i in range(ngroup) ]
vw_ret.columns = [f"col_{i+1}" for i in range(ngroup) ]

ew_ret['high_low'] = ew_ret[f"col_{ngroup}"] - ew_ret["col_1"]
vw_ret['high_low'] = vw_ret[f"col_{ngroup}"] - vw_ret["col_1"]

ew_other = ew_ret.loc[:,['high_low']]
ew_other = ew_other.stack()
ew_other.name = 'Ew_ret'
vw_other = vw_ret.loc[:,['high_low']]
vw_other = vw_other.stack()
vw_other.name = 'Vw_ret'

other = pd.concat([ew_other, vw_other], axis=1, ignore_index=False)
other = other.reset_index()
other = other.rename(columns={'level_1':f"{sort_factor}_g{ngroup}"})
other = other.set_index(['TRDMNT', f"{sort_factor}_g{ngroup}"])
month_result = pd.concat([month_result, other], axis=0, ignore_index=False)
month_result.sort_index(inplace=True)

month_result.to_csv(os.path.join(op_path, f"{sort_factor}_month_result.csv"))

# 如果 index 是 PeriodIndex，用 to_timestamp
if isinstance(ew_ret.index, pd.PeriodIndex):
    ew_ret.index = ew_ret.index.to_timestamp(how="end")

if isinstance(vw_ret.index, pd.PeriodIndex):
    vw_ret.index = vw_ret.index.to_timestamp(how="end")

### 得到各组及high_low的收益、t值
ew_stat = func.get_stat(ew_ret, max_lag = 3)
vw_stat = func.get_stat(vw_ret, max_lag = 3)

ew_stat.to_csv(os.path.join(op_path, f"{sort_factor}_ew_result.csv"))
vw_stat.to_csv(os.path.join(op_path, f"{sort_factor}_vw_result.csv"))
