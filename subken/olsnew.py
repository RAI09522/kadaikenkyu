import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

FILE_PATH = 'steam_market_data.csv'

df_raw = pd.read_csv(FILE_PATH)

print(df_raw.head())
print(df_raw.columns)

# 数値化
cols = ['Demand_D', 'Visibility_V', 'Quality_Q', 'Price_P']

for col in cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

# クリーニング
df = df_raw[
    (df_raw['Demand_D'] > 0) &
    (df_raw['Visibility_V'] > 0) &
    (df_raw['Quality_Q'] > 0) &
    (df_raw['Price_P'] > 0)
].dropna().copy()

print("cleaned:", len(df))

# 空なら終了
if len(df) == 0:
    print("有効データ0件")
    exit()

formula = 'np.log(Demand_D) ~ np.log(Visibility_V) + np.log(Quality_Q) + np.log(Price_P)'

results = smf.ols(formula, data=df).fit()

print(results.summary())