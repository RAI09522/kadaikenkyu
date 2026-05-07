import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------------------------------------
# 1. 現実のスクレイピングデータを想定したデータフレームの準備
# （※ここではSteam APIから取得したと仮定した擬似データを生成）
# ---------------------------------------------------------
np.random.seed(42)
N_empirical = 1000 # サンプル数1000件のゲームデータ

# 現実の分布に近い形でデータを生成（対数正規分布などを想定）
Visibility_data = np.random.lognormal(mean=5.0, sigma=1.5, size=N_empirical) # フォロワー数
Quality_data = np.random.uniform(2.0, 9.8, size=N_empirical) # レビュースコア(2~9.8)
Price_data = np.random.uniform(5.0, 60.0, size=N_empirical) # 価格帯($5~$60)

# 現実市場を模倣した需要データの生成（可視性が強く効く世界線を想定）
# ln(D) = 2.0 + 1.2*ln(V) + 0.6*ln(Q) - 0.3*ln(P) + noise
log_Demand = (
    2.0 
    + 1.2 * np.log(Visibility_data) 
    + 0.6 * np.log(Quality_data) 
    - 0.3 * np.log(Price_data) 
    + np.random.normal(0, 0.5, N_empirical) # これが現実のノイズ(σ)
)
Demand_data = np.exp(log_Demand)

# データフレームの作成
df = pd.DataFrame({
    'Demand': Demand_data,
    'Visibility': Visibility_data,
    'Quality': Quality_data,
    'Price': Price_data
})

# ---------------------------------------------------------
# 2. OLS（通常最小二乗法）の実行
# ---------------------------------------------------------
# Log-Logモデルを定義
formula = 'np.log(Demand) ~ np.log(Visibility) + np.log(Quality) + np.log(Price)'

# 回帰分析の実行
model = smf.ols(formula, data=df).fit()

# 結果の出力
print(model.summary())

# シミュレーションに代入するパラメータの抽出
alpha_empirical = model.params['np.log(Visibility)']
beta_empirical = model.params['np.log(Quality)']
sigma_empirical = np.sqrt(model.mse_resid) # 誤差項の標準偏差（ボラティリティ）

print("\n--- シミュレーション用 抽出パラメータ ---")
print(f"可視性の弾力性 (α): {alpha_empirical:.4f}")
print(f"品質の弾力性 (β): {beta_empirical:.4f}")
print(f"確率的ショック (σ): {sigma_empirical:.4f}")
