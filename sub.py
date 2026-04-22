import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_long_term_analysis(tickers, start_date="2023-01-01", end_date="2024-12-31"):
    base_amount = 100000 # 毎月の基本積立額
    
    for ticker in tickers:
        print(f"--- {ticker} の分析を開始 ---")
        
        # 1. データの自動取得
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty: continue
        
        # 指標の計算
        df = data[['Close']].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['sma200'] = df['Close'].rolling(window=200).mean()
        # ボラティリティ（20日移動標準偏差の年率換算）
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # 市場（S&P500）のボラティリティ取得
        spy = yf.download("^GSPC", start=start_date, end=end_date)['Close']
        mkt_vol = spy.pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol

        # 2. DVA-AAM ロジック適用
        def calculate_dva(row):
            # ※本来は動的なPER/PBRが必要ですが、長期分析用に一旦一定値または推定値で計算
            # 実際にはここに取得した財務データを結合できます
            pe_target, pe_curr = 35, 30 
            pbr_target, pbr_curr = 15, 12
            p_avg = df['Close'].expanding().mean().shift(1).iloc[df.index.get_loc(row.name)]
            
            # 生存・損切り判定
            f_trend = 1 if row['Close'] > (row['sma200'] * 0.7) else 0
            f_panic = 1 if row['Close'] > (row['prev_close'] * 0.8) else 0
            omega = 1 if row['Close'] > (row['sma200'] * 0.8) else 0
            signal = f_trend * f_panic
            
            # 安全装置 & 加速装置
            phi = max(0, 1 - (row['vol'] - (row['mkt_vol'] * 1.5)) / 10)
            dyn_lambda = 2.0 * (pe_target / pe_curr) * (1 + (pbr_target / pbr_curr))
            gap = (p_avg - row['Close']) / p_avg if not pd.isna(p_avg) else 0
            psi = np.exp(dyn_lambda * gap)
            
            return base_amount * phi * psi * omega * signal

        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)
        df['DCA_Amount'] = base_amount # 比較用のドルコスト平均法

        # 3. グラフ化
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['DVA_Amount'], label='DVA-AAM (Dynamic)', color='blue')
        plt.axhline(y=base_amount, color='red', linestyle='--', label='DCA (Fixed)')
        plt.title(f"{ticker} Investment Amount Over Time")
        plt.legend()
        plt.show()

# 分析したい銘柄リスト
my_tickers = ["NVDA", "AAPL", "TSLA", "MSFT"]
run_long_term_analysis(my_tickers)