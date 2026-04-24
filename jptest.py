import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_full_comparison_experiment(tickers, start_date="2014-01-01"):
    # --- 基本予算設定 ---
    monthly_budget = 500  # 月間予算（ドル）
    daily_base = monthly_budget / 20
    weekly_base = monthly_budget / 4
    
    print(f"--- 実験開始: {start_date} 以降の全積立手法を比較 ---")
    
    # 市場平均（S&P 500）データ
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【解析対象: {ticker}】")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        # --- 1. 提案手法 DVA-AAM ---
        df['DVA_Amt'] = 0.0
        dva_shares = 0.0
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
            if pd.isna(ema) or pd.isna(bb_l):
                amt = daily_base
            else:
                # 方程式: I(t) = B * F * Psi * Phi * T
                ideal_s = (daily_base * (i + 1)) / ema
                f_t = 1.0 + max(0, min(1.0, ((ideal_s - dva_shares) / ideal_s) * 2.0)) if ideal_s > 0 else 1.0
                psi_t = np.exp(6.0 * ((ema - price) / ema)) * (2.5 if price < bb_l else 1.0)
                phi_t = max(0.5, 1 - (df['vol'].iloc[i] - (df['mkt_vol'].iloc[i] * 1.2)) / 5)
                t_t = 1.15 if price > ema else 0.85
                amt = daily_base * f_t * psi_t * phi_t * t_t
            
            amt = max(amt, daily_base * 0.1)
            df.iloc[i, df.columns.get_loc('DVA_Amt')] = amt
            dva_shares += (amt / price)

        # --- 2. 毎日積立 (Daily DCA) ---
        dca_daily_shares = (daily_base / df['Close']).sum()
        
        # --- 3. 毎週積立 (Weekly DCA) ---
        # 毎週水曜日に投資すると仮定
        weekly_df = df[df.index.dayofweek == 2].copy()
        dca_weekly_shares = (weekly_base / weekly_df['Close']).sum()

        # --- 4. 毎月積立 (Monthly DCA) ---
        # 月末の最終営業日に投資
        monthly_df = df.resample('ME').last()
        dca_monthly_shares = (monthly_budget / monthly_df['Close']).sum()

        # --- 結果集計 ---
        methods = [
            ("提案DVA-AAM", df['DVA_Amt'].sum(), dva_shares),
            ("毎日積立(DCA)", daily_base * len(df), dca_daily_shares),
            ("毎週積立(DCA)", weekly_base * len(weekly_df), dca_weekly_shares),
            ("毎月積立(DCA)", monthly_budget * len(monthly_df), dca_monthly_shares)
        ]

        print("-" * 75)
        print(f"{'投資手法':<15} | {'平均取得単価':<12} | {'累計株数':<12} | {'最安比':<6}")
        print("-" * 75)
        
        # 単価でソートして表示
        sorted_methods = sorted(methods, key=lambda x: x[1]/x[2])
        best_price = sorted_methods[0][1] / sorted_methods[0][2]
        
        for name, total_spent, shares in sorted_methods:
            avg_price = total_spent / shares
            diff = (avg_price / best_price - 1) * 100
            print(f"{name:<15} | ${avg_price:>10.2f} | {shares:>10.2f}株 | +{diff:>5.2f}%")
        print("-" * 75)

if __name__ == "__main__":
    run_full_comparison_experiment(["NVDA", "AAPL", "TSLA"], start_date="2014-01-01")
