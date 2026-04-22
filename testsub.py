import yfinance as yf
import pandas as pd
import numpy as np

def run_strategy_tournament(tickers, start_date="2023-01-01"):
    monthly_budget = 100000
    daily_base = monthly_budget / 20  # ここで定義
    
    print("市場データを取得中...")
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): 
        spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【銘柄分析: {ticker}】データを取得中...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['bb_low'] = df['ema20'] - (std20 * 2)

        # --- DVA 累積株数連動ロジックの実行 ---
        df['DVA_Amount'] = 0.0
        current_shares = 0.0
        
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
            vol = df['vol'].iloc[i]
            m_vol = df['mkt_vol'].iloc[i]

            if pd.isna(ema) or pd.isna(bb_l):
                invest_amt = daily_base
            else:
                # 1. 理想の株数ターゲット（毎日DCAの想定株数）
                ideal_shares_pace = (daily_base * (i + 1)) / ema
                
                # 2. 株数フィードバック（不足分を補正）
                share_gap = (ideal_shares_pace - current_shares) / ideal_shares_pace if ideal_shares_pace > 0 else 0
                share_boost = 1.0 + max(0, min(1.0, share_gap * 2.0))
                
                # 3. 価格ロジック
                gap = (ema - price) / ema
                psi = np.exp(5.0 * gap) * (2.5 if price < bb_l else 1.0)
                phi = max(0.5, 1 - (vol - (m_vol * 1.5)) / 5)
                tf = 1.2 if price > ema else 0.8
                
                invest_amt = daily_base * phi * psi * tf * share_boost

            # 更新
            df.iloc[i, df.columns.get_loc('DVA_Amount')] = invest_amt
            current_shares += (invest_amt / price)

        # --- 結果計算 ---
        dva_total_spent = df['DVA_Amount'].sum()
        dva_avg = dva_total_spent / current_shares if current_shares > 0 else 0

        # 他の手法（比較用）
        dca_daily_avg = (daily_base * len(df)) / (daily_base / df['Close']).sum()
        
        weekly_df = df[df.index.dayofweek == 2].copy()
        dca_weekly_avg = (25000 * len(weekly_df)) / (25000 / weekly_df['Close']).sum()

        monthly_df = df.resample('ME').last()
        dca_monthly_avg = (monthly_budget * len(monthly_df)) / (monthly_budget / monthly_df['Close']).sum()

        # ランキング
        results = [
            ("提案DVA(株数連動型)", dva_avg),
            ("毎日積立(DCA)", dca_daily_avg),
            ("毎週積立(DCA)", dca_weekly_avg),
            ("毎月積立(DCA)", dca_monthly_avg)
        ]
        results.sort(key=lambda x: x[1])

        print("-" * 65)
        print(f" {ticker} 実験結果ランキング")
        print("-" * 65)
        for i, (name, price) in enumerate(results, 1):
            diff = (price / results[0][1] - 1) * 100
            print(f" {i}位: {name:<20} | ${price:<8.2f} (最安比 +{diff:.2f}%)")
        print(f" [参考] DVA最終保有株数: {current_shares:.2f} 株")
        print("-" * 65)

# 実験実行
run_strategy_tournament(["NVDA", "AAPL", "TSLA", "8088.T"])