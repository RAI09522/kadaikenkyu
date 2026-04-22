import yfinance as yf
import pandas as pd
import numpy as np

def run_strategy_tournament(tickers, start_date="2023-01-01"):
    monthly_budget = 100000
    daily_base = monthly_budget / 20
    
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

        # --- DVA 実行 ---
        df['DVA_Amount'] = 0.0
        dva_shares = 0.0
        
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
            
            if pd.isna(ema) or pd.isna(bb_l):
                invest_amt = daily_base
            else:
                # 株数フィードバック
                ideal_shares_pace = (daily_base * (i + 1)) / ema
                share_gap = (ideal_shares_pace - dva_shares) / ideal_shares_pace if ideal_shares_pace > 0 else 0
                share_boost = 1.0 + max(0, min(1.0, share_gap * 2.0))
                
                # 価格ロジック
                gap = (ema - price) / ema
                psi = np.exp(5.0 * gap) * (2.5 if price < bb_l else 1.0)
                phi = max(0.5, 1 - (df['vol'].iloc[i] - (df['mkt_vol'].iloc[i] * 1.5)) / 5)
                tf = 1.2 if price > ema else 0.8
                invest_amt = daily_base * phi * psi * tf * share_boost

            df.iloc[i, df.columns.get_loc('DVA_Amount')] = invest_amt
            dva_shares += (invest_amt / price)

        # --- DCA 比較用の計算 ---
        # 1. 毎日積立
        dca_daily_shares = (daily_base / df['Close']).sum()
        dca_daily_price = (daily_base * len(df)) / dca_daily_shares if dca_daily_shares > 0 else 0
        
        # 2. 毎週積立
        weekly_df = df[df.index.dayofweek == 2].copy()
        dca_weekly_shares = (25000 / weekly_df['Close']).sum()
        dca_weekly_price = (25000 * len(weekly_df)) / dca_weekly_shares if dca_weekly_shares > 0 else 0

        # 3. 毎月積立
        monthly_df = df.resample('ME').last()
        dca_monthly_shares = (monthly_budget / monthly_df['Close']).sum()
        dca_monthly_price = (monthly_budget * len(monthly_df)) / dca_monthly_shares if dca_monthly_shares > 0 else 0

        # --- ランキングデータの整理 ---
        dva_total_spent = df['DVA_Amount'].sum() # ここで定義
        
        methods = [
            {"name": "提案DVA(株数連動)", "price": dva_total_spent / dva_shares if dva_shares > 0 else 0, "shares": dva_shares},
            {"name": "毎日積立(DCA)", "price": dca_daily_price, "shares": dca_daily_shares},
            {"name": "毎週積立(DCA)", "price": dca_weekly_price, "shares": dca_weekly_shares},
            {"name": "毎月積立(DCA)", "price": dca_monthly_price, "shares": dca_monthly_shares}
        ]

        # --- ランキング表示 ---
        print("-" * 70)
        print(f" {ticker} 実験結果ダブルランキング")
        print("-" * 70)
        
        # 取得単価ランキング（低い順）
        print("【取得単価ランキング】（安いほど効率的）")
        price_rank = sorted(methods, key=lambda x: x['price'])
        best_price = price_rank[0]['price']
        for i, m in enumerate(price_rank, 1):
            if best_price > 0:
                diff = (m['price'] / best_price - 1) * 100
                print(f" {i}位: {m['name']:<15} | ${m['price']:>8.2f} (最安比 +{diff:.2f}%)")
            
        print("\n【最終保有株数ランキング】（多いほど資産増）")
        # 保有株数ランキング（多い順）
        share_rank = sorted(methods, key=lambda x: x['shares'], reverse=True)
        best_share = share_rank[0]['shares']
        for i, m in enumerate(share_rank, 1):
            if best_share > 0:
                diff = (1 - m['shares'] / best_share) * 100
                print(f" {i}位: {m['name']:<15} | {m['shares']:>8.2f} 株 (最多比 -{diff:.2f}%)")
        print("-" * 70)

# 実行
run_strategy_tournament(["NVDA", "AAPL", "TSLA", "8088.T"])