import yfinance as yf
import pandas as pd
import numpy as np

def run_strategy_tournament(tickers, start_date="2023-01-01"):
    monthly_budget = 100000
    
    # 1. 市場平均（S&P 500）のデータ取得をループの外で1回だけ行う
    print("市場データを取得中...")
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): 
        spy.columns = spy.columns.get_level_values(0)
    
    # 市場ボラの基本計算
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【銘柄分析: {ticker}】データを取得中...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)

        # 市場ボラを銘柄のデータフレームのインデックスに合わせる（ここを修正）
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())

        # 指標計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma50'] = df['Close'].rolling(window=50).mean()
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        # --- DVA (提案モデル) の計算 ---
        def calculate_dva(row):
            price = float(row['Close'])
            sma_ref = row['sma50'] if not pd.isna(row['sma50']) else price
            gap = (sma_ref - price) / sma_ref if sma_ref > 0 else 0
            
            # 安全装置
            phi = max(0.1, 1 - (row['vol'] - (row['mkt_vol'] * 2.0)) / 5)
            # 加速装置
            psi = np.exp(3.0 * gap)
            # 生存判定
            signal = 1 if price > sma_ref * 0.5 else 0
            
            return (monthly_budget / 20) * phi * psi * signal

        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)
        
        # DVAの結果計算
        dva_total_spent = df['DVA_Amount'].sum()
        dva_total_shares = (df['DVA_Amount'] / df['Close']).sum()
        dva_avg = dva_total_spent / dva_total_shares if dva_total_shares > 0 else 0

        # --- 2. DCA 毎日積立 ---
        daily_amt = monthly_budget / 20
        dca_daily_avg = (daily_amt * len(df)) / (daily_amt / df['Close']).sum()

        # --- 3. DCA 毎週積立 (水曜) ---
        weekly_df = df[df.index.dayofweek == 2].copy()
        dca_weekly_avg = (25000 * len(weekly_df)) / (25000 / weekly_df['Close']).sum()

        # --- 4. DCA 毎月積立 (月末) ---
        monthly_df = df.resample('ME').last()
        dca_monthly_avg = (monthly_budget * len(monthly_df)) / (monthly_budget / monthly_df['Close']).sum()

        # --- ランキング化 ---
        results = [
            ("提案DVAモデル", dva_avg),
            ("毎日積立(DCA)", dca_daily_avg),
            ("毎週積立(DCA)", dca_weekly_avg),
            ("毎月積立(DCA)", dca_monthly_avg)
        ]
        results.sort(key=lambda x: x[1])

        print("-" * 65)
        print(f" {ticker} 取得単価ランキング (低いほど優秀)")
        print("-" * 65)
        for i, (name, price) in enumerate(results, 1):
            diff_from_best = (price / results[0][1] - 1) * 100
            print(f" {i}位: {name:<12} | ${price:<8.2f} (最安比 +{diff_from_best:.2f}%)")
        print("-" * 65)

run_strategy_tournament(["NVDA", "AAPL", "TSLA"])