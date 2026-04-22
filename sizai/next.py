import yfinance as yf
import pandas as pd
import numpy as np

def run_final_experiment(tickers, start_date="2023-01-01"):
    monthly_budget = 100000 
    results_summary = []

    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    # 市場ボラ（欠損値は平均で埋める）
    mkt_vol = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    mkt_vol_mean = mkt_vol.mean()

    for ticker in tickers:
        stock_obj = yf.Ticker(ticker)
        df = stock_obj.history(start=start_date)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma200'] = df['Close'].rolling(window=200).mean()
        df['sma50'] = df['Close'].rolling(window=50).mean() # 200日線の代わりの保険
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol.fillna(mkt_vol_mean)

       # calculate_dva 関数の中身を以下のように微調整します

def calculate_dva(row):
    price = float(row['Close'])
    # 基準線をSMA50に据え、トレンドへの追従性を高める
    sma_ref = row['sma50'] if not pd.isna(row['sma50']) else price
    prev = row['prev_close'] if not pd.isna(row['prev_close']) else price
    
    # 1. 生存判定（トレンドに乗り続けるよう緩和）
    # 50日線の50%以上なら積極的に継続
    signal = 1 if (price > sma_ref * 0.5) and (price > prev * 0.7) else 0
    
    # 2. 安全装置（Phi）
    # 市場との連動性を重視
    phi = max(0.5, 1 - (row['vol'] - (row['mkt_vol'] * 2.0)) / 5) 
    
    # 3. 加速装置（Psi） - ここが肝！
    # 基準を「過去平均」ではなく「現在のトレンド（SMA50）」との乖離にする
    gap = (sma_ref - price) / sma_ref
    # 割安（gap > 0）の時は爆発的に買い、割高（gap < 0）でも一定量は買う
    psi = np.exp(4.0 * gap) if gap > 0 else np.exp(1.0 * gap)
    
    # 1日あたりの標準投資額
    daily_base = monthly_budget / 20
    return daily_base * phi * psi * signal

        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)

        # --- 統計計算 ---
        monthly_df = df.resample('ME').last()
        dca_total_spent = len(monthly_df) * monthly_budget
        dca_total_shares = (monthly_budget / monthly_df['Close']).sum()
        dca_avg_price = dca_total_spent / dca_total_shares

        dva_total_spent = df['DVA_Amount'].sum()
        dva_total_shares = (df['DVA_Amount'] / df['Close']).sum()
        dva_avg_price = dva_total_spent / dva_total_shares if dva_total_shares > 0 else 0
        
        reduction = (1 - dva_avg_price / dca_avg_price) * 100 if dva_avg_price > 0 else 0

        results_summary.append({
            "Ticker": ticker,
            "DVA_Avg": round(dva_avg_price, 2),
            "DCA_Avg": round(dca_avg_price, 2),
            "Reduction": round(reduction, 2)
        })

    # 結果表示
    print("\n" + "="*60)
    print(f"{'Ticker':<8} | {'DVA Avg($)':<12} | {'DCA(MonthEnd)($)':<16} | {'削減率(%)':<10}")
    print("-" * 60)
    for res in results_summary:
        print(f"{res['Ticker']:<8} | ${res['DVA_Avg']:<11} | ${res['DCA_Avg']:<15} | {res['Reduction']}%")
    print("="*60)

run_final_experiment(["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"])