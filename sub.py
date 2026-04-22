import yfinance as yf
import pandas as pd
import numpy as np

def run_final_experiment(tickers, start_date="2023-01-01"):
    monthly_budget = 100000  # 毎月の積立予算
    results_summary = []

    # 1. 市場平均（S&P 500）のデータ取得
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    mkt_vol = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        stock_obj = yf.Ticker(ticker)
        df = stock_obj.history(start=start_date)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma200'] = df['Close'].rolling(window=200).mean()
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol

        # --- DVA-AAM 計算ロジック ---
        def calculate_dva(row):
            if pd.isna(row['sma200']) or pd.isna(row['mkt_vol']) or pd.isna(row['prev_close']):
                return 0
            
            price, sma, prev = float(row['Close']), float(row['sma200']), float(row['prev_close'])
            
            # 生存・損切り・安全装置
            signal = 1 if (price > sma * 0.7) and (price > prev * 0.8) else 0
            omega = 1 if price > (sma * 0.8) else 0
            phi = max(0, 1 - (float(row['vol']) - (float(row['mkt_vol']) * 1.5)) / 10)
            
            # 動的ラムダ（固定基準値を使用）
            dyn_lambda = 4.0 # PER/PBRを考慮した強めの設定
            
            # 期待値計算（直近20日平均より安いか）
            p_ref = df['Close'].iloc[max(0, df.index.get_loc(row.name)-20):df.index.get_loc(row.name)].mean()
            p_ref = p_ref if not pd.isna(p_ref) else price
            gap = (p_ref - price) / p_ref
            psi = np.exp(dyn_lambda * gap)
            
            # 日次ベースの投資額（予算を営業日数20で割ったものをベースにする）
            daily_base = monthly_budget / 20
            return daily_base * phi * psi * omega * signal

        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)

        # --- DCA（毎月末の終値で積立）の計算 ---
        # 月ごとの最終行を抽出
        monthly_df = df.resample('ME').last() 
        dca_total_spent = len(monthly_df) * monthly_budget
        dca_total_shares = (monthly_budget / monthly_df['Close']).sum()
        dca_avg_price = dca_total_spent / dca_total_shares

        # --- DVAの合計統計 ---
        dva_total_spent = df['DVA_Amount'].sum()
        dva_total_shares = (df['DVA_Amount'] / df['Close']).sum()
        dva_avg_price = dva_total_spent / dva_total_shares if dva_total_shares > 0 else 0

        # --- 削減率の計算 ---
        reduction = (1 - dva_avg_price / dca_avg_price) * 100 if dca_avg_price > 0 else 0

        results_summary.append({
            "Ticker": ticker,
            "DVA_Avg": round(dva_avg_price, 2),
            "DCA_Avg": round(dca_avg_price, 2),
            "Reduction": round(reduction, 2),
            "Total_Invested": int(dva_total_spent)
        })

    # 結果表示
    print("\n" + "="*60)
    print(f"{'Ticker':<8} | {'DVA Avg($)':<12} | {'DCA(MonthEnd)($)':<16} | {'削減率(%)':<10}")
    print("-" * 60)
    for res in results_summary:
        print(f"{res['Ticker']:<8} | ${res['DVA_Avg']:<11} | ${res['DCA_Avg']:<15} | {res['Reduction']}%")
    print("="*60)
    print("※DCAは毎月末の終値で100,000円ずつ購入したと仮定")

run_final_experiment(["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"])