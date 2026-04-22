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

        def calculate_dva(row):
            # 必要なデータがない期間は、50日線で代用するか、判定を緩める
            price = float(row['Close'])
            # 200日線がなければ50日線、それもなければ現在価格を使用
            sma_ref = row['sma200'] if not pd.isna(row['sma200']) else (row['sma50'] if not pd.isna(row['sma50']) else price)
            prev = row['prev_close'] if not pd.isna(row['prev_close']) else price
            
            # 生存判定（条件を少し緩和：SMAの60%以上ならGO）
            signal = 1 if (price > sma_ref * 0.6) and (price > prev * 0.7) else 0
            
            # 安全装置（あまりにもボラが高い時だけ止める）
            vol_val = row['vol'] if not pd.isna(row['vol']) else 0.3
            m_vol_val = row['mkt_vol'] if not pd.isna(row['mkt_vol']) else 0.2
            phi = max(0.1, 1 - (vol_val - (m_vol_val * 2.0)) / 5) # 最低でも0.1は買う
            
            # 加速装置
            # 直近20日の平均を基準にする
            idx = df.index.get_loc(row.name)
            p_ref = df['Close'].iloc[max(0, idx-20):idx].mean() if idx > 0 else price
            gap = (p_ref - price) / p_ref if p_ref > 0 else 0
            psi = np.exp(3.0 * gap) # ラムダを3.0に固定して感度を調整
            
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