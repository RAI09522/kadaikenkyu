import yfinance as yf
import pandas as pd
import numpy as np

def run_ultimate_comparison(tickers, start_date="2023-01-01"):
    monthly_budget = 100000 
    # 0=月, 1=火, 2=水, 3=木, 4=金
    weekday_map = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI'}
    
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    mkt_vol_mean = mkt_vol.mean()

    for ticker in tickers:
        print(f"\n分析中: {ticker}...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma50'] = df['Close'].rolling(window=50).mean()
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol.reindex(df.index).fillna(mkt_vol_mean)
        df['day_of_week'] = df.index.dayofweek # 曜日を取得

        # --- DVA-AAM 計算ロジック ---
        def calculate_dva(row):
            price = float(row['Close'])
            sma_ref = row['sma50'] if not pd.isna(row['sma50']) else price
            prev = row['prev_close'] if not pd.isna(row['prev_close']) else price
            
            signal = 1 if (price > sma_ref * 0.6) and (price > prev * 0.7) else 0
            phi = max(0.1, 1 - (row['vol'] - (row['mkt_vol'] * 2.0)) / 5)
            
            idx = df.index.get_loc(row.name)
            p_ref = df['Close'].iloc[max(0, idx-20):idx].mean() if idx > 0 else price
            psi = np.exp(3.0 * ((p_ref - price) / p_ref)) if p_ref > 0 else 1
            
            return (monthly_budget / 20) * phi * psi * signal

        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)
        
        # DVAの結果計算
        dva_total_spent = df['DVA_Amount'].sum()
        dva_total_shares = (df['DVA_Amount'] / df['Close']).sum()
        dva_avg = dva_total_spent / dva_total_shares if dva_total_shares > 0 else 0

        # --- 各曜日のDCA計算 ---
        dca_results = {}
        for day_num, day_name in weekday_map.items():
            # その曜日だけの行を抽出
            dca_df = df[df['day_of_week'] == day_num].copy()
            if dca_df.empty:
                dca_results[day_name] = 0
                continue
            
            # 週に1回、25000円分買う設定
            dca_spent = len(dca_df) * 25000
            dca_shares = (25000 / dca_df['Close']).sum()
            dca_results[day_name] = round(dca_spent / dca_shares, 2)

        # --- 結果表示 ---
        print("-" * 65)
        print(f"{ticker} 取得単価比較（DVA vs 全曜日DCA）")
        print("-" * 65)
        print(f" 提案DVAモデル : ${round(dva_avg, 2)}")
        for day, price in dca_results.items():
            if price > 0:
                diff = (1 - dva_avg / price) * 100
                print(f" DCA ({day}積立) : ${price:<8} (DVAの優位性: {diff:.2f}%)")
            else:
                print(f" DCA ({day}積立) : データ不足")
        print("-" * 65)

# 分析実行
run_ultimate_comparison(["NVDA", "AAPL", "TSLA"])