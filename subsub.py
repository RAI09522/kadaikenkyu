import yfinance as yf
import pandas as pd
import numpy as np

def run_ultimate_comparison(tickers, start_date="2023-01-01"):
    monthly_budget = 100000 
    # 比較する曜日設定 (MON=月, TUE=火, WED=水, THU=木, FRI=金)
    weekdays = {'MON': 'W-MON', 'TUE': 'W-TUE', 'WED': 'W-WED', 'THU': 'W-THU', 'FRI': 'W-FRI'}
    
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
        df['mkt_vol'] = mkt_vol.fillna(mkt_vol_mean)

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
        dva_avg = df['DVA_Amount'].sum() / (df['DVA_Amount'] / df['Close']).sum()

        # --- 各曜日のDCA計算 ---
        dca_results = {}
        for day_name, day_code in weekdays.items():
            # その曜日だけのデータを抽出
            dca_df = df.resample(day_code).last().dropna()
            # 週あたり予算を25,000円として計算
            shares = (25000 / dca_df['Close']).sum()
            spent = len(dca_df) * 25000
            dca_results[day_name] = round(spent / shares, 2)

        # --- 結果表示 ---
        print("-" * 65)
        print(f"{ticker} 取得単価比較表")
        print("-" * 65)
        print(f" 提案DVAモデル : ${round(dva_avg, 2)}")
        for day, price in dca_results.items():
            diff = (1 - dva_avg / price) * 100
            print(f" DCA ({day}積立) : ${price:<8} (DVAの優位性: {diff:.2f}%)")
        print("-" * 65)

run_ultimate_comparison(["NVDA", "AAPL", "TSLA"])