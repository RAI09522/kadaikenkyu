import yfinance as yf
import pandas as pd
import numpy as np

def run_performance_test(tickers, start_date="2023-01-01"):
    base_amount = 100000 
    results_summary = []

    # 市場平均（S&P 500）のボラティリティを計算
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    mkt_vol = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        stock_obj = yf.Ticker(ticker)
        df = stock_obj.history(start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 財務指標
        try:
            info = stock_obj.info
            curr_pe, curr_pbr = info.get('trailingPE', 35), info.get('priceToBook', 15)
        except:
            curr_pe, curr_pbr = 35, 15

        # 指標計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma200'] = df['Close'].rolling(window=200).mean()
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol

        def calculate_dva(row):
            if pd.isna(row['sma200']) or pd.isna(row['mkt_vol']): return 0
            price, sma, prev = float(row['Close']), float(row['sma200']), float(row['prev_close'])
            
            # 生存・損切り・安全装置
            signal = 1 if (price > sma * 0.7) and (price > prev * 0.8) else 0
            omega = 1 if price > (sma * 0.8) else 0
            phi = max(0, 1 - (float(row['vol']) - (float(row['mkt_vol']) * 1.5)) / 10)
            
            # 動的ラムダと加速装置
            dyn_lambda = 2.0 * (35 / curr_pe) * (1 + (15 / curr_pbr))
            current_idx = df.index.get_loc(row.name)
            p_avg = df['Close'].iloc[:current_idx].mean() if current_idx > 0 else price
            psi = np.exp(dyn_lambda * (p_avg - price) / p_avg)
            
            return base_amount * phi * psi * omega * signal

        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)
        df['DCA_Amount'] = base_amount

        # 統計計算
        dva_total = df['DVA_Amount'].sum()
        dca_total = df['DCA_Amount'].sum()
        dva_shares = (df['DVA_Amount'] / df['Close']).sum()
        dca_shares = (df['DCA_Amount'] / df['Close']).sum()
        
        dva_avg = dva_total / dva_shares if dva_shares > 0 else 0
        dca_avg = dca_total / dca_shares if dca_shares > 0 else 0
        reduction = (1 - dva_avg / dca_avg) * 100 if dca_avg > 0 else 0

        results_summary.append({
            "銘柄": ticker,
            "DVA単価": round(dva_avg, 2),
            "DCA単価": round(dca_avg, 2),
            "削減率(%)": round(reduction, 2),
            "投資総額比": round(dva_total / dca_total, 2)
        })

    # 結果を一括表示
    print("\n" + "="*50)
    print(f"{'銘柄':<8} | {'DVA単価':<10} | {'DCA単価':<10} | {'単価削減率':<10}")
    print("-" * 50)
    for res in results_summary:
        print(f"{res['銘柄']:<8} | ${res['DVA単価']:<9} | ${res['DCA_Avg']:<9} | {res['削減率(%)']}%")
    print("="*50)

run_performance_test(["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"])