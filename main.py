import yfinance as yf
import pandas as pd
import numpy as np

def run_share_count_experiment(tickers, start_date="2014-01-01"):
    monthly_budget = 500
    daily_base = monthly_budget / 20
    weekly_base = monthly_budget / 4
    
    print(f"--- 10年間・累積取得株数比較 ({start_date}～) ---")

    # 市場データ
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【銘柄: {ticker}】")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change(fill_method=None).rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        methods_data = []

        # 1. 提案DVA
        dva_shares, dva_spent = 0.0, 0.0
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema, bb_l = df['ema20'].iloc[i], df['bb_low'].iloc[i]
            if pd.isna(ema) or pd.isna(bb_l):
                amt = daily_base
            else:
                ideal_s = (daily_base * (i + 1)) / ema
                f_t = 1.0 + max(0, min(1.0, ((ideal_s - dva_shares) / ideal_s) * 2.0)) if ideal_s > 0 else 1.0
                psi_t = np.exp(6.0 * ((ema - price) / ema)) * (2.5 if price < bb_l else 1.0)
                phi_t = max(0.5, 1 - (df['vol'].iloc[i] - (df['mkt_vol'].iloc[i] * 1.2)) / 5)
                t_t = 1.15 if price > ema else 0.85
                amt = daily_base * f_t * psi_t * phi_t * t_t
            amt = max(amt, daily_base * 0.1)
            dva_spent += amt
            dva_shares += (amt / price)
        methods_data.append(("★提案DVA", dva_spent, dva_shares))

        # 2. 毎日積立
        methods_data.append(("毎日積立", daily_base * len(df), (daily_base / df['Close']).sum()))

        # 3. 曜日別
        days = ["月曜積立", "火曜積立", "水曜積立", "木曜積立", "金曜積立"]
        for d in range(5):
            w_df = df[df.index.dayofweek == d]
            methods_data.append((days[d], weekly_base * len(w_df), (weekly_base / w_df['Close']).sum()))

        # 4. 毎月 (月初・月末)
        m_first = df.resample('MS').first()
        methods_data.append(("毎月(月初)", monthly_budget * len(m_first), (monthly_budget / m_first['Close']).sum()))
        m_last = df.resample('ME').last()
        methods_data.append(("毎月(月末)", monthly_budget * len(m_last), (monthly_budget / m_last['Close']).sum()))

        # --- 株数が多い順にランキング ---
        results = sorted(methods_data, key=lambda x: x[2], reverse=True)
        
        print("-" * 70)
        print(f"{'順位':<2} | {'手法名':<10} | {'取得株数':<12} | {'平均単価':<10} | {'総投資額'}")
        print("-" * 70)
        for i, (name, spent, shares) in enumerate(results):
            avg_p = spent / shares
            print(f"{i+1:>2} | {name:<10} | {shares:>10.2f} 株 | ${avg_p:>8.2f} | ${spent:>8.0f}")
        print("-" * 70)

if __name__ == "__main__":
    run_share_count_experiment(["NVDA", "AAPL", "TSLA", "8088.T"])