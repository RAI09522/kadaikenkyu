import yfinance as yf
import pandas as pd
import numpy as np

def run_ultimate_comparison(tickers, start_date="2014-01-01"):
    monthly_budget = 500
    daily_base = monthly_budget / 20
    weekly_base = monthly_budget / 4
    
    print(f"--- 10年間一括検証開始 ({start_date}～) ---")

    # 市場平均データ（安全装置用）
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【解析対象: {ticker}】")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        # --- 手法エントリーリスト ---
        methods_data = []

        # A. 提案手法 DVA-AAM
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
        methods_data.append(("★提案DVA-AAM", dva_spent, dva_shares))

        # B. 毎日積立
        methods_data.append(("毎日積立", daily_base * len(df), (daily_base / df['Close']).sum()))

        # C. 各曜日積立 (月～金)
        day_names = ["月曜積立", "火曜積立", "水曜積立", "木曜積立", "金曜積立"]
        for d in range(5):
            w_df = df[df.index.dayofweek == d]
            methods_data.append((day_names[d], weekly_base * len(w_df), (weekly_base / w_df['Close']).sum()))

        # D. 毎月積立 (月初・月末)
        m_first = df.resample('MS').first()
        m_last = df.resample('ME').last()
        methods_data.append(("毎月(月初)積立", monthly_budget * len(m_first), (monthly_budget / m_first['Close']).sum()))
        methods_data.append(("毎月(月末)積立", monthly_budget * len(m_last), (monthly_budget / m_last['Close']).sum()))

        # --- ランキング表示 ---
        results = []
        for name, spent, shares in methods_data:
            avg_p = spent / shares
            results.append({'name': name, 'avg_p': avg_p, 'shares': shares})
        
        # 単価が安い順に並び替え
        results = sorted(results, key=lambda x: x['avg_p'])
        
        print("-" * 65)
        print(f"{'順位':<2} | {'手法名':<12} | {'平均取得単価':<12} | {'単価の差(%)'}")
        print("-" * 65)
        best_p = results[0]['avg_p']
        for i, res in enumerate(results):
            diff = (res['avg_p'] / best_p - 1) * 100
            print(f"{i+1:>2} | {res['name']:<12} | ${res['avg_p']:>10.2f} | +{diff:>5.2f}%")
        print("-" * 65)

if __name__ == "__main__":
    run_ultimate_comparison(["NVDA", "AAPL", "TSLA"])
