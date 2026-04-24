import yfinance as yf
import pandas as pd
import numpy as np

def run_dva_resurrection(tickers, start_date="2014-01-01"):
    monthly_budget = 50000
    daily_base = monthly_budget / 20
    weekly_base = monthly_budget / 4
    
    print(f"--- DVA復活・日本株バックテスト ---")

    # 日経平均（安全装置用）
    mkt = yf.download("^N225", start=start_date, progress=False)
    if isinstance(mkt.columns, pd.MultiIndex): mkt.columns = mkt.columns.get_level_values(0)
    mkt_vol = mkt['Close'].pct_change(fill_method=None).rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【解析: {ticker}】")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['m_vol'] = mkt_vol.reindex(df.index).ffill().fillna(mkt_vol.mean())
        df['vol'] = df['Close'].pct_change(fill_method=None).rolling(window=20).std() * np.sqrt(252)
        df['ema'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['std'] = df['Close'].rolling(window=20).std()
        df['bb_l'] = df['ema'] - (df['std'] * 2)

        # --- DVA復活ロジック ---
        dva_shares, dva_spent = 0.0, 0.0
        
        for i in range(len(df)):
            try:
                p = float(df['Close'].iloc[i])
                e = float(df['ema'].iloc[i])
                b = float(df['bb_l'].iloc[i])
                v = float(df['vol'].iloc[i])
                mv = float(df['m_vol'].iloc[i])

                # 異常値チェック
                if p <= 0 or e <= 0 or np.isnan(e):
                    amt = daily_base
                else:
                    # 1. 株数フィードバック (F)
                    ideal = (daily_base * (i + 1)) / e
                    f_t = 1.0 + max(0, min(1.0, ((ideal - dva_shares) / ideal) * 2.0)) if ideal > 0.01 else 1.0
                    
                    # 2. 指数加速 (Psi) - 指数爆発を防ぐため上限を3.0に
                    gap = (e - p) / e
                    psi_t = np.exp(min(2.0, 5.0 * gap)) 
                    if p < b: psi_t *= 2.0
                    
                    # 3. 安全装置 (Phi)
                    phi_t = max(0.5, 1 - (v - (mv * 1.2)) / 5) if not np.isnan(v) else 1.0
                    
                    # 4. トレンド (T)
                    t_t = 1.1 if p > e else 0.9
                    
                    amt = daily_base * f_t * psi_t * phi_t * t_t
                
                # 数値の最終バリデーション
                if not np.isfinite(amt) or amt <= 0: amt = daily_base
            except:
                amt = daily_base

            amt = max(amt, daily_base * 0.1)
            dva_spent += amt
            dva_shares += (amt / p)

        # 手法リスト作成
        results = [("★提案DVA", dva_spent, dva_shares)]
        results.append(("毎日積立", daily_base * len(df), (daily_base / df['Close']).sum()))
        
        for d, name in enumerate(["月曜", "火曜", "水曜", "木曜", "金曜"]):
            w_df = df[df.index.dayofweek == d]
            results.append((f"{name}積立", weekly_base * len(w_df), (weekly_base / w_df['Close']).sum()))
        
        m_f = df.resample('MS').first()
        results.append(("毎月(月初)", monthly_budget * len(m_f), (monthly_budget / m_f['Close']).sum()))
        m_l = df.resample('ME').last()
        results.append(("毎月(月末)", monthly_budget * len(m_l), (monthly_budget / m_l['Close']).sum()))

        # --- 表示 (絶対にnanを出さない) ---
        clean_results = []
        for n, sp, sh in results:
            # 万が一nanが混じっても0として扱う
            shares = sh if (np.isfinite(sh) and sh > 0) else 1e-9
            clean_results.append((n, sp, shares))

        sorted_res = sorted(clean_results, key=lambda x: x[2], reverse=True)
        
        print("-" * 80)
        print(f"{'順位':<2} | {'手法名':<10} | {'取得株数':>12} | {'平均単価':>10} | {'総投資額':>12}")
        print("-" * 80)
        for i, (name, spent, shares) in enumerate(sorted_res):
            avg = spent / shares
            print(f"{i+1:>2} | {name:<10} | {shares:>12.2f} 株 | {int(avg):>10} 円 | {int(spent):>12} 円")
        print("-" * 80)

if __name__ == "__main__":
    run_dva_resurrection(["8058.T", "8088.T", "7203.T"])