import yfinance as yf
import pandas as pd
import numpy as np

def run_custom_dca_experiment(tickers, start_date="2014-01-01"):
    # ==========================================
    # 【編集エリア】ここを変えて実験条件を設定
    # ==========================================
    monthly_budget = 500   # 月間の総予算
    
    # 毎週積立の設定
    weekly_day = 2         # 何曜日に買うか（0:月, 1:火, 2:水, 3:木, 4:金）
    
    # 毎月積立の設定
    monthly_mode = 'last'  # 'first'（月初） or 'last'（月末）
    # ==========================================

    daily_base = monthly_budget / 20
    weekly_base = monthly_budget / 4
    
    print(f"--- 実験条件 ---")
    print(f"予算: {monthly_budget}ドル/月")
    print(f"毎週積立: 曜日の設定値={weekly_day}")
    print(f"毎月積立: タイミング={monthly_mode}")
    print(f"----------------")

    # 市場平均データ
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【解析: {ticker}】")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算 (DVA用)
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        # --- 1. 提案手法 DVA-AAM (比較用) ---
        dva_shares = 0.0
        dva_total_spent = 0.0
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
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
            dva_total_spent += amt
            dva_shares += (amt / price)

        # --- 2. 毎日積立 (Daily DCA) ---
        dca_daily_shares = (daily_base / df['Close']).sum()
        dca_daily_spent = daily_base * len(df)
        
        # --- 3. 毎週積立 (Weekly DCA) ---
        # 指定した曜日のデータだけ抽出
        weekly_df = df[df.index.dayofweek == weekly_day].copy()
        dca_weekly_shares = (weekly_base / weekly_df['Close']).sum()
        dca_weekly_spent = weekly_base * len(weekly_df)

        # --- 4. 毎月積立 (Monthly DCA) ---
        # 指定したタイミング（月初or月末）のデータ抽出
        if monthly_mode == 'first':
            monthly_df = df.resample('MS').first() # 月初
        else:
            monthly_df = df.resample('ME').last()  # 月末
        dca_monthly_shares = (monthly_budget / monthly_df['Close']).sum()
        dca_monthly_spent = monthly_budget * len(monthly_df)

        # --- 結果まとめ ---
        stats = [
            ("提案DVA-AAM", dva_total_spent, dva_shares),
            ("毎日積立", dca_daily_spent, dca_daily_shares),
            (f"毎週積立(曜日:{weekly_day})", dca_weekly_spent, dca_weekly_shares),
            (f"毎月積立({monthly_mode})", dca_monthly_spent, dca_monthly_shares)
        ]
        
        print(f"{'手法':<18} | {'平均単価':<10} | {'累計株数':<10}")
        for name, spent, shares in stats:
            avg = spent / shares if shares > 0 else 0
            print(f"{name:<18} | ${avg:>8.2f} | {shares:>8.2f}株")

# 実験実行
run_custom_dca_experiment(["NVDA", "AAPL", "TSLA"], start_date="2014-01-01")
