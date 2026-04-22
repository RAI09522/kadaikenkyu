import yfinance as yf
import pandas as pd
import numpy as np

def run_japan_stock_system(tickers, start_date="2023-01-01"):
    # 日本株向けの予算設定
    monthly_budget = 100000 
    daily_base = monthly_budget / 20
    
    # 日本市場のベンチマーク（TOPIXまたは日経平均）
    print("日本市場データを取得中...")
    mkt = yf.download("^N225", start=start_date, progress=False) # 日経225
    if isinstance(mkt.columns, pd.MultiIndex): mkt.columns = mkt.columns.get_level_values(0)
    mkt_vol_raw = mkt['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【日本株分析: {ticker}】解析中...")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        # --- DVAロジック（日本株調整済み） ---
        df['DVA_Amount'] = 0.0
        dva_shares = 0.0
        
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
            
            if pd.isna(ema) or pd.isna(bb_l):
                invest_amt = daily_base
            else:
                # 株数フィードバック
                ideal_shares_pace = (daily_base * (i + 1)) / ema
                share_gap = (ideal_shares_pace - dva_shares) / ideal_shares_pace if ideal_shares_pace > 0 else 0
                share_boost = 1.0 + max(0, min(1.0, share_gap * 2.0))
                
                # 日本株は米国株よりボラティリティが低いため、感度(gapの係数)を少し強める(5.0->6.0)
                gap = (ema - price) / ema
                psi = np.exp(6.0 * gap) * (2.5 if price < bb_l else 1.0)
                phi = max(0.5, 1 - (df['vol'].iloc[i] - (df['mkt_vol'].iloc[i] * 1.2)) / 5)
                tf = 1.1 if price > ema else 0.9 # 日本株はレンジ相場が多いため順張り係数を控えめに
                
                invest_amt = daily_base * phi * psi * tf * share_boost

            df.iloc[i, df.columns.get_loc('DVA_Amount')] = invest_amt
            dva_shares += (invest_amt / price)

        # --- DCA計算 ---
        dca_daily_shares = (daily_base / df['Close']).sum()
        dca_daily_price = (daily_base * len(df)) / dca_daily_shares if dca_daily_shares > 0 else 0
        
        # --- 結果表示 ---
        total_spent = df['DVA_Amount'].sum()
        dva_price = total_spent / dva_shares if dva_shares > 0 else 0
        
        print("-" * 60)
        print(f" {ticker} 日本株DVAレポート")
        print("-" * 60)
        print(f" 提案DVA平均単価: {dva_price:>10.2f} 円")
        print(f" 毎日積立平均単価: {dca_daily_price:>10.2f} 円")
        diff = (1 - dva_price / dca_daily_price) * 100 if dca_daily_price > 0 else 0
        print(f" 優位性(単価抑制率): {diff:.2f} %")
        print(f" 最終確保株数: {dva_shares:>10.2f} 株")
        print("-" * 60)

# 日本株の実験対象（例：岩谷産業、トヨタ、ソニー、三菱商事）
run_japan_stock_system(["8088.T", "7203.T", "6758.T", "8058.T"])