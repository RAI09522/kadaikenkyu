import yfinance as yf
import pandas as pd
import numpy as np

def run_japan_stock_system(tickers, start_date="2023-01-01"):
    monthly_budget = 100000 
    daily_base = monthly_budget / 20
    
    print("日本市場（日経225）データを取得中...")
    mkt = yf.download("^N225", start=start_date, progress=False)
    if isinstance(mkt.columns, pd.MultiIndex): mkt.columns = mkt.columns.get_level_values(0)
    # 市場ボラを計算
    mkt_vol_raw = mkt['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【日本株分析: {ticker}】解析中...")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: 
            print(f"データが空です: {ticker}")
            continue
        # マルチインデックス対策
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        # タイムゾーンを合わせる
        df.index = df.index.tz_localize(None)
        mkt_vol_raw.index = mkt_vol_raw.index.tz_localize(None)
        
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        # --- DVAロジック ---
        df['DVA_Amount'] = 0.0
        dva_shares = 0.0
        
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
            vol = df['vol'].iloc[i]
            m_vol = df['mkt_vol'].iloc[i]
            
            # データ不足期間の回避
            if pd.isna(ema) or pd.isna(bb_l) or pd.isna(vol):
                invest_amt = daily_base
            else:
                # 理想の株数（目標設定）
                ideal_shares_pace = (daily_base * (i + 1)) / ema
                share_gap = (ideal_shares_pace - dva_shares) / ideal_shares_pace if ideal_shares_pace > 0 else 0
                share_boost = 1.0 + max(0, min(1.0, share_gap * 2.0))
                
                # 価格ロジック（感度を日本株向けに調整）
                gap = (ema - price) / ema
                psi = np.exp(6.0 * gap) * (2.5 if price < bb_l else 1.0)
                phi = max(0.5, 1 - (vol - (m_vol * 1.2)) / 5)
                tf = 1.1 if price > ema else 0.9
                
                invest_amt = daily_base * phi * psi * tf * share_boost

            # 投資額が極端に0にならないようにセーフティをかける
            invest_amt = max(invest_amt, daily_base * 0.1)
            
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
        print(f" 提案DVA平均単価: {dva_price:>12.2f} 円")
        print(f" 毎日積立平均単価: {dca_daily_price:>12.2f} 円")
        diff = (1 - dva_price / dca_daily_price) * 100 if dca_daily_price > 0 else 0
        print(f" 優位性(単価抑制率): {diff:.2f} %")
        print(f" 最終確保株数: {dva_shares:>12.2f} 株")
        print("-" * 60)

run_japan_stock_system(["8088.T", "7203.T", "6758.T", "8058.T"])