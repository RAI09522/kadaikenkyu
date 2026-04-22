import yfinance as yf
import pandas as pd
import numpy as np

def run_japan_stock_system(tickers, start_date="2023-01-01"):
    monthly_budget = 100000 
    daily_base = monthly_budget / 20
    
    print("市場データを取得中...")
    # 日経平均を取得し、インデックスを単純な日付にする
    mkt = yf.download("^N225", start=start_date, progress=False)
    if isinstance(mkt.columns, pd.MultiIndex): mkt.columns = mkt.columns.get_level_values(0)
    mkt.index = mkt.index.date # タイムゾーンを除去して「日付のみ」に
    
    # 市場ボラ計算
    mkt_vol_series = mkt['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【分析中: {ticker}】")
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = df.index.date # タイムゾーンを除去

        # 市場データと個別銘柄データを日付でガッチャンコする（これが確実）
        combined = pd.DataFrame(index=df.index)
        combined['Close'] = df['Close']
        combined['mkt_vol'] = mkt_vol_series.reindex(df.index).ffill().fillna(mkt_vol_series.mean())
        
        # 指標計算
        combined['vol'] = combined['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        combined['ema20'] = combined['Close'].ewm(span=20, adjust=False).mean()
        std20 = combined['Close'].rolling(window=20).std()
        combined['bb_low'] = combined['ema20'] - (std20 * 2)

        # --- DVAロジック ---
        dva_shares = 0.0
        total_spent = 0.0
        
        for i in range(len(combined)):
            row = combined.iloc[i]
            price = float(row['Close'])
            ema = row['ema20']
            bb_l = row['bb_low']
            
            if pd.isna(ema) or pd.isna(bb_l):
                invest_amt = daily_base
            else:
                # 株数フィードバック
                ideal_shares = (daily_base * (i + 1)) / ema
                share_gap = (ideal_shares - dva_shares) / ideal_shares if ideal_shares > 0 else 0
                share_boost = 1.0 + max(0, min(1.0, share_gap * 2.0))
                
                # 日本株向け価格ロジック
                gap = (ema - price) / ema
                psi = np.exp(6.0 * gap) * (2.5 if price < bb_l else 1.0)
                phi = max(0.5, 1 - (row['vol'] - (row['mkt_vol'] * 1.2)) / 5)
                tf = 1.1 if price > ema else 0.9
                
                invest_amt = daily_base * phi * psi * tf * share_boost

            # セーフティ：最低限の購入
            invest_amt = max(invest_amt, daily_base * 0.1)
            
            total_spent += invest_amt
            dva_shares += (invest_amt / price)

        # --- DCA計算 ---
        dca_shares = (daily_base / combined['Close']).sum()
        dca_price = (daily_base * len(combined)) / dca_shares

        # --- 結果表示 ---
        dva_price = total_spent / dva_shares if dva_shares > 0 else 0
        
        print("-" * 50)
        print(f" 提案DVA単価: {dva_price:>10.2f} 円")
        print(f" 毎日積立単価: {dca_price:>10.2f} 円")
        diff = (1 - dva_price / dca_price) * 100
        print(f" 単価抑制率  : {diff:.2f} %")
        print(f" 最終確保株数: {dva_shares:>10.2f} 株")
        print("-" * 50)

run_japan_stock_system(["8088.T", "7203.T", "6758.T", "8058.T"])