import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_final_experiment(tickers, start_date="2014-01-01"):
    """
    DVA-AAM Ver 3.0: 累積株数フィードバック搭載型
    10年間のヒストリカル・バックテスト実行スクリプト
    """
    # --- 基本設定 ---
    monthly_budget = 500  # 月間予算（ドル）
    daily_base = monthly_budget / 20
    
    print(f"--- 実験開始: {start_date} 以降のデータを解析中 ---")
    
    # 市場平均（S&P 500）のボラティリティ取得
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    results_summary = []

    for ticker in tickers:
        print(f"\n【解析対象: {ticker}】")
        # データ取得
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算 (EMA, ボリンジャーバンド, 個別ボラ)
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['bb_low'] = df['ema20'] - (df['Close'].rolling(window=20).std() * 2)

        # --- DVA-AAM アルゴリズム実行 ---
        df['DVA_Amount'] = 0.0
        dva_shares = 0.0
        
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            bb_l = df['bb_low'].iloc[i]
            vol = df['vol'].iloc[i]
            m_vol = df['mkt_vol'].iloc[i]
            
            if pd.isna(ema) or pd.isna(bb_l):
                invest_amt = daily_base
            else:
                # 1. 株数フィードバック (F)
                ideal_shares = (daily_base * (i + 1)) / ema
                share_gap = (ideal_shares - dva_shares) / ideal_shares if ideal_shares > 0 else 0
                f_t = 1.0 + max(0, min(1.0, share_gap * 2.0))
                
                # 2. 価格乖離加速 (Psi)
                gap = (ema - price) / ema
                psi_t = np.exp(6.0 * gap) * (2.5 if price < bb_l else 1.0)
                
                # 3. 市場連動安全装置 (Phi)
                phi_t = max(0.5, 1 - (vol - (m_vol * 1.2)) / 5)
                
                # 4. トレンド追従 (T)
                t_t = 1.15 if price > ema else 0.85
                
                # 最終投資額の算出
                invest_amt = daily_base * f_t * psi_t * phi_t * t_t

            # 資金ショート防止のセーフティ
            invest_amt = max(invest_amt, daily_base * 0.1)
            
            df.iloc[i, df.columns.get_loc('DVA_Amount')] = invest_amt
            dva_shares += (invest_amt / price)

        # --- DCA (毎日積立) の計算 ---
        dca_shares = (daily_base / df['Close']).sum()
        dca_price = (daily_base * len(df)) / dca_shares
        
        # --- 結果の集計 ---
        dva_total_spent = df['DVA_Amount'].sum()
        dva_price = dva_total_spent / dva_shares
        last_price = float(df['Close'].iloc[-1])
        
        results_summary.append({
            'ticker': ticker,
            'dva_price': dva_price,
            'dca_price': dca_price,
            'dva_shares': dva_shares,
            'dca_shares': dca_shares,
            'final_value': dva_shares * last_price
        })

        # --- 結果表示 ---
        print(f" 提案DVA単価: ${dva_price:>8.2f} (DCA比: {((dva_price/dca_price)-1)*100:>+6.2f}%)")
        print(f" 最終累計株数: {dva_shares:>10.2f} 株")

    # --- 最終ランキング出力 ---
    print("\n" + "="*50)
    print(" 実験結果サマリー (10年バックテスト)")
    print("="*50)
    for res in results_summary:
        print(f"【{res['ticker']}】")
        print(f"  単価抑制率: {((1 - res['dva_price']/res['dca_price'])*100):.2f}% 優秀")
        print(f"  最終資産額: ${res['final_value']:,.2f}")
    print("="*50)

# 実験の実行
if __name__ == "__main__":
    # 米国市場を代表するボラティリティの異なる銘柄を選定
    run_final_experiment(["NVDA", "AAPL", "TSLA"], start_date="2014-01-01")
