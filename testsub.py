import yfinance as yf
import pandas as pd
import numpy as np

def run_strategy_tournament(tickers, start_date="2023-01-01"):
    monthly_budget = 100000
    
    print("市場データを取得中...")
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): 
        spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n【銘柄分析: {ticker}】データを取得中...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)

        # --- 指標計算 (BB / EMA / Vol) ---
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # EMA20 (指数平滑移動平均)
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        # ボリンジャーバンド (-2σ)
        std20 = df['Close'].rolling(window=20).std()
        df['bb_low'] = df['ema20'] - (std20 * 2)

        # --- DVA 最強ロジック関数 ---
        def calculate_dva(row):
            price = float(row['Close'])
            ema = row['ema20']
            bb_l = row['bb_low']
            
            if pd.isna(ema) or pd.isna(bb_l): 
                return monthly_budget / 20 # データ不足期間は定額積立

            # 1. 加速装置 (Psi)
            # EMAより安いほど加速、さらにBB-2σ以下ならボーナスブースト
            gap = (ema - price) / ema
            bb_boost = 2.5 if price < bb_l else 1.0 # 2.5倍に強化
            psi = np.exp(5.0 * gap) * bb_boost # 感度を5.0にアップ
            
            # 2. 安全装置 (Phi) ＆ 順張り追従
            # 上昇トレンド中は投資額を1.2倍にして「置いていかれ」を防止
            trend_follow = 1.2 if price > ema else 0.8
            phi = max(0.5, 1 - (row['vol'] - (row['mkt_vol'] * 1.5)) / 5)
            
            # 3. 生存判定
            signal = 1 if price > ema * 0.5 else 0
            
            return (monthly_budget / 20) * phi * psi * trend_follow * signal

        # 投資実行額の計算
        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)
        
        # --- 結果計算 ---
        dva_shares = (df['DVA_Amount'] / df['Close']).sum()
        dva_avg = df['DVA_Amount'].sum() / dva_shares if dva_shares > 0 else 0

        dca_daily_avg = ( (monthly_budget/20) * len(df) ) / ( (monthly_budget/20) / df['Close'] ).sum()
        
        weekly_df = df[df.index.dayofweek == 2].copy()
        dca_weekly_avg = (25000 * len(weekly_df)) / (25000 / weekly_df['Close']).sum()

        monthly_df = df.resample('ME').last()
        dca_monthly_avg = (monthly_budget * len(monthly_df)) / (monthly_budget / monthly_df['Close']).sum()
        # --- DVA 累積株数連動ロジック ---
        df['Cumulative_Shares'] = 0.0
        df['Actual_Investment'] = 0.0

        # ループで1日ずつ計算（累積が必要なため）
        current_shares = 0.0
        
        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            ema = df['ema20'].iloc[i]
            
            # 1. 理想の株数（毎日一定額買った場合のシミュレーション）
            # 比較対象（毎日DCA）が今何株持っているはずか
            ideal_shares = (daily_base * (i + 1)) / ema # 簡易的な目標設定
            
            # 2. 株数ギャップ（足りないほどプラス）
            # 自分が理想より少なければ、その分を埋めるためにブーストをかける
            share_gap = (ideal_shares - current_shares) / ideal_shares if ideal_shares > 0 else 0
            share_boost = 1.0 + max(0, share_gap * 2.0) # 足りない分に応じて最大2倍程度まで
            
            # 3. 従来の価格ロジック（Psi, Phi）
            gap = (ema - price) / ema
            psi = np.exp(5.0 * gap) * (2.5 if price < df['bb_low'].iloc[i] else 1.0)
            phi = max(0.5, 1 - (df['vol'].iloc[i] - (df['mkt_vol'].iloc[i] * 1.5)) / 5)
            tf = 1.2 if price > ema else 0.8
            
            # 4. 最終投資額（株数ブーストを乗算）
            invest_amt = daily_base * phi * psi * tf * share_boost
            
            # 更新
            bought_shares = invest_amt / price
            current_shares += bought_shares
            df.iloc[i, df.columns.get_loc('DVA_Amount')] = invest_amt
            df.iloc[i, df.columns.get_loc('Cumulative_Shares')] = current_shares

        # ランキング表示
        results = [
            ("提案DVAモデル(BB+EMA)", dva_avg),
            ("毎日積立(DCA)", dca_daily_avg),
            ("毎週積立(DCA)", dca_weekly_avg),
            ("毎月積立(DCA)", dca_monthly_avg)
        ]
        results.sort(key=lambda x: x[1])

        print("-" * 65)
        print(f" {ticker} 最終決戦ランキング")
        print("-" * 65)
        for i, (name, price) in enumerate(results, 1):
            diff = (price / results[0][1] - 1) * 100
            print(f" {i}位: {name:<20} | ${price:<8.2f} (最安比 +{diff:.2f}%)")
        print("-" * 65)

# 実行（ここに実験したい株のコードを入力）
run_strategy_tournament(["NVDA", "AAPL", "TSLA", "8088.T"]) 