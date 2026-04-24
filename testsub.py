import yfinance as yf
import pandas as pd
import numpy as np

def run_ultimate_strategy(tickers, start_date="2020-01-01"):
    monthly_budget = 100000
    daily_base = monthly_budget / 20
    max_cash_multiple = 3  # 1日最大投資倍率

    print("市場データ取得中...")
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    mkt_vol = spy['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n===== {ticker} 分析開始 =====")
        df = yf.Ticker(ticker).history(start=start_date)

        if df.empty:
            print("データなし")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ===== 指標 =====
        df['return'] = df['Close'].pct_change()
        df['vol'] = df['return'].rolling(20).std() * np.sqrt(252)
        df['ema20'] = df['Close'].ewm(span=20).mean()
        df['ema50'] = df['Close'].ewm(span=50).mean()
        std20 = df['Close'].rolling(20).std()
        df['bb_low'] = df['ema20'] - 2 * std20

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 市場ボラ同期
        df['mkt_vol'] = mkt_vol.reindex(df.index).ffill().fillna(mkt_vol.mean())

        # ===== 投資シミュ =====
        shares = 0
        cash_spent = 0
        portfolio_values = []

        for i in range(len(df)):
            price = df['Close'].iloc[i]

            if pd.isna(price):
                portfolio_values.append(shares * price)
                continue

            ema20 = df['ema20'].iloc[i]
            ema50 = df['ema50'].iloc[i]
            bb_low = df['bb_low'].iloc[i]
            vol = df['vol'].iloc[i]
            mktv = df['mkt_vol'].iloc[i]
            rsi = df['rsi'].iloc[i]

            invest = daily_base

            # ===== ① 割安ロジック =====
            if not pd.isna(ema20):
                gap = (ema20 - price) / ema20
                invest *= np.exp(5 * gap)

            # ボリバン下
            if not pd.isna(bb_low) and price < bb_low:
                invest *= 2.0

            # ===== ② トレンド =====
            if not pd.isna(ema50):
                if price > ema50:
                    invest *= 1.2
                else:
                    invest *= 0.7

            # ===== ③ RSI =====
            if not pd.isna(rsi):
                if rsi < 30:
                    invest *= 1.5
                elif rsi > 70:
                    invest *= 0.6

            # ===== ④ ボラ制御 =====
            if not pd.isna(vol) and not pd.isna(mktv):
                risk_factor = 1 - max(0, (vol - mktv * 1.5)) / 5
                invest *= max(0.5, risk_factor)

            # ===== ⑤ 株数フィードバック =====
            if not pd.isna(ema20):
                ideal_shares = (daily_base * (i + 1)) / ema20
                gap = (ideal_shares - shares) / ideal_shares if ideal_shares > 0 else 0
                invest *= (1 + min(1, max(0, gap * 2)))

            # ===== ⑥ 暴落ブースト =====
            if not pd.isna(ema20):
                drop = (ema20 - price) / ema20
                if drop > 0.1:
                    invest *= 2.5
                elif drop > 0.2:
                    invest *= 4.0

            # ===== ⑦ 上限制御 =====
            invest = min(invest, daily_base * max_cash_multiple)

            # ===== 購入 =====
            shares += invest / price
            cash_spent += invest

            portfolio_values.append(shares * price)

        df['portfolio'] = portfolio_values

        # ===== 評価 =====
        returns = df['portfolio'].pct_change().dropna()

        total_return = df['portfolio'].iloc[-1] / cash_spent - 1 if cash_spent > 0 else 0
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # 最大ドローダウン
        cummax = df['portfolio'].cummax()
        drawdown = (df['portfolio'] - cummax) / cummax
        mdd = drawdown.min()

        avg_price = cash_spent / shares if shares > 0 else 0

        print("------ 結果 ------")
        print(f"総投資額: {cash_spent:,.0f} 円")
        print(f"最終資産: {df['portfolio'].iloc[-1]:,.0f} 円")
        print(f"リターン: {total_return*100:.2f}%")
        print(f"取得単価: {avg_price:.2f}")
        print(f"保有株数: {shares:.2f}")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"最大DD: {mdd*100:.2f}%")
        print("------------------")


# 実行
run_ultimate_strategy(["NVDA", "AAPL", "TSLA", "8088.T"])