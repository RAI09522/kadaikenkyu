import yfinance as yf
import pandas as pd
import numpy as np

def compare_strategies(ticker, start="2020-01-01"):
    monthly_budget = 100000
    daily_base = monthly_budget / 20

    print(f"\n===== {ticker} 比較開始 =====")

    # データ取得
    df = yf.Ticker(ticker).history(start=start)
    if df.empty:
        print("データなし")
        return

    # 指標
    df['ret'] = df['Close'].pct_change()
    df['ema20'] = df['Close'].ewm(span=20).mean()
    df['ema50'] = df['Close'].ewm(span=50).mean()
    std20 = df['Close'].rolling(20).std()
    df['bb_low'] = df['ema20'] - 2 * std20

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # =========================
    # ① DVA（最強版簡易）
    # =========================
    dva_shares = 0
    dva_cash = 0
    dva_values = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]

        if pd.isna(price):
            dva_values.append(dva_shares * price)
            continue

        invest = daily_base

        ema20 = df['ema20'].iloc[i]
        ema50 = df['ema50'].iloc[i]
        bb_low = df['bb_low'].iloc[i]
        rsi = df['rsi'].iloc[i]

        if not pd.isna(ema20):
            gap = (ema20 - price) / ema20
            invest *= np.exp(5 * gap)

        if not pd.isna(bb_low) and price < bb_low:
            invest *= 2

        if not pd.isna(ema50):
            invest *= 1.2 if price > ema50 else 0.7

        if not pd.isna(rsi):
            if rsi < 30:
                invest *= 1.5
            elif rsi > 70:
                invest *= 0.6

        invest = min(invest, daily_base * 3)

        dva_shares += invest / price
        dva_cash += invest
        dva_values.append(dva_shares * price)

    df['DVA'] = dva_values

    # =========================
    # ② DCA（毎日）
    # =========================
    dca_daily_shares = (daily_base / df['Close']).cumsum()
    df['DCA_daily'] = dca_daily_shares * df['Close']

    # =========================
    # ③ DCA（毎週）
    # =========================
    weekly = df[df.index.dayofweek == 2].copy()  # 水曜
    weekly_shares = (25000 / weekly['Close']).cumsum()
    weekly['shares'] = weekly_shares

    df['DCA_weekly'] = np.nan
    last = 0
    for i in range(len(df)):
        date = df.index[i]
        if date in weekly.index:
            last = weekly.loc[date, 'shares']
        df.iloc[i, df.columns.get_loc('DCA_weekly')] = last * df['Close'].iloc[i]

    # =========================
    # ④ DCA（毎月）
    # =========================
    monthly = df.resample('ME').last()
    monthly_shares = (monthly_budget / monthly['Close']).cumsum()
    monthly['shares'] = monthly_shares

    df['DCA_monthly'] = np.nan
    last = 0
    for i in range(len(df)):
        date = df.index[i]
        if date in monthly.index:
            last = monthly.loc[date, 'shares']
        df.iloc[i, df.columns.get_loc('DCA_monthly')] = last * df['Close'].iloc[i]

    # =========================
    # 評価関数
    # =========================
    def evaluate(series, name):
        series = series.dropna()
        returns = series.pct_change().dropna()

        total_invest = monthly_budget * ((series.index[-1] - series.index[0]).days / 30)

        total_return = series.iloc[-1] / total_invest - 1 if total_invest > 0 else 0
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        cummax = series.cummax()
        dd = (series - cummax) / cummax
        mdd = dd.min()

        return {
            "name": name,
            "final": series.iloc[-1],
            "return": total_return,
            "sharpe": sharpe,
            "mdd": mdd
        }

    results = [
        evaluate(df['DVA'], "DVA"),
        evaluate(df['DCA_daily'], "DCA(毎日)"),
        evaluate(df['DCA_weekly'], "DCA(毎週)"),
        evaluate(df['DCA_monthly'], "DCA(毎月)")
    ]

    # =========================
    # 結果表示
    # =========================
    print("\n------ 結果 ------")
    for r in results:
        print(f"{r['name']:<12} | "
              f"最終資産: {r['final']:>10,.0f} | "
              f"リターン: {r['return']*100:>6.2f}% | "
              f"Sharpe: {r['sharpe']:>5.2f} | "
              f"MDD: {r['mdd']*100:>6.2f}%")
    print("------------------")

# 実行
compare_strategies("NVDA")
compare_strategies("AAPL")
compare_strategies("TSLA")
compare_strategies("8088.T")