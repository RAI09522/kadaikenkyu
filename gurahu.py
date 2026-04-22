import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_report_graphs(tickers, start_date="2023-01-01"):
    monthly_budget = 100000
    daily_base = monthly_budget / 20

    # 市場データの取得
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    mkt_vol_raw = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"グラフ生成中: {ticker}...")
        df = yf.download(ticker, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['bb_low'] = df['ema20'] - (std20 * 2)
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())

        # DVA投資額計算
        def calc_dva(row):
            p, ema, bbl = float(row['Close']), row['ema20'], row['bb_low']
            if pd.isna(ema) or pd.isna(bbl): return daily_base
            gap = (ema - p) / ema
            psi = np.exp(5.0 * gap) * (2.5 if p < bbl else 1.0)
            phi = max(0.5, 1 - (row['vol'] - (row['mkt_vol'] * 1.5)) / 5)
            tf = 1.2 if p > ema else 0.8
            return daily_base * phi * psi * tf

        df['DVA_Amount'] = df.apply(calc_dva, axis=1)
        dva_avg = df['DVA_Amount'].sum() / (df['DVA_Amount'] / df['Close']).sum()

        # グラフ描画
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

        # 上段：株価とBB/EMA
        ax1.plot(df.index, df['Close'], color='black', label='Price', linewidth=1)
        ax1.plot(df.index, df['ema20'], color='blue', alpha=0.3, label='EMA20')
        ax1.fill_between(df.index, df['bb_low'], df['ema20'], color='gray', alpha=0.1, label='Buy Zone')
        ax1.axhline(y=dva_avg, color='red', linestyle='--', label=f'DVA Avg: ${dva_avg:.2f}')
        ax1.set_title(f'{ticker} Strategy Analysis (DVA vs Market)')
        ax1.legend(loc='upper left')

        # 下段：投資額の推移
        ax2.fill_between(df.index, 0, df['DVA_Amount'], color='orange', alpha=0.5, label='DVA Investment ($)')
        ax2.axhline(y=daily_base, color='black', linestyle=':', alpha=0.5, label='DCA Base')
        ax2.set_ylabel('Daily Invest ($)')
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{ticker}_experiment.png')
        print(f"保存完了: {ticker}_experiment.png")

generate_report_graphs(["NVDA", "TSLA"])