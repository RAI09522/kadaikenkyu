import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_long_term_analysis(tickers, start_date="2023-01-01"):
    base_amount = 100000 
    
    # 1. 市場平均（S&P 500）のボラティリティを計算
    spy = yf.download("^GSPC", start=start_date)
    # マルチインデックス対策：列をフラットにする
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    mkt_vol = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"\n--- {ticker} の分析を開始 ---")
        stock_obj = yf.Ticker(ticker)
        df = stock_obj.history(start=start_date)
        
        if df.empty:
            print(f"{ticker} のデータ取得に失敗しました。")
            continue

        # マルチインデックス対策：列名を単純化
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 財務指標（エラー回避のためデフォルト値を設定）
        try:
            info = stock_obj.info
            curr_pe = info.get('trailingPE', 35)
            curr_pbr = info.get('priceToBook', 15)
        except:
            curr_pe, curr_pbr = 35, 15

        # 必要な指標の計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma200'] = df['Close'].rolling(window=200).mean()
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol

        # DVA-AAM 計算エンジン
        def calculate_dva(row):
            # データの欠損がある期間（開始直後など）はスキップ
            if pd.isna(row['sma200']) or pd.isna(row['mkt_vol']) or pd.isna(row['prev_close']):
                return 0
            
            # 生存・損切り判定 (数値として取得)
            price = float(row['Close'])
            sma = float(row['sma200'])
            prev = float(row['prev_close'])
            
            f_trend = 1 if price > (sma * 0.7) else 0
            f_panic = 1 if price > (prev * 0.8) else 0
            omega = 1 if price > (sma * 0.8) else 0
            
            # 安全装置
            phi = max(0, 1 - (float(row['vol']) - (float(row['mkt_vol']) * 1.5)) / 10)
            
            # 動的ラムダ (簡易版)
            dyn_lambda = 2.0 * (35 / curr_pe) * (1 + (15 / curr_pbr))
            
            # 加速装置
            # その時点までの平均単価を算出
            current_idx = df.index.get_loc(row.name)
            p_avg = df['Close'].iloc[:current_idx].mean() if current_idx > 0 else price
            
            gap = (p_avg - price) / p_avg if p_avg > 0 else 0
            psi = np.exp(dyn_lambda * gap)
            
            return base_amount * phi * psi * omega * f_trend * f_panic

        # 計算実行
        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)

        # 4. 結果の可視化
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['Close'], label='Stock Price', color='black')
        plt.plot(df.index, df['sma200'], label='SMA200', color='gray', linestyle='--')
        plt.title(f"{ticker} Analysis")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.fill_between(df.index, df['DVA_Amount'], color='blue', alpha=0.3, label='DVA-AAM Investment')
        plt.axhline(y=base_amount, color='red', linestyle='--', label='Normal DCA')
        plt.ylabel("Investment Amount")
        plt.legend()
        plt.tight_layout()
        
        # グラフをファイル保存（VSCode等で表示されない場合用）
        plt.savefig(f"{ticker}_analysis.png")
        plt.show()
        print(f"✅ {ticker} の分析完了。画像 '{ticker}_analysis.png' を保存しました。")

# 実行する銘柄リスト
run_long_term_analysis(["NVDA", "AAPL", "TSLA"])

# 各銘柄の累積投資額と取得株数の簡易計算
total_dva_spent = df['DVA_Amount'].sum()
total_dca_spent = df['DCA_Amount'].sum()

# 取得単価の比較（投資額 / その時の株価 の合計）
shares_dva = (df['DVA_Amount'] / df['Close']).sum()
shares_dca = (df['DCA_Amount'] / df['Close']).sum()

avg_price_dva = total_dva_spent / shares_dva
avg_price_dca = total_dca_spent / shares_dca

print(f"【最終比較結果】")
print(f"DVA平均取得単価: {avg_price_dva:.2f}")
print(f"DCA平均取得単価: {avg_price_dca:.2f}")
print(f"単価削減率: {(1 - avg_price_dva/avg_price_dca)*100:.2f}%")