import yfinance as yf
import pandas as pd
import numpy as np

def run_performance_test(tickers, start_date="2023-01-01"):
    base_amount = 100000 
    results_summary = []

    # 1. 市場平均（S&P 500）のボラティリティを計算
    print("市場データを取得中...")
    spy = yf.download("^GSPC", start=start_date, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    
    # 市場ボラ（20日移動標準偏差）
    mkt_vol = spy['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

    for ticker in tickers:
        print(f"{ticker} を分析中...")
        stock_obj = yf.Ticker(ticker)
        # history() から progress 引数を削除しました
        df = stock_obj.history(start=start_date)
        
        if df.empty:
            print(f"  -> {ticker} のデータが取得できませんでした。")
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 財務指標の取得
        try:
            info = stock_obj.info
            curr_pe = info.get('trailingPE', 35)
            curr_pbr = info.get('priceToBook', 15)
        except:
            curr_pe, curr_pbr = 35, 15

        # 指標計算
        df['prev_close'] = df['Close'].shift(1)
        df['sma200'] = df['Close'].rolling(window=200).mean()
        df['vol'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['mkt_vol'] = mkt_vol

        def calculate_dva(row):
            if pd.isna(row['sma200']) or pd.isna(row['mkt_vol']) or pd.isna(row['prev_close']):
                return 0
            
            price = float(row['Close'])
            sma = float(row['sma200'])
            prev = float(row['prev_close'])
            
            # 生存・損切り・安全装置
            # トレンド維持(SMA200の70%上) かつ パニック拒否(前日比-20%以内)
            signal = 1 if (price > sma * 0.7) and (price > prev * 0.8) else 0
            # 200日線付近での生存判定
            omega = 1 if price > (sma * 0.8) else 0
            # 安全装置（市場ボラとの乖離）
            phi = max(0, 1 - (float(row['vol']) - (float(row['mkt_vol']) * 1.5)) / 10)
            
            # 動的ラムダ
            dyn_lambda = 2.0 * (35 / curr_pe) * (1 + (15 / curr_pbr))
            
            # 加速装置
            current_idx = df.index.get_loc(row.name)
            p_avg = df['Close'].iloc[:current_idx].mean() if current_idx > 0 else price
            gap = (p_avg - price) / p_avg if p_avg > 0 else 0
            psi = np.exp(dyn_lambda * gap)
            
            return base_amount * phi * psi * omega * signal

        # 投資シミュレーション実行
        df['DVA_Amount'] = df.apply(calculate_dva, axis=1)
        df['DCA_Amount'] = base_amount

        # パフォーマンス統計
        dva_total = df['DVA_Amount'].sum()
        dca_total = df['DCA_Amount'].sum()
        
        # 取得株数 (投資額 / その日の価格)
        dva_shares = (df['DVA_Amount'] / df['Close']).sum()
        dca_shares = (df['DCA_Amount'] / df['Close']).sum()
        
        dva_avg = dva_total / dva_shares if dva_shares > 0 else 0
        dca_avg = dca_total / dca_shares if dca_shares > 0 else 0
        reduction = (1 - dva_avg / dca_avg) * 100 if dca_avg > 0 else 0

        results_summary.append({
            "銘柄": ticker,
            "DVA単価": round(dva_avg, 2),
            "DCA単価": round(dca_avg, 2),
            "削減率": round(reduction, 2)
        })

    # 最終結果表示
    print("\n" + "="*55)
    print(f"{'Ticker':<8} | {'DVA Avg($)':<12} | {'DCA Avg($)':<12} | {'削減率(%)':<10}")
    print("-" * 55)
    for res in results_summary:
        print(f"{res['銘柄']:<8} | ${res['DVA単価']:<11} | ${res['DCA単価']:<11} | {res['削減率']}%")
    print("="*55)

# 分析実行
run_performance_test(["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"])