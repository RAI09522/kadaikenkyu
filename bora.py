import yfinance as yf
import numpy as np
import pandas as pd

def calculate_volatility(ticker_symbol, start_date, end_date):
    """
    指定した銘柄と期間のヒストリカル・ボラティリティ（年率）を計算する関数
    """
    print(f"[{ticker_symbol}] のデータを取得中 ({start_date} 〜 {end_date})...")
    
    try:
        # 株価データを取得
        stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if stock_data.empty:
            print("データが取得できませんでした。ティッカーシンボルや日付を確認してください。")
            return None

        # 日次の終値を取得
        close_prices = stock_data['Close']
        
        # 日次収益率（前日比のパーセンテージ変化）を計算
        daily_returns = close_prices.pct_change().dropna()
        
        # 日次収益率の標準偏差を計算（日次のボラティリティ）
        daily_volatility = daily_returns.std()
        
        # 年率換算のボラティリティを計算 (1年の営業日を約252日とする)
        # 算出式: 日次ボラティリティ × √252
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # 結果をパーセンテージ表記に変換
        annualized_volatility_pct = annualized_volatility * 100
        
        # float型として抽出 (yfinanceのバージョンによるSeries/DataFrameの仕様差異に対応)
        if isinstance(annualized_volatility_pct, pd.Series):
            annualized_volatility_pct = annualized_volatility_pct.iloc[0]
            
        print("-" * 30)
        print(f"銘柄: {ticker_symbol}")
        print(f"期間: {start_date} 〜 {end_date}")
        print(f"年率ボラティリティ: {annualized_volatility_pct:.2f}%")
        print("-" * 30)
        
        return annualized_volatility_pct

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

# ===== 実行例 =====
if __name__ == "__main__":
    # 例: トヨタ自動車 (日本の銘柄は末尾に '.T' をつける)
    ticker = "3349.T" 
    
    # 別の例: Apple (米国株の場合)
    # ticker = "AAPL"
    
    # 期間を指定 (YYYY-MM-DD形式)
    start = "2016-01-01"
    end = "2025-12-31"
    
    calculate_volatility(ticker, start, end)