import yfinance as yf
import pandas as pd
import numpy as np


def normalize_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    df = df.sort_index()
    return df


def get_market_config(ticker, monthly_budget_us=500, monthly_budget_jp=50000):
    """
    ticker の市場設定を返す
    .T は日本株として扱う
    """
    if ticker.endswith(".T"):
        return {
            "market_name": "JP",
            "benchmark": "^N225",   # 日本株の市場ボラ用
            "monthly_budget": monthly_budget_jp,
            "currency_symbol": "¥",
            "currency_code": "JPY",
        }
    else:
        return {
            "market_name": "US",
            "benchmark": "^GSPC",   # 米国株の市場ボラ用
            "monthly_budget": monthly_budget_us,
            "currency_symbol": "$",
            "currency_code": "USD",
        }


def calc_dva_result(df, daily_base):
    """
    提案DVAの累積投資額・累積取得株数を返す
    """
    dva_shares, dva_spent = 0.0, 0.0

    for i in range(len(df)):
        price = float(df["Close"].iloc[i])
        ema = df["ema20"].iloc[i]
        bb_l = df["bb_low"].iloc[i]
        vol = df["vol"].iloc[i]
        mkt_vol = df["mkt_vol"].iloc[i]

        if pd.isna(price) or price <= 0:
            continue

        if pd.isna(ema) or pd.isna(bb_l):
            amt = daily_base
        else:
            # 1) 理想保有株数
            ideal_s = (daily_base * (i + 1)) / ema if ema > 0 else 0.0

            # 2) 進捗補正
            if ideal_s > 0:
                shortage_ratio = (ideal_s - dva_shares) / ideal_s
                f_t = 1.0 + max(0.0, min(1.0, shortage_ratio * 2.0))
            else:
                f_t = 1.0

            # 3) 割安度補正
            mispricing = (ema - price) / ema if ema > 0 else 0.0
            psi_t = np.exp(6.0 * mispricing) * (2.5 if price < bb_l else 1.0)

            # 4) ボラ抑制
            if pd.isna(vol) or pd.isna(mkt_vol):
                phi_t = 1.0
            else:
                phi_t = max(0.5, 1.0 - (vol - (mkt_vol * 1.2)) / 5.0)

            # 5) トレンド補正
            t_t = 1.15 if price > ema else 0.85

            # 6) 最終投資額
            amt = daily_base * f_t * psi_t * phi_t * t_t

        # 最低投資額
        amt = max(amt, daily_base * 0.1)

        dva_spent += amt
        dva_shares += amt / price

    return dva_spent, dva_shares


def run_share_count_experiment(
    tickers,
    start_date="2014-01-01",
    monthly_budget_us=500,
    monthly_budget_jp=50000,
):
    print(f"--- 累積取得株数比較 ({start_date}～) ---")

    # 市場ボラのキャッシュ
    benchmark_vol_cache = {}

    for ticker in tickers:
        cfg = get_market_config(
            ticker,
            monthly_budget_us=monthly_budget_us,
            monthly_budget_jp=monthly_budget_jp
        )

        monthly_budget = cfg["monthly_budget"]
        daily_base = monthly_budget / 20
        weekly_base = monthly_budget / 4
        benchmark = cfg["benchmark"]
        cur = cfg["currency_symbol"]

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']}】")

        # ベンチマークの市場ボラ
        if benchmark not in benchmark_vol_cache:
            bmk = yf.download(
                benchmark,
                start=start_date,
                auto_adjust=True,
                progress=False
            )
            bmk = normalize_df(bmk)
            if bmk.empty:
                print(f"  ベンチマーク取得失敗: {benchmark}")
                continue

            benchmark_vol_cache[benchmark] = (
                bmk["Close"].pct_change().rolling(window=20).std() * np.sqrt(252)
            )

        mkt_vol_raw = benchmark_vol_cache[benchmark]

        # 個別株データ
        df = yf.download(
            ticker,
            start=start_date,
            auto_adjust=True,
            progress=False
        )
        df = normalize_df(df)
        if df.empty:
            print("  データ取得失敗 or データなし")
            continue

        # 指標計算
        df["mkt_vol"] = mkt_vol_raw.reindex(df.index).ffill().fillna(mkt_vol_raw.mean())
        df["vol"] = df["Close"].pct_change().rolling(window=20).std() * np.sqrt(252)
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["bb_low"] = df["ema20"] - (df["Close"].rolling(window=20).std() * 2)

        methods_data = []

        # 1. 提案DVA
        dva_spent, dva_shares = calc_dva_result(df, daily_base)
        methods_data.append(("★提案DVA", dva_spent, dva_shares))

        # 2. 毎日積立
        daily_spent = daily_base * len(df)
        daily_shares = (daily_base / df["Close"]).sum()
        methods_data.append(("毎日積立", daily_spent, daily_shares))

        # 3. 曜日別積立
        day_names = ["月曜積立", "火曜積立", "水曜積立", "木曜積立", "金曜積立"]
        for d in range(5):
            w_df = df[df.index.dayofweek == d]
            if len(w_df) == 0:
                methods_data.append((day_names[d], 0.0, 0.0))
                continue
            spent = weekly_base * len(w_df)
            shares = (weekly_base / w_df["Close"]).sum()
            methods_data.append((day_names[d], spent, shares))

        # 4. 毎月（月初・月末）
        m_first = df.resample("MS").first().dropna(subset=["Close"])
        m_last = df.resample("ME").last().dropna(subset=["Close"])

        methods_data.append((
            "毎月(月初)",
            monthly_budget * len(m_first),
            (monthly_budget / m_first["Close"]).sum() if len(m_first) else 0.0
        ))
        methods_data.append((
            "毎月(月末)",
            monthly_budget * len(m_last),
            (monthly_budget / m_last["Close"]).sum() if len(m_last) else 0.0
        ))

        # ランキング
        results = sorted(methods_data, key=lambda x: x[2], reverse=True)

        print("-" * 78)
        print(f"{'順位':<2} | {'手法名':<10} | {'取得株数':<14} | {'平均単価':<14} | {'総投資額'}")
        print("-" * 78)

        for i, (name, spent, shares) in enumerate(results, start=1):
            avg_p = spent / shares if shares > 0 else np.nan
            avg_p_str = f"{cur}{avg_p:,.2f}" if pd.notna(avg_p) else "-"
            spent_str = f"{cur}{spent:,.0f}"
            shares_str = f"{shares:,.4f} 株"
            print(f"{i:>2} | {name:<10} | {shares_str:>14} | {avg_p_str:>14} | {spent_str}")

        print("-" * 78)


if __name__ == "__main__":
    run_share_count_experiment(
        ["NVDA", "AAPL", "TSLA", "8088.T"],
        start_date="2014-01-01",
        monthly_budget_us=500,      # 米国株は 500 USD / 月
        monthly_budget_jp=50000     # 日本株は 50,000 JPY / 月
    )
