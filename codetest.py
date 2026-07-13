import yfinance as yf
import pandas as pd
import numpy as np


def normalize_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)

    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    df = df.sort_index()
    return df


def get_market_config(ticker, monthly_budget_us=500, monthly_budget_jp=50000):
    if ticker.endswith(".T"):
        return {
            "market_name": "JP",
            "benchmark": "^N225",
            "monthly_budget": monthly_budget_jp,
            "currency_symbol": "¥",
            "currency_code": "JPY",
            "lot_size": 100,   # 必要なら 100 に変更可能
        }
    else:
        return {
            "market_name": "US",
            "benchmark": "^GSPC",
            "monthly_budget": monthly_budget_us,
            "currency_symbol": "$",
            "currency_code": "USD",
            "lot_size": 1,
        }


def calc_annualized_vol(close_series, window=20):
    returns = close_series.pct_change(fill_method=None)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calc_indicators(df, mkt_vol_raw):
    out = df.copy()
    out["mkt_vol"] = mkt_vol_raw.reindex(out.index).ffill().fillna(mkt_vol_raw.mean())
    out["vol"] = calc_annualized_vol(out["Close"], window=20)
    out["ema20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["bb_low"] = out["ema20"] - (out["Close"].rolling(window=20).std() * 2)
    return out


def execute_integer_buy(cash, price, lot_size=1):
    if pd.isna(price) or price <= 0:
        return 0, 0.0, cash

    max_shares = int(np.floor(cash / price + 1e-12))
    buy_shares = (max_shares // lot_size) * lot_size

    if buy_shares <= 0:
        return 0, 0.0, cash

    cost = buy_shares * float(price)
    remaining_cash = cash - cost
    return buy_shares, cost, remaining_cash


def simulate_integer_strategy(df, allocation_series, lot_size=1):
    shares = 0
    spent = 0.0
    cash = 0.0

    allocation_series = allocation_series.reindex(df.index).fillna(0.0)

    for dt in df.index:
        price = df.at[dt, "Close"]
        cash += float(allocation_series.at[dt])

        buy_shares, cost, cash = execute_integer_buy(cash, price, lot_size=lot_size)

        shares += buy_shares
        spent += cost

    return spent, shares, cash


def build_daily_allocation(df, daily_base):
    return pd.Series(daily_base, index=df.index, dtype=float)


def build_weekday_allocation(df, weekly_base, weekday):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    target_idx = df.index[df.index.dayofweek == weekday]
    alloc.loc[target_idx] = weekly_base
    return alloc


def build_monthly_first_allocation(df, monthly_budget):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    first_idx = df.groupby(df.index.to_period("M")).head(1).index
    alloc.loc[first_idx] = monthly_budget
    return alloc


def build_monthly_last_allocation(df, monthly_budget):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    last_idx = df.groupby(df.index.to_period("M")).tail(1).index
    alloc.loc[last_idx] = monthly_budget
    return alloc


def calc_dva_result_integer(df, daily_base, lot_size=1):
    shares = 0
    spent = 0.0
    cash = 0.0

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
            ideal_s = (daily_base * (i + 1)) / ema if ema > 0 else 0.0

            if ideal_s > 0:
                shortage_ratio = (ideal_s - shares) / ideal_s
                f_t = 1.0 + max(0.0, min(1.0, shortage_ratio * 2.0))
            else:
                f_t = 1.0

            mispricing = (ema - price) / ema if ema > 0 else 0.0
            psi_t = np.exp(6.0 * mispricing) * (2.5 if price < bb_l else 1.0)

            if pd.isna(vol) or pd.isna(mkt_vol):
                phi_t = 1.0
            else:
                phi_t = max(0.5, 1.0 - (vol - (mkt_vol * 1.2)) / 5.0)

            t_t = 1.15 if price > ema else 0.85

            amt = daily_base * f_t * psi_t * phi_t * t_t

        amt = max(amt, daily_base * 0.1)

        cash += amt
        buy_shares, cost, cash = execute_integer_buy(cash, price, lot_size=lot_size)

        shares += buy_shares
        spent += cost

    return spent, shares, cash


def run_share_count_experiment(
    tickers,
    start_date="2014-01-01",
    monthly_budget_us=500,
    monthly_budget_jp=50000,
):
    print(f"--- 累積取得株数比較（整数株のみ・繰越現金あり）({start_date}～) ---")
    print("DEBUG: INTEGER SHARE VERSION")

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
        lot_size = cfg["lot_size"]

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']}】")

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

            benchmark_vol_cache[benchmark] = calc_annualized_vol(bmk["Close"], window=20)

        mkt_vol_raw = benchmark_vol_cache[benchmark]

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

        df = calc_indicators(df, mkt_vol_raw)

        methods_data = []

        dva_spent, dva_shares, dva_cash = calc_dva_result_integer(df, daily_base, lot_size=lot_size)
        methods_data.append(("★提案DVA", dva_spent, dva_shares, dva_cash))

        daily_alloc = build_daily_allocation(df, daily_base)
        daily_spent, daily_shares, daily_cash = simulate_integer_strategy(df, daily_alloc, lot_size=lot_size)
        methods_data.append(("毎日積立", daily_spent, daily_shares, daily_cash))

        day_names = ["月曜積立", "火曜積立", "水曜積立", "木曜積立", "金曜積立"]
        for d in range(5):
            weekday_alloc = build_weekday_allocation(df, weekly_base, d)
            spent, shares, cash = simulate_integer_strategy(df, weekday_alloc, lot_size=lot_size)
            methods_data.append((day_names[d], spent, shares, cash))

        first_alloc = build_monthly_first_allocation(df, monthly_budget)
        spent_first, shares_first, cash_first = simulate_integer_strategy(df, first_alloc, lot_size=lot_size)
        methods_data.append(("毎月(月初)", spent_first, shares_first, cash_first))

        last_alloc = build_monthly_last_allocation(df, monthly_budget)
        spent_last, shares_last, cash_last = simulate_integer_strategy(df, last_alloc, lot_size=lot_size)
        methods_data.append(("毎月(月末)", spent_last, shares_last, cash_last))

        results = sorted(methods_data, key=lambda x: x[2], reverse=True)

        print("-" * 100)
        print(f"{'順位':<2} | {'手法名':<10} | {'取得株数':<10} | {'平均単価':<14} | {'約定総額':<14} | {'繰越現金'}")
        print("-" * 100)

        for rank, (name, spent, shares, cash) in enumerate(results, start=1):
            avg_p = spent / shares if shares > 0 else np.nan
            avg_p_str = f"{cur}{avg_p:,.2f}" if pd.notna(avg_p) else "-"
            spent_str = f"{cur}{spent:,.0f}"
            cash_str = f"{cur}{cash:,.0f}"
            shares_str = f"{shares:,d} 株"

            print(f"{rank:>2} | {name:<10} | {shares_str:>10} | {avg_p_str:>14} | {spent_str:>14} | {cash_str}")

        print("-" * 100)


if __name__ == "__main__":
    run_share_count_experiment(
        ["1301.T", "1332.T", "1377.T", "1379.T", "1375.T", "1376.T", "1380.T", "1381.T", "1382.T", "1383.T", "1605.T", "1662.T", "1663.T", "1515.T", "1518.T", "1514.T", "1801.T", "1802.T", "1803.T", "1812.T", "1808.T", "1810.T", "1811.T", "1804.T", "1914.T", "1716.T", "1430.T", "1431.T", "1436.T", "1439.T", "145A.T", "2001.T", "2002.T", "2004.T", "2201.T", "2206.T", "2003.T", "2009.T", "2204.T", "2208.T", "2215.T", "2586.T", "2934.T", "2936.T", "2937.T", "2938.T", "3101.T", "3104.T", "3106.T", "3107.T", "3109.T", "3103.T", "3111.T", "3123.T", "3202.T", "3204.T", "278A.T", "3861.T", "3863.T", "3941.T", "3880.T", "3865.T", "3708.T", "3892.T", "3896.T", "3953.T", "4004.T", "3405.T", "3407.T", "4005.T", "4021.T", "4119.T", "277A.T", "247A.T", "2930.T", "4502.T", "4503.T", "4507.T", "4519.T", "4523.T", "197A.T", "130A.T", "2160.T", "219A.T", "206A.T", "149A.T", "5019.T", "5020.T", "5021.T", "5017.T", "3315.T", "5011.T", "5015.T", "5108.T", "5110.T", "5101.T", "5105.T", "7282.T", "5189.T", "5199.T", "5103.T", "5201.T", "5202.T", "5204.T", "5332.T", "5333.T", "5337.T", "3157.T", "5401.T", "5411.T", "5406.T", "5444.T", "5463.T", "5408.T", "5451.T", "5449.T", "5713.T", "5714.T", "5706.T", "5711.T", "5741.T", "1491.T", "5938.T", "5943.T", "5929.T", "5947.T", "7270.T", "2961.T", "2962.T", "296A.T", "6326.T", "6367.T", "6301.T", "6302.T", "6305.T", "1909.T", "6501.T", "6503.T", "6504.T", "6506.T", "6752.T", "241A.T", "7203.T", "7201.T", "7267.T", "7261.T", "7269.T", "4543.T", "7731.T", "7733.T", "7741.T", "7751.T", "7705.T", "218A.T", "7951.T", "7974.T", "8001.T", "8015.T", "8058.T", "168A.T", "2121.T", "2307.T", "2317.T", "2326.T", "2327.T", "2332.T", "2323.T", "2329.T", "2330.T", "2339.T", "147A.T", "137A.T", "135A.T", "148A.T", "8031.T", "8002.T", "2667.T", "2668.T", "2689.T", "2693.T", "2700.T", "3070.T", "3140.T", "280A.T", "2670.T", "2678.T", "2681.T", "2685.T", "2695.T", "2652.T", "2653.T", "2654.T", "2656.T", "2666.T", "138A.T", "141A.T", "154A.T", "245A.T", "3185.T", "8306.T", "8316.T", "8411.T", "7182.T", "8308.T", "8601.T", "8604.T", "8628.T", "8698.T", "8473.T", "254A.T", "3113.T", "8795.T", "8750.T", "8766.T", "8725.T", "8630.T", "8591.T", "8593.T", "8253.T", "8515.T", "8572.T", "196A.T", "2388.T", "8801.T", "8802.T", "8804.T", "8830.T", "3289.T", "1435.T", "146A.T", "160A.T", "3208.T", "2981.T", "2983.T", "2986.T", "2997.T", "2120.T", "2124.T", "2127.T", "212A.T", "2130.T", "2122.T", "2134.T", "2136.T", "2139.T", "2152.T", "142A.T", "143A.T", "155A.T", "156A.T", "157A.T", "9501.T", "9503.T", "9502.T", "9506.T", "9508.T", "9020.T", "9021.T", "9022.T", "9042.T", "9041.T", "9101.T", "9104.T", "9107.T", "9110.T", "9119.T", "9202.T", "9201.T", "9301.T", "9302.T", "9303.T", "9364.T", "9304.T"],
        start_date="2014-01-01",
        monthly_budget_us=500,
        monthly_budget_jp=50000
    )
