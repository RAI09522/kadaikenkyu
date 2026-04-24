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
        ["NVDA", "AAPL", "TSLA", "8088.T", "1443.T"],
        start_date="2014-01-01",
        monthly_budget_us=500,
        monthly_budget_jp=50000
    )
