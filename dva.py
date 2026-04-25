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


def get_market_config(ticker, total_budget_us=60000, total_budget_jp=6000000):
    if ticker.endswith(".T"):
        return {
            "market_name": "JP",
            "benchmark": "^N225",
            "total_budget": total_budget_jp,
            "currency_symbol": "¥",
            "currency_code": "JPY",
            "lot_size": 100,   # 日本株は100株単元
        }
    else:
        return {
            "market_name": "US",
            "benchmark": "^GSPC",
            "total_budget": total_budget_us,
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
    contribution = 0.0

    allocation_series = allocation_series.reindex(df.index).fillna(0.0)

    for dt in df.index:
        price = df.at[dt, "Close"]
        add_cash = float(allocation_series.at[dt])

        cash += add_cash
        contribution += add_cash

        buy_shares, cost, cash = execute_integer_buy(cash, price, lot_size=lot_size)

        shares += buy_shares
        spent += cost

    return spent, shares, cash, contribution


def get_month_groups(df):
    return list(df.groupby(df.index.to_period("M")))


def calc_monthly_budget_fixed_total(df, total_budget):
    month_groups = get_month_groups(df)
    n_months = len(month_groups)

    if n_months == 0:
        return 0.0, 0

    monthly_budget = total_budget / n_months
    return monthly_budget, n_months


def build_daily_allocation_fixed_total(df, total_budget):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        n_days = len(m_df)
        if n_days == 0:
            continue
        alloc.loc[m_df.index] = monthly_budget / n_days

    return alloc


def build_weekday_allocation_fixed_total(df, total_budget, weekday):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        idx = m_df.index[m_df.index.dayofweek == weekday]

        # その月に対象曜日の取引日が無ければ、月初営業日に全額入れる
        if len(idx) == 0:
            alloc.loc[m_df.index[0]] += monthly_budget
        else:
            alloc.loc[idx] = monthly_budget / len(idx)

    return alloc


def build_monthly_first_allocation_fixed_total(df, total_budget):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        alloc.loc[m_df.index[0]] = monthly_budget

    return alloc


def build_monthly_last_allocation_fixed_total(df, total_budget):
    alloc = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        alloc.loc[m_df.index[-1]] = monthly_budget

    return alloc


def build_dva_allocation_fixed_total(df, total_budget):
    """
    DVAの「重み」は維持しつつ、各月の合計拠出額を monthly_budget に固定する。
    つまり:
        1) 各日について raw_amt（生のDVA強度）を作る
        2) 月内で正規化して合計を月予算に一致させる
    """
    alloc = pd.Series(0.0, index=df.index, dtype=float)

    monthly_budget, n_months = calc_monthly_budget_fixed_total(df, total_budget)
    if n_months == 0:
        return alloc

    # DVAの進捗用に、理論上の「重み計算用保有株数」を小数株で追跡
    # これは配分の滑らかさのための内部変数であり、実際の約定株数ではない
    virtual_shares = 0.0
    cumulative_target_budget = 0.0

    for _, m_df in get_month_groups(df):
        n_days = len(m_df)
        if n_days == 0:
            continue

        daily_base = monthly_budget / n_days
        raw_amts = []

        temp_virtual_shares = virtual_shares

        for j in range(n_days):
            row = m_df.iloc[j]
            price = float(row["Close"])
            ema = row["ema20"]
            bb_l = row["bb_low"]
            vol = row["vol"]
            mkt_vol = row["mkt_vol"]

            if pd.isna(price) or price <= 0:
                raw_amt = 0.0
            elif pd.isna(ema) or pd.isna(bb_l):
                raw_amt = daily_base
            else:
                # 月予算固定版では、理想株数の分子も「実際の累計予算目標」に合わせる
                target_budget_t = cumulative_target_budget + daily_base * (j + 1)
                ideal_s = target_budget_t / ema if ema > 0 else 0.0

                if ideal_s > 0:
                    shortage_ratio = (ideal_s - temp_virtual_shares) / ideal_s
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

                raw_amt = daily_base * f_t * psi_t * phi_t * t_t
                raw_amt = max(raw_amt, daily_base * 0.1)

            raw_amts.append(raw_amt)

            # 仮の株数更新（重み作成用）
            if price > 0:
                temp_virtual_shares += raw_amt / price

        raw_amts = np.array(raw_amts, dtype=float)
        raw_sum = raw_amts.sum()

        if raw_sum <= 0:
            normalized = np.full(n_days, monthly_budget / n_days)
        else:
            normalized = monthly_budget * (raw_amts / raw_sum)

        alloc.loc[m_df.index] = normalized

        # 次月用の理論株数を更新
        for j in range(n_days):
            price = float(m_df["Close"].iloc[j])
            if price > 0:
                virtual_shares += normalized[j] / price

        cumulative_target_budget += monthly_budget

    return alloc


def run_share_count_experiment_fixed_budget(
    tickers,
    start_date="2014-01-01",
    end_date="2023-12-31",   # 10年固定にしたいのでデフォルトを明示
    total_budget_us=60000,
    total_budget_jp=6000000,
):
    print(f"--- 累積取得株数比較（総予算固定版・整数株のみ・繰越現金あり）({start_date} ～ {end_date}) ---")
    print("DEBUG: FIXED TOTAL BUDGET VERSION")

    benchmark_vol_cache = {}

    for ticker in tickers:
        cfg = get_market_config(
            ticker,
            total_budget_us=total_budget_us,
            total_budget_jp=total_budget_jp
        )

        total_budget = cfg["total_budget"]
        benchmark = cfg["benchmark"]
        cur = cfg["currency_symbol"]
        lot_size = cfg["lot_size"]

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']}】")

        if benchmark not in benchmark_vol_cache:
            bmk = yf.download(
                benchmark,
                start=start_date,
                end=end_date,
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
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        df = normalize_df(df)

        if df.empty:
            print("  データ取得失敗 or データなし")
            continue

        df = calc_indicators(df, mkt_vol_raw)

        monthly_budget, n_months = calc_monthly_budget_fixed_total(df, total_budget)

        print(f"  対象月数: {n_months} か月")
        print(f"  総予算: {cur}{total_budget:,.0f}")
        print(f"  月あたり予算(自動計算): {cur}{monthly_budget:,.2f}")

        methods_data = []

        # 1. 提案DVA（総予算固定）
        dva_alloc = build_dva_allocation_fixed_total(df, total_budget)
        dva_spent, dva_shares, dva_cash, dva_contrib = simulate_integer_strategy(
            df, dva_alloc, lot_size=lot_size
        )
        methods_data.append(("★提案DVA", dva_spent, dva_shares, dva_cash, dva_contrib))

        # 2. 毎日積立（総予算固定）
        daily_alloc = build_daily_allocation_fixed_total(df, total_budget)
        daily_spent, daily_shares, daily_cash, daily_contrib = simulate_integer_strategy(
            df, daily_alloc, lot_size=lot_size
        )
        methods_data.append(("毎日積立", daily_spent, daily_shares, daily_cash, daily_contrib))

        # 3. 曜日別積立（総予算固定）
        day_names = ["月曜積立", "火曜積立", "水曜積立", "木曜積立", "金曜積立"]
        for d in range(5):
            weekday_alloc = build_weekday_allocation_fixed_total(df, total_budget, d)
            spent, shares, cash, contrib = simulate_integer_strategy(
                df, weekday_alloc, lot_size=lot_size
            )
            methods_data.append((day_names[d], spent, shares, cash, contrib))

        # 4. 毎月（月初・月末）
        first_alloc = build_monthly_first_allocation_fixed_total(df, total_budget)
        spent_first, shares_first, cash_first, contrib_first = simulate_integer_strategy(
            df, first_alloc, lot_size=lot_size
        )
        methods_data.append(("毎月(月初)", spent_first, shares_first, cash_first, contrib_first))

        last_alloc = build_monthly_last_allocation_fixed_total(df, total_budget)
        spent_last, shares_last, cash_last, contrib_last = simulate_integer_strategy(
            df, last_alloc, lot_size=lot_size
        )
        methods_data.append(("毎月(月末)", spent_last, shares_last, cash_last, contrib_last))

        results = sorted(methods_data, key=lambda x: x[2], reverse=True)

        print("-" * 122)
        print(f"{'順位':<2} | {'手法名':<10} | {'取得株数':<10} | {'平均単価':<14} | {'約定総額':<14} | {'繰越現金':<14} | {'拠出総額'}")
        print("-" * 122)

        for rank, (name, spent, shares, cash, contrib) in enumerate(results, start=1):
            avg_p = spent / shares if shares > 0 else np.nan
            avg_p_str = f"{cur}{avg_p:,.2f}" if pd.notna(avg_p) else "-"
            spent_str = f"{cur}{spent:,.0f}"
            cash_str = f"{cur}{cash:,.0f}"
            contrib_str = f"{cur}{contrib:,.0f}"
            shares_str = f"{shares:,d} 株"

            print(
                f"{rank:>2} | {name:<10} | {shares_str:>10} | "
                f"{avg_p_str:>14} | {spent_str:>14} | {cash_str:>14} | {contrib_str}"
            )

        print("-" * 122)
        print(f"  ※ 原則として「約定総額 + 繰越現金 = 拠出総額」となる")
        print(f"  ※ この版では、全手法の拠出総額が同一になるよう固定している")


if __name__ == "__main__":
    run_share_count_experiment_fixed_budget(
        ["NVDA", "AAPL", "TSLA", "8088.T", "1443.T"],
        start_date="2014-01-01",
        end_date="2023-12-31",   # 10年間
        total_budget_us=60000,   # 500 USD × 12 × 10年
        total_budget_jp=6000000  # 50,000 JPY × 12 × 10年
    )
