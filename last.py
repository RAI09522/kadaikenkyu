import re
import sys

# ================================
# 依存ライブラリの事前チェック
# ================================
MISSING = []

try:
    import yfinance as yf
except ModuleNotFoundError:
    MISSING.append("yfinance")

try:
    import pandas as pd
except ModuleNotFoundError:
    MISSING.append("pandas")

try:
    import numpy as np
except ModuleNotFoundError:
    MISSING.append("numpy")

if MISSING:
    names = ", ".join(MISSING)
    print("必要ライブラリが不足しています:", names)
    print("次を実行してください:")
    print("  py -m pip install yfinance pandas numpy")
    sys.exit(1)


# ================================
# 設定
# ================================
CONFIG = {
    "start_date": "2018-01-01",
    "end_date": "2023-12-31",

    # 総予算固定
    "total_budget_us": 60000,      # 例: 500 USD × 12 × 10年
    "total_budget_jp": 6000000,    # 例: 50,000 JPY × 12 × 10年

    "tickers": ["NVDA", "AAPL", "TSLA", "8088.T", "1443.T"],

    # ハイブリッド比率
    # dca_ratio = 0.60 -> 60% DCA / 40% DVA
    "default_dca_ratio": 0.60,

    # DVA側の購入基準
    "score_threshold": 0.30,

    # 執行ロット合成係数
    # q_exec = λ*q_score + (1-λ)*q_cash
    "execution_mix_lambda": 0.45,

    # DVA側スコア重み
    # B案なので trend を強めにしている
    "score_weights": {
        "shortage": 0.18,
        "value": 0.10,
        "bb": 0.05,
        "trend": 0.45,
        "cash_pressure": 0.17,
        "risk": 0.05,
    },
}

# ================================
# 単元株の個別上書き
# 優先順位:
# 1. この辞書
# 2. 日本株(.T)は100株
# 3. それ以外は1株
# ================================
LOT_SIZE_OVERRIDES = {
    "8088.T": 100,
    "1443.T": 100,
    # 例:
    # "1306.T": 10,
    # "1343.T": 10,
}

# ================================
# 銘柄ごとの DCA 比率上書き
# 優先順位:
# 1. この辞書
# 2. CONFIG["default_dca_ratio"]
# ================================
HYBRID_RATIO_OVERRIDES = {
    # 例:
    # "8088.T": 0.70,   # 70% DCA / 30% DVA
    # "NVDA": 0.50,     # 50% DCA / 50% DVA
}


# ================================
# 基本ユーティリティ
# ================================
def clip(x, low, high):
    return max(low, min(high, x))


def validate_ratio(x, name="ratio"):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise ValueError(f"{name} は 0.0〜1.0 の範囲で指定してください: {x}")
    return x


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


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

    return df.sort_index()


def get_lot_size(ticker: str) -> int:
    if ticker in LOT_SIZE_OVERRIDES:
        return int(LOT_SIZE_OVERRIDES[ticker])

    if ticker.endswith(".T"):
        return 100

    return 1


def get_dca_ratio(ticker: str) -> float:
    if ticker in HYBRID_RATIO_OVERRIDES:
        return validate_ratio(HYBRID_RATIO_OVERRIDES[ticker], f"{ticker} dca_ratio")
    return validate_ratio(CONFIG["default_dca_ratio"], "default_dca_ratio")


def get_market_config(ticker, total_budget_us=60000, total_budget_jp=6000000):
    lot_size = get_lot_size(ticker)

    if ticker.endswith(".T"):
        return {
            "market_name": "JP",
            "benchmark": "^N225",
            "total_budget": total_budget_jp,
            "currency_symbol": "¥",
            "currency_code": "JPY",
            "lot_size": lot_size,
            "max_lots_per_trade": 5,  # DVA側の追加ロット上限
        }

    return {
        "market_name": "US",
        "benchmark": "^GSPC",
        "total_budget": total_budget_us,
        "currency_symbol": "$",
        "currency_code": "USD",
        "lot_size": lot_size,
        "max_lots_per_trade": 5,
    }


def calc_annualized_vol(close_series, window=20):
    returns = close_series.pct_change(fill_method=None)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calc_indicators(df, mkt_vol_raw):
    out = df.copy()
    out["mkt_vol"] = mkt_vol_raw.reindex(out.index).ffill().fillna(mkt_vol_raw.mean())
    out["vol"] = calc_annualized_vol(out["Close"], window=20)

    out["ema20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["ema60"] = out["Close"].ewm(span=60, adjust=False).mean()

    out["bb_low"] = out["ema20"] - (out["Close"].rolling(window=20).std() * 2)

    out["ema20_lag5"] = out["ema20"].shift(5)
    return out


def get_month_groups(df):
    return list(df.groupby(df.index.to_period("M")))


def calc_monthly_budget_fixed_total(df, total_budget):
    month_groups = get_month_groups(df)
    n_months = len(month_groups)
    if n_months == 0:
        return 0.0, 0
    return total_budget / n_months, n_months


def build_monthly_contribution_schedule(df, total_budget):
    """
    毎月最初の営業日に、その月の予算を入金する
    """
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        if len(m_df) > 0:
            schedule.loc[m_df.index[0]] = monthly_budget

    return schedule


# ================================
# DVA側スコア
# B案なので trend を強く使う
# ================================
def calc_hybrid_dva_score_row(
    price,
    ema20,
    ema60,
    ema20_lag5,
    bb_low,
    vol,
    mkt_vol,
    cumulative_contribution,
    shares,
    cash_available,
    lot_size,
    weights,
):
    if pd.isna(price) or price <= 0 or pd.isna(ema20) or ema20 <= 0:
        parts = {
            "shortage": 0.0,
            "value": 0.0,
            "bb": 0.0,
            "trend": 0.0,
            "cash_pressure": 0.0,
            "risk": 0.0,
        }
        return 0.0, parts

    # 目標保有株数との不足
    ideal_shares = cumulative_contribution / ema20 if ema20 > 0 else 0.0
    if ideal_shares > 0:
        shortage = clip((ideal_shares - shares) / ideal_shares, 0.0, 1.0)
    else:
        shortage = 0.0

    # 軽い割安度
    value = clip((ema20 - price) / (0.25 * ema20), 0.0, 1.0)

    # BB下限割れ
    bb = 1.0 if (pd.notna(bb_low) and price < bb_low) else 0.0

    # トレンド成分
    trend_price = 1.0 if price > ema20 else 0.0
    trend_cross = 1.0 if (pd.notna(ema60) and ema20 > ema60) else 0.0

    if pd.notna(ema20_lag5) and ema20_lag5 > 0:
        trend_slope = clip((ema20 - ema20_lag5) / (0.05 * ema20_lag5), 0.0, 1.0)
    else:
        trend_slope = 0.0

    trend = 0.4 * trend_price + 0.4 * trend_cross + 0.2 * trend_slope
    trend = clip(trend, 0.0, 1.0)

    # 現金圧力
    lot_cost = price * lot_size if price > 0 else np.inf
    if np.isfinite(lot_cost) and lot_cost > 0:
        cash_pressure = clip(((cash_available / lot_cost) - 1.0) / 4.0, 0.0, 1.0)
    else:
        cash_pressure = 0.0

    # リスク減点
    if pd.isna(vol) or pd.isna(mkt_vol) or mkt_vol <= 0:
        risk = 0.0
    else:
        risk = clip((vol / (1.25 * mkt_vol)) - 1.0, 0.0, 1.0)

    score_raw = (
        weights["shortage"] * shortage
        + weights["value"] * value
        + weights["bb"] * bb
        + weights["trend"] * trend
        + weights["cash_pressure"] * cash_pressure
        - weights["risk"] * risk
    )

    score = clip(score_raw, 0.0, 1.0)

    parts = {
        "shortage": shortage,
        "value": value,
        "bb": bb,
        "trend": trend,
        "cash_pressure": cash_pressure,
        "risk": risk,
    }
    return score, parts


# ================================
# 純DCA
# ================================
def execute_monthly_first(df, contribution_schedule, lot_size=1):
    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        add_cash = float(contribution_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        if add_cash > 0 and price > 0:
            lot_cost = price * lot_size
            buy_lots = int(cash // lot_cost)
            buy_shares = buy_lots * lot_size
            cost = buy_shares * price

            if buy_shares > 0:
                shares += buy_shares
                spent += cost
                cash -= cost
                trade_count += 1

    return {
        "spent": spent,
        "shares": shares,
        "cash": cash,
        "contribution": contribution,
        "trade_count": trade_count,
    }


def execute_monthly_last(df, contribution_schedule, lot_size=1):
    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    for _, m_df in get_month_groups(df):
        for dt in m_df.index:
            add_cash = float(contribution_schedule.at[dt])
            cash += add_cash
            contribution += add_cash

        last_dt = m_df.index[-1]
        price = float(df.at[last_dt, "Close"])

        if price > 0:
            lot_cost = price * lot_size
            buy_lots = int(cash // lot_cost)
            buy_shares = buy_lots * lot_size
            cost = buy_shares * price

            if buy_shares > 0:
                shares += buy_shares
                spent += cost
                cash -= cost
                trade_count += 1

    return {
        "spent": spent,
        "shares": shares,
        "cash": cash,
        "contribution": contribution,
        "trade_count": trade_count,
    }


# ================================
# ハイブリッド戦略
# DCA部分: 月初に必ず買う
# DVA部分: 日次判定で追加ロットを執行
# ================================
def execute_hybrid_dca_dva(
    df,
    contribution_schedule,
    dca_ratio,
    lot_size,
    score_threshold,
    max_lots_per_trade,
    weights,
    execution_mix_lambda,
):
    dca_ratio = validate_ratio(dca_ratio, "dca_ratio")
    dva_ratio = 1.0 - dca_ratio

    shares = 0
    spent = 0.0

    core_cash = 0.0   # DCA側
    sat_cash = 0.0    # DVA側

    contribution = 0.0
    trade_count = 0
    dca_trade_count = 0
    dva_trade_count = 0

    trade_logs = []
    score_logs = []

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        ema20 = df.at[dt, "ema20"]
        ema60 = df.at[dt, "ema60"]
        ema20_lag5 = df.at[dt, "ema20_lag5"]
        bb_low = df.at[dt, "bb_low"]
        vol = df.at[dt, "vol"]
        mkt_vol = df.at[dt, "mkt_vol"]

        add_cash = float(contribution_schedule.at[dt])

        # 月初入金
        if add_cash > 0:
            add_cash_dca = add_cash * dca_ratio
            add_cash_dva = add_cash * dva_ratio

            core_cash += add_cash_dca
            sat_cash += add_cash_dva
            contribution += add_cash

            # DCA部分は月初に即執行
            if price > 0:
                lot_cost = price * lot_size
                buy_lots_dca = int(core_cash // lot_cost)
                buy_shares_dca = buy_lots_dca * lot_size
                cost_dca = buy_shares_dca * price

                if buy_shares_dca > 0:
                    shares += buy_shares_dca
                    spent += cost_dca
                    core_cash -= cost_dca
                    trade_count += 1
                    dca_trade_count += 1

                    trade_logs.append({
                        "date": dt,
                        "side": "DCA",
                        "price": price,
                        "score": np.nan,
                        "buy_lots": buy_lots_dca,
                        "buy_shares": buy_shares_dca,
                        "cost": cost_dca,
                        "core_cash_after": core_cash,
                        "sat_cash_after": sat_cash,
                        "shares_after": shares,
                    })

        # DVA部分の日次スコア
        score, parts = calc_hybrid_dva_score_row(
            price=price,
            ema20=ema20,
            ema60=ema60,
            ema20_lag5=ema20_lag5,
            bb_low=bb_low,
            vol=vol,
            mkt_vol=mkt_vol,
            cumulative_contribution=contribution,
            shares=shares,
            cash_available=sat_cash,
            lot_size=lot_size,
            weights=weights,
        )

        lot_cost = price * lot_size if price > 0 else np.inf
        affordable_lots_sat = int(sat_cash // lot_cost) if lot_cost > 0 else 0

        if affordable_lots_sat >= 1 and score >= score_threshold:
            q_score = clip((score - score_threshold) / (1.0 - score_threshold), 0.0, 1.0)
            q_cash = clip(affordable_lots_sat / max_lots_per_trade, 0.0, 1.0)
            q_exec = execution_mix_lambda * q_score + (1.0 - execution_mix_lambda) * q_cash

            wish_lots_sat = 1 + int(np.floor(q_exec * (max_lots_per_trade - 1)))
            buy_lots_sat = min(affordable_lots_sat, wish_lots_sat)
        else:
            q_score = 0.0
            q_cash = 0.0
            q_exec = 0.0
            wish_lots_sat = 0
            buy_lots_sat = 0

        buy_shares_sat = buy_lots_sat * lot_size
        cost_sat = buy_shares_sat * price

        score_logs.append({
            "date": dt,
            "price": price,
            "score": score,
            "q_score": q_score,
            "q_cash": q_cash,
            "q_exec": q_exec,
            "affordable_lots_sat": affordable_lots_sat,
            "wish_lots_sat": wish_lots_sat,
            "buy_lots_sat": buy_lots_sat,
            "buy_shares_sat": buy_shares_sat,
            "cost_sat": cost_sat,
            "core_cash": core_cash,
            "sat_cash": sat_cash,
            "shortage": parts["shortage"],
            "value": parts["value"],
            "bb": parts["bb"],
            "trend": parts["trend"],
            "cash_pressure": parts["cash_pressure"],
            "risk": parts["risk"],
        })

        if buy_shares_sat > 0:
            shares += buy_shares_sat
            spent += cost_sat
            sat_cash -= cost_sat
            trade_count += 1
            dva_trade_count += 1

            trade_logs.append({
                "date": dt,
                "side": "DVA",
                "price": price,
                "score": score,
                "buy_lots": buy_lots_sat,
                "buy_shares": buy_shares_sat,
                "cost": cost_sat,
                "core_cash_after": core_cash,
                "sat_cash_after": sat_cash,
                "shares_after": shares,
                "q_score": q_score,
                "q_cash": q_cash,
                "q_exec": q_exec,
                "shortage": parts["shortage"],
                "value": parts["value"],
                "bb": parts["bb"],
                "trend": parts["trend"],
                "cash_pressure": parts["cash_pressure"],
                "risk": parts["risk"],
            })

    total_cash = core_cash + sat_cash

    return {
        "spent": spent,
        "shares": shares,
        "cash": total_cash,
        "core_cash": core_cash,
        "sat_cash": sat_cash,
        "contribution": contribution,
        "trade_count": trade_count,
        "dca_trade_count": dca_trade_count,
        "dva_trade_count": dva_trade_count,
        "trade_logs": pd.DataFrame(trade_logs),
        "score_logs": pd.DataFrame(score_logs),
        "dca_ratio": dca_ratio,
        "dva_ratio": dva_ratio,
    }


# ================================
# 結果表示
# ================================
def format_result_row(name, result, currency_symbol, ticker, mode):
    shares = result["shares"]
    spent = result["spent"]
    cash = result["cash"]
    contribution = result["contribution"]
    trade_count = result["trade_count"]
    avg_price = (spent / shares) if shares > 0 else np.nan

    row = {
        "ticker": ticker,
        "mode": mode,
        "手法名": name,
        "取得株数": shares,
        "平均単価": avg_price,
        "約定総額": spent,
        "繰越現金": cash,
        "拠出総額": contribution,
        "売買回数": trade_count,
        "currency": currency_symbol,
    }

    if "core_cash" in result:
        row["DCA現金"] = result["core_cash"]
        row["DVA現金"] = result["sat_cash"]
        row["DCA回数"] = result.get("dca_trade_count", 0)
        row["DVA回数"] = result.get("dva_trade_count", 0)

    return row


def print_results_table(results):
    results = sorted(results, key=lambda x: x["取得株数"], reverse=True)

    print("-" * 150)
    print(f"{'順位':<2} | {'手法名':<24} | {'取得株数':<10} | {'平均単価':<14} | {'約定総額':<14} | {'繰越現金':<14} | {'拠出総額':<14} | {'売買回数'}")
    print("-" * 150)

    for rank, r in enumerate(results, start=1):
        cur = r["currency"]
        shares_str = f"{r['取得株数']:,d} 株"
        avg_str = f"{cur}{r['平均単価']:,.2f}" if pd.notna(r["平均単価"]) else "-"
        spent_str = f"{cur}{r['約定総額']:,.0f}"
        cash_str = f"{cur}{r['繰越現金']:,.0f}"
        contrib_str = f"{cur}{r['拠出総額']:,.0f}"
        trades_str = f"{r['売買回数']}回"

        print(
            f"{rank:>2} | {r['手法名']:<24} | {shares_str:>10} | {avg_str:>14} | "
            f"{spent_str:>14} | {cash_str:>14} | {contrib_str:>14} | {trades_str}"
        )

    print("-" * 150)


# ================================
# データ取得
# ================================
def download_price_data(ticker, start_date, end_date):
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        return normalize_df(df)
    except Exception as e:
        print(f"  データ取得エラー: {ticker} -> {e}")
        return pd.DataFrame()


# ================================
# メイン
# ================================
def run_all():
    benchmark_vol_cache = {}

    print(f"--- ハイブリッド B案: DCA + DVA ({CONFIG['start_date']} ～ {CONFIG['end_date']}) ---")
    print("DEBUG: CONSOLE ONLY / FLEXIBLE HYBRID RATIO VERSION")
    print("DEBUG: CSV保存なし / 画像保存なし / 単元株上書き対応 / 比率自由設定")

    for ticker in CONFIG["tickers"]:
        cfg = get_market_config(
            ticker,
            total_budget_us=CONFIG["total_budget_us"],
            total_budget_jp=CONFIG["total_budget_jp"],
        )

        total_budget = cfg["total_budget"]
        benchmark = cfg["benchmark"]
        cur = cfg["currency_symbol"]
        lot_size = cfg["lot_size"]
        max_lots_per_trade = cfg["max_lots_per_trade"]

        dca_ratio = get_dca_ratio(ticker)
        dva_ratio = 1.0 - dca_ratio

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']} / 単元: {lot_size}株】")

        if ticker in LOT_SIZE_OVERRIDES:
            print(f"  単元設定: 個別上書き適用 -> {LOT_SIZE_OVERRIDES[ticker]}株")
        else:
            if ticker.endswith(".T"):
                print("  単元設定: 日本株デフォルト -> 100株")
            else:
                print("  単元設定: 米国株デフォルト -> 1株")

        if ticker in HYBRID_RATIO_OVERRIDES:
            print(f"  比率設定: 個別上書き適用 -> DCA {dca_ratio:.0%} / DVA {dva_ratio:.0%}")
        else:
            print(f"  比率設定: 既定値適用 -> DCA {dca_ratio:.0%} / DVA {dva_ratio:.0%}")

        if benchmark not in benchmark_vol_cache:
            bmk = download_price_data(benchmark, CONFIG["start_date"], CONFIG["end_date"])
            if bmk.empty:
                print(f"  ベンチマーク取得失敗: {benchmark}")
                continue
            benchmark_vol_cache[benchmark] = calc_annualized_vol(bmk["Close"], window=20)

        mkt_vol_raw = benchmark_vol_cache[benchmark]

        df = download_price_data(ticker, CONFIG["start_date"], CONFIG["end_date"])
        if df.empty:
            print("  データ取得失敗 or データなし")
            continue

        df = calc_indicators(df, mkt_vol_raw)
        contribution_schedule = build_monthly_contribution_schedule(df, total_budget)
        monthly_budget, n_months = calc_monthly_budget_fixed_total(df, total_budget)

        print(f"  対象月数: {n_months} か月")
        print(f"  総予算: {cur}{total_budget:,.0f}")
        print(f"  月予算: {cur}{monthly_budget:,.2f}")
        print(f"  スコア閾値: {CONFIG['score_threshold']:.2f}")
        print(f"  執行合成係数 λ: {CONFIG['execution_mix_lambda']:.2f}")

        # 基準戦略
        monthly_first = execute_monthly_first(df, contribution_schedule, lot_size=lot_size)
        monthly_last = execute_monthly_last(df, contribution_schedule, lot_size=lot_size)

        # 純DVA (0% DCA / 100% DVA)
        pure_dva = execute_hybrid_dca_dva(
            df=df,
            contribution_schedule=contribution_schedule,
            dca_ratio=0.0,
            lot_size=lot_size,
            score_threshold=CONFIG["score_threshold"],
            max_lots_per_trade=max_lots_per_trade,
            weights=CONFIG["score_weights"],
            execution_mix_lambda=CONFIG["execution_mix_lambda"],
        )

        # ハイブリッド
        hybrid_result = execute_hybrid_dca_dva(
            df=df,
            contribution_schedule=contribution_schedule,
            dca_ratio=dca_ratio,
            lot_size=lot_size,
            score_threshold=CONFIG["score_threshold"],
            max_lots_per_trade=max_lots_per_trade,
            weights=CONFIG["score_weights"],
            execution_mix_lambda=CONFIG["execution_mix_lambda"],
        )

        results = [
            format_result_row("毎月(月初) 100%DCA", monthly_first, cur, ticker, "monthly_first"),
            format_result_row("毎月(月末) 100%DCA", monthly_last, cur, ticker, "monthly_last"),
            format_result_row("純DVA 0/100", pure_dva, cur, ticker, "pure_dva"),
            format_result_row(
                f"ハイブリッド {int(round(dca_ratio*100))}/{int(round(dva_ratio*100))}",
                hybrid_result,
                cur,
                ticker,
                "hybrid",
            ),
        ]

        print_results_table(results)

        # ハイブリッド詳細
        print("  ハイブリッド内訳:")
        print(f"    DCA側繰越現金: {cur}{hybrid_result['core_cash']:,.0f}")
        print(f"    DVA側繰越現金: {cur}{hybrid_result['sat_cash']:,.0f}")
        print(f"    DCA執行回数: {hybrid_result['dca_trade_count']}回")
        print(f"    DVA執行回数: {hybrid_result['dva_trade_count']}回")

        if not hybrid_result["trade_logs"].empty:
            print("  ハイブリッド直近5件の約定:")
            for _, row in hybrid_result["trade_logs"].tail(5).iterrows():
                side = row["side"]
                score_text = "-" if pd.isna(row.get("score", np.nan)) else f"{row['score']:.3f}"
                print(
                    f"    {row['date'].date()} | {side} | 価格 {cur}{row['price']:,.2f} | "
                    f"スコア {score_text} | 買付 {int(row['buy_shares'])}株 | 約定 {cur}{row['cost']:,.0f}"
                )
        else:
            print("  ハイブリッド約定はありません")

    print("\n完了。")


if __name__ == "__main__":
    run_all()
