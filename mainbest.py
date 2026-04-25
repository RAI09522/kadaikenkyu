import re
import sys
from pathlib import Path
from itertools import product

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
    "start_date": "2014-01-01",
    "end_date": "2023-12-31",

    # 総予算固定
    "total_budget_us": 60000,      # 500 USD × 12 × 10年
    "total_budget_jp": 6000000,    # 50,000 JPY × 12 × 10年

    "tickers": ["NVDA", "AAPL", "TSLA", "8088.T", "1443.T"],

    # 購入基準スコア
    "default_score_threshold": 0.45,

    # 執行ロット決定: q_exec = λ*q_score + (1-λ)*q_cash
    "execution_mix_lambda": 0.70,

    # DVAスコア重み
    "default_weights": {
        "shortage": 0.35,
        "value": 0.35,
        "bb": 0.15,
        "trend": 0.15,
        "risk": 0.20,
    },

    # 重いので通常はFalse推奨
    "grid_search": {
        "enabled": False,
        "thresholds": [0.40, 0.45, 0.50, 0.55],
        "lambdas": [0.60, 0.70, 0.80],
        "shortage_weights": [0.25, 0.35, 0.45],
        "value_weights": [0.25, 0.35, 0.45],
        "bb_weights": [0.10, 0.15, 0.20],
        "trend_weights": [0.05, 0.10, 0.15],
        "risk_weights": [0.10, 0.20, 0.30],
    },
}

# ================================
# 単元株の個別上書き
# ここに銘柄ごとの売買単位を明示できる
# 優先順位:
# 1. この辞書
# 2. 日本株(.T)は100
# 3. それ以外は1
# ================================
LOT_SIZE_OVERRIDES = {
    "8088.T": 100,
    "1443.T": 100,
    # 例:
    # "1306.T": 10,
    # "1343.T": 10,
}


# ================================
# 基本ユーティリティ
# ================================
def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def clip(x, low, high):
    return max(low, min(high, x))


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
            "max_lots_per_trade": 3,
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
    out["bb_low"] = out["ema20"] - (out["Close"].rolling(window=20).std() * 2)
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
    毎月の最初の営業日に、その月の予算を入金する
    """
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        if len(m_df) > 0:
            schedule.loc[m_df.index[0]] = monthly_budget

    return schedule


# ================================
# DVAスコア
# ================================
def calc_dva_score_row(price, ema, bb_low, vol, mkt_vol, cumulative_contribution, shares, weights):
    """
    実用研究向け DVA スコア

    shortage: 理想保有株数との差
    value   : EMAよりどれだけ割安か
    bb      : BB下限割れ
    trend   : 価格がEMA下なら逆張り加点
    risk    : 個別ボラが市場より高すぎると減点
    """
    if pd.isna(price) or price <= 0 or pd.isna(ema) or ema <= 0:
        parts = {
            "shortage": 0.0,
            "value": 0.0,
            "bb": 0.0,
            "trend": 0.0,
            "risk": 0.0,
        }
        return 0.0, parts

    ideal_shares = cumulative_contribution / ema if ema > 0 else 0.0

    if ideal_shares > 0:
        shortage = clip((ideal_shares - shares) / ideal_shares, 0.0, 1.0)
    else:
        shortage = 0.0

    value = clip((ema - price) / (0.20 * ema), 0.0, 1.0)
    bb = 1.0 if (pd.notna(bb_low) and price < bb_low) else 0.0

    if pd.isna(vol) or pd.isna(mkt_vol) or mkt_vol <= 0:
        risk = 0.0
    else:
        risk = clip((vol / (1.2 * mkt_vol)) - 1.0, 0.0, 1.0)

    trend = 1.0 if price < ema else 0.0

    score_raw = (
        weights["shortage"] * shortage
        + weights["value"] * value
        + weights["bb"] * bb
        + weights["trend"] * trend
        - weights["risk"] * risk
    )

    score = clip(score_raw, 0.0, 1.0)

    parts = {
        "shortage": shortage,
        "value": value,
        "bb": bb,
        "trend": trend,
        "risk": risk,
    }
    return score, parts


# ================================
# 執行ルール
# 強制買いなし
# スコアが高く、現金余力が厚いほどロット数を増やす
# ================================
def execute_practical_dva(
    df,
    contribution_schedule,
    lot_size=1,
    score_threshold=0.45,
    max_lots_per_trade=3,
    weights=None,
    execution_mix_lambda=0.70,
):
    if weights is None:
        weights = CONFIG["default_weights"]

    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    trade_logs = []
    score_logs = []

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        ema = df.at[dt, "ema20"]
        bb_low = df.at[dt, "bb_low"]
        vol = df.at[dt, "vol"]
        mkt_vol = df.at[dt, "mkt_vol"]

        add_cash = float(contribution_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        score, parts = calc_dva_score_row(
            price=price,
            ema=ema,
            bb_low=bb_low,
            vol=vol,
            mkt_vol=mkt_vol,
            cumulative_contribution=contribution,
            shares=shares,
            weights=weights,
        )

        lot_cost = price * lot_size if price > 0 else np.inf
        affordable_lots = int(cash // lot_cost) if lot_cost > 0 else 0

        if affordable_lots >= 1 and score >= score_threshold:
            q_score = clip((score - score_threshold) / (1.0 - score_threshold), 0.0, 1.0)

            if max_lots_per_trade <= 1:
                q_cash = 0.0
            else:
                q_cash = clip((affordable_lots - 1) / (max_lots_per_trade - 1), 0.0, 1.0)

            q_exec = execution_mix_lambda * q_score + (1.0 - execution_mix_lambda) * q_cash
            wish_lots = 1 + int(np.floor(q_exec * (max_lots_per_trade - 1)))
            buy_lots = min(affordable_lots, wish_lots)
        else:
            q_score = 0.0
            q_cash = 0.0
            q_exec = 0.0
            wish_lots = 0
            buy_lots = 0

        buy_shares = buy_lots * lot_size
        cost = buy_shares * price

        score_logs.append({
            "date": dt,
            "price": price,
            "ema20": ema,
            "bb_low": bb_low,
            "vol": vol,
            "mkt_vol": mkt_vol,
            "cash_before_trade": cash,
            "contribution_cum": contribution,
            "shares_before_trade": shares,
            "score": score,
            "q_score": q_score,
            "q_cash": q_cash,
            "q_exec": q_exec,
            "affordable_lots": affordable_lots,
            "wish_lots": wish_lots,
            "buy_lots": buy_lots,
            "buy_shares": buy_shares,
            "cost": cost,
            "shortage": parts["shortage"],
            "value": parts["value"],
            "bb": parts["bb"],
            "trend": parts["trend"],
            "risk": parts["risk"],
        })

        if buy_shares > 0:
            shares += buy_shares
            spent += cost
            cash -= cost
            trade_count += 1

            trade_logs.append({
                "date": dt,
                "price": price,
                "score": score,
                "q_score": q_score,
                "q_cash": q_cash,
                "q_exec": q_exec,
                "buy_lots": buy_lots,
                "buy_shares": buy_shares,
                "cost": cost,
                "cash_after": cash,
                "contribution_cum": contribution,
                "shares_after": shares,
                "shortage": parts["shortage"],
                "value": parts["value"],
                "bb": parts["bb"],
                "trend": parts["trend"],
                "risk": parts["risk"],
            })

    return {
        "spent": spent,
        "shares": shares,
        "cash": cash,
        "contribution": contribution,
        "trade_count": trade_count,
        "trade_logs": pd.DataFrame(trade_logs),
        "score_logs": pd.DataFrame(score_logs),
        "weights": weights,
        "score_threshold": score_threshold,
        "execution_mix_lambda": execution_mix_lambda,
    }


# ================================
# ベースライン戦略
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
# 結果表示
# ================================
def format_result_row(name, result, currency_symbol, ticker, mode):
    shares = result["shares"]
    spent = result["spent"]
    cash = result["cash"]
    contribution = result["contribution"]
    trade_count = result["trade_count"]
    avg_price = (spent / shares) if shares > 0 else np.nan

    return {
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


def print_results_table(results):
    results = sorted(results, key=lambda x: x["取得株数"], reverse=True)

    print("-" * 136)
    print(f"{'順位':<2} | {'手法名':<14} | {'取得株数':<10} | {'平均単価':<14} | {'約定総額':<14} | {'繰越現金':<14} | {'拠出総額':<14} | {'売買回数'}")
    print("-" * 136)

    for rank, r in enumerate(results, start=1):
        cur = r["currency"]
        shares_str = f"{r['取得株数']:,d} 株"
        avg_str = f"{cur}{r['平均単価']:,.2f}" if pd.notna(r["平均単価"]) else "-"
        spent_str = f"{cur}{r['約定総額']:,.0f}"
        cash_str = f"{cur}{r['繰越現金']:,.0f}"
        contrib_str = f"{cur}{r['拠出総額']:,.0f}"
        trades_str = f"{r['売買回数']}回"

        print(
            f"{rank:>2} | {r['手法名']:<14} | {shares_str:>10} | {avg_str:>14} | "
            f"{spent_str:>14} | {cash_str:>14} | {contrib_str:>14} | {trades_str}"
        )

    print("-" * 136)


# ================================
# グリッドサーチ
# ================================
def grid_search_dva(df, contribution_schedule, lot_size, max_lots_per_trade, grid_config):
    rows = []
    best_result = None
    best_metric = None

    combos = product(
        grid_config["thresholds"],
        grid_config["lambdas"],
        grid_config["shortage_weights"],
        grid_config["value_weights"],
        grid_config["bb_weights"],
        grid_config["trend_weights"],
        grid_config["risk_weights"],
    )

    for threshold, mix_lambda, w_shortage, w_value, w_bb, w_trend, w_risk in combos:
        weights = {
            "shortage": w_shortage,
            "value": w_value,
            "bb": w_bb,
            "trend": w_trend,
            "risk": w_risk,
        }

        result = execute_practical_dva(
            df=df,
            contribution_schedule=contribution_schedule,
            lot_size=lot_size,
            score_threshold=threshold,
            max_lots_per_trade=max_lots_per_trade,
            weights=weights,
            execution_mix_lambda=mix_lambda,
        )

        shares = result["shares"]
        spent = result["spent"]
        cash = result["cash"]
        trades = result["trade_count"]
        avg_price = spent / shares if shares > 0 else np.nan

        rows.append({
            "threshold": threshold,
            "mix_lambda": mix_lambda,
            "w_shortage": w_shortage,
            "w_value": w_value,
            "w_bb": w_bb,
            "w_trend": w_trend,
            "w_risk": w_risk,
            "shares": shares,
            "spent": spent,
            "cash": cash,
            "trades": trades,
            "avg_price": avg_price,
        })

        metric = (
            shares,
            spent,
            -cash,
            -(avg_price if pd.notna(avg_price) else 10**18),
        )

        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_result = {
                "threshold": threshold,
                "mix_lambda": mix_lambda,
                "weights": weights,
                "result": result,
                "search_rows": rows,
            }

    return best_result


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

    print(f"--- 実用研究向け DVA 一括実行 ({CONFIG['start_date']} ～ {CONFIG['end_date']}) ---")
    print("DEBUG: DVA CONSOLE ONLY VERSION")
    print("DEBUG: CSV保存なし / 画像保存なし / 単元株上書き対応")

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

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']} / 単元: {lot_size}株】")

        if ticker in LOT_SIZE_OVERRIDES:
            print(f"  単元設定: 個別上書き適用 -> {LOT_SIZE_OVERRIDES[ticker]}株")
        else:
            if ticker.endswith(".T"):
                print("  単元設定: 日本株デフォルト -> 100株")
            else:
                print("  単元設定: 米国株デフォルト -> 1株")

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
        print(f"  購入基準スコア: {CONFIG['default_score_threshold']:.2f}")
        print(f"  執行合成係数 λ: {CONFIG['execution_mix_lambda']:.2f}")

        # 既定DVA
        default_dva = execute_practical_dva(
            df=df,
            contribution_schedule=contribution_schedule,
            lot_size=lot_size,
            score_threshold=CONFIG["default_score_threshold"],
            max_lots_per_trade=max_lots_per_trade,
            weights=CONFIG["default_weights"],
            execution_mix_lambda=CONFIG["execution_mix_lambda"],
        )

        # ベースライン
        monthly_first = execute_monthly_first(df, contribution_schedule, lot_size=lot_size)
        monthly_last = execute_monthly_last(df, contribution_schedule, lot_size=lot_size)

        optimized_dva = None
        best_weights = None
        best_threshold = None
        best_lambda = None

        if CONFIG["grid_search"]["enabled"]:
            best_result = grid_search_dva(
                df=df,
                contribution_schedule=contribution_schedule,
                lot_size=lot_size,
                max_lots_per_trade=max_lots_per_trade,
                grid_config=CONFIG["grid_search"],
            )

            if best_result is not None:
                optimized_dva = best_result["result"]
                best_weights = best_result["weights"]
                best_threshold = best_result["threshold"]
                best_lambda = best_result["mix_lambda"]

        results = [
            format_result_row("実用DVA(既定)", default_dva, cur, ticker, "default"),
            format_result_row("毎月(月初)", monthly_first, cur, ticker, "monthly_first"),
            format_result_row("毎月(月末)", monthly_last, cur, ticker, "monthly_last"),
        ]

        if optimized_dva is not None:
            results.append(format_result_row("実用DVA(最適)", optimized_dva, cur, ticker, "optimized"))

        print_results_table(results)

        if not default_dva["trade_logs"].empty:
            print("  既定DVAの直近5件の約定:")
            for _, row in default_dva["trade_logs"].tail(5).iterrows():
                print(
                    f"    {row['date'].date()} | 価格 {cur}{row['price']:,.2f} | "
                    f"スコア {row['score']:.3f} | q_score {row['q_score']:.3f} | "
                    f"q_cash {row['q_cash']:.3f} | q_exec {row['q_exec']:.3f} | "
                    f"買付 {int(row['buy_shares'])}株 | 約定 {cur}{row['cost']:,.0f}"
                )
        else:
            print("  既定DVAの約定はありません")

        if optimized_dva is not None:
            print("  最適化パラメータ:")
            print(
                f"    threshold={best_threshold:.2f}, lambda={best_lambda:.2f}, "
                f"shortage={best_weights['shortage']:.2f}, value={best_weights['value']:.2f}, "
                f"bb={best_weights['bb']:.2f}, trend={best_weights['trend']:.2f}, "
                f"risk={best_weights['risk']:.2f}"
            )

    print("\n完了。")


if __name__ == "__main__":
    run_all()
