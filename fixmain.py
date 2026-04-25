import re
from pathlib import Path
from itertools import product

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================
# 設定
# ==================================
CONFIG = {
    "start_date": "2014-01-01",
    "end_date": "2023-12-31",
    "total_budget_us": 60000,
    "total_budget_jp": 6000000,
    "tickers": ["NVDA", "AAPL", "TSLA", "8088.T", "1443.T"],
    "output_dir": "./output_dva_practical",
    "default_score_threshold": 0.45,
    "default_score_weight": 0.70,   # λ
    "default_weights": {
        "shortage": 0.35,
        "value": 0.35,
        "bb": 0.15,
        "trend": 0.15,
        "risk": 0.20,
    },
    "grid_search": {
        "enabled": True,
        "thresholds": [0.40, 0.45, 0.50, 0.55],
        "score_mix_lambdas": [0.60, 0.70, 0.80],
        "shortage_weights": [0.25, 0.35, 0.45],
        "value_weights": [0.25, 0.35, 0.45],
        "bb_weights": [0.10, 0.15, 0.20],
        "trend_weights": [0.05, 0.10, 0.15],
        "risk_weights": [0.10, 0.20, 0.30],
    },
}


# ==================================
# 基本ユーティリティ
# ==================================
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


def get_market_config(ticker, total_budget_us=60000, total_budget_jp=6000000):
    if ticker.endswith(".T"):
        return {
            "market_name": "JP",
            "benchmark": "^N225",
            "total_budget": total_budget_jp,
            "currency_symbol": "¥",
            "currency_code": "JPY",
            "lot_size": 100,
            "max_lots_per_trade": 3,
        }
    else:
        return {
            "market_name": "US",
            "benchmark": "^GSPC",
            "total_budget": total_budget_us,
            "currency_symbol": "$",
            "currency_code": "USD",
            "lot_size": 1,
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
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    monthly_budget, _ = calc_monthly_budget_fixed_total(df, total_budget)

    for _, m_df in get_month_groups(df):
        if len(m_df) == 0:
            continue
        schedule.loc[m_df.index[0]] = monthly_budget

    return schedule


# ==================================
# DVAスコア
# ==================================
def calc_dva_score_row(price, ema, bb_low, vol, mkt_vol, cumulative_contribution, shares, weights):
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


# ==================================
# 執行ルール
# ==================================
def execute_practical_dva(
    df,
    contribution_schedule,
    lot_size=1,
    score_threshold=0.45,
    score_mix_lambda=0.70,   # λ
    max_lots_per_trade=3,
    weights=None,
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
        affordable_lots = int(cash // lot_cost) if lot_cost > 0 and np.isfinite(lot_cost) else 0
        capped_affordable_lots = min(affordable_lots, max_lots_per_trade)

        if score >= score_threshold and capped_affordable_lots >= 1:
            # スコア強度
            q_score = clip((score - score_threshold) / (1.0 - score_threshold), 0.0, 1.0)

            # 現金ストック強度
            if max_lots_per_trade > 1:
                q_cash = clip((affordable_lots - 1) / (max_lots_per_trade - 1), 0.0, 1.0)
            else:
                q_cash = 0.0

            # 執行強度
            q_exec = score_mix_lambda * q_score + (1.0 - score_mix_lambda) * q_cash

            # 希望ロット数
            wish_lots = 1 + int(np.floor(q_exec * (max_lots_per_trade - 1)))
            buy_lots = min(capped_affordable_lots, wish_lots)
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
        "score_mix_lambda": score_mix_lambda,
    }


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


# ==================================
# 保存・表示
# ==================================
def save_histogram(score_logs, output_png, title):
    if score_logs is None or score_logs.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(score_logs["score"].dropna(), bins=20, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def save_csv(df, path):
    if df is None:
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


# ==================================
# グリッドサーチ
# ==================================
def grid_search_dva(
    df,
    contribution_schedule,
    lot_size,
    max_lots_per_trade,
    grid_config,
):
    rows = []
    best_result = None
    best_metric = None

    combos = product(
        grid_config["thresholds"],
        grid_config["score_mix_lambdas"],
        grid_config["shortage_weights"],
        grid_config["value_weights"],
        grid_config["bb_weights"],
        grid_config["trend_weights"],
        grid_config["risk_weights"],
    )

    for threshold, score_mix_lambda, w_shortage, w_value, w_bb, w_trend, w_risk in combos:
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
            score_mix_lambda=score_mix_lambda,
            max_lots_per_trade=max_lots_per_trade,
            weights=weights,
        )

        shares = result["shares"]
        spent = result["spent"]
        cash = result["cash"]
        trades = result["trade_count"]
        avg_price = spent / shares if shares > 0 else np.nan

        rows.append({
            "threshold": threshold,
            "score_mix_lambda": score_mix_lambda,
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
                "score_mix_lambda": score_mix_lambda,
                "weights": weights,
                "result": result,
            }

    search_df = pd.DataFrame(rows).sort_values(
        by=["shares", "spent", "cash"], ascending=[False, False, True]
    ).reset_index(drop=True)

    return best_result, search_df


# ==================================
# メイン
# ==================================
def run_all():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_vol_cache = {}
    summary_rows = []
    best_param_rows = []

    print(f"--- 実用研究向け DVA 一括実行 ({CONFIG['start_date']} ～ {CONFIG['end_date']}) ---")
    print("DEBUG: PRACTICAL DVA VARIABLE LOT VERSION")

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
        ticker_key = safe_name(ticker)

        print(f"\n【銘柄: {ticker} / 市場: {cfg['market_name']} / 通貨: {cfg['currency_code']} / 単元: {lot_size}株】")

        if benchmark not in benchmark_vol_cache:
            bmk = yf.download(
                benchmark,
                start=CONFIG["start_date"],
                end=CONFIG["end_date"],
                auto_adjust=True,
                progress=False,
            )
            bmk = normalize_df(bmk)

            if bmk.empty:
                print(f"  ベンチマーク取得失敗: {benchmark}")
                continue

            benchmark_vol_cache[benchmark] = calc_annualized_vol(bmk["Close"], window=20)

        mkt_vol_raw = benchmark_vol_cache[benchmark]

        df = yf.download(
            ticker,
            start=CONFIG["start_date"],
            end=CONFIG["end_date"],
            auto_adjust=True,
            progress=False,
        )
        df = normalize_df(df)

        if df.empty:
            print("  データ取得失敗 or データなし")
            continue

        df = calc_indicators(df, mkt_vol_raw)
        contribution_schedule = build_monthly_contribution_schedule(df, total_budget)
        monthly_budget, n_months = calc_monthly_budget_fixed_total(df, total_budget)

        print(f"  対象月数: {n_months} か月")
        print(f"  総予算: {cur}{total_budget:,.0f}")
        print(f"  月予算: {cur}{monthly_budget:,.2f}")

        default_dva = execute_practical_dva(
            df=df,
            contribution_schedule=contribution_schedule,
            lot_size=lot_size,
            score_threshold=CONFIG["default_score_threshold"],
            score_mix_lambda=CONFIG["default_score_weight"],
            max_lots_per_trade=max_lots_per_trade,
            weights=CONFIG["default_weights"],
        )

        monthly_first = execute_monthly_first(df, contribution_schedule, lot_size=lot_size)
        monthly_last = execute_monthly_last(df, contribution_schedule, lot_size=lot_size)

        optimized_dva = None
        search_df = pd.DataFrame()

        if CONFIG["grid_search"]["enabled"]:
            best_result, search_df = grid_search_dva(
                df=df,
                contribution_schedule=contribution_schedule,
                lot_size=lot_size,
                max_lots_per_trade=max_lots_per_trade,
                grid_config=CONFIG["grid_search"],
            )
            optimized_dva = best_result["result"]
            best_weights = best_result["weights"]
            best_threshold = best_result["threshold"]
            best_lambda = best_result["score_mix_lambda"]

            best_param_rows.append({
                "ticker": ticker,
                "threshold": best_threshold,
                "score_mix_lambda": best_lambda,
                "w_shortage": best_weights["shortage"],
                "w_value": best_weights["value"],
                "w_bb": best_weights["bb"],
                "w_trend": best_weights["trend"],
                "w_risk": best_weights["risk"],
                "shares": optimized_dva["shares"],
                "spent": optimized_dva["spent"],
                "cash": optimized_dva["cash"],
                "trades": optimized_dva["trade_count"],
            })

        save_csv(default_dva["trade_logs"], output_dir / f"{ticker_key}_default_trade_log.csv")
        save_csv(default_dva["score_logs"], output_dir / f"{ticker_key}_default_score_log.csv")
        save_histogram(default_dva["score_logs"], output_dir / f"{ticker_key}_default_score_hist.png", f"{ticker} Default DVA Score Histogram")

        if optimized_dva is not None:
            save_csv(optimized_dva["trade_logs"], output_dir / f"{ticker_key}_optimized_trade_log.csv")
            save_csv(optimized_dva["score_logs"], output_dir / f"{ticker_key}_optimized_score_log.csv")
            save_histogram(optimized_dva["score_logs"], output_dir / f"{ticker_key}_optimized_score_hist.png", f"{ticker} Optimized DVA Score Histogram")
            save_csv(search_df, output_dir / f"{ticker_key}_grid_search.csv")

        results = [
            format_result_row("実用DVA(既定)", default_dva, cur, ticker, "default"),
            format_result_row("毎月(月初)", monthly_first, cur, ticker, "monthly_first"),
            format_result_row("毎月(月末)", monthly_last, cur, ticker, "monthly_last"),
        ]
        if optimized_dva is not None:
            results.append(format_result_row("実用DVA(最適)", optimized_dva, cur, ticker, "optimized"))

        print_results_table(results)

        for row in results:
            summary_rows.append(row)

        if not default_dva["trade_logs"].empty:
            print("  既定DVAの直近5件の約定:")
            for _, row in default_dva["trade_logs"].tail(5).iterrows():
                print(
                    f"    {row['date'].date()} | 価格 {cur}{row['price']:,.2f} | "
                    f"スコア {row['score']:.3f} | q_score {row['q_score']:.3f} | "
                    f"q_cash {row['q_cash']:.3f} | ロット {int(row['buy_lots'])} | 約定 {cur}{row['cost']:,.0f}"
                )
        else:
            print("  既定DVAの約定はありません")

        if optimized_dva is not None:
            print("  最適化パラメータ:")
            print(
                f"    threshold={best_threshold:.2f}, lambda={best_lambda:.2f}, "
                f"shortage={best_weights['shortage']:.2f}, value={best_weights['value']:.2f}, "
                f"bb={best_weights['bb']:.2f}, trend={best_weights['trend']:.2f}, risk={best_weights['risk']:.2f}"
            )

        print("  ※ スコア未達では買わない")
        print("  ※ 強制買いはなし")
        print("  ※ スコアが高く、かつ現金ストックが大きいほどロット数が増える")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["ticker", "取得株数"], ascending=[True, False])
        save_csv(summary_df, output_dir / "summary_results.csv")

    best_params_df = pd.DataFrame(best_param_rows)
    if not best_params_df.empty:
        save_csv(best_params_df, output_dir / "best_params_summary.csv")

    print(f"\n完了。出力先: {output_dir}")


if __name__ == "__main__":
    run_all()
