import sys
from itertools import product
import pandas as pd
import numpy as np

# yfinance等のインポートチェック省略（前回と同様の環境で動きます）
import yfinance as yf

# ================================
# 設定 (CONFIG)
# ================================
CONFIG = {
    "start_date": "2014-01-01",
    "end_date": "2023-12-31",
    "total_budget_jp": 6000000,
    "ticker": "8088.T", # 今回は最適化のターゲットを1銘柄に絞ります
    "benchmark": "^N225",
    "lot_size": 100,
    "max_lots_per_trade": 3,
    
    # 固定しておくベース設定
    "trading_days_yr": 252,
    "use_dow_shield": True,
    "default_weights": {"shortage": 0.35, "value": 0.35, "bb": 0.15, "trend": 0.15, "risk": 0.20},

    # ★ グリッドサーチ（総当たり探索）するパラメータの範囲
    "grid_search_params": {
        "window_size": [20, 50],             # 短期波(20日) vs 中期波(50日)
        "bb_sd_mult": [2.0, 2.5],            # BBの幅(-2σ vs -2.5σ)
        "value_drop_ratio": [0.15, 0.20],    # 割安とみなす下落率(15% vs 20%)
        "score_threshold": [0.40, 0.45],     # 買うための合格ライン
        "execution_mix_lambda": [0.5, 0.8]   # 意欲とお財布の比率（0.5=半々, 0.8=意欲重視）
    }
}

# ================================
# 基本ユーティリティ＆指標計算
# ================================
def clip(x, low, high):
    return max(low, min(high, x))

def calc_annualized_vol(close_series, window, trading_days):
    returns = close_series.pct_change(fill_method=None)
    return returns.rolling(window=window).std() * np.sqrt(trading_days)

def calc_indicators(df, mkt_vol_raw, win, bb_mult, ann_days):
    out = df.copy()
    out["mkt_vol"] = mkt_vol_raw.reindex(out.index).ffill().fillna(mkt_vol_raw.mean())
    out["vol"] = calc_annualized_vol(out["Close"], window=win, trading_days=ann_days)
    out["ema"] = out["Close"].ewm(span=win, adjust=False).mean()
    out["bb_sd"] = out["Close"].rolling(window=win).std()
    out["bb_low"] = out["ema"] - (out["bb_sd"] * bb_mult)

    if "High" in out.columns and "Low" in out.columns:
        out["recent_high"] = out["High"].rolling(window=win).max()
        out["prev_high"] = out["recent_high"].shift(win)
        out["recent_low"] = out["Low"].rolling(window=win).min()
        out["prev_low"] = out["recent_low"].shift(win)
        out["dow_downtrend"] = (out["recent_high"] < out["prev_high"]) & (out["recent_low"] < out["prev_low"])
    else:
        out["dow_downtrend"] = False
    return out

def get_month_groups(df):
    return list(df.groupby(df.index.to_period("M")))

def build_monthly_contribution_schedule(df, total_budget):
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    month_groups = get_month_groups(df)
    n_months = len(month_groups)
    monthly_budget = total_budget / n_months if n_months > 0 else 0
    for _, m_df in month_groups:
        if len(m_df) > 0:
            schedule.loc[m_df.index[0]] = monthly_budget
    return schedule

# ================================
# DVAスコア計算＆執行ロジック
# ================================
def calc_dva_score_row(price, ema, bb_low, vol, mkt_vol, contribution, shares, weights, is_dow_downtrend, v_drop, r_mult=1.2):
    if pd.isna(price) or price <= 0 or pd.isna(ema) or ema <= 0: return 0.0
    ideal_shares = contribution / ema if ema > 0 else 0.0
    shortage = clip((ideal_shares - shares) / ideal_shares, 0.0, 1.0) if ideal_shares > 0 else 0.0
    value = clip((ema - price) / (v_drop * ema), 0.0, 1.0)
    bb = 1.0 if (pd.notna(bb_low) and price < bb_low) else 0.0
    risk = clip((vol / (r_mult * mkt_vol)) - 1.0, 0.0, 1.0) if (pd.notna(vol) and mkt_vol > 0) else 0.0
    trend = 1.0 if price < ema else 0.0

    score_raw = (weights["shortage"] * shortage + weights["value"] * value + weights["bb"] * bb + weights["trend"] * trend - weights["risk"] * risk)
    if CONFIG["use_dow_shield"] and is_dow_downtrend: score_raw = 0.0
    return clip(score_raw, 0.0, 1.0)

def execute_dva_for_grid(df, contrib_schedule, lot_size, max_lots, params):
    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0

    # 月末スイープ用の判定
    month_groups = get_month_groups(df)
    last_days = set(m.index[-1] for m in month_groups if len(m) > 0)

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        ema = df.at[dt, "ema"]
        bb_low = df.at[dt, "bb_low"]
        vol = df.at[dt, "vol"]
        mkt_vol = df.at[dt, "mkt_vol"]
        is_dow_downtrend = bool(df.at[dt, "dow_downtrend"]) if "dow_downtrend" in df.columns else False

        add_cash = float(contrib_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        score = calc_dva_score_row(price, ema, bb_low, vol, mkt_vol, contribution, shares, CONFIG["default_weights"], is_dow_downtrend, params["value_drop_ratio"])

        lot_cost = price * lot_size if price > 0 else np.inf
        affordable_lots = int(cash // lot_cost) if lot_cost > 0 else 0

        buy_lots = 0
        if affordable_lots >= 1 and score >= params["score_threshold"]:
            q_score = clip((score - params["score_threshold"]) / (1.0 - params["score_threshold"]), 0.0, 1.0)
            q_cash = clip((affordable_lots - 1) / (max_lots - 1), 0.0, 1.0) if max_lots > 1 else 0.0
            q_exec = params["execution_mix_lambda"] * q_score + (1.0 - params["execution_mix_lambda"]) * q_cash
            wish_lots = 1 + int(np.floor(q_exec * (max_lots - 1)))
            buy_lots = min(affordable_lots, wish_lots)

        if buy_lots > 0:
            buy_sh = buy_lots * lot_size
            cost = buy_sh * price
            shares += buy_sh
            spent += cost
            cash -= cost

        # 月末スイープ(強制買付)
        if dt in last_days and cash > 0 and price > 0:
            sweep_lots = int(cash // (price * lot_size))
            if sweep_lots > 0:
                buy_sh = sweep_lots * lot_size
                cost = buy_sh * price
                shares += buy_sh
                spent += cost
                cash -= cost

    return shares, (spent / shares) if shares > 0 else 0

# ================================
# グリッドサーチ実行機能
# ================================
def run_grid_search():
    print(f"--- 黄金パラメータ探索（グリッドサーチ）開始 ---")
    ticker = CONFIG["ticker"]
    benchmark = CONFIG["benchmark"]
    
    # データの取得
    df_raw = yf.download(ticker, start=CONFIG["start_date"], end=CONFIG["end_date"], auto_adjust=True, progress=False)
    bmk_raw = yf.download(benchmark, start=CONFIG["start_date"], end=CONFIG["end_date"], auto_adjust=True, progress=False)
    
    if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
    if isinstance(bmk_raw.columns, pd.MultiIndex): bmk_raw.columns = bmk_raw.columns.get_level_values(0)

    contrib_sched = build_monthly_contribution_schedule(df_raw, CONFIG["total_budget_jp"])
    
    # 探索するパラメータの組み合わせを生成
    param_grid = CONFIG["grid_search_params"]
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"対象銘柄: {ticker} / 組み合わせ総数: {len(combinations)}パターン")
    print("計算中...\n")

    results = []
    for i, params in enumerate(combinations):
        # パラメータに合わせて指標を再計算
        df = calc_indicators(
            df_raw, 
            calc_annualized_vol(bmk_raw["Close"], params["window_size"], CONFIG["trading_days_yr"]),
            win=params["window_size"], 
            bb_mult=params["bb_sd_mult"], 
            ann_days=CONFIG["trading_days_yr"]
        )
        
        # シミュレーション実行
        shares, avg_price = execute_dva_for_grid(df, contrib_sched, CONFIG["lot_size"], CONFIG["max_lots_per_trade"], params)
        
        params["取得株数"] = shares
        params["平均単価"] = avg_price
        results.append(params)

    # 結果を株数が多い順にソート
    results_df = pd.DataFrame(results).sort_values(by=["取得株数", "平均単価"], ascending=[False, True])
    
    print("=== ✨ 最適パラメータ トップ5 ✨ ===")
    print(results_df.head(5).to_string(index=False))

if __name__ == "__main__":
    run_grid_search()