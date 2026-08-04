# dva_research_final_academic.py
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import pickle
import itertools
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


# =========================================================
# 0. 設定
# =========================================================

TICKERS = [
    "1443.T", "4733.T", "3046.T", "4519.T", "3994.T", "3445.T", "4194.T", "9603.T", "3659.T", "4684.T",
    "7832.T", "7550.T", "9766.T", "7974.T", "6702.T", "4307.T", "9843.T", "3349.T", "2791.T", "9627.T",
    "4478.T", "464A.T", "9158.T", "7699.T", "4375.T", "6574.T", "2986.T", "5590.T", "6030.T", "2586.T",
    "6094.T", "2998.T", "6521.T", "9162.T", "3652.T", "4170.T", "9348.T", "2160.T", "4582.T", "7777.T",
    "9166.T", "7806.T", "3479.T", "3491.T", "4575.T", "4592.T", "3773.T", "7803.T", "4628.T", "2782.T",
    "8871.T", "8928.T", "3437.T", "2790.T", "3944.T", "3690.T", "2689.T", "4624.T", "3710.T", "3951.T",
    "3958.T", "4623.T", "3955.T", "3067.T", "3484.T", "3020.T", "3435.T", "3134.T", "3943.T", "3698.T",
    "4635.T", "3096.T", "3426.T", "3667.T", "3080.T", "3011.T", "8844.T", "2991.T", "8869.T", "8904.T",
    "2370.T", "4588.T", "4013.T", "4893.T", "4881.T", "3911.T", "2385.T", "4892.T", "6999.T", "8388.T",
    "6532.T", "6460.T", "9519.T", "9601.T", "8362.T", "6961.T", "4911.T", "7383.T", "5032.T", "9310.T",
    "8084.T", "3778.T", "4443.T", "2726.T", "7203.T",
]

BENCHMARK = "^N225"
START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

IS_END = "2020-12-31"
OOS_START = "2021-01-01"

TOTAL_BUDGET_JP = 6_000_000
LOT_SIZE = 100

WORK_DIR = "./dva_study"
CACHE_DIR = os.path.join(WORK_DIR, "cache")
RESULT_DIR = os.path.join(WORK_DIR, "results")
SAFE_DIR = "/mnt/user-data/outputs/dva_study"

for d in [WORK_DIR, CACHE_DIR, RESULT_DIR, SAFE_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# 1. 保存・共通ユーティリティ
# =========================================================

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df.sort_index()


def save_df(df: pd.DataFrame, name: str):
    p_csv = os.path.join(RESULT_DIR, f"{name}.csv")
    p_pkl = os.path.join(RESULT_DIR, f"{name}.pkl")
    df.to_csv(p_csv, index=False, encoding="utf-8-sig")
    df.to_pickle(p_pkl)

    s_csv = os.path.join(SAFE_DIR, f"{name}.csv")
    s_pkl = os.path.join(SAFE_DIR, f"{name}.pkl")
    df.to_csv(s_csv, index=False, encoding="utf-8-sig")
    df.to_pickle(s_pkl)


def save_obj(obj, name: str):
    p = os.path.join(RESULT_DIR, f"{name}.pkl")
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    s = os.path.join(SAFE_DIR, f"{name}.pkl")
    with open(s, "wb") as f:
        pickle.dump(obj, f)


def save_text(text: str, name: str):
    p = os.path.join(RESULT_DIR, f"{name}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    s = os.path.join(SAFE_DIR, f"{name}.txt")
    with open(s, "w", encoding="utf-8") as f:
        f.write(text)


def clip(x, low, high):
    return max(low, min(high, x))


def zscore_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def is_full_10y(df):
    start_ok = df.index.min() <= pd.Timestamp("2016-02-01")
    end_ok = df.index.max() >= pd.Timestamp("2025-12-01")
    return start_ok and end_ok


# =========================================================
# 2. データ取得
# =========================================================

def fetch_one(ticker, start, end, retries=3, pause=1.0):
    last_err = None
    for i in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = normalize_df(df)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(pause * (i + 1))
    print(f"[FAIL] {ticker}: {last_err}")
    return pd.DataFrame()


def fetch_all_data():
    meta = {"ok": [], "fail": []}

    bmk_path = os.path.join(CACHE_DIR, f"{BENCHMARK}.pkl")
    if not os.path.exists(bmk_path):
        print(f"fetch benchmark: {BENCHMARK}")
        bmk = fetch_one(BENCHMARK, START_DATE, END_DATE)
        bmk.to_pickle(bmk_path)

    for i, tk in enumerate(TICKERS, 1):
        p = os.path.join(CACHE_DIR, f"{tk}.pkl")
        if os.path.exists(p):
            meta["ok"].append(tk)
            continue

        df = fetch_one(tk, START_DATE, END_DATE)
        if df.empty:
            meta["fail"].append(tk)
        else:
            df.to_pickle(p)
            meta["ok"].append(tk)

        if i % 10 == 0 or i == len(TICKERS):
            print(f"fetch {i}/{len(TICKERS)}")

    save_obj(meta, "fetch_meta")
    return meta


def load_benchmark():
    return pd.read_pickle(os.path.join(CACHE_DIR, f"{BENCHMARK}.pkl"))


def load_ticker(ticker):
    return pd.read_pickle(os.path.join(CACHE_DIR, f"{ticker}.pkl"))


# =========================================================
# 3. 指標
# =========================================================

def calc_annualized_vol(close_series, window=20):
    returns = close_series.pct_change(fill_method=None)
    return returns.rolling(window=window).std() * np.sqrt(252)


def calc_indicators(df, mkt_vol_raw):
    out = df.copy()
    out["mkt_vol"] = mkt_vol_raw.reindex(out.index).ffill().fillna(mkt_vol_raw.mean())
    out["vol"] = calc_annualized_vol(out["Close"], window=20)
    out["ema20"] = out["Close"].ewm(span=20, adjust=False).mean()
    std20 = out["Close"].rolling(window=20).std()
    out["bb_low"] = out["ema20"] - 2.0 * std20
    return out


def get_month_groups(df):
    return list(df.groupby(df.index.to_period("M")))


def build_monthly_contribution_schedule(df, total_budget):
    schedule = pd.Series(0.0, index=df.index, dtype=float)
    groups = get_month_groups(df)
    n_months = len(groups)
    if n_months == 0:
        return schedule, 0.0, 0
    monthly_budget = total_budget / n_months
    for _, m_df in groups:
        if len(m_df) > 0:
            schedule.loc[m_df.index[0]] = monthly_budget
    return schedule, monthly_budget, n_months


# =========================================================
# 4. DVA
# =========================================================

@dataclass
class DVAParams:
    score_threshold: float = 0.55
    w_shortage: float = 0.35
    w_value: float = 0.35
    w_bb: float = 0.15
    w_trend: float = 0.15
    w_risk: float = -0.20
    value_band: float = 0.20
    risk_ratio: float = 1.20
    max_lots_per_trade: int = 2
    lot_size: int = 100


def execute_dva(df, contribution_schedule, params: DVAParams):
    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0
    trade_count = 0

    for dt in df.index:
        price = float(df.at[dt, "Close"])
        ema = df.at[dt, "ema20"]
        bb_low = df.at[dt, "bb_low"]
        vol = df.at[dt, "vol"]
        mkt_vol = df.at[dt, "mkt_vol"]

        add_cash = float(contribution_schedule.at[dt])
        cash += add_cash
        contribution += add_cash

        if pd.isna(price) or price <= 0 or pd.isna(ema) or ema <= 0:
            continue

        ideal_shares = contribution / ema if ema > 0 else 0.0
        shortage = clip((ideal_shares - shares) / ideal_shares, 0.0, 1.0) if ideal_shares > 0 else 0.0
        value = clip((ema - price) / (params.value_band * ema), 0.0, 1.0)
        bb = 1.0 if (pd.notna(bb_low) and price < bb_low) else 0.0

        if pd.isna(vol) or pd.isna(mkt_vol) or mkt_vol <= 0:
            risk = 0.0
        else:
            risk = clip((vol / (params.risk_ratio * mkt_vol)) - 1.0, 0.0, 1.0)

        trend = 1.0 if price < ema else 0.0

        score_raw = (
            params.w_shortage * shortage
            + params.w_value * value
            + params.w_bb * bb
            + params.w_trend * trend
            + params.w_risk * risk
        )
        score = clip(score_raw, 0.0, 1.0)

        lot_cost = price * params.lot_size
        affordable_lots = int(cash // lot_cost) if lot_cost > 0 else 0

        if affordable_lots >= 1 and score >= params.score_threshold:
            q = clip((score - params.score_threshold) / (1.0 - params.score_threshold), 0.0, 1.0)
            wish_lots = 1 + int(np.floor(q * (params.max_lots_per_trade - 1)))
            buy_lots = min(affordable_lots, wish_lots)
        else:
            buy_lots = 0

        buy_shares = buy_lots * params.lot_size
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


# =========================================================
# 5. DCA基準
# =========================================================

def execute_dca_A_monthly_realistic(df, contribution_schedule, lot_size=100):
    month_groups = get_month_groups(df)

    shares = 0
    spent = 0.0
    cash = 0.0
    contribution = 0.0

    month_logs = []

    for _, m_df in month_groups:
        if len(m_df) == 0:
            continue

        first_dt = m_df.index[0]
        last_dt = m_df.index[-1]

        p_first = float(df.at[first_dt, "Close"])
        p_last = float(df.at[last_dt, "Close"])
        avg_price = (p_first + p_last) / 2.0

        add_cash = float(contribution_schedule.at[first_dt])
        cash += add_cash
        contribution += add_cash

        lot_cost = avg_price * lot_size
        buy_lots = int(cash // lot_cost) if lot_cost > 0 else 0
        buy_shares = buy_lots * lot_size
        cost = buy_shares * avg_price

        shares += buy_shares
        spent += cost
        cash -= cost

        month_logs.append({
            "month": str(m_df.index[0].to_period("M")),
            "avg_price": avg_price,
            "buy_shares": buy_shares,
            "cash_after": cash,
        })

    avg_price_total = spent / shares if shares > 0 else np.nan
    return {
        "shares": shares,
        "spent": spent,
        "cash": cash,
        "contribution": contribution,
        "avg_price": avg_price_total,
        "logs": pd.DataFrame(month_logs),
    }


def execute_dca_B_lumpsum_reference(df, total_budget=6_000_000, lot_size=100):
    prices = []
    for _, m_df in get_month_groups(df):
        if len(m_df) == 0:
            continue
        first_dt = m_df.index[0]
        last_dt = m_df.index[-1]
        p_first = float(df.at[first_dt, "Close"])
        p_last = float(df.at[last_dt, "Close"])
        prices.append((p_first + p_last) / 2.0)

    if len(prices) == 0:
        return None

    ref_price = float(np.mean(prices))
    lots = int(total_budget // (ref_price * lot_size))
    shares = lots * lot_size
    spent = shares * ref_price
    cash = total_budget - spent

    return {
        "shares": shares,
        "spent": spent,
        "cash": cash,
        "contribution": total_budget,
        "avg_price": ref_price,
    }


# =========================================================
# 6. 銘柄評価
# =========================================================

def evaluate_one_ticker(df, mkt_vol_raw, params: DVAParams, total_budget=6_000_000, lot_size=100):
    df_ind = calc_indicators(df, mkt_vol_raw)
    schedule, monthly_budget, n_months = build_monthly_contribution_schedule(df_ind, total_budget)

    dva = execute_dva(df_ind, schedule, params)
    dca_a = execute_dca_A_monthly_realistic(df_ind, schedule, lot_size=lot_size)
    dca_b = execute_dca_B_lumpsum_reference(df_ind, total_budget=total_budget, lot_size=lot_size)

    p_end = float(df_ind["Close"].iloc[-1])

    def mix_eval(ref_price):
        if not np.isfinite(ref_price) or ref_price <= 0:
            return {
                "extra_shares": 0, "mix_shares": np.nan, "mix_avg_price": np.nan,
                "dca_pure_shares": np.nan, "eff_mix": np.nan, "eff_dca": np.nan, "improvement": np.nan
            }

        extra_lots = int(dva["cash"] // (ref_price * lot_size))
        extra_shares = extra_lots * lot_size
        extra_cost = extra_shares * ref_price

        mix_shares = dva["shares"] + extra_shares
        mix_cost = dva["spent"] + extra_cost
        mix_avg_price = mix_cost / mix_shares if mix_shares > 0 else np.nan

        dca_lots_pure = int(total_budget // (ref_price * lot_size))
        dca_pure_shares = dca_lots_pure * lot_size

        eff_mix = (mix_shares * p_end) / total_budget if total_budget > 0 else np.nan
        eff_dca = (dca_pure_shares * p_end) / total_budget if total_budget > 0 else np.nan
        improvement = eff_mix / eff_dca if (pd.notna(eff_dca) and eff_dca > 0) else np.nan

        return {
            "extra_shares": extra_shares,
            "mix_shares": mix_shares,
            "mix_avg_price": mix_avg_price,
            "dca_pure_shares": dca_pure_shares,
            "eff_mix": eff_mix,
            "eff_dca": eff_dca,
            "improvement": improvement,
        }

    a = mix_eval(dca_a["avg_price"])
    b = mix_eval(dca_b["avg_price"] if dca_b is not None else np.nan)

    return {
        "dva_shares": dva["shares"],
        "dva_spent": dva["spent"],
        "dva_cash": dva["cash"],
        "dva_avg_price": dva["spent"] / dva["shares"] if dva["shares"] > 0 else np.nan,
        "dva_trades": dva["trade_count"],
        "P_end": p_end,

        "A_P_dca": dca_a["avg_price"],
        "A_extra_shares": a["extra_shares"],
        "A_mix_shares": a["mix_shares"],
        "A_mix_avg_price": a["mix_avg_price"],
        "A_dca_pure_shares": a["dca_pure_shares"],
        "A_eff_mix": a["eff_mix"],
        "A_eff_dca": a["eff_dca"],
        "A_improvement": a["improvement"],

        "B_P_dca": dca_b["avg_price"] if dca_b is not None else np.nan,
        "B_extra_shares": b["extra_shares"],
        "B_mix_shares": b["mix_shares"],
        "B_mix_avg_price": b["mix_avg_price"],
        "B_dca_pure_shares": b["dca_pure_shares"],
        "B_eff_mix": b["eff_mix"],
        "B_eff_dca": b["eff_dca"],
        "B_improvement": b["improvement"],

        "n_months": n_months,
        "monthly_budget": monthly_budget,
        "n_days": len(df_ind),
    }


# =========================================================
# 7. ベースライン
# =========================================================

def run_baseline():
    bmk = load_benchmark()
    mkt_vol_raw = calc_annualized_vol(bmk["Close"], window=20)

    rows = []
    params = DVAParams()

    for i, tk in enumerate(TICKERS, 1):
        df = load_ticker(tk)
        res = evaluate_one_ticker(df, mkt_vol_raw, params, TOTAL_BUDGET_JP, LOT_SIZE)
        rows.append({
            "ticker": tk,
            "full_10y": is_full_10y(df),
            **res
        })
        if i % 20 == 0 or i == len(TICKERS):
            print(f"baseline {i}/{len(TICKERS)}")

    out = pd.DataFrame(rows)
    save_df(out, "baseline_105")
    return out


# =========================================================
# 8. 特徴量
# =========================================================

def hurst_exponent(ts, min_lag=2, max_lag=100):
    ts = np.asarray(ts, dtype=float)
    ts = ts[np.isfinite(ts)]
    if len(ts) < max_lag * 4:
        max_lag = max(min_lag + 5, len(ts) // 4)
    if max_lag <= min_lag:
        return np.nan

    lags = range(min_lag, max_lag)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        s = np.std(diff)
        tau.append(s if s > 0 else np.nan)

    lags_arr = np.array(list(lags))
    tau_arr = np.array(tau)
    mask = np.isfinite(tau_arr) & (tau_arr > 0)
    if mask.sum() < 5:
        return np.nan

    poly = np.polyfit(np.log(lags_arr[mask]), np.log(tau_arr[mask]), 1)
    return float(poly[0])


def max_drawdown(close):
    x = np.asarray(close, dtype=float)
    peak = np.maximum.accumulate(x)
    dd = (x - peak) / peak
    return float(dd.min())


def annualized_return(close):
    x = np.asarray(close, dtype=float)
    if len(x) < 2:
        return np.nan
    years = len(x) / 252
    return float((x[-1] / x[0]) ** (1 / years) - 1)


def extract_features():
    bmk = load_benchmark()
    rows = []

    for tk in TICKERS:
        df = load_ticker(tk)
        common = df.index.intersection(bmk.index)
        px = df.loc[common, "Close"]
        bpx = bmk.loc[common, "Close"]

        r = px.pct_change(fill_method=None)
        rb = bpx.pct_change(fill_method=None)

        rows.append({
            "ticker": tk,
            "sigma": float(r.std() * np.sqrt(252)),
            "hurst": hurst_exponent(np.log(px.values)),
            "corr_bmk": float(r.corr(rb)),
            "mu": annualized_return(px.values),
            "mdd": max_drawdown(px.values),
            "log_vol": float(np.log(df["Volume"].mean() + 1)) if "Volume" in df.columns else np.nan,
            "n_days": len(df),
            "full_10y": is_full_10y(df),
        })

    out = pd.DataFrame(rows)
    save_df(out, "features")
    return out


# =========================================================
# 9. グリッドサーチ
# =========================================================

def simplex_weights(step=0.20, lo=0.05, hi=0.65):
    out = []
    vals = np.arange(lo, hi + 1e-9, step)
    for ws in vals:
        for wv in vals:
            for wbb in vals:
                wt = 1.0 - ws - wv - wbb
                if lo <= wt <= hi:
                    out.append((
                        round(float(ws), 4),
                        round(float(wv), 4),
                        round(float(wbb), 4),
                        round(float(wt), 4),
                    ))
    return sorted(list(set(out)))


def build_stage1_grid():
    weights = simplex_weights(step=0.20, lo=0.05, hi=0.65)
    thresholds = [0.40, 0.55, 0.65]
    w_risks = [-0.20, -0.10]
    value_bands = [0.15, 0.25]
    risk_ratios = [1.00, 1.30]
    max_lots = [2]

    params_list = []
    for ws, wv, wbb, wt in weights:
        for thr in thresholds:
            for wr in w_risks:
                for vb in value_bands:
                    for rr in risk_ratios:
                        for ml in max_lots:
                            params_list.append(DVAParams(
                                score_threshold=thr,
                                w_shortage=ws,
                                w_value=wv,
                                w_bb=wbb,
                                w_trend=wt,
                                w_risk=wr,
                                value_band=vb,
                                risk_ratio=rr,
                                max_lots_per_trade=ml,
                                lot_size=LOT_SIZE,
                            ))
    return params_list


def build_stage2_grid(top_rows, n_local=200, delta=0.05, seed=42):
    rng = np.random.default_rng(seed)
    params_list = []

    for _, row in top_rows.iterrows():
        base = DVAParams(
            score_threshold=float(row["score_threshold"]),
            w_shortage=float(row["w_shortage"]),
            w_value=float(row["w_value"]),
            w_bb=float(row["w_bb"]),
            w_trend=float(row["w_trend"]),
            w_risk=float(row["w_risk"]),
            value_band=float(row["value_band"]),
            risk_ratio=float(row["risk_ratio"]),
            max_lots_per_trade=int(row["max_lots_per_trade"]),
            lot_size=LOT_SIZE
        )

        for _ in range(60):
            w = np.array([
                base.w_shortage + rng.uniform(-delta, delta),
                base.w_value + rng.uniform(-delta, delta),
                base.w_bb + rng.uniform(-delta, delta),
                base.w_trend + rng.uniform(-delta, delta),
            ])
            w = np.clip(w, 0.05, 0.70)
            w = w / w.sum()

            params_list.append(DVAParams(
                score_threshold=float(np.clip(base.score_threshold + rng.uniform(-0.05, 0.05), 0.30, 0.75)),
                w_shortage=float(round(w[0], 4)),
                w_value=float(round(w[1], 4)),
                w_bb=float(round(w[2], 4)),
                w_trend=float(round(w[3], 4)),
                w_risk=base.w_risk,
                value_band=float(np.clip(base.value_band + rng.uniform(-0.05, 0.05), 0.08, 0.35)),
                risk_ratio=float(np.clip(base.risk_ratio + rng.uniform(-0.10, 0.10), 0.90, 1.60)),
                max_lots_per_trade=int(np.clip(base.max_lots_per_trade + rng.integers(-1, 2), 1, 4)),
                lot_size=LOT_SIZE
            ))

    uniq = []
    seen = set()
    for p in params_list:
        key = tuple(asdict(p).items())
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    if len(uniq) > n_local:
        idx = rng.choice(len(uniq), n_local, replace=False)
        uniq = [uniq[i] for i in idx]
    return uniq


def evaluate_param_set(params: DVAParams, tickers_subset, mkt_vol_raw, split=None):
    rows = []
    for tk in tickers_subset:
        df = load_ticker(tk)
        if split == "is":
            df = df.loc[:IS_END].copy()
        elif split == "oos":
            df = df.loc[OOS_START:].copy()

        if df.empty or len(df) < 60:
            continue

        try:
            res = evaluate_one_ticker(df, mkt_vol_raw, params, TOTAL_BUDGET_JP, LOT_SIZE)
        except Exception:
            continue

        rows.append({
            "ticker": tk,
            "A_improvement": res["A_improvement"],
            "B_improvement": res["B_improvement"],
            "A_eff_mix": res["A_eff_mix"],
            "A_eff_dca": res["A_eff_dca"],
            "B_eff_mix": res["B_eff_mix"],
            "B_eff_dca": res["B_eff_dca"],
        })

    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return {"n": 0, "A_mean": np.nan, "A_median": np.nan, "A_winrate": np.nan,
                "B_mean": np.nan, "B_median": np.nan, "B_winrate": np.nan}

    aA = tmp["A_improvement"].dropna()
    aB = tmp["B_improvement"].dropna()

    return {
        "n": len(tmp),
        "A_mean": float(aA.mean()) if len(aA) else np.nan,
        "A_median": float(aA.median()) if len(aA) else np.nan,
        "A_winrate": float((aA > 1).mean()) if len(aA) else np.nan,
        "B_mean": float(aB.mean()) if len(aB) else np.nan,
        "B_median": float(aB.median()) if len(aB) else np.nan,
        "B_winrate": float((aB > 1).mean()) if len(aB) else np.nan,
    }


def run_grid_stage(params_list, name):
    bmk = load_benchmark()
    mkt_vol_raw = calc_annualized_vol(bmk["Close"], window=20)

    full10 = [tk for tk in TICKERS if is_full_10y(load_ticker(tk))]
    rows = []
    t0 = time.time()

    for i, p in enumerate(params_list, 1):
        all_res = evaluate_param_set(p, TICKERS, mkt_vol_raw, split=None)
        full_res = evaluate_param_set(p, full10, mkt_vol_raw, split=None)
        is_res = evaluate_param_set(p, full10, mkt_vol_raw, split="is")
        oos_res = evaluate_param_set(p, full10, mkt_vol_raw, split="oos")

        row = {
            **asdict(p),
            "all_A_mean": all_res["A_mean"],
            "all_A_median": all_res["A_median"],
            "all_A_winrate": all_res["A_winrate"],
            "all_B_mean": all_res["B_mean"],
            "all_B_median": all_res["B_median"],
            "all_B_winrate": all_res["B_winrate"],
            "full_A_mean": full_res["A_mean"],
            "full_B_mean": full_res["B_mean"],
            "is_A_mean": is_res["A_mean"],
            "is_B_mean": is_res["B_mean"],
            "oos_A_mean": oos_res["A_mean"],
            "oos_B_mean": oos_res["B_mean"],
        }
        rows.append(row)

        if i % 25 == 0 or i == 1:
            mid = pd.DataFrame(rows)
            mid["combo_score"] = mid["all_A_mean"].fillna(0) * 0.5 + mid["oos_A_mean"].fillna(mid["all_A_mean"]).fillna(0) * 0.5
            save_df(mid, f"{name}_partial")
            elapsed = time.time() - t0
            eta = elapsed / i * (len(params_list) - i)
            print(f"{name}: {i}/{len(params_list)} elapsed={elapsed:.1f}s eta={eta/60:.1f}min")

    out = pd.DataFrame(rows)
    out["combo_score"] = out["all_A_mean"].fillna(0) * 0.5 + out["oos_A_mean"].fillna(out["all_A_mean"]).fillna(0) * 0.5
    save_df(out, name)
    return out


# =========================================================
# 10. 動的パラメータ
# =========================================================

def build_dynamic_params(features_df, alpha=0.10, beta=0.30, gamma=0.15, delta=0.05, eta=0.15, zeta=0.10):
    f = features_df.set_index("ticker").copy()

    z_sigma = zscore_series(f["sigma"])
    z_corr = zscore_series(f["corr_bmk"])
    z_liq = zscore_series(f["log_vol"])

    params_by_ticker = {}

    for tk, row in f.iterrows():
        s_ = z_sigma.loc[tk]
        c_ = z_corr.loc[tk]
        l_ = z_liq.loc[tk]
        h_ = row["hurst"]

        thr = clip(0.55 - alpha * s_ + beta * (h_ - 0.5), 0.30, 0.75)
        vb = clip(0.20 * (1.0 + gamma * s_), 0.08, 0.35)
        rr = clip(1.20 - delta * c_, 0.90, 1.60)

        adjust = zeta * (0.5 - h_)
        w_shortage = 0.35
        w_bb = 0.15
        w_risk = -0.20
        w_value = 0.35 + adjust
        w_trend = 0.15 - adjust

        total = w_shortage + w_value + w_bb + w_trend
        w_shortage /= total
        w_value /= total
        w_bb /= total
        w_trend /= total

        max_lots = int(np.clip(round(2 + eta * l_), 1, 4))

        params_by_ticker[tk] = DVAParams(
            score_threshold=float(thr),
            w_shortage=float(w_shortage),
            w_value=float(w_value),
            w_bb=float(w_bb),
            w_trend=float(w_trend),
            w_risk=float(w_risk),
            value_band=float(vb),
            risk_ratio=float(rr),
            max_lots_per_trade=max_lots,
            lot_size=LOT_SIZE
        )

    return params_by_ticker


def run_dynamic_search():
    features = pd.read_pickle(os.path.join(RESULT_DIR, "features.pkl"))
    bmk = load_benchmark()
    mkt_vol_raw = calc_annualized_vol(bmk["Close"], window=20)

    alphas = [0.0, 0.10, 0.20]
    betas  = [0.0, 0.30, 0.60]
    gammas = [0.0, 0.15, 0.30]
    deltas = [0.0, 0.05]
    etas   = [0.0, 0.15, 0.30]
    zetas  = [0.0, 0.10]

    rows = []
    total = len(alphas)*len(betas)*len(gammas)*len(deltas)*len(etas)*len(zetas)

    for i, (a, b, g, d, e, z) in enumerate(itertools.product(alphas, betas, gammas, deltas, etas, zetas), 1):
        pmap = build_dynamic_params(features, alpha=a, beta=b, gamma=g, delta=d, eta=e, zeta=z)
        vals_A, vals_B = [], []

        for tk in TICKERS:
            df = load_ticker(tk)
            res = evaluate_one_ticker(df, mkt_vol_raw, pmap[tk], TOTAL_BUDGET_JP, LOT_SIZE)
            if pd.notna(res["A_improvement"]):
                vals_A.append(res["A_improvement"])
            if pd.notna(res["B_improvement"]):
                vals_B.append(res["B_improvement"])

        rows.append({
            "alpha": a, "beta": b, "gamma": g, "delta": d, "eta": e, "zeta": z,
            "A_mean": float(np.mean(vals_A)) if vals_A else np.nan,
            "A_median": float(np.median(vals_A)) if vals_A else np.nan,
            "A_winrate": float(np.mean(np.array(vals_A) > 1)) if vals_A else np.nan,
            "B_mean": float(np.mean(vals_B)) if vals_B else np.nan,
            "B_median": float(np.median(vals_B)) if vals_B else np.nan,
            "B_winrate": float(np.mean(np.array(vals_B) > 1)) if vals_B else np.nan,
        })

        if i % 20 == 0 or i == 1:
            save_df(pd.DataFrame(rows), "dynamic_search_partial")
            print(f"dynamic {i}/{total}")

    out = pd.DataFrame(rows)
    save_df(out, "dynamic_search")
    return out


# =========================================================
# 11. 統計
# =========================================================

def paired_test(mix_vals, dca_vals):
    mix = np.asarray(mix_vals, dtype=float)
    dca = np.asarray(dca_vals, dtype=float)
    mask = np.isfinite(mix) & np.isfinite(dca)
    mix = mix[mask]
    dca = dca[mask]

    if len(mix) < 3:
        return {"n": len(mix), "wilcoxon_p": np.nan, "t_p": np.nan, "cohen_d": np.nan}

    diff = mix - dca

    try:
        _, wilcoxon_p = stats.wilcoxon(mix, dca, alternative="greater")
    except Exception:
        wilcoxon_p = np.nan

    try:
        _, t_p = stats.ttest_rel(mix, dca, alternative="greater")
    except Exception:
        t_p = np.nan

    cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

    return {
        "n": int(len(mix)),
        "wilcoxon_p": float(wilcoxon_p) if pd.notna(wilcoxon_p) else np.nan,
        "t_p": float(t_p) if pd.notna(t_p) else np.nan,
        "cohen_d": float(cohen_d) if pd.notna(cohen_d) else np.nan,
        "diff_mean": float(diff.mean()),
        "diff_median": float(np.median(diff)),
    }


def bootstrap_ci(values, n_boot=3000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"stat": np.nan, "lo": np.nan, "hi": np.nan}

    n = len(v)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = v[rng.integers(0, n, n)]
        boots[i] = np.mean(sample)

    return {
        "stat": float(np.mean(v)),
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
    }


def summarize_group(df, label, imp_col, eff_mix_col, eff_dca_col):
    vals = df[imp_col].dropna()
    test = paired_test(df[eff_mix_col], df[eff_dca_col])
    ci = bootstrap_ci(vals)

    return {
        "group": label,
        "metric": imp_col,
        "n": int(vals.shape[0]),
        "mean_improvement": float(vals.mean()) if len(vals) else np.nan,
        "median_improvement": float(vals.median()) if len(vals) else np.nan,
        "std_improvement": float(vals.std()) if len(vals) else np.nan,
        "min_improvement": float(vals.min()) if len(vals) else np.nan,
        "max_improvement": float(vals.max()) if len(vals) else np.nan,
        "winrate": float((vals > 1).mean()) if len(vals) else np.nan,
        "bootstrap_mean": ci["stat"],
        "bootstrap_lo95": ci["lo"],
        "bootstrap_hi95": ci["hi"],
        "wilcoxon_p": test["wilcoxon_p"],
        "paired_t_p": test["t_p"],
        "cohen_d": test["cohen_d"],
        "eff_diff_mean": test.get("diff_mean", np.nan),
        "eff_diff_median": test.get("diff_median", np.nan),
    }


def build_academic_report(baseline_df, grid_df=None, dynamic_df=None):
    rows = []

    all_df = baseline_df.copy()
    full_df = baseline_df[baseline_df["full_10y"]].copy()
    short_df = baseline_df[~baseline_df["full_10y"]].copy()

    for gname, gdf in [("all_105", all_df), ("full_10y", full_df), ("short_history", short_df)]:
        rows.append(summarize_group(gdf, gname, "A_improvement", "A_eff_mix", "A_eff_dca"))
        rows.append(summarize_group(gdf, gname, "B_improvement", "B_eff_mix", "B_eff_dca"))

    out = pd.DataFrame(rows)
    save_df(out, "academic_report_baseline")

    text = []
    text.append("=== Baseline Academic Summary ===")
    for _, r in out.iterrows():
        text.append(
            f"{r['group']} | {r['metric']} | n={r['n']} | "
            f"mean={r['mean_improvement']:.4f} | median={r['median_improvement']:.4f} | "
            f"winrate={r['winrate']:.4%} | CI95=({r['bootstrap_lo95']:.4f}, {r['bootstrap_hi95']:.4f}) | "
            f"wilcoxon_p={r['wilcoxon_p']:.6g} | t_p={r['paired_t_p']:.6g} | d={r['cohen_d']:.4f}"
        )

    if grid_df is not None and not grid_df.empty:
        best = grid_df.sort_values("combo_score", ascending=False).head(10)
        save_df(best, "academic_report_grid_top10")
        text.append("")
        text.append("=== Grid Top10 by combo_score ===")
        text.append(best.to_string(index=False))

    if dynamic_df is not None and not dynamic_df.empty:
        best_dyn = dynamic_df.sort_values("A_mean", ascending=False).head(10)
        save_df(best_dyn, "academic_report_dynamic_top10")
        text.append("")
        text.append("=== Dynamic Top10 by A_mean ===")
        text.append(best_dyn.to_string(index=False))

    save_text("\n".join(text), "academic_report_summary")
    return out


# =========================================================
# 12. 実行入口
# =========================================================

def build_only():
    print("module loaded only")


def run_basics():
    fetch_all_data()
    features = extract_features()
    baseline = run_baseline()
    build_academic_report(baseline_df=baseline)
    return features, baseline


def run_grid_priority():
    fetch_all_data()
    features = extract_features()
    baseline = run_baseline()

    stage1_params = build_stage1_grid()
    stage1 = run_grid_stage(stage1_params, name="grid_stage1")

    top5 = stage1.sort_values("combo_score", ascending=False).head(5)
    save_df(top5, "grid_stage1_top5")

    stage2_params = build_stage2_grid(top5, n_local=200, delta=0.05)
    stage2 = run_grid_stage(stage2_params, name="grid_stage2")

    merged = pd.concat([stage1, stage2], ignore_index=True)
    merged["combo_score"] = merged["all_A_mean"].fillna(0) * 0.5 + merged["oos_A_mean"].fillna(merged["all_A_mean"]).fillna(0) * 0.5
    save_df(merged, "grid_all")

    build_academic_report(baseline_df=baseline, grid_df=merged)
    return merged


def run_dynamic_only():
    fetch_all_data()
    features = extract_features()
    baseline = run_baseline()
    dynamic = run_dynamic_search()
    build_academic_report(baseline_df=baseline, dynamic_df=dynamic)
    return dynamic


def run_all():
    fetch_all_data()
    features = extract_features()
    baseline = run_baseline()

    stage1_params = build_stage1_grid()
    stage1 = run_grid_stage(stage1_params, name="grid_stage1")
    top5 = stage1.sort_values("combo_score", ascending=False).head(5)
    save_df(top5, "grid_stage1_top5")

    stage2_params = build_stage2_grid(top5, n_local=200, delta=0.05)
    stage2 = run_grid_stage(stage2_params, name="grid_stage2")

    grid_all = pd.concat([stage1, stage2], ignore_index=True)
    grid_all["combo_score"] = grid_all["all_A_mean"].fillna(0) * 0.5 + grid_all["oos_A_mean"].fillna(grid_all["all_A_mean"]).fillna(0) * 0.5
    save_df(grid_all, "grid_all")

    dynamic = run_dynamic_search()
    build_academic_report(baseline_df=baseline, grid_df=grid_all, dynamic_df=dynamic)

    return {
        "features": features,
        "baseline": baseline,
        "grid_all": grid_all,
        "dynamic": dynamic,
    }


if __name__ == "__main__":
    # 実行を誤爆しないよう、デフォルトは build_only
    # 必要なときだけ下を差し替える:
    # run_basics()
    # run_grid_priority()
    # run_dynamic_only()
    # run_all()
    build_only()
