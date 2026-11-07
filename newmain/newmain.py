# -*- coding: utf-8 -*-
"""
DVA/DCA 比較シミュレータ v3 (査読指摘対応・170銘柄/TOPIX17業種対応版)
====================================================================
v2 からの変更:
  [V1] 銘柄ユニバースを外部CSV(tickers.csv)から読込: 列 = ticker,sector
       - 例: 7203.T,輸送用機器  (TOPIX17業種名は任意文字列でOK)
       - CSVが無い場合は内蔵の旧105銘柄リスト(sector="unknown")で動作
  [V2] 業種別(sector)集計を summarize() に追加
  [V3] ユニバースメタデータに選抜規則の記述欄を追加 (SELECTION_RULE を編集して明記)
  [V4] 生存バイアス・短履歴の扱いを fetch_meta.json に自動記録

v2 で対応済みの査読指摘:
  [M1] 統計単位 = 1銘柄1観測
  [M2] full_10y / short_history 層別集計
  [M3] DCA-B = 端株許容ノーフリクション月次DCA (ルックアヘッド排除)
  [M4] パラメータ選抜は IS(2016-2020) のみ。OOS は報告専用
  [M5] 取引コスト(デフォルト0.1%)。コスト0も併記
  [M6] 再現性メタデータJSON出力

使い方:
  1) 同じフォルダに tickers.csv を置く (列: ticker,sector)
  2) pip install yfinance pandas numpy scipy
  3) python3 simulator_v3.py baseline   # ベースライン評価 (数分)
     python3 simulator_v3.py grid       # グリッド探索 (数時間規模)
"""
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache"
RESULTS = BASE / "results"
for d in (CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- 研究設定 (ex ante 固定) ----------------
START = "2016-01-01"
END = "2025-12-31"          # 終了日(含む)
FETCH_END = "2026-01-01"    # yfinance endは排他的
IS_END = "2020-12-31"       # In-Sample 終了 (パラメータ選抜はここまでの期間のみ)
BUDGET_TOTAL = 6_000_000    # 総予算 600万円
N_MONTHS = 120              # 10年 x 12ヶ月
MONTHLY_BUDGET = BUDGET_TOTAL / N_MONTHS  # 50,000円/月
LOT = 100                   # 100株単位 (JPX標準売買単位, 2018-10統一)
COST_RATE = 0.001           # 売買手数料 0.1% (買付時)
FULL_10Y_MIN_DAYS = 2400    # 10年フル判定: 営業日数の閾値
FULL_10Y_LATEST_FIRST = "2016-03-31"  # または初回取引日がこれ以前

# ★論文用: 選抜規則をここに記述 (fetch_meta.json に自動記録される)
SELECTION_RULE = (
    "TOPIX17業種の各業種から10銘柄ずつ抽出した計170銘柄。"
    "業種内の選抜基準: 【ここに記入: 例) 2016-01-01時点の時価総額上位10社】"
)

# 内蔵フォールバック: 旧105銘柄 (tickers.csv が無い場合のみ使用)
FALLBACK_TICKERS = [
    "1443.T","4733.T","3046.T","4519.T","3994.T","3445.T","4194.T","9603.T","3659.T","4684.T",
    "7832.T","7550.T","9766.T","7974.T","6702.T","4307.T","9843.T","3349.T","2791.T","9627.T",
    "4478.T","464A.T","9158.T","7699.T","4375.T","6574.T","2986.T","5590.T","6030.T","2586.T",
    "6094.T","2998.T","6521.T","9162.T","3652.T","4170.T","9348.T","2160.T","4582.T","7777.T",
    "9166.T","7806.T","3479.T","3491.T","4575.T","4592.T","3773.T","7803.T","4628.T","2782.T",
    "8871.T","8928.T","3437.T","2790.T","3944.T","3690.T","2689.T","4624.T","3710.T","3951.T",
    "3958.T","4623.T","3955.T","3067.T","3484.T","3020.T","3435.T","3134.T","3943.T","3698.T",
    "4635.T","3096.T","3426.T","3667.T","3080.T","3011.T","8844.T","2991.T","8869.T","8904.T",
    "2370.T","4588.T","4013.T","4893.T","4881.T","3911.T","2385.T","4892.T","6999.T","8388.T",
    "6532.T","6460.T","9519.T","9601.T","8362.T","6961.T","4911.T","7383.T","5032.T","9310.T",
    "8084.T","3778.T","4443.T","2726.T","7203.T",
]

# ---------------- ユニバース読込 ----------------
def load_universe():
    """tickers.csv (ticker,sector) があればそれを使用。無ければ内蔵105銘柄。"""
    fp = BASE / "tickers.csv"
    if fp.exists():
        u = pd.read_csv(fp, dtype=str)
        u.columns = [c.strip().lower() for c in u.columns]
        assert "ticker" in u.columns, "tickers.csv に ticker 列が必要です"
        if "sector" not in u.columns:
            u["sector"] = "unknown"
        u["ticker"] = u["ticker"].str.strip()
        u["sector"] = u["sector"].fillna("unknown").str.strip()
        u = u.drop_duplicates("ticker").reset_index(drop=True)
        print(f"universe: {len(u)} tickers from tickers.csv, sectors={u['sector'].nunique()}")
        return u
    print(f"universe: fallback {len(FALLBACK_TICKERS)} tickers (sector=unknown)")
    return pd.DataFrame({"ticker": FALLBACK_TICKERS, "sector": "unknown"})

# ---------------- データ取得 ----------------
def fetch_all(universe: pd.DataFrame, refresh: bool = False) -> dict:
    import yfinance as yf
    data, failed = {}, []
    tickers = universe["ticker"].tolist()
    for tk in tickers + ["^N225"]:
        fp = CACHE / f"{tk.replace('^','IDX_')}.pkl"
        if fp.exists() and not refresh:
            data[tk] = pd.read_pickle(fp)
            continue
        try:
            df = yf.download(tk, start=START, end=FETCH_END, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if len(df) < 250:
                failed.append((tk, f"too_short:{len(df)}"))
                continue
            df.to_pickle(fp)
            data[tk] = df
        except Exception as e:
            failed.append((tk, str(e)))
        time.sleep(0.15)
    sec_counts = universe["sector"].value_counts().to_dict()
    meta = {
        "fetch_date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": "Yahoo Finance via yfinance (auto_adjust=True: 分割・配当調整済み終値)",
        "period": [START, END],
        "is_oos_split": IS_END,
        "budget_total_jpy": BUDGET_TOTAL, "monthly_budget_jpy": MONTHLY_BUDGET,
        "lot_size": LOT, "cost_rate": COST_RATE,
        "n_tickers_requested": len(tickers),
        "n_tickers_ok": len([k for k in data if k != "^N225"]),
        "sector_counts": sec_counts,
        "failed": failed,
        "selection_rule": SELECTION_RULE,
        "limitations": [
            "生存バイアス: 現在上場中の銘柄を過去に遡って評価。上場廃止・業種再編で消えた銘柄は含まない。",
            "短履歴銘柄: 上場が2016年以降の銘柄は full_10y=False として層別集計。",
            "yfinance は研究・教育用途のOSSラッパー。正式な市場データベンダーではない。",
        ],
    }
    (RESULTS / "fetch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return data

# ---------------- 月次評価エンジン ----------------
def month_groups(df: pd.DataFrame, start=None, end=None):
    sub = df
    if start: sub = sub[sub.index > pd.Timestamp(start)]
    if end:   sub = sub[sub.index <= pd.Timestamp(end)]
    if len(sub) == 0:
        return []
    return list(sub.groupby(sub.index.to_period("M")))

def sim_dca_A(df, cost=COST_RATE):
    """DCA-A: 現実的月次DCA。月初値で100株単位購入、不足分は現金繰越。"""
    cash, shares, spent, trades = 0.0, 0, 0.0, 0
    for _, gg in month_groups(df):
        cash += MONTHLY_BUDGET
        price = float(gg["Close"].iloc[0])
        lot_cost = price * LOT * (1 + cost)
        lots = int(cash // lot_cost)
        if lots > 0:
            pay = lots * price * LOT
            shares += lots * LOT
            spent += pay * (1 + cost)
            cash -= pay * (1 + cost)
            trades += 1
    p_end = float(df["Close"].iloc[-1])
    return {"shares": shares, "spent": spent, "cash": cash, "trades": trades,
            "p_end": p_end, "value": shares * p_end + cash}

def sim_dca_B(df):
    """DCA-B(再定義): 端株許容・ノーフリクション月次定額購入。将来情報を含まない。"""
    shares = 0.0
    for _, gg in month_groups(df):
        shares += MONTHLY_BUDGET / float(gg["Close"].iloc[0])
    p_end = float(df["Close"].iloc[-1])
    return {"shares": shares, "spent": BUDGET_TOTAL, "p_end": p_end, "value": shares * p_end}

def dva_score_series(df):
    """DVAスコア構成要素 (時点tまでの情報のみ使用)"""
    c = df["Close"].astype(float)
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    ma60 = c.rolling(60).mean()
    out = pd.DataFrame(index=df.index)
    out["value"] = (ma20 - c) / ma20
    z = (c - ma20) / sd20.replace(0, np.nan)
    out["bb"] = (-z / 2.0).clip(-1, 1)
    out["trend"] = ((ma20 - ma60) / ma60).clip(-0.05, 0.05) / 0.05
    return out.fillna(0.0)

def sim_dva_mix(df, params, cost=COST_RATE):
    """DVA混合: 月次ベース買付 + スコア>=閾値のとき追加買付 (100株単位)"""
    sc = dva_score_series(df)
    cash, shares, spent, trades, dva_trades = 0.0, 0, 0.0, 0, 0
    cum_target = 0.0
    for _, gg in month_groups(df):
        cash += MONTHLY_BUDGET
        cum_target += MONTHLY_BUDGET
        row = sc.loc[gg.index[0]]
        price = float(gg["Close"].iloc[0])
        port_val = shares * price
        shortage = max(0.0, (cum_target - port_val) / cum_target) if cum_target > 0 else 0.0
        score = (params["w_shortage"] * shortage
                 + params["w_value"] * row["value"] * 5
                 + params["w_bb"] * row["bb"]
                 + params["w_trend"] * row["trend"]
                 + params["w_risk"] * max(0.0, -row["trend"]))
        lot_cost = price * LOT * (1 + cost)
        if cash >= lot_cost:  # ベース買付
            pay = price * LOT
            shares += LOT; spent += pay * (1 + cost); cash -= pay * (1 + cost); trades += 1
        if (score >= params["score_threshold"]
                and shares * price < cum_target * params.get("deploy_cap", 1.5)):
            add_lots = min(params["max_lots_per_trade"], int(cash // lot_cost))
            if add_lots > 0:
                pay = add_lots * price * LOT
                shares += add_lots * LOT; spent += pay * (1 + cost)
                cash -= pay * (1 + cost); dva_trades += 1
    p_end = float(df["Close"].iloc[-1])
    return {"shares": shares, "spent": spent, "cash": cash, "trades": trades,
            "dva_trades": dva_trades, "p_end": p_end, "value": shares * p_end + cash}

# ---------------- 特徴量 ----------------
def hurst_exponent(ts, max_lag=100):
    ts = np.asarray(ts, dtype=float)
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    tau = np.where(tau <= 0, 1e-12, tau)
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])

def extract_features(tk, df, bmk):
    c = df["Close"].astype(float)
    r = c.pct_change().dropna()
    rb = bmk["Close"].astype(float).pct_change().dropna()
    j = r.to_frame("r").join(rb.to_frame("b"), how="inner").dropna()
    return {"ticker": tk, "sigma": float(r.std() * np.sqrt(252)),
            "hurst": hurst_exponent(c.values[-500:]) if len(c) >= 500 else np.nan,
            "corr_bmk": float(j["r"].corr(j["b"])) if len(j) > 60 else np.nan,
            "mu": float(r.mean() * 252),
            "mdd": float((c / c.cummax() - 1.0).min()),
            "log_vol": float(np.log(df["Volume"].astype(float).replace(0, np.nan).mean())),
            "n_days": int(len(df)), "first_date": str(df.index[0].date())}

# ---------------- 1銘柄評価 (1銘柄=1観測) ----------------
def evaluate_ticker(df, params, cost=COST_RATE):
    out = {}
    for tag, (s, e) in {"all": (None, None), "is": (None, IS_END), "oos": (IS_END, None)}.items():
        n_m = len(month_groups(df, s, e))
        if n_m < 6:
            out[f"{tag}_ok"] = False
            continue
        sub = df
        if s: sub = sub[sub.index > pd.Timestamp(s)]
        if e: sub = sub[sub.index <= pd.Timestamp(e)]
        a = sim_dca_A(sub, cost)
        b = sim_dca_B(sub)
        d = sim_dva_mix(sub, params, cost)
        budget_period = MONTHLY_BUDGET * n_m
        eff_a = a["value"] / budget_period
        eff_b = b["value"] / budget_period
        eff_d = d["value"] / budget_period
        out[f"{tag}_ok"] = True
        out[f"{tag}_n_months"] = n_m
        out[f"{tag}_A_improvement"] = eff_d / eff_a if eff_a > 0 else np.nan
        out[f"{tag}_B_improvement"] = eff_d / eff_b if eff_b > 0 else np.nan
        if tag == "all":
            out.update({"dva_shares": d["shares"], "dva_trades": d["trades"],
                        "dva_extra_trades": d["dva_trades"], "dcaA_trades": a["trades"],
                        "P_end": d["p_end"]})
    return out

# ---------------- 集計・検定 ----------------
def _summarize_block(sub, grp_name, rows):
    from scipy import stats
    for metric in ["all_A_improvement", "all_B_improvement",
                   "is_A_improvement", "oos_A_improvement",
                   "is_B_improvement", "oos_B_improvement"]:
        v = sub[metric].dropna().astype(float)
        if len(v) < 3:
            rows.append({"group": grp_name, "metric": metric, "n": int(len(v))})
            continue
        diff = v - 1.0
        try:
            w_p = stats.wilcoxon(diff).pvalue
        except Exception:
            w_p = np.nan
        rng = np.random.default_rng(42)
        boot = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(2000)]
        rows.append({"group": grp_name, "metric": metric, "n": int(len(v)),
                     "mean": v.mean(), "median": v.median(), "std": v.std(),
                     "min": v.min(), "max": v.max(),
                     "winrate": float((v > 1).mean()),
                     "ci_lo": float(np.percentile(boot, 2.5)),
                     "ci_hi": float(np.percentile(boot, 97.5)),
                     "wilcoxon_p": w_p, "t_p": stats.ttest_1samp(v, 1.0).pvalue,
                     "cohen_d": float(diff.mean() / diff.std()) if diff.std() > 0 else np.nan})

def summarize(df_res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    _summarize_block(df_res, "all", rows)
    _summarize_block(df_res[df_res["full_10y"]], "full_10y", rows)
    _summarize_block(df_res[~df_res["full_10y"]], "short_history", rows)
    # 業種別 (TOPIX17業種など)
    for sec, sub in df_res.groupby("sector"):
        _summarize_block(sub, f"sector:{sec}", rows)
    return pd.DataFrame(rows)

# ---------------- パイプライン ----------------
BASELINE_PARAMS = {"score_threshold": 0.55, "w_shortage": 0.35, "w_value": 0.35,
                   "w_bb": 0.15, "w_trend": 0.15, "w_risk": -0.20,
                   "max_lots_per_trade": 2, "deploy_cap": 1.5}

def run_baseline(data, universe, cost=COST_RATE, tag=""):
    rows = []
    sec_map = dict(zip(universe["ticker"], universe["sector"]))
    for tk in universe["ticker"]:
        if tk not in data:
            continue
        df = data[tk]
        r = evaluate_ticker(df, BASELINE_PARAMS, cost)
        r["ticker"] = tk
        r["sector"] = sec_map.get(tk, "unknown")
        r["full_10y"] = bool(len(df) >= FULL_10Y_MIN_DAYS or
                             str(df.index[0].date()) <= FULL_10Y_LATEST_FIRST)
        r["n_days"] = len(df)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / f"baseline_v3{tag}.csv", index=False)
    summ = summarize(res)
    summ.to_csv(RESULTS / f"baseline_v3_summary{tag}.csv", index=False)
    return res, summ

def grid_params():
    """IS選抜用の粗グリッド (過学習抑制のため候補数を抑制)"""
    gs = []
    for thr in [0.35, 0.40, 0.45, 0.55, 0.65]:
        for ws in [0.35, 0.50, 0.65]:
            for wv in [0.05, 0.15, 0.30]:
                for wb in [0.05, 0.15]:
                    wt = round(1.0 - ws - wv - wb, 4)
                    if wt < 0 or wt > 0.35:
                        continue
                    for wr in [-0.2, -0.1]:
                        for ml in [1, 2, 3]:
                            gs.append({"score_threshold": thr, "w_shortage": ws,
                                       "w_value": wv, "w_bb": wb, "w_trend": wt,
                                       "w_risk": wr, "max_lots_per_trade": ml,
                                       "deploy_cap": 1.5})
    return gs

def run_grid(data, universe, cost=COST_RATE):
    """グリッド探索: 選抜は IS のみ (select_score = is_A_mean)。OOS は報告専用。"""
    gs = grid_params()
    rows, t0 = [], time.time()
    tickers = [tk for tk in universe["ticker"] if tk in data]
    for pi, p in enumerate(gs):
        isA, oosA, isB, oosB = [], [], [], []
        for tk in tickers:
            r = evaluate_ticker(data[tk], p, cost)
            if r.get("is_ok"):
                isA.append(r["is_A_improvement"]); isB.append(r["is_B_improvement"])
            if r.get("oos_ok"):
                oosA.append(r["oos_A_improvement"]); oosB.append(r["oos_B_improvement"])
        rows.append({**p,
                     "is_A_mean": float(np.nanmean(isA)),
                     "is_A_win": float(np.mean(np.array(isA) > 1)),
                     "oos_A_mean": float(np.nanmean(oosA)),
                     "oos_A_win": float(np.mean(np.array(oosA) > 1)),
                     "is_B_mean": float(np.nanmean(isB)),
                     "oos_B_mean": float(np.nanmean(oosB))})
        if (pi + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(RESULTS / "grid_v3_partial.csv", index=False)
            print(f"grid {pi+1}/{len(gs)} elapsed={time.time()-t0:.0f}s", flush=True)
    gdf = pd.DataFrame(rows)
    gdf["select_score"] = gdf["is_A_mean"]  # ★選抜基準はISのみ (OOS非使用)
    gdf = gdf.sort_values("select_score", ascending=False)
    gdf.to_csv(RESULTS / "grid_v3.csv", index=False)
    return gdf

def run_features(data, universe):
    rows = []
    bmk = data.get("^N225")
    for tk in universe["ticker"]:
        if tk in data and bmk is not None:
            rows.append(extract_features(tk, data[tk], bmk))
    fdf = pd.DataFrame(rows)
    fdf.to_csv(RESULTS / "features_v3.csv", index=False)
    return fdf

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    universe = load_universe()
    data = fetch_all(universe)
    print(f"data ok: {len(data)-1}/{len(universe)} tickers")
    if mode == "baseline":
        res, summ = run_baseline(data, universe, cost=COST_RATE, tag="")
        res0, summ0 = run_baseline(data, universe, cost=0.0, tag="_nocost")
        fdf = run_features(data, universe)
        print(summ.to_string())
    elif mode == "grid":
        gdf = run_grid(data, universe)
        print(gdf.head(10).to_string())
    elif mode == "features":
        print(run_features(data, universe).describe().to_string())
