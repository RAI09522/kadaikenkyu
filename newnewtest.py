# newtest.py
import pandas as pd
import numpy as np
import yfinance as yf
from itertools import product

CONFIG = {
    "start_date":"2014-01-01","end_date":"2023-12-31",
    "total_budget_jp":6000000,"ticker":"8088.T","benchmark":"^N225",
    "lot_size":100,"max_lots_per_trade":3,"trading_days_yr":252,
    "use_dow_shield":True,
    "default_weights":{"shortage":0.35,"value":0.35,"bb":0.15,"trend":0.15,"risk":0.20},
    "grid_search_params":{
        "window_size":[20,50],
        "bb_sd_mult":[2.0,2.5],
        "value_drop_ratio":[0.15,0.20],
        "score_threshold":[0.40,0.45],
        "execution_mix_lambda":[0.5,0.8]
    }
}

def clip(x,l,h): return max(l,min(h,x))

def normalize(df):
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=df.columns.get_level_values(0)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"]=df["Adj Close"]
    if "High" not in df.columns: df["High"]=df["Close"]
    if "Low" not in df.columns: df["Low"]=df["Close"]
    return df

def calc_annualized_vol(s,w,d):
    return s.pct_change(fill_method=None).rolling(w).std()*np.sqrt(d)

def calc_indicators(df,mkt,win,bb,days):
    o=df.copy()
    o["mkt_vol"]=mkt.reindex(o.index).ffill().fillna(mkt.mean())
    o["vol"]=calc_annualized_vol(o["Close"],win,days)
    o["ema"]=o["Close"].ewm(span=win,adjust=False).mean()
    o["bb_sd"]=o["Close"].rolling(win).std()
    o["bb_low"]=o["ema"]-o["bb_sd"]*bb
    o["recent_high"]=o["High"].rolling(win).max()
    o["prev_high"]=o["recent_high"].shift(win)
    o["recent_low"]=o["Low"].rolling(win).min()
    o["prev_low"]=o["recent_low"].shift(win)
    o["dow_downtrend"]=(o["recent_high"]<o["prev_high"])&(o["recent_low"]<o["prev_low"])
    return o

def get_month_groups(df):
    return [g for _,g in df.groupby(df.index.to_period("M"))]

def build_monthly_contribution_schedule(df,total):
    s=pd.Series(0.0,index=df.index)
    groups=get_month_groups(df)
    m=total/len(groups) if groups else 0
    for g in groups:
        if not g.empty: s.loc[g.index[0]]=m
    return s

def calc_dva_score_row(price,ema,bb_low,vol,mkt,contrib,shares,w,down,vdrop,r=1.2):
    if pd.isna(price) or pd.isna(ema) or ema<=0:return 0
    ideal=contrib/ema
    shortage=clip((ideal-shares)/ideal,0,1) if ideal>0 else 0
    value=clip((ema-price)/(vdrop*ema),0,1)
    bb=1 if pd.notna(bb_low) and price<bb_low else 0
    risk=clip((vol/(r*mkt))-1,0,1) if pd.notna(vol) and mkt>0 else 0
    trend=1 if price<ema else 0
    score=w["shortage"]*shortage+w["value"]*value+w["bb"]*bb+w["trend"]*trend-w["risk"]*risk
    if CONFIG["use_dow_shield"] and down: score=0
    return clip(score,0,1)

def execute_dva_for_grid(df,sched,lot,maxlots,p):
    shares=0;spent=0.0;cash=0.0;contrib=0.0
    last={g.index[-1] for g in get_month_groups(df) if not g.empty}
    for dt in df.index:
        price=float(df.loc[dt,"Close"]); lot_cost=price*lot
        cash+=float(sched.loc[dt]); contrib+=float(sched.loc[dt])
        score=calc_dva_score_row(price,df.loc[dt,"ema"],df.loc[dt,"bb_low"],df.loc[dt,"vol"],df.loc[dt,"mkt_vol"],contrib,shares,CONFIG["default_weights"],bool(df.loc[dt,"dow_downtrend"]),p["value_drop_ratio"])
        afford=int(cash//lot_cost); buy=0
        if afford>0 and score>=p["score_threshold"]:
            qs=clip((score-p["score_threshold"])/(1-p["score_threshold"]),0,1)
            qc=clip((afford-1)/(maxlots-1),0,1) if maxlots>1 else 1
            q=p["execution_mix_lambda"]*qs+(1-p["execution_mix_lambda"])*qc
            buy=min(afford,1+int(np.floor(q*(maxlots-1))))
        if buy:
            sh=buy*lot; cost=sh*price; shares+=sh; spent+=cost; cash-=cost
        if dt in last:
            sw=int(cash//lot_cost)
            if sw:
                sh=sw*lot; cost=sh*price; shares+=sh; spent+=cost; cash-=cost
    return shares,(spent/shares if shares else 0)

def run_grid_search():
    d=normalize(yf.download(CONFIG["ticker"],start=CONFIG["start_date"],end=CONFIG["end_date"],auto_adjust=True,progress=False))
    b=normalize(yf.download(CONFIG["benchmark"],start=CONFIG["start_date"],end=CONFIG["end_date"],auto_adjust=True,progress=False))
    if d.empty or b.empty: raise ValueError("データ取得失敗")
    sched=build_monthly_contribution_schedule(d,CONFIG["total_budget_jp"])
    keys,vals=zip(*CONFIG["grid_search_params"].items())
    res=[]
    for v in product(*vals):
        p=dict(zip(keys,v))
        ind=calc_indicators(d,calc_annualized_vol(b["Close"],p["window_size"],CONFIG["trading_days_yr"]),p["window_size"],p["bb_sd_mult"],CONFIG["trading_days_yr"])
        sh,avg=execute_dva_for_grid(ind,sched,CONFIG["lot_size"],CONFIG["max_lots_per_trade"],p)
        p["取得株数"]=sh;p["平均単価"]=avg;res.append(p)
    out=pd.DataFrame(res).sort_values(["取得株数","平均単価"],ascending=[False,True])
    out.to_csv("grid_search_result.csv",index=False,encoding="utf-8-sig")
    print(out.head())

if __name__=="__main__":
    run_grid_search()
