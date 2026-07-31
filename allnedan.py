# -*- coding: utf-8 -*-
"""
指定日時の東証銘柄価格を一括取得するスクリプト

使い方例:
  1) 日足（その日の終値）
     python bulk_price_tse.py --datetime "2026-07-30"

  2) 日時指定（近い足を返す）
     python bulk_price_tse.py --datetime "2026-07-30 10:15" --interval 1m

  3) CSV保存先を指定
     python bulk_price_tse.py --datetime "2026-07-30 10:15" --interval 5m --output prices.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


TOKYO_TZ = ZoneInfo("Asia/Tokyo")

RAW_TICKERS = [
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
    "8084.T","3778.T","4443.T","2726.T",
    # 重複分が貼られていたので、この下は入れなくてOK
]


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


TICKERS = unique_keep_order(RAW_TICKERS)


def parse_target_datetime(dt_str: str) -> tuple[datetime, bool]:
    """
    returns:
      target_dt_jst, has_time
    """
    dt_str = dt_str.strip()
    formats_with_time = [
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    date_only_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]

    for fmt in formats_with_time:
        try:
            dt = datetime.strptime(dt_str, fmt).replace(tzinfo=TOKYO_TZ)
            return dt, True
        except ValueError:
            pass

    for fmt in date_only_formats:
        try:
            dt = datetime.strptime(dt_str, fmt).replace(tzinfo=TOKYO_TZ)
            return dt, False
        except ValueError:
            pass

    raise ValueError(
        "日時フォーマット不正です。例: '2026-07-30' または '2026-07-30 10:15'"
    )


def choose_intraday_interval(target_dt: datetime, preferred: str | None) -> str:
    """
    Yahoo Finance / yfinance の一般的な制限を踏まえた簡易選択:
      - 1m は直近7日程度
      - intraday(<1d) は直近60日程度
    """
    now_jst = datetime.now(TOKYO_TZ)
    age = now_jst - target_dt

    valid_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}

    if preferred:
        if preferred not in valid_intervals:
            raise ValueError(f"interval は {sorted(valid_intervals)} のいずれかです: {preferred}")

        if preferred == "1m" and age > timedelta(days=7):
            raise ValueError("1m は通常、直近7日程度までしか取得できません。")
        if preferred in valid_intervals and preferred != "1d" and age > timedelta(days=60):
            raise ValueError("1日未満の足は通常、直近60日程度までしか取得できません。")
        return preferred

    # 自動選択
    if age <= timedelta(days=7):
        return "1m"
    elif age <= timedelta(days=60):
        return "5m"
    else:
        raise ValueError(
            "この日時は古すぎるため intraday 取得不可の可能性が高いです。"
            "日付だけ指定して日足終値を取得するか、別データソースを使ってください。"
        )


def flatten_downloaded(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    yf.download の返り値を ticker -> DataFrame に揃える
    """
    out = {}

    if df.empty:
        return out

    # 複数銘柄時は MultiIndex になりやすい
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))

        # [field, ticker] か [ticker, field] かの両対応
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if set(df.columns.get_level_values(0)) & fields:
            # columns = (field, ticker)
            for t in tickers:
                if t in level1:
                    sub = df.xs(t, axis=1, level=1).copy()
                    if not sub.empty:
                        out[t] = sub
        else:
            # columns = (ticker, field)
            for t in tickers:
                if t in level0:
                    sub = df[t].copy()
                    if not sub.empty:
                        out[t] = sub
    else:
        # 単一銘柄ケース
        if len(tickers) == 1:
            out[tickers[0]] = df.copy()

    return out


def get_daily_prices(target_dt: datetime, tickers: list[str]) -> pd.DataFrame:
    """
    指定日の終値ベース。休場日は直前営業日を返す。
    """
    start = (target_dt - timedelta(days=10)).date().isoformat()
    end = (target_dt + timedelta(days=2)).date().isoformat()

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    splitted = flatten_downloaded(raw, tickers)

    rows = []
    target_date = target_dt.date()

    for ticker in tickers:
        sdf = splitted.get(ticker)
        if sdf is None or sdf.empty:
            rows.append({
                "ticker": ticker,
                "requested_at_jst": target_dt.isoformat(),
                "actual_bar_time_jst": None,
                "Open": None,
                "High": None,
                "Low": None,
                "Close": None,
                "Adj Close": None,
                "Volume": None,
                "status": "NO_DATA",
            })
            continue

        idx = pd.to_datetime(sdf.index)
        if idx.tz is None:
            idx = idx.tz_localize(TOKYO_TZ)
        else:
            idx = idx.tz_convert(TOKYO_TZ)

        sdf = sdf.copy()
        sdf.index = idx

        # 当日以前の最後の営業日を採用
        candidates = sdf[sdf.index.date <= target_date]
        if candidates.empty:
            rows.append({
                "ticker": ticker,
                "requested_at_jst": target_dt.isoformat(),
                "actual_bar_time_jst": None,
                "Open": None,
                "High": None,
                "Low": None,
                "Close": None,
                "Adj Close": None,
                "Volume": None,
                "status": "NO_PREV_TRADING_DAY",
            })
            continue

        row = candidates.iloc[-1]
        actual_ts = candidates.index[-1]

        rows.append({
            "ticker": ticker,
            "requested_at_jst": target_dt.isoformat(),
            "actual_bar_time_jst": actual_ts.isoformat(),
            "Open": row.get("Open"),
            "High": row.get("High"),
            "Low": row.get("Low"),
            "Close": row.get("Close"),
            "Adj Close": row.get("Adj Close"),
            "Volume": row.get("Volume"),
            "status": "OK",
        })

    return pd.DataFrame(rows)


def get_intraday_prices(target_dt: datetime, tickers: list[str], interval: str) -> pd.DataFrame:
    """
    指定日時に一番近い足を返す
    """
    if interval == "1m":
        pad_before = timedelta(days=1)
        pad_after = timedelta(days=1)
    else:
        pad_before = timedelta(days=2)
        pad_after = timedelta(days=1)

    start = (target_dt - pad_before).strftime("%Y-%m-%d")
    end = (target_dt + pad_after).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    splitted = flatten_downloaded(raw, tickers)

    rows = []

    for ticker in tickers:
        sdf = splitted.get(ticker)
        if sdf is None or sdf.empty:
            rows.append({
                "ticker": ticker,
                "requested_at_jst": target_dt.isoformat(),
                "actual_bar_time_jst": None,
                "Open": None,
                "High": None,
                "Low": None,
                "Close": None,
                "Adj Close": None,
                "Volume": None,
                "status": "NO_DATA",
            })
            continue

        idx = pd.to_datetime(sdf.index)
        if idx.tz is None:
            idx = idx.tz_localize(TOKYO_TZ)
        else:
            idx = idx.tz_convert(TOKYO_TZ)

        sdf = sdf.copy()
        sdf.index = idx

        # 指定日時以前の直近足を優先
        prev_rows = sdf[sdf.index <= target_dt]
        if not prev_rows.empty:
            picked = prev_rows.iloc[-1]
            actual_ts = prev_rows.index[-1]
            status = "OK_PREV_BAR"
        else:
            # なければ以後の最初の足
            next_rows = sdf[sdf.index > target_dt]
            if next_rows.empty:
                rows.append({
                    "ticker": ticker,
                    "requested_at_jst": target_dt.isoformat(),
                    "actual_bar_time_jst": None,
                    "Open": None,
                    "High": None,
                    "Low": None,
                    "Close": None,
                    "Adj Close": None,
                    "Volume": None,
                    "status": "NO_NEAR_BAR",
                })
                continue
            picked = next_rows.iloc[0]
            actual_ts = next_rows.index[0]
            status = "OK_NEXT_BAR"

        rows.append({
            "ticker": ticker,
            "requested_at_jst": target_dt.isoformat(),
            "actual_bar_time_jst": actual_ts.isoformat(),
            "Open": picked.get("Open"),
            "High": picked.get("High"),
            "Low": picked.get("Low"),
            "Close": picked.get("Close"),
            "Adj Close": picked.get("Adj Close"),
            "Volume": picked.get("Volume"),
            "status": status,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datetime",
        required=True,
        help='例: "2026-07-30" または "2026-07-30 10:15"',
    )
    parser.add_argument(
        "--interval",
        default=None,
        help='日時指定時のみ有効。例: 1m, 5m, 15m, 30m, 60m, 90m, 1h',
    )
    parser.add_argument(
        "--output",
        default="prices_output.csv",
        help="出力CSVファイル名",
    )

    args = parser.parse_args()

    target_dt, has_time = parse_target_datetime(args.datetime)

    tickers = unique_keep_order(TICKERS)

    if has_time:
        interval = choose_intraday_interval(target_dt, args.interval)
        result = get_intraday_prices(target_dt, tickers, interval)
    else:
        result = get_daily_prices(target_dt, tickers)

    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"保存完了: {args.output}")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
