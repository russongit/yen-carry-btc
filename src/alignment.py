"""Timing alignment, PREREGISTRATION.md sections 3 to 5.

Builds one row per NYSE trading day from 2015-01-02 to 2026-09-30 with every
series placed on the day it was known, the section 4 day-over-day derived
variables, and a flag column for every day a timing rule changed something.
Also writes the primary hourly BTC path for section 8.4.

Outputs:
    data/processed/aligned_daily.csv
    data/processed/btc_hourly.csv

No percentiles, rolling windows, signals, or regressions are computed here.

Usage:
    python src/alignment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

import ingestion as ing

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# Section 3: burn-in start and analysis end. Rows start one month earlier internally so the
# first day in the window has a previous trading day to difference against.
WINDOW_START, WINDOW_END = pd.Timestamp("2015-01-02"), pd.Timestamp("2026-09-30")
BUILD_START = pd.Timestamp("2014-12-01")

NY = "America/New_York"
# DEVIATIONS.md, BTC sampling rule (approved alignment plan): the price at target time T is the close of the
# hourly candle that ends at T (candles are labeled by open time).
BTC_TARGETS = {"noon": 12, "close": 16, "1700": 17}  # section 5 rules 2 and 3; section 8.1 and 8.5
# DEVIATIONS.md, BTC sampling rule, zero-volume part: walk back to the last candle with trades; leave
# blank if that candle ended more than 6 hours before T (mirrors the section 4 gap limit).
MAX_STALE_HOURS = 6

# Section 4, r_dxyx: ICE exponents for the five non-yen currencies, rescaled by 1/0.864.
DXYX_WEIGHTS = {"DEXUSEU": -0.576, "DEXUSUK": -0.119, "DEXCAUS": 0.091, "DEXSDUS": 0.042, "DEXSZUS": 0.036}
DXYX_SCALE = 1 / 0.864
FX_SERIES = ["DEXJPUS", "DEXUSEU", "DEXUSUK", "DEXCAUS", "DEXSDUS", "DEXSZUS"]


class AlignmentError(RuntimeError):
    pass


# ---------------------------------------------------------------- calendar

def nyse_calendar() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Section 3: NYSE trading days, cross-checked against S&P 500 dates (DEVIATIONS.md, red-team documentation items)."""
    cal = xcals.get_calendar("XNYS", start=BUILD_START - pd.Timedelta(days=10), end=WINDOW_END + pd.Timedelta(days=10))
    days = cal.sessions_in_range(BUILD_START, WINDOW_END).tz_localize(None)
    spx = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(RAW / "yfinance" / "SPX.csv")["Date"]))
    cal_window = days[days >= WINDOW_START]
    missing, extra = cal_window.difference(spx), spx.difference(cal_window)
    if len(missing) or len(extra):
        raise AlignmentError(f"NYSE calendar and S&P 500 dates disagree: calendar only {list(missing.date)}, "
                             f"S&P 500 only {list(extra.date)}")
    early = cal.early_closes.tz_localize(None) if cal.early_closes.tz else cal.early_closes
    return days, pd.DatetimeIndex(early)


# ---------------------------------------------------------------- helpers

def _fred(sid: str) -> pd.Series:
    df = pd.read_csv(RAW / "fred" / f"{sid}.csv", na_values=["."])
    return df.set_index(pd.to_datetime(df["observation_date"]))[sid].astype(float)


def _gap_return(level: pd.Series, days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Section 5 rule 5: log change between consecutive available values on NYSE days.

    Returns (log change, missing-day flag, multi-day flag). A day with no value gets a blank
    change; the next available change spans both days and is flagged. Nothing is forward-filled.
    """
    on_days = level.reindex(days)
    avail = on_days.dropna()
    change = np.log(avail).diff()
    pos = pd.Series(np.arange(len(days)), index=days)
    spans = pos.reindex(avail.index).diff() > 1
    return change.reindex(days), on_days.isna(), spans.reindex(days, fill_value=False)


def _carried_level(level: pd.Series, days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rule 7 logic: the latest value on or before each NYSE date; flags closed and spanning days.

    Returns (level on NYSE days, closed flag, multi-day flag). Closed means no new value on that
    calendar date. Multi-day means the change since the previous NYSE day includes more than
    one new value (the market traded on a day NYSE was closed) or follows closed days.
    """
    obs = level.dropna().sort_index()
    carried = obs.reindex(obs.index.union(days)).ffill().reindex(days)
    closed = ~days.isin(obs.index)
    # Number of new values since the previous NYSE day.
    upto = obs.index.searchsorted(days, side="right")
    counts = np.diff(upto, prepend=upto[0] - 1)
    after_closed = pd.Series(closed, index=days).shift(1, fill_value=False)
    multi = (pd.Series(counts, index=days) > 1) | (after_closed & ~pd.Series(closed, index=days))
    return carried, pd.Series(closed, index=days), multi


# ---------------------------------------------------------------- BTC

def load_bitstamp() -> pd.DataFrame:
    """Primary BTC source after the section 4 fallback (DEVIATIONS.md). CCXT rows: [ms, o, h, l, c, v]."""
    rows = []
    for f in sorted((RAW / "bitstamp_btcusd_1h").glob("*.json")):
        rows.extend(json.loads(f.read_bytes()))
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    return df.sort_index()


def load_binance() -> pd.DataFrame:
    """Robustness check 2 source. Binance klines: [open ms, o, h, l, c, v, ...]."""
    rows = []
    for f in sorted((RAW / "binance_btcusdt_1h").glob("*.json")):
        rows.extend(r[:6] for r in json.loads(f.read_bytes()))
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"]).drop_duplicates("t")
    df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
    return df.astype(float).sort_index()


def sample_btc(hourly: pd.DataFrame, days: pd.DatetimeIndex, target_hour: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Price at target_hour New York on each NYSE day (DEVIATIONS.md, BTC sampling rule).

    Uses the close of the candle that opens one hour before the target, so the price is the last
    trade at or before the target time. New York times are converted with the America/New_York
    time zone, so daylight saving is handled. If that candle is missing or had no trades, the
    close of the most recent candle with trades is used and flagged stale; if that candle ended
    more than MAX_STALE_HOURS before the target, the value is blank and flagged missing.
    """
    traded = hourly[hourly["volume"] > 0]
    candle_open = (days + pd.Timedelta(hours=target_hour - 1)).tz_localize(NY).tz_convert("UTC")
    pos = traded.index.searchsorted(candle_open, side="right") - 1
    value, stale, missing = [], [], []
    for p, want in zip(pos, candle_open):
        if p < 0:
            value.append(np.nan); stale.append(False); missing.append(True)
            continue
        got = traded.index[p]
        hours_back = (want - got) / pd.Timedelta(hours=1)
        if hours_back > MAX_STALE_HOURS:
            value.append(np.nan); stale.append(False); missing.append(True)
        else:
            value.append(float(traded["close"].iloc[p])); stale.append(hours_back > 0); missing.append(False)
    idx = days
    return pd.Series(value, index=idx), pd.Series(stale, index=idx), pd.Series(missing, index=idx)


# ---------------------------------------------------------------- COT

def cot_daily(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Section 5 rule 9: each report is usable from the first NYSE day after its actual release date.

    Reports with no published release date are never usable (DEVIATIONS.md, 2026-10-19); on those
    days COT_net keeps the last report that was actually released.
    """
    raw = ing.load_cot_yen()
    resolved, excluded, unresolved = ing.cot_release_dates()
    if unresolved:
        raise AlignmentError(f"COT reports without a release date: {unresolved}")
    num = lambda c: pd.to_numeric(raw[c].str.strip(), errors="raise")
    raw = raw.assign(cot_net=(num("Lev_Money_Positions_Long_All") - num("Lev_Money_Positions_Short_All"))
                     / num("Open_Interest_All"))
    raw = raw[raw["as_of"].isin(resolved.keys())].copy()
    raw["release"] = pd.to_datetime([resolved[d][0] for d in raw["as_of"]])
    # First NYSE day strictly after the release date (rule 9).
    raw["usable_from"] = days[days.searchsorted(raw["release"], side="right").clip(max=len(days) - 1)]
    raw.loc[raw["release"] >= days[-1], "usable_from"] = pd.NaT
    raw = raw.dropna(subset=["usable_from"]).sort_values(["usable_from", "as_of"])
    # If several reports become usable the same day (catch-up periods), the latest as-of wins.
    latest = raw.groupby("usable_from").last()
    out = latest[["as_of", "cot_net"]].reindex(days).ffill()
    out["as_of"] = pd.to_datetime(out["as_of"])
    return out.rename(columns={"as_of": "cot_as_of", "cot_net": "COT_net"})


# ---------------------------------------------------------------- build

def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    days, early = nyse_calendar()
    d = pd.DataFrame(index=days)
    d.index.name = "date"
    d["nyse_early_close"] = days.isin(early)  # approved addition; no other change on these days

    # FX, section 5 rules 1 and 5: noon New York rates for date t, never forward-filled.
    logret = {}
    for sid in FX_SERIES:
        level = _fred(sid)
        d[sid] = level.reindex(days)
        logret[sid], miss, spans = _gap_return(level, days)
        if sid == "DEXJPUS":
            d["fx_missing"], d["fx_multi_day"] = miss, spans
        elif not miss.equals(d["fx_missing"]):
            raise AlignmentError(f"{sid} is missing on different days from DEXJPUS")
    d["r_yen"] = -logret["DEXJPUS"]  # section 4: positive means the yen appreciated
    d["r_dxyx"] = DXYX_SCALE * sum(w * logret[s] for s, w in DXYX_WEIGHTS.items())  # blank if any input blank
    d["r_eurusd"] = logret["DEXUSEU"]  # robustness check 3
    dtw = _fred("DTWEXBGS")  # robustness check 4; H.10 series, so rule 5 applies
    d["DTWEXBGS"] = dtw.reindex(days)
    d["r_dtwexbgs"], _, d["dtw_multi_day"] = _gap_return(dtw, days)

    # Equities, VIX, ETFs, section 5 rule 2: closing values on day t (Close; Adj Close for ACWI and FXY per section 4).
    for name, col in [("SPX", "Close"), ("VIX", "Close"), ("ACWI", "Adj Close"), ("FXY", "Adj Close")]:
        df = pd.read_csv(RAW / "yfinance" / f"{name}.csv")
        d[name] = df.set_index(pd.to_datetime(df["Date"]))[col].reindex(days)
    if d.loc[d.index >= WINDOW_START, ["SPX", "VIX", "ACWI", "FXY"]].isna().any().any():
        raise AlignmentError("Equity, VIX, or ETF close missing on an NYSE day")
    d["r_spx"] = np.log(d["SPX"]).diff()
    d["dVIX"] = d["VIX"].diff()

    # US yields, section 5 rule 8 (not lagged) and the DEVIATIONS.md US Treasury holiday rule: on days the Treasury
    # market is closed while NYSE is open, the level carries over (change 0) and is flagged.
    for sid, name in [("DGS2", "US2Y"), ("DGS10", "US10Y"), ("DGS3MO", "US3M")]:
        lvl, closed, multi = _carried_level(_fred(sid), days)
        d[name] = lvl
        if sid == "DGS2":
            d["ust_holiday"], d["ust_multi_day"] = closed, multi
    d["dUS2Y"] = d["US2Y"].diff() * 100  # basis points (section 4)
    d["dUS10Y"] = d["US10Y"].diff() * 100  # robustness check 5
    d["rf"] = d["US3M"] / 100 / 252  # section 4

    # Japan 2Y, section 5 rules 6 and 7: latest Tokyo 3 p.m. close on or before calendar date t.
    frames = []
    for name in ing.JGB_FILES:
        df = pd.read_csv(RAW / "mof_jgb" / name, skiprows=1, na_values="-", encoding="cp932", dtype={"Date": str})
        dt = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
        frames.append(pd.Series(df.loc[dt.notna(), "2Y"].astype(float).values, index=dt[dt.notna()]))
    jgb = pd.concat(frames)
    jgb = jgb[~jgb.index.duplicated(keep="first")].sort_index()
    d["JP2Y"], d["jp_holiday"], d["jp_multi_day"] = _carried_level(jgb, days)
    d["dJP2Y"] = d["JP2Y"].diff() * 100  # basis points; 0 on Japanese holidays (rule 7)
    d["dJP2Y_lag1"] = d["dJP2Y"].shift(1)  # rule 6: robustness check 7 and any trading rule
    d["spread_lag1"] = (d["US2Y"] - d["JP2Y"]).shift(1)  # section 4: level, lagged one trading day

    # BTC, section 5 rules 2 to 4 and the DEVIATIONS.md BTC sampling rule. Weekends and NYSE holidays fall
    # inside the next trading day's return (rule 4); nothing is dropped.
    btc = load_bitstamp()
    binance = load_binance()
    stale_any = pd.Series(False, index=days)
    for label, hour in BTC_TARGETS.items():
        d[f"BTC_{label}"], stale, miss = sample_btc(btc, days, hour)
        d[f"r_btc_{label}"] = np.log(d[f"BTC_{label}"]).diff()
        stale_any |= stale
        d[f"btc_missing_{label}"] = miss
        d[f"BTCB_{label}"], _, d[f"btcb_missing_{label}"] = sample_btc(binance, days, hour)  # robustness check 2
        d[f"r_btcb_{label}"] = np.log(d[f"BTCB_{label}"]).diff()
    d["btc_stale"] = stale_any

    # COT, section 5 rule 9 (exploratory only, D2).
    d = d.join(cot_daily(days))

    d = d[d.index >= WINDOW_START]
    hourly = btc[(btc.index >= pd.Timestamp("2017-01-01", tz="UTC")) & (btc.index < pd.Timestamp("2026-10-01", tz="UTC"))]
    hourly = hourly.rename_axis("open_utc")
    return d, hourly


def report(d: pd.DataFrame) -> None:
    print(f"Rows: {len(d):,} NYSE trading days, {d.index.min().date()} to {d.index.max().date()}")
    print("\nFlags (days):")
    for c in [c for c in d.columns if d[c].dtype == bool]:
        print(f"  {c:<22} {int(d[c].sum()):>5}")
    btc_window = d.index >= "2017-01-04"
    print("\nBlank values in the window (BTC counted from 2017-01-04):")
    for c in [c for c in d.columns if d[c].dtype != bool and c != "cot_as_of"]:
        mask = btc_window if "btc" in c.lower() else slice(None)
        n = int(d.loc[mask, c].isna().sum())
        if n:
            print(f"  {c:<22} {n:>5}")


def main() -> None:
    d, hourly = build()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "aligned_daily.csv", date_format="%Y-%m-%d")
    hourly.to_csv(OUT / "btc_hourly.csv")
    report(d)
    print(f"\nWrote {(OUT / 'aligned_daily.csv').relative_to(ROOT)} and {(OUT / 'btc_hourly.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
