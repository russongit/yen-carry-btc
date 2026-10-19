"""Raw data ingestion, PREREGISTRATION.md section 4.

Downloads every series in section 4 to data/raw/ exactly as served, logs each
file's download time (UTC) and SHA-256 hash in data/raw/MANIFEST.csv, checks
Coinbase hourly coverage against the section 4 fallback rule, and prints row
counts, date ranges and missing-value counts per series.

Nothing here computes returns, signals, or any derived variable.

Usage:
    python src/ingestion.py            # download files not already in data/raw/
    python src/ingestion.py --refresh  # re-download everything

Existing raw files are not re-downloaded by default, so their logged hashes stay
valid. yfinance has no raw file to save; its DataFrame is written to CSV as
received (auto_adjust=False, so both Close and Adj Close are kept).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.csv"
MANIFEST_COLS = ["file", "source", "url", "download_utc", "sha256", "bytes"]

# Section 3: burn-in start and analysis end.
SAMPLE_START = "2015-01-02"
SAMPLE_END = "2026-09-30"

# Section 4, BTC source fallback: coverage 2017-01-01 to 2026-09-30, no gap > 6 hours.
# Hours are UTC; the window ends at 2026-10-01 00:00 UTC, which is after
# 17:00 New York on 2026-09-30 (the last BTC_1700 observation needed).
BTC_START = datetime(2017, 1, 1, tzinfo=timezone.utc)
BTC_END = datetime(2026, 10, 1, tzinfo=timezone.utc)
MAX_GAP_HOURS = 6

FRED_SERIES = {
    "DEXJPUS": "USDJPY, JPY per USD, noon New York",
    "DEXUSEU": "EURUSD, USD per EUR, noon New York",
    "DEXUSUK": "GBPUSD, USD per GBP, noon New York",
    "DEXCAUS": "USDCAD, CAD per USD, noon New York",
    "DEXSDUS": "USDSEK, SEK per USD, noon New York",
    "DEXSZUS": "USDCHF, CHF per USD, noon New York",
    "DGS3MO": "3-month Treasury constant maturity",
    "DGS2": "2-year Treasury constant maturity",
    "DGS10": "10-year Treasury constant maturity (robustness)",
    "DTWEXBGS": "Broad dollar index (robustness)",
}
YF_TICKERS = {"^GSPC": "SPX", "^VIX": "VIX", "ACWI": "ACWI", "FXY": "FXY"}
JGB_FILES = {
    "jgbcme_all.csv": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv",
    "jgbcme.csv": "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv",
}
JGB_HEADER = ["Date", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y", "40Y"]
CFTC_FILES = ["fin_fut_txt_2006_2016.zip"] + [f"fut_fin_txt_{y}.zip" for y in range(2017, 2027)]
CFTC_URL = "https://www.cftc.gov/files/dea/history/"
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}
CFTC_YEN_CODE = "097741"

# Section 5 rule 9: COT is usable from the first trading day after its actual release date.
# The TFF files carry only the as-of date, so release dates come from CFTC's release
# schedule page (archived copies for past years; the live page lists only the current year)
# and from CFTC press releases for disrupted periods.
COT_SCHEDULE_URL = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm"
COT_SCHEDULE_SNAPSHOTS = ["20150515", "20160520", "20170602", "20180413", "20190716", "20200727",
                          "20210618", "20220523", "20230929", "20240625", "20250617"]
COT_PRESS_RELEASES = ["7864-19", "8655-23", "8662-23", "9138-25", "9147-25"]
# Disrupted periods where the published schedule does not hold. Reports with as-of dates in
# these windows need a release date from a press release, or they are unresolved.
COT_DISRUPTIONS = [
    # 2018-19 lapse in appropriations. 7864-19 gives only a cadence (first release Fri 2019-02-01,
    # then one Tuesday and one Friday each week until current), not dates.
    ("2018-12-24", "2019-03-05", "7864-19"),
    # ION cyber incident. 8662-23 says catch-up begins with the report originally due 2023-02-03
    # and finishes "by mid-March", without dates.
    ("2023-01-31", "2023-03-14", "8662-23"),
    # 2025 lapse in appropriations. 9147-25 (latest revision of 9138-25) lists every new date.
    ("2025-09-30", "2025-12-23", "9147-25"),
]

COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Keep requests' default user-agent: FRED stalls browser and unrecognized ones.
HTTP = requests.Session()


class SourceError(RuntimeError):
    """A source failed or its format differs from section 4. Stop and log it."""


# ---------------------------------------------------------------- manifest

def _load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def _write_manifest(rows: dict[str, dict]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def save_raw(path: Path, content: bytes, source: str, url: str, manifest: dict) -> None:
    """Write bytes unmodified and record the hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    rel = path.relative_to(RAW).as_posix()
    manifest[rel] = {
        "file": rel,
        "source": source,
        "url": url,
        "download_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _get(url: str, params: dict | None = None, timeout: int = 60, tries: int = 4) -> requests.Response:
    for attempt in range(tries):
        try:
            r = HTTP.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise SourceError(f"{url} failed: {e}") from e
            time.sleep(2 * (attempt + 1))
    raise SourceError(f"{url} failed after {tries} tries")


# ---------------------------------------------------------------- downloads

def fetch_fred(manifest: dict, refresh: bool) -> None:
    for sid in FRED_SERIES:
        path = RAW / "fred" / f"{sid}.csv"
        if path.exists() and not refresh:
            continue
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        r = _get(url)
        header = r.text.splitlines()[0].strip()
        if header != f"observation_date,{sid}":
            raise SourceError(f"FRED {sid}: unexpected header {header!r}")
        save_raw(path, r.content, "FRED", url, manifest)
        print(f"  saved {path.relative_to(ROOT)}")


def fetch_jgb(manifest: dict, refresh: bool) -> None:
    for name, url in JGB_FILES.items():
        path = RAW / "mof_jgb" / name
        if path.exists() and not refresh:
            continue
        r = _get(url)
        save_raw(path, r.content, "MoF", url, manifest)
        print(f"  saved {path.relative_to(ROOT)}")


def fetch_yfinance(manifest: dict, refresh: bool) -> None:
    import yfinance as yf

    for ticker, name in YF_TICKERS.items():
        path = RAW / "yfinance" / f"{name}.csv"
        if path.exists() and not refresh:
            continue
        # yfinance's end date is exclusive.
        df = yf.download(ticker, start=SAMPLE_START, end="2026-10-01", auto_adjust=False,
                         actions=False, progress=False, multi_level_index=False)
        if df.empty:
            raise SourceError(f"yfinance {ticker}: no data returned")
        content = df.to_csv().encode("utf-8")
        save_raw(path, content, "yfinance", f"yfinance:{ticker} auto_adjust=False", manifest)
        print(f"  saved {path.relative_to(ROOT)}")


def fetch_cftc(manifest: dict, refresh: bool) -> None:
    for name in CFTC_FILES:
        path = RAW / "cftc_tff" / name
        if path.exists() and not refresh:
            continue
        r = _get(CFTC_URL + name, timeout=120)
        save_raw(path, r.content, "CFTC TFF futures only", CFTC_URL + name, manifest)
        print(f"  saved {path.relative_to(ROOT)}")


def fetch_cot_release(manifest: dict, refresh: bool) -> None:
    folder = RAW / "cftc_release"
    sources = [(f"schedule_{ts}.html", f"https://web.archive.org/web/{ts}id_/{COT_SCHEDULE_URL}", "Wayback Machine copy of CFTC COT release schedule")
               for ts in COT_SCHEDULE_SNAPSHOTS]
    sources.append((f"schedule_live_{datetime.now(timezone.utc):%Y%m%d}.html", COT_SCHEDULE_URL, "CFTC COT release schedule"))
    sources += [(f"press_{p}.html", f"https://www.cftc.gov/PressRoom/PressReleases/{p}", "CFTC press release")
                for p in COT_PRESS_RELEASES]
    for name, url, source in sources:
        if name.startswith("schedule_live") and any(folder.glob("schedule_live_*.html")) and not refresh:
            continue
        path = folder / name
        if path.exists() and not refresh:
            continue
        r = _get(url, timeout=120, tries=6)
        save_raw(path, r.content, source, url, manifest)
        print(f"  saved {path.relative_to(ROOT)}")
        if "web.archive.org" in url:
            time.sleep(8)  # the Wayback Machine refuses rapid requests


def _page_text(raw: bytes) -> str:
    t = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="replace"))
    return re.sub(r"\s+", " ", html.unescape(t).replace("\xa0", " "))


def parse_schedule(raw: bytes) -> list[date]:
    """Release dates from one copy of the release schedule page (year, month, then day numbers)."""
    t = _page_text(raw)
    i, j = t.find("tentative schedule"), t.find("*Delayed")
    if i < 0 or j < i:
        raise SourceError("CFTC release schedule: no schedule table found")
    seg = t[i:j].split(".", 1)[1]  # drop the sentence ending "through YYYY."
    out, year, month = [], None, None
    for tok in seg.split():
        if re.fullmatch(r"20\d\d", tok):
            year, month = int(tok), None
        elif tok in MONTHS:
            month = MONTHS[tok]
        elif re.fullmatch(r"\d{1,2}\*{0,2}", tok) and year and month:  # "*" holiday delay, "**" Monday data
            out.append(date(year, month, int(tok.rstrip("*"))))
    return out


def parse_press_schedule(raw: bytes) -> dict[date, date]:
    """As-of date -> new publish date, from a press release table of (as-of, original, new) dates."""
    t = _page_text(raw)
    triples = re.findall(r"(\d\d/\d\d/\d{4}) (\d\d/\d\d/\d{4}) (\d\d/\d\d/\d{4})", t)
    to_date = lambda s: datetime.strptime(s, "%m/%d/%Y").date()
    return {to_date(a): to_date(n) for a, _, n in triples}


def load_cot_yen() -> pd.DataFrame:
    """Raw yen rows from every TFF file, with the as-of date parsed."""
    frames = []
    for name in CFTC_FILES:
        with zipfile.ZipFile(RAW / "cftc_tff" / name) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"), dtype=str, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["CFTC_Contract_Market_Code"].str.strip() == CFTC_YEN_CODE].copy()
        raw_dates = df["Report_Date_as_YYYY-MM-DD"].str.strip()
        # The 2006-2016 file writes this column as "12/27/2016 12:00:00 AM" despite its name.
        dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce").fillna(
            pd.to_datetime(raw_dates, format="%m/%d/%Y %I:%M:%S %p", errors="coerce"))
        if dates.isna().any():
            raise SourceError(f"CFTC {name}: unparseable report dates {raw_dates[dates.isna()].tolist()[:3]}")
        df["as_of"] = dates.dt.date
        frames.append(df)
    return pd.concat(frames).drop_duplicates("as_of").sort_values("as_of").reset_index(drop=True)


def cot_release_dates(first_as_of: str = "2014-12-23", last_as_of: str = SAMPLE_END) -> tuple[dict, list, list]:
    """Map each yen as-of date to its release date. Returns (resolved, excluded, unresolved).

    excluded: reports in a disrupted period whose press release gives no date. Per DEVIATIONS.md
    (2026-10-19) they are never usable; COT_net keeps the last released report.
    unresolved: anything else without a release date, which must stop the run.
    """
    folder = RAW / "cftc_release"
    scheduled: set[date] = set()
    for f in sorted(folder.glob("schedule_*.html")):
        scheduled.update(parse_schedule(f.read_bytes()))
    overrides: dict[date, date] = {}
    for f in sorted(folder.glob("press_*.html")):
        overrides.update(parse_press_schedule(f.read_bytes()))

    as_of = [d for d in load_cot_yen()["as_of"]
             if date.fromisoformat(first_as_of) <= d <= date.fromisoformat(last_as_of)]
    windows = [(date.fromisoformat(a), date.fromisoformat(b), src) for a, b, src in COT_DISRUPTIONS]
    resolved, excluded, unresolved = {}, [], []
    releases = sorted(scheduled)
    for d in as_of:
        window = next((w for w in windows if w[0] <= d <= w[1]), None)
        if window:
            if d in overrides:
                resolved[d] = (overrides[d], f"press release {window[2]}")
            else:
                excluded.append((d, f"disrupted period, no date in press release {window[2]}"))
            continue
        # Normal weeks: the first scheduled release after the as-of date, within 10 days.
        nxt = next((r for r in releases if d < r <= d + timedelta(days=10)), None)
        if nxt is None:
            unresolved.append((d, "no scheduled release within 10 days"))
        else:
            resolved[d] = (nxt, "release schedule")

    # Two normal-week reports must never share one release date.
    by_release: dict[date, list] = {}
    for d, (r, src) in resolved.items():
        by_release.setdefault(r, []).append(d)
    for r, ds in by_release.items():
        if len(ds) > 1:
            for d in ds:
                resolved.pop(d)
                unresolved.append((d, f"shares release date {r} with another report"))
    return resolved, sorted(excluded), sorted(unresolved)


def _paged_hours(start: datetime, end: datetime, step_hours: int):
    t = start
    while t < end:
        yield t, min(t + timedelta(hours=step_hours), end)
        t += timedelta(hours=step_hours)


def fetch_coinbase(manifest: dict, refresh: bool) -> None:
    """Hourly BTC-USD candles, 300 per request (the API maximum)."""
    folder = RAW / "coinbase_btcusd_1h"
    pages = list(_paged_hours(BTC_START, BTC_END, 300))
    for i, (a, b) in enumerate(pages):
        path = folder / f"{a:%Y%m%dT%H}.json"
        if path.exists() and not refresh:
            continue
        # Coinbase's end bound is inclusive, so stop one hour before the next page.
        params = {"granularity": 3600, "start": a.isoformat(), "end": (b - timedelta(hours=1)).isoformat()}
        r = _get(COINBASE_URL, params=params)
        if not isinstance(r.json(), list):
            raise SourceError(f"Coinbase page {a}: unexpected response {r.text[:200]!r}")
        save_raw(path, r.content, "Coinbase Exchange", r.url, manifest)
        time.sleep(0.15)
        if i % 50 == 0:
            print(f"  Coinbase page {i + 1}/{len(pages)}")


def fetch_binance(manifest: dict, refresh: bool) -> None:
    """Hourly BTCUSDT klines (robustness series), 1,000 per request."""
    folder = RAW / "binance_btcusdt_1h"
    for i, (a, b) in enumerate(_paged_hours(BTC_START, BTC_END, 1000)):
        path = folder / f"{a:%Y%m%dT%H}.json"
        if path.exists() and not refresh:
            continue
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 1000,
                  "startTime": int(a.timestamp() * 1000), "endTime": int(b.timestamp() * 1000) - 1}
        r = _get(BINANCE_URL, params=params)
        save_raw(path, r.content, "Binance", r.url, manifest)
        time.sleep(0.1)


def fetch_bitstamp(manifest: dict, refresh: bool) -> None:
    """Section 4 fallback: Bitstamp BTC/USD hourly via CCXT. Only run if Coinbase fails coverage."""
    import ccxt

    ex = ccxt.bitstamp({"enableRateLimit": True})
    folder = RAW / "bitstamp_btcusd_1h"
    for a, b in _paged_hours(BTC_START, BTC_END, 1000):
        path = folder / f"{a:%Y%m%dT%H}.json"
        if path.exists() and not refresh:
            continue
        rows = ex.fetch_ohlcv("BTC/USD", "1h", since=int(a.timestamp() * 1000), limit=1000)
        rows = [row for row in rows if row[0] < int(b.timestamp() * 1000)]
        # CCXT returns parsed lists, not the exchange's bytes; this is saved as received.
        save_raw(path, json.dumps(rows).encode("utf-8"), "Bitstamp via CCXT", f"ccxt:bitstamp BTC/USD 1h since={a.isoformat()}", manifest)


# ---------------------------------------------------------------- coverage

def load_hourly(folder: Path, ts_index: int, ts_unit: str) -> pd.DatetimeIndex:
    stamps = []
    for f in sorted(folder.glob("*.json")):
        stamps.extend(row[ts_index] for row in json.loads(f.read_bytes()))
    idx = pd.to_datetime(pd.Series(stamps, dtype="int64"), unit=ts_unit, utc=True)
    return pd.DatetimeIndex(idx.drop_duplicates().sort_values())


def coverage(stamps: pd.DatetimeIndex) -> dict:
    """Longest run of missing hourly candles in [BTC_START, BTC_END)."""
    grid = pd.date_range(BTC_START, BTC_END, freq="h", inclusive="left")
    present = grid.isin(stamps)
    longest, run, run_end = 0, 0, None
    for t, ok in zip(grid, present):
        run = 0 if ok else run + 1
        if run > longest:
            longest, run_end = run, t
    gap_start = run_end - timedelta(hours=longest - 1) if longest else None
    return {
        "expected_hours": len(grid),
        "present_hours": int(present.sum()),
        "missing_hours": int((~present).sum()),
        "first": stamps.min() if len(stamps) else None,
        "last": stamps.max() if len(stamps) else None,
        "longest_gap_hours": longest,
        "longest_gap_start": gap_start,
        "passes": longest <= MAX_GAP_HOURS,
    }


def _print_coverage(name: str, c: dict) -> None:
    print(f"  {name}: {c['present_hours']:,} of {c['expected_hours']:,} hours present, "
          f"{c['missing_hours']:,} missing; first {c['first']}, last {c['last']}")
    print(f"  longest gap: {c['longest_gap_hours']} hours starting {c['longest_gap_start']}")


# ---------------------------------------------------------------- summary

def _window(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = pd.to_datetime(df[col])
    return df[(d >= SAMPLE_START) & (d <= SAMPLE_END)]


def _summary_row(name: str, df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    w = _window(df, date_col)
    return {
        "series": name,
        "raw_rows": len(df),
        "raw_first": pd.to_datetime(df[date_col]).min().date(),
        "raw_last": pd.to_datetime(df[date_col]).max().date(),
        "window_rows": len(w),
        "window_missing": int(w[value_col].isna().sum()),
    }


def summarize(btc_source: str) -> pd.DataFrame:
    rows = []
    for sid in FRED_SERIES:
        df = pd.read_csv(RAW / "fred" / f"{sid}.csv", na_values=["."])
        rows.append(_summary_row(f"FRED {sid}", df, "observation_date", sid))

    for name in YF_TICKERS.values():
        df = pd.read_csv(RAW / "yfinance" / f"{name}.csv")
        col = "Adj Close" if name in ("ACWI", "FXY") else "Close"
        rows.append(_summary_row(f"yfinance {name} ({col})", df, "Date", col))

    # Section 4: skiprows=1, na_values="-". The current-month file must match the historical structure;
    # any difference is handled here and reported. MoF files are Shift-JIS (cp932).
    frames = []
    for name in JGB_FILES:
        df = pd.read_csv(RAW / "mof_jgb" / name, skiprows=1, na_values="-", encoding="cp932", dtype={"Date": str})
        if list(df.columns) != JGB_HEADER:
            raise SourceError(f"MoF {name}: header {list(df.columns)} differs from section 4")
        dates = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
        extra = df.loc[dates.isna(), "Date"]
        if len(extra):
            print(f"  MoF {name}: dropped {len(extra)} non-data row(s): "
                  + "; ".join(repr(x) for x in extra.fillna("<blank>").str.strip()))
        df = df[dates.notna()].copy()
        df["Date"] = dates[dates.notna()]
        frames.append(df)
        rows.append(_summary_row(f"MoF {name} (2Y)", df, "Date", "2Y"))
    jgb = pd.concat(frames).drop_duplicates("Date").sort_values("Date")
    rows.append(_summary_row("MoF JGB combined (2Y)", jgb, "Date", "2Y"))

    yen = load_cot_yen()[["as_of"]].rename(columns={"as_of": "date"})
    yen["value"] = 0
    rows.append(_summary_row(f"CFTC TFF yen ({CFTC_YEN_CODE}) reports", yen, "date", "value"))

    for label, folder, ts_index, unit in [
        ("Coinbase BTC-USD 1h", RAW / "coinbase_btcusd_1h", 0, "s"),
        ("Binance BTCUSDT 1h", RAW / "binance_btcusdt_1h", 0, "ms"),
        ("Bitstamp BTC/USD 1h", RAW / "bitstamp_btcusd_1h", 0, "ms"),
    ]:
        if not folder.exists():
            continue
        stamps = load_hourly(folder, ts_index, unit)
        c = coverage(stamps)
        rows.append({
            "series": label + (" [primary]" if label.startswith(btc_source) else ""),
            "raw_rows": len(stamps),
            "raw_first": stamps.min().date(),
            "raw_last": stamps.max().date(),
            "window_rows": c["present_hours"],
            "window_missing": c["missing_hours"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    try:
        for label, fn in [("FRED", fetch_fred), ("MoF JGB", fetch_jgb), ("yfinance", fetch_yfinance),
                          ("CFTC TFF", fetch_cftc), ("CFTC release dates", fetch_cot_release),
                          ("Coinbase", fetch_coinbase), ("Binance", fetch_binance)]:
            print(f"{label}:")
            fn(manifest, args.refresh)
            _write_manifest(manifest)

        print("\nCoinbase coverage check (section 4 fallback rule):")
        cb = coverage(load_hourly(RAW / "coinbase_btcusd_1h", 0, "s"))
        _print_coverage("Coinbase BTC-USD", cb)
        btc_source = "Coinbase"
        if cb["passes"]:
            print(f"  PASS: no gap longer than {MAX_GAP_HOURS} hours. Coinbase is the primary BTC source.")
        else:
            print(f"  FAIL: gap longer than {MAX_GAP_HOURS} hours. Switching to the Bitstamp fallback.")
            fetch_bitstamp(manifest, args.refresh)
            _write_manifest(manifest)
            bs = coverage(load_hourly(RAW / "bitstamp_btcusd_1h", 0, "ms"))
            _print_coverage("Bitstamp BTC/USD", bs)
            btc_source = "Bitstamp"
            print("  Log this switch in DEVIATIONS.md (section 4 BTC source fallback, decided before analysis).")
    finally:
        _write_manifest(manifest)

    print(f"\nSeries summary (window = {SAMPLE_START} to {SAMPLE_END}; BTC window = "
          f"{BTC_START:%Y-%m-%d %H:%M} to {BTC_END:%Y-%m-%d %H:%M} UTC, hourly):")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(summarize(btc_source).to_string(index=False))
    print(f"\nManifest: {MANIFEST.relative_to(ROOT)} ({len(manifest)} files)")

    print("\nCOT release dates (section 5 rule 9):")
    resolved, excluded, unresolved = cot_release_dates()
    sources = pd.Series([src for _, src in resolved.values()]).value_counts()
    print(f"  {len(resolved)} yen reports resolved: " + ", ".join(f"{n} from {s}" for s, n in sources.items()))
    print(f"  {len(excluded)} report(s) never usable (no published release date; see DEVIATIONS.md):")
    for d, why in excluded:
        print(f"    as-of {d}: {why}")
    if unresolved:
        print(f"  STOP: {len(unresolved)} report(s) have no release date:")
        for d, why in unresolved:
            print(f"    as-of {d}: {why}")
        raise SourceError("COT release dates incomplete; decide how to handle the reports above before alignment")


if __name__ == "__main__":
    main()
