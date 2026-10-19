"""Look-ahead tests for src/alignment.py (PREREGISTRATION.md section 5).

The core test scrambles every raw value that became known after a cutoff date and
checks that nothing in the aligned table on or before the cutoff changes. The other
tests pin down each timing rule on small synthetic inputs or on the real data.

Run from the repository root:  python -m pytest tests/test_lookahead.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import alignment as al  # noqa: E402
import ingestion as ing  # noqa: E402

SCRAMBLE = 1.37  # multiplier applied to every value known only after the cutoff


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return al.build()[0]


# ---------------------------------------------------------------- 1. future data cannot change the past

def _scrambled_build(monkeypatch, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rebuild with every input that became known after the cutoff scaled by SCRAMBLE."""
    orig_fred = al._fred
    monkeypatch.setattr(al, "_fred", lambda sid: orig_fred(sid).where(lambda s: s.index <= cutoff, lambda s: s * SCRAMBLE))

    # BTC: anything after midnight New York at the end of the cutoff day.
    btc_cut = (cutoff + pd.Timedelta(days=1)).tz_localize(al.NY).tz_convert("UTC")
    for loader in ("load_bitstamp", "load_binance"):
        orig = getattr(al, loader)

        def scrambled(orig=orig):
            df = orig().copy()
            later = df.index >= btc_cut
            df.loc[later, ["open", "high", "low", "close"]] *= SCRAMBLE
            return df
        monkeypatch.setattr(al, loader, scrambled)

    # Yahoo closes and MoF JGB yields are read with pd.read_csv inside build().
    orig_read = pd.read_csv

    def read_csv(path, *args, **kwargs):
        df = orig_read(path, *args, **kwargs)
        p = str(path).replace("\\", "/")
        if "/yfinance/" in p:
            later = pd.to_datetime(df["Date"]) > cutoff
        elif "/mof_jgb/" in p:
            later = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce") > cutoff
        else:
            return df
        num = df.select_dtypes("float").columns  # prices and yields; volume is not used
        df.loc[later, num] = df.loc[later, num] * SCRAMBLE
        return df
    monkeypatch.setattr(pd, "read_csv", read_csv)

    # COT: scramble every report not yet released before the cutoff day ended.
    resolved, excluded, _ = ing.cot_release_dates()
    unreleased = {d for d, (r, _) in resolved.items() if pd.Timestamp(r) >= cutoff} | {d for d, _ in excluded}
    orig_cot = ing.load_cot_yen

    def cot():
        df = orig_cot().copy()
        later = df["as_of"].isin(unreleased)
        df.loc[later, "Lev_Money_Positions_Long_All"] = (
            pd.to_numeric(df.loc[later, "Lev_Money_Positions_Long_All"]) * 3 + 1000).astype(int).astype(str)
        return df
    monkeypatch.setattr(ing, "load_cot_yen", cot)

    return al.build()[0]


@pytest.mark.parametrize("cutoff", ["2018-01-02", "2019-02-20", "2024-08-05", "2026-07-31"])
def test_future_data_does_not_change_past(baseline, monkeypatch, cutoff):
    cutoff = pd.Timestamp(cutoff)
    scrambled = _scrambled_build(monkeypatch, cutoff)
    past = baseline.index <= cutoff
    pd.testing.assert_frame_equal(baseline.loc[past], scrambled.loc[past])
    # The scramble must actually reach the future, or the test proves nothing.
    future = baseline.index > cutoff
    assert not baseline.loc[future, "DEXJPUS"].equals(scrambled.loc[future, "DEXJPUS"])
    assert not baseline.loc[future, "BTC_1700"].equals(scrambled.loc[future, "BTC_1700"])


# ---------------------------------------------------------------- 2. BTC sampling

def _hourly(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float) + 100, "volume": 1.0}, index=idx)


def test_btc_uses_close_of_candle_ending_at_target_across_dst():
    hourly = _hourly("2024-03-07", "2024-03-13")
    days = pd.DatetimeIndex(["2024-03-08", "2024-03-11"])  # EST, then EDT after 2024-03-10
    for target, open_ny in [(12, 11), (16, 15), (17, 16)]:
        value, stale, missing = al.sample_btc(hourly, days, target)
        expected_utc = (days + pd.Timedelta(hours=open_ny)).tz_localize(al.NY).tz_convert("UTC")
        assert list(value) == list(hourly.loc[expected_utc, "close"])
        assert not stale.any() and not missing.any()
    # The 16:00 New York candle opens at 21:00 UTC in winter and 20:00 UTC in summer.
    assert list(expected_utc.hour) == [21, 20]


def test_btc_ignores_the_candle_after_target():
    hourly = _hourly("2024-08-04", "2024-08-07")
    days = pd.DatetimeIndex(["2024-08-05"])
    before, _, _ = al.sample_btc(hourly, days, 17)
    after_open = pd.Timestamp("2024-08-05 17:00", tz=al.NY).tz_convert("UTC")
    hourly.loc[hourly.index >= after_open, "close"] *= SCRAMBLE
    after, _, _ = al.sample_btc(hourly, days, 17)
    assert before.equals(after)


@pytest.mark.parametrize("last_traded_open_ny, expect_value", [(5, True), (4, False)])
def test_btc_stale_limit_is_six_hours(last_traded_open_ny, expect_value):
    hourly = _hourly("2024-08-04", "2024-08-07")
    ny = hourly.index.tz_convert(al.NY)
    no_trades = (ny.date == pd.Timestamp("2024-08-05").date()) & (ny.hour > last_traded_open_ny) & (ny.hour <= 11)
    hourly.loc[no_trades, "volume"] = 0.0
    value, stale, missing = al.sample_btc(hourly, pd.DatetimeIndex(["2024-08-05"]), 12)
    last = hourly.loc[pd.Timestamp(f"2024-08-05 {last_traded_open_ny:02d}:00", tz=al.NY).tz_convert("UTC"), "close"]
    if expect_value:  # candle 05:00-06:00 ended exactly 6 hours before 12:00
        assert value.iloc[0] == last and stale.iloc[0] and not missing.iloc[0]
    else:             # candle 04:00-05:00 ended 7 hours before 12:00
        assert np.isnan(value.iloc[0]) and missing.iloc[0] and not stale.iloc[0]


# ---------------------------------------------------------------- 3. FX (rule 5)

def test_fx_missing_fixing_is_blank_not_zero():
    days = pd.DatetimeIndex(["2024-10-10", "2024-10-11", "2024-10-14", "2024-10-15"])
    level = pd.Series([150.0, 151.0, np.nan, 152.0], index=days)
    change, missing, spans = al._gap_return(level, days)
    assert np.isnan(change["2024-10-14"]) and missing["2024-10-14"]
    assert change["2024-10-15"] == pytest.approx(np.log(152 / 151)) and spans["2024-10-15"]
    assert not spans["2024-10-11"]


def test_fx_real_data_missing_days_are_blank(baseline):
    flagged = baseline[baseline["fx_missing"]]
    assert len(flagged) > 0
    assert flagged[["r_yen", "r_dxyx", "r_eurusd"]].isna().all().all()


# ---------------------------------------------------------------- 4. Treasuries (decision B) and JGBs (rules 6, 7)

def test_carried_level_closed_day_and_spanning_day():
    # NYSE: Mon, Tue, Wed, Fri (closed Thu). Tokyo: Mon, Tue, Thu, Fri (closed Wed).
    days = pd.DatetimeIndex(["2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29"])
    obs = pd.Series([1.00, 1.02, 1.10, 1.12], index=pd.DatetimeIndex(["2024-11-25", "2024-11-26", "2024-11-28", "2024-11-29"]))
    level, closed, multi = al._carried_level(obs, days)
    assert level["2024-11-27"] == 1.02 and closed["2024-11-27"]  # Tokyo closed: no change
    assert level["2024-11-29"] == 1.12 and multi["2024-11-29"]  # Thursday's Tokyo move lands on Friday
    assert not closed["2024-11-29"]


def test_closed_days_have_zero_change(baseline):
    assert (baseline.loc[baseline["ust_holiday"], "dUS2Y"] == 0).all()
    assert (baseline.loc[baseline["jp_holiday"], "dJP2Y"] == 0).all()
    assert baseline.loc["2018-01-02", "jp_holiday"] and baseline.loc["2018-01-02", "dJP2Y"] == 0


def test_lagged_columns_use_previous_day_only(baseline):
    b = baseline
    pd.testing.assert_series_equal(b["dJP2Y_lag1"].iloc[1:], b["dJP2Y"].shift(1).iloc[1:], check_names=False)
    expected = (b["US2Y"] - b["JP2Y"]).shift(1)
    pd.testing.assert_series_equal(b["spread_lag1"].iloc[1:], expected.iloc[1:], check_names=False)


# ---------------------------------------------------------------- 5. COT (rule 9)

def test_cot_never_used_on_or_before_release(baseline):
    resolved, excluded, _ = ing.cot_release_dates()
    used = baseline[["cot_as_of"]].dropna()
    release = used["cot_as_of"].map(lambda d: pd.Timestamp(resolved[pd.Timestamp(d).date()][0]))
    assert (release < used.index).all()


def test_cot_holiday_release_first_appears_next_trading_day(baseline):
    s = pd.to_datetime(baseline["cot_as_of"])
    assert s["2024-12-30"] < pd.Timestamp("2024-12-24")
    assert s["2024-12-31"] == pd.Timestamp("2024-12-24")


def test_cot_excluded_reports_never_appear(baseline):
    _, excluded, _ = ing.cot_release_dates()
    used = set(pd.to_datetime(baseline["cot_as_of"].dropna()).dt.date)
    assert used.isdisjoint({d for d, _ in excluded})
    assert len(excluded) == 18


# ---------------------------------------------------------------- 6. rolling and expanding statistics

# alignment.py computes none. Each expanding percentile added in src/signals.py (section 6)
# gets its own test here, per the project rule.
