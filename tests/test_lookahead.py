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
# One aligned column per raw input source: FRED, Yahoo, MoF, Bitstamp, Binance, CFTC.
SCRAMBLED_SOURCES = ["DEXJPUS", "DEXUSEU", "DEXUSUK", "DEXCAUS", "DEXSDUS", "DEXSZUS", "DTWEXBGS",
                     "US2Y", "US10Y", "US3M", "SPX", "VIX", "ACWI", "FXY", "JP2Y",
                     "BTC_noon", "BTC_close", "BTC_1700", "BTCB_noon", "BTCB_close", "BTCB_1700", "COT_net"]


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
    pd.testing.assert_frame_equal(baseline.loc[past], scrambled.loc[past], check_exact=True)
    # The scramble must reach every input source, or the test proves nothing for that source.
    future = baseline.index > cutoff
    for col in SCRAMBLED_SOURCES:
        assert not baseline.loc[future, col].equals(scrambled.loc[future, col]), f"scramble did not reach {col}"


def _truncated_build(monkeypatch, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rebuild with every input that became known after the cutoff removed, and the calendar ending at it."""
    monkeypatch.setattr(al, "WINDOW_END", cutoff)
    orig_fred = al._fred
    monkeypatch.setattr(al, "_fred", lambda sid: orig_fred(sid).loc[:cutoff])

    btc_cut = (cutoff + pd.Timedelta(days=1)).tz_localize(al.NY).tz_convert("UTC")
    for loader in ("load_bitstamp", "load_binance"):
        orig = getattr(al, loader)
        monkeypatch.setattr(al, loader, lambda orig=orig: orig().loc[lambda df: df.index < btc_cut])

    orig_read = pd.read_csv

    def read_csv(path, *args, **kwargs):
        df = orig_read(path, *args, **kwargs)
        p = str(path).replace("\\", "/")
        if "/yfinance/" in p:
            return df[pd.to_datetime(df["Date"]) <= cutoff]
        if "/mof_jgb/" in p:
            return df[~(pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce") > cutoff)]
        return df
    monkeypatch.setattr(pd, "read_csv", read_csv)

    resolved, _, _ = ing.cot_release_dates()
    unreleased = {d for d, (r, _) in resolved.items() if pd.Timestamp(r) >= cutoff}
    orig_cot = ing.load_cot_yen
    monkeypatch.setattr(ing, "load_cot_yen", lambda: orig_cot().loc[
        lambda df: (df["as_of"] <= cutoff.date()) & ~df["as_of"].isin(unreleased)].reset_index(drop=True))

    return al.build()[0]


@pytest.mark.parametrize("cutoff", ["2018-01-02", "2019-02-20", "2024-08-05"])
def test_truncated_run_matches_full_run(baseline, monkeypatch, cutoff):
    """Run on data cut off at t; every value up to t must equal the full run."""
    cutoff = pd.Timestamp(cutoff)
    truncated = _truncated_build(monkeypatch, cutoff)
    assert truncated.index.max() == cutoff
    pd.testing.assert_frame_equal(baseline.loc[:cutoff], truncated, check_exact=True)


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


# Spot-checked Bitstamp values: (date, column, value, UTC open time of the candle whose close is used).
PINNED_BTC = [
    ("2024-08-05", "BTC_noon", 54385.0, "2024-08-05 15:00"),   # summer, EDT = UTC-4
    ("2024-08-05", "BTC_close", 53483.0, "2024-08-05 19:00"),
    ("2024-08-05", "BTC_1700", 54407.0, "2024-08-05 20:00"),
    ("2018-01-02", "BTC_noon", 13689.09, "2018-01-02 16:00"),  # winter, EST = UTC-5
    ("2018-01-02", "BTC_close", 14839.76, "2018-01-02 20:00"),
    ("2018-01-02", "BTC_1700", 14802.0, "2018-01-02 21:00"),
]


@pytest.mark.parametrize("day, column, value, candle_open_utc", PINNED_BTC)
def test_btc_pinned_spot_check_values(baseline, day, column, value, candle_open_utc):
    """Expected values are the spot-checked closes (docs/spot_check.md), hard-coded here."""
    assert baseline.loc[day, column] == value
    hour = {"BTC_noon": 11, "BTC_close": 15, "BTC_1700": 16}[column]
    expected_open = pd.Timestamp(f"{day} {hour:02d}:00", tz=al.NY).tz_convert("UTC")
    assert expected_open == pd.Timestamp(candle_open_utc, tz="UTC")


@pytest.mark.parametrize("target", [12, 16, 17])
@pytest.mark.parametrize("day", ["2024-08-05", "2018-01-02"])
def test_btc_ignores_candles_from_target_onward(target, day):
    hourly = _hourly("2017-12-30", "2018-01-04") if day.startswith("2018") else _hourly("2024-08-04", "2024-08-07")
    days = pd.DatetimeIndex([day])
    before, _, _ = al.sample_btc(hourly, days, target)
    target_utc = pd.Timestamp(f"{day} {target:02d}:00", tz=al.NY).tz_convert("UTC")
    hourly.loc[hourly.index >= target_utc, "close"] *= SCRAMBLE  # the candle opening at T and everything later
    after, _, _ = al.sample_btc(hourly, days, target)
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


# r_dxyx computed from the H.10 noon rates for 2024-08-05 to 2024-08-09 typed in
# tests/verify_dxyx_formula.py, with the section 4 weights; independent of the FRED files.
PINNED_DXYX = {
    "2024-08-06": 0.0022843389556390613,
    "2024-08-07": -0.0004361586255433554,
    "2024-08-08": 0.000902187556346464,
    "2024-08-09": -0.0012550694558881796,
}


@pytest.mark.parametrize("day, value", PINNED_DXYX.items())
def test_r_dxyx_matches_h10_values(baseline, day, value):
    assert baseline.loc[day, "r_dxyx"] == pytest.approx(value, rel=1e-12, abs=0)


def test_fx_real_data_missing_days_are_blank(baseline):
    flagged = baseline[baseline["fx_missing"]]
    assert len(flagged) > 0
    assert flagged[["r_yen", "r_dxyx", "r_eurusd"]].isna().all().all()


# ---------------------------------------------------------------- 4. Treasuries (DEVIATIONS.md US Treasury holiday rule) and JGBs (rules 6, 7)

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
    pd.testing.assert_series_equal(b["dJP2Y_lag1"].iloc[1:], b["dJP2Y"].shift(1).iloc[1:], check_names=False,
                                   check_exact=True)
    expected = (b["US2Y"] - b["JP2Y"]).shift(1)
    pd.testing.assert_series_equal(b["spread_lag1"].iloc[1:], expected.iloc[1:], check_names=False, check_exact=True)


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


# ---------------------------------------------------------------- 6. rolling and expanding statistics (src/signals.py)

import signals as sg  # noqa: E402

SIGNAL_CUTOFFS = ["2018-06-29", "2020-03-16", "2024-08-05"]
SIGNAL_STATS = ["yen_move5", "yen_vol20", "yen_vol20_pct80", "yen_move5_pct90", "yen_move5_pct85",
                "yen_move5_pct95", "spx_ret5", "vix_pct", "spx_ret5_pct",
                "vix_change5", "vix_change5_pct2", "spx_ret5_pct2"]
SIGNAL_OUTPUTS = ["S_yen", "S_yen_p85", "S_yen_p95", "S_vix", "S_eq", "S_vix2", "S_eq2", "signal_unavailable",
                  "yen_move5_long_window"]
for _name in ["S_yen", "S_yen_p85", "S_yen_p95", "S_vix", "S_eq", "S_vix2", "S_eq2"]:
    SIGNAL_OUTPUTS += [f"{_name}_episode", f"{_name}_episode_start"]


@pytest.fixture(scope="module")
def signals_full(baseline):
    return sg.compute(baseline)


@pytest.fixture(scope="module")
def signals_truncated(baseline, signals_full):
    """Signals rerun on aligned data cut off at each date, with the calibrated q held fixed."""
    out = signals_full[0]
    q = dict(q_v=int(out["q_v"].iloc[0]), q_e=int(out["q_e"].iloc[0]),
             q_v2=float(out["q_v2"].iloc[0]), q_e2=float(out["q_e2"].iloc[0]))
    return {c: sg.compute(baseline.loc[:c], **q)[0] for c in SIGNAL_CUTOFFS}


@pytest.mark.parametrize("cutoff", SIGNAL_CUTOFFS)
@pytest.mark.parametrize("column", SIGNAL_STATS + SIGNAL_OUTPUTS)
def test_signal_statistic_uses_data_through_t_only(signals_full, signals_truncated, column, cutoff):
    """One test per rolling or expanding statistic: a run cut off at t reproduces every value up to t."""
    full = signals_full[0].loc[:cutoff, column]
    pd.testing.assert_series_equal(full, signals_truncated[cutoff][column], check_exact=True)


def test_expanding_percentile_matches_numpy_through_t(signals_full):
    out = signals_full[0]
    for day in ["2018-01-02", "2021-06-30", "2024-08-05"]:
        hist = out.loc[:day, "yen_move5"].dropna().to_numpy()
        assert out.loc[day, "yen_move5_pct90"] == pytest.approx(np.percentile(hist, 90))


def test_calibration_ignores_test_period(baseline, signals_full):
    """Scrambling VIX, S&P 500 and the yen inputs after the training period must not change q_V, q_E, or the grid."""
    d = baseline.copy()
    after = d.index > sg.TRAIN_END
    d.loc[after, ["VIX", "SPX", "DEXJPUS", "r_yen"]] *= SCRAMBLE
    _, grid, _ = sg.compute(d)
    pd.testing.assert_frame_equal(grid, signals_full[1], check_exact=True)


def test_calibration_on_data_cut_at_training_end_matches_full_run(baseline, signals_full):
    """q_V, q_E, q_V2, q_E2 and both grids from data ending at the training end equal the full run."""
    out, grid, grid2 = sg.compute(baseline.loc[:sg.TRAIN_END])
    pd.testing.assert_frame_equal(grid, signals_full[1], check_exact=True)
    pd.testing.assert_frame_equal(grid2, signals_full[2], check_exact=True)
    for q in ["q_v", "q_e", "q_v2", "q_e2"]:
        assert out[q].iloc[0] == signals_full[0][q].iloc[0]


def test_vol_burn_in_on_real_data(baseline, signals_full):
    """yen_vol20 is blank until 20 daily returns exist (approved burn-in rule)."""
    first_vol = signals_full[0]["yen_vol20"].first_valid_index()
    twentieth_return = baseline["r_yen"].dropna().index[19]
    assert first_vol == twentieth_return


def test_pick_primary_on_synthetic_ties():
    grid = pd.DataFrame({"signal": ["vix"] * 3 + ["equity"] * 3, "q": [97, 98, 99, 1, 2, 3],
                         "abs_diff": [0.02, 0.01, 0.01, 0.01, 0.01, 0.02]})
    assert sg.pick_primary(grid, "vix", prefer_high=True) == 99
    assert sg.pick_primary(grid, "equity", prefer_high=False) == 1


def test_calibration_tie_break_prefers_fewer_firings(signals_full):
    grid = signals_full[1]
    eq = grid[grid["signal"] == "equity"]
    best = eq[np.isclose(eq["abs_diff"], eq["abs_diff"].min(), rtol=0, atol=1e-12)]
    assert int(eq.loc[eq["chosen"], "q"].iloc[0]) == int(best["q"].min())
    vix = grid[grid["signal"] == "vix"]
    best = vix[np.isclose(vix["abs_diff"], vix["abs_diff"].min(), rtol=0, atol=1e-12)]
    assert int(vix.loc[vix["chosen"], "q"].iloc[0]) == int(best["q"].max())


def test_secondary_calibration_ignores_test_period(baseline, signals_full):
    """Scrambling VIX, S&P 500 and the yen inputs after the training period must not change the secondary grid."""
    d = baseline.copy()
    after = d.index > sg.TRAIN_END
    d.loc[after, ["VIX", "SPX", "DEXJPUS", "r_yen"]] *= SCRAMBLE
    _, _, grid2 = sg.compute(d)
    pd.testing.assert_frame_equal(grid2, signals_full[2], check_exact=True)


def test_secondary_choice_is_eligible_and_follows_logged_order(signals_full):
    grid2 = signals_full[2]
    chosen = grid2[grid2["chosen"]]
    assert len(chosen) == 2 and chosen["eligible"].all()
    assert (chosen["training_days"] <= 2 * chosen["yen_training_days"]).all()
    for name, prefer_high in [("vix_change5", True), ("equity", False)]:
        g = grid2[(grid2["signal"] == name) & grid2["eligible"]]
        best = g[g["episode_diff"] == g["episode_diff"].min()]
        best = best[best["day_diff"] == best["day_diff"].min()]
        expected = best["q"].max() if prefer_high else best["q"].min()
        assert float(chosen.loc[chosen["signal"] == name, "q"].iloc[0]) == expected


def test_pick_secondary_on_synthetic_grid():
    grid = pd.DataFrame({
        "signal": ["equity"] * 4,
        "q": [0.5, 0.6, 0.7, 70.0],
        "episode_diff": [1, 1, 1, 0],
        "day_diff": [5, 3, 3, 700],
        "eligible": [True, True, True, False],  # the exact episode match fires too often: excluded
    })
    assert sg.pick_secondary(grid, "equity", prefer_high=False) == 0.6
    assert sg.pick_secondary(grid.assign(signal="vix_change5"), "vix_change5", prefer_high=True) == 0.7


def test_yen_vol_rules_on_synthetic_returns():
    days = pd.bdate_range("2024-01-01", periods=30)
    r = pd.Series(np.linspace(-0.01, 0.01, 30), index=days)
    d = pd.DataFrame({"r_yen": r, "fx_multi_day": False}, index=days)
    vol = sg.yen_vol(d)
    assert np.isnan(vol.iloc[18])  # fewer than 20 returns so far
    assert vol.iloc[25] == pytest.approx(r.iloc[6:26].std(ddof=1) * np.sqrt(252))
    # Two missing days and two multi-day returns leave 16 usable: blank.
    d2 = d.copy()
    d2.iloc[[20, 22], 0] = np.nan
    d2.iloc[[21, 23], 1] = True
    assert np.isnan(sg.yen_vol(d2).iloc[25])
    # One missing and one multi-day leave 18: defined, and the multi-day return is not used.
    d3 = d.copy()
    d3.iloc[20, 0] = np.nan
    d3.iloc[21, 1] = True
    usable = r.iloc[6:26].drop(r.index[[20, 21]])
    assert sg.yen_vol(d3).iloc[25] == pytest.approx(usable.std(ddof=1) * np.sqrt(252))


def test_yen_move_uses_last_fixing_on_or_before_t_minus_5():
    days = pd.bdate_range("2024-10-07", periods=8)
    usdjpy = pd.Series([150, 151, 152, 153, 154, 155, 156, 157], index=days, dtype=float)
    usdjpy.iloc[1] = np.nan  # no fixing on 2024-10-08
    move, long_window = sg.yen_move(pd.DataFrame({"DEXJPUS": usdjpy}))
    assert move.iloc[6] == pytest.approx(-np.log(156 / 150)) and long_window.iloc[6]
    assert move.iloc[7] == pytest.approx(-np.log(157 / 152)) and not long_window.iloc[7]


def test_unavailable_days_are_zero_and_burn_in_is_blank(signals_full):
    out = signals_full[0]
    unavailable = out[out["signal_unavailable"]]
    assert len(unavailable) == 44 and (unavailable["S_yen"] == 0).all()
    assert out.loc[:"2017-12-29", ["S_yen", "S_yen_p85", "S_yen_p95", "S_vix", "S_eq"]].isna().all().all()


def test_episode_gap_of_ten_trading_days():
    s = pd.Series(0, index=pd.RangeIndex(40), dtype="Int8")
    s.iloc[[0, 10, 21, 22]] = 1  # 10 days apart: same episode; 11 days apart: new episode
    ep, start = sg.episodes(s)
    assert list(ep.iloc[[0, 10, 21, 22]]) == [1, 1, 2, 2]
    assert list(start[start].index) == [0, 21]
