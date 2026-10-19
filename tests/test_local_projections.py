"""Tests for the regression samples in src/local_projections.py. Nothing here fits a model.

Run from the repository root:  python -m pytest tests/test_local_projections.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import local_projections as lp  # noqa: E402

N_FX_MULTI_DAY = 23  # DEVIATIONS.md (pre-analysis): fx_multi_day days, 2018-01-02 to 2026-09-30
BASE = ["H1a", "H4", "M1", "H1d", "H1d_h0", "decomp_H1a", "decomp_H1b_h5"] + [f"H1b_h{h}" for h in range(1, 11)]


@pytest.fixture(scope="module")
def data():
    return lp.load()


@pytest.fixture(scope="module")
def panels(data):
    return lp.daily_panel(data), lp.weekly_panel(data)


@pytest.fixture(scope="module")
def specs(panels):
    return lp.registry(panels[0])


@pytest.fixture(scope="module")
def samples(data):
    return {sub: lp.build_samples(sub, data) for sub in ["full", "train", "test"]}


# ---------------------------------------------------------------- fx_multi_day exclusion

def test_exclusion_rule_holds_for_every_daily_spec_using_fx_terms(samples, panels):
    """Rule, not list: no estimation sample with a fixing-based daily return (dependent variable or
    regressor) contains a flagged row (DEVIATIONS.md, extended fx_multi_day rule)."""
    flagged = set(panels[0].index[panels[0]["fx_multi_day"]])
    checked = 0
    for sub, built in samples.items():
        for name, s in built.items():
            if s.spec.frequency == "daily" and s.spec.uses_fx_terms:
                assert flagged.isdisjoint(s.index), f"{name} ({sub}) contains fx_multi_day rows"
                checked += 1
    assert checked > 100  # every daily spec in all three subsamples


def test_exactly_23_rows_dropped_in_full_sample(panels, samples):
    daily, weekly = panels
    spec = samples["full"]["H1a"].spec
    with_rule = lp.make_sample(spec, daily, weekly, "full")
    without_rule = lp.make_sample(spec, daily, weekly, "full", _apply_fx_rule=False)
    dropped = without_rule.index.difference(with_rule.index)
    flagged = daily.index[daily["fx_multi_day"]]
    assert len(flagged) == N_FX_MULTI_DAY
    assert len(dropped) == N_FX_MULTI_DAY
    assert dropped.equals(flagged)  # all 23 would otherwise be complete rows
    assert with_rule.dropped_fx_multi_day.equals(flagged)


@pytest.mark.parametrize("name", ["H1a", "H1b_h5", "H1d", "M1", "H4", "rc3_H1d", "rc3_M1", "rc5_H1d"])
def test_each_core_spec_drops_the_same_23_rows(samples, panels, name):
    flagged = panels[0].index[panels[0]["fx_multi_day"]]
    assert samples["full"][name].dropped_fx_multi_day.equals(flagged)


def test_registry_covers_every_logged_analysis(specs):
    groups = {s.group for s in specs}
    required = {"H1a", "H1b", "H1d", "H4", "M1", "decomposition",
                "rc1", "rc2", "rc3", "rc4", "rc5", "rc6", "rc7", "rc8", "rc10", "rc11"}
    assert required <= groups
    assert {s.h for s in specs if s.group == "H1b"} == set(range(1, 11))
    # Every daily regression contains a fixing-based daily return, so the rule reaches all of them.
    for s in specs:
        if s.frequency == "daily":
            assert s.uses_fx_terms, f"{s.name} has no fixing-based daily return"


def test_rule_covers_dependent_variable():
    spec = lp.Spec("x", "test", "r_yen", ("dJP2Y", "dUS2Y"))
    assert spec.uses_fx_terms
    assert lp.Spec("x", "test", "r_btc_noon", ("r_eurusd",)).uses_fx_terms
    assert lp.Spec("x", "test", "r_btc_noon", ("r_dtwexbgs",)).uses_fx_terms
    assert not lp.Spec("x", "test", "r_btc_noon", ("dJP2Y", "dUS2Y")).uses_fx_terms


@pytest.mark.parametrize("group, var", [("rc2", "btc"), ("rc3", "r_dxyx"), ("rc4", "r_dxyx"),
                                        ("rc5", "dUS2Y"), ("rc11", None)])
def test_substitution_checks_cover_every_base_regression_with_the_variable(specs, group, var):
    by = {s.name: s for s in specs}
    if var == "btc":
        expected = {n for n in BASE if any(v.startswith(("r_btc_", "lp_btc_")) for v in (by[n].y, *by[n].x))}
    elif var is None:
        expected = set(BASE)
    else:
        expected = {n for n in BASE if by[n].uses(var)}
    got = {s.name.split("_", 1)[1] for s in specs if s.group == group}
    assert got == expected


def test_targeted_checks_follow_section_9(specs):
    targets = {g: sorted(s.name for s in specs if s.group == g) for g in ["rc6", "rc7", "rc8"]}
    assert targets == {"rc6": ["rc6_H1a", "rc6_H1b_h5"], "rc7": ["rc7_H1d"], "rc8": ["rc8_H1d"]}
    assert all(s.y == "r_btc_noon" and s.x == lp.X_H1A for s in specs if s.group == "rc10")


def test_weekly_check_is_outside_the_rule(samples):
    for name, s in samples["full"].items():
        if s.spec.frequency == "weekly":
            assert len(s.dropped_fx_multi_day) == 0
            assert s.spec.group == "rc1"


def test_check4_also_drops_dtw_multi_day(samples, panels):
    dtw = set(panels[0].index[panels[0]["dtw_multi_day"]])
    for name, s in samples["full"].items():
        if s.spec.group == "rc4":
            assert dtw.isdisjoint(s.index)


# ---------------------------------------------------------------- variable construction

def test_verdict_specs_are_h1a_h1b5_h1d(specs):
    assert sorted(s.name for s in specs if s.verdict) == ["H1a", "H1b_h5", "H1d"]


def test_regressors_match_preregistration(specs):
    by = {s.name: s for s in specs}
    assert by["H1a"].x == ("r_yen", "S_x_ryen", "S", "r_spx", "dVIX", "r_dxyx", "dUS2Y", "r_btc_noon_lag1")
    assert by["H1b_h5"].x == ("r_yen", "S_x_ryen", "S", "r_spx", "dVIX", "r_dxyx", "dUS2Y", "r_btc_1700")
    assert by["H1d"].x == ("dJP2Y", "dUS2Y", "r_spx", "dVIX", "r_dxyx", "r_btc_1700")
    assert by["M1"].y == "r_yen" and by["M1"].x == ("dJP2Y", "dUS2Y", "r_spx", "dVIX", "r_dxyx")
    assert by["H4"].x[-1] == "post_S_ryen"


def test_lp_dependent_is_exactly_h_trading_days_ahead(data, panels):
    daily = panels[0]
    for day in ["2018-03-01", "2024-08-05"]:
        pos = data.index.get_loc(pd.Timestamp(day))
        for h in [1, 5, 10]:
            expected = np.log(data["BTC_1700"].iloc[pos + h] / data["BTC_1700"].iloc[pos])
            assert daily.loc[day, f"lp_btc_h{h}"] == expected


def test_lp_dependent_never_reaches_past_sample_end(panels):
    daily = panels[0]
    assert daily[f"lp_btc_h10"].loc["2026-09-17":].isna().all()  # last 10 trading days


def test_interactions_and_lags(panels):
    daily = panels[0]
    pd.testing.assert_series_equal(daily["S_x_ryen"], daily["S"] * daily["r_yen"], check_names=False)
    assert (daily.loc[daily.index < lp.POST2024, "post2024"] == 0).all()
    assert (daily.loc[daily.index >= lp.POST2024, "post2024"] == 1).all()
    assert daily["r_btc_noon_lag1"].iloc[1:].equals(daily["r_btc_noon"].shift(1).iloc[1:])


def test_fit_call_is_pinned():
    assert lp.STATSMODELS_VERSION == "0.15.0"
    assert lp.fit_kwargs(0) == {"cov_type": "HAC",
                                "cov_kwds": {"maxlags": 5, "kernel": "bartlett", "use_correction": True},
                                "use_t": False}
    assert lp.fit_kwargs(10)["cov_kwds"]["maxlags"] == 10


# ---------------------------------------------------------------- winsorization (check 11)

def test_winsor_bounds_use_training_period_only(panels):
    daily = panels[0]
    scrambled = daily.copy()
    scrambled.loc[scrambled.index > lp.TRAIN_END, ["r_yen", "r_btc_noon"]] *= 50
    for col in ["r_yen", "r_btc_noon"]:
        assert lp.winsor_bounds(daily, col) == lp.winsor_bounds(scrambled, col)


def test_interaction_formed_after_winsorizing_r_yen(panels):
    daily = panels[0]
    cols = ["r_btc_noon", *lp.X_H1A]
    w = lp.winsorize(daily, cols)
    lo, hi = lp.winsor_bounds(daily, "r_yen")
    assert w["r_yen"].min() >= lo and w["r_yen"].max() <= hi
    pd.testing.assert_series_equal(w["S_x_ryen"], w["S"] * w["r_yen"], check_names=False)
    assert w["S"].equals(daily["S"])


# ---------------------------------------------------------------- weekly (check 1) and leave-one-out (check 10)

def test_weekly_signal_is_any_day_thursday_to_wednesday(data):
    a = data.loc["2017-12-01":"2018-03-01"].copy()
    a["S_yen"] = 0.0
    a.loc["2018-01-25", "S_yen"] = 1.0  # Thursday: belongs to the week ending Wednesday 2018-01-31
    w = lp.weekly_panel(a)
    assert w.loc["2018-01-31", "S"] == 1.0
    assert w.loc["2018-01-24", "S"] == 0.0 and w.loc["2018-02-07", "S"] == 0.0


def test_weekly_anchor_uses_last_fixing_on_or_before_wednesday(data):
    a = data.loc["2017-12-01":"2018-03-01"].copy()
    a.loc["2018-01-31", "DEXJPUS"] = np.nan  # no fixing on the anchor Wednesday
    w = lp.weekly_panel(a)
    expected = -np.log(a.loc["2018-01-30", "DEXJPUS"] / a.loc["2018-01-24", "DEXJPUS"])
    assert w.loc["2018-01-31", "r_yen"] == pytest.approx(expected, rel=1e-15)


def test_leave_one_out_drops_whole_episode_span(samples, panels):
    daily = panels[0]
    for name, s in samples["full"].items():
        if s.spec.group == "rc10":
            start, end = s.spec.drop_span
            assert not ((s.index >= start) & (s.index <= end)).any()
            ep_days = daily.index[daily["S_yen_episode"] == int(name.rsplit("ep", 1)[1])]
            assert ep_days.min() == start and ep_days.max() == end
