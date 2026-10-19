"""Tests for src/event_study.py on synthetic data only. No H1c outcome is computed here.

Run from the repository root:  python -m pytest tests/test_event_study.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import event_study as es  # noqa: E402

DAYS = pd.bdate_range("2020-01-01", periods=700)


def synthetic(alpha=0.001, beta=1.5, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    r_spx = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    r_btc = alpha + beta * r_spx + pd.Series(rng.normal(0, noise, len(DAYS)), index=DAYS)
    return r_btc, r_spx


# ---------------------------------------------------------------- event days and episodes

def test_trading_day_mapping_shifts_to_next_day_and_flags():
    assert es.to_trading_day("2020-01-06", DAYS) == (pd.Timestamp("2020-01-06"), False)
    assert es.to_trading_day("2020-01-04", DAYS) == (pd.Timestamp("2020-01-06"), True)  # Saturday


def test_cluster_merges_within_10_trading_days():
    d = [DAYS[100], DAYS[110], DAYS[121], DAYS[300]]
    assert es.cluster(d, DAYS) == [[DAYS[100], DAYS[110]], [DAYS[121]], [DAYS[300]]]


# ---------------------------------------------------------------- market model and CARs

def test_estimation_window_is_minus_260_to_minus_11_minus_exclusions():
    pos = es.estimation_positions(400, set())
    assert pos.min() == 140 and pos.max() == 389 and len(pos) == 250
    excl = es.exclusion_positions([300], 700)
    assert excl == set(range(299, 311))
    pos2 = es.estimation_positions(400, excl)
    assert len(pos2) == 250 - 12 and not (set(pos2) & excl)


def test_market_model_recovers_parameters_and_car_arithmetic():
    r_btc, r_spx = synthetic(noise=1e-9)
    res = es.event_result("X", DAYS[400], r_btc, r_spx, set())
    assert res.alpha == pytest.approx(0.001, abs=1e-8) and res.beta == pytest.approx(1.5, abs=1e-6)
    r_btc2 = r_btc.copy()
    r_btc2.iloc[399:403] += [0.01, -0.02, 0.03, 0.005]  # days -1..+2 of the event
    res2 = es.event_result("X", DAYS[400], r_btc2, r_spx, set())
    assert res2.cars["car_m1_p2"] == pytest.approx(0.025, abs=1e-6)
    assert res2.raw["raw_m1_p2"] == pytest.approx(float(r_btc2.iloc[399:403].sum()), rel=1e-12)
    assert res2.cars["car_m1_p5"] == pytest.approx(0.025, abs=1e-6)


def test_short_estimation_window_is_flagged():
    r_btc, r_spx = synthetic()
    excl = set(range(150, 220))  # removes 70 of 250 estimation days
    res = es.event_result("X", DAYS[400], r_btc, r_spx, excl)
    assert res.n_est == 180 and res.flag_short_estimation


def test_car_variance_patell_form():
    r_btc, r_spx = synthetic()
    res = es.event_result("X", DAYS[400], r_btc, r_spx, set())
    est = es.estimation_positions(400, set())
    x = r_spx.to_numpy()
    w = np.arange(399, 403)
    L, T = 4, len(est)
    manual = res.resid_var * (L + L ** 2 / T + (x[w].sum() - L * x[est].mean()) ** 2 / ((x[est] - x[est].mean()) ** 2).sum())
    assert res.car_var_primary == pytest.approx(manual, rel=1e-12)


# ---------------------------------------------------------------- H1c test (section 2 table)

def cars_with_positive_rank(rank: int | None) -> np.ndarray:
    """Six CARs with distinct magnitudes 1..6; the one with magnitude `rank` positive (None: all negative)."""
    c = -np.arange(1, 7, dtype=float)
    if rank is not None:
        c[rank - 1] = rank
    return c


@pytest.mark.parametrize("rank, p", [(None, 0.015625), (1, 0.03125), (2, 0.046875), (3, 0.078125),
                                     (4, 0.109375), (6, 0.21875)])
def test_exact_wilcoxon_reproduces_section_2_table(rank, p):
    assert es.h1c_test(cars_with_positive_rank(rank))["p_one_sided"] == pytest.approx(p, abs=1e-12)


def test_h1c_requires_mean_and_median_negative():
    assert es.h1c_test(cars_with_positive_rank(3))["pass"]          # p = 0.078
    assert not es.h1c_test(cars_with_positive_rank(4))["pass"]      # p = 0.109
    lopsided = np.array([-0.01, -0.02, -0.03, -0.04, -0.05, 0.40])  # five small falls, one large rise: mean > 0
    r = es.h1c_test(lopsided)
    assert r["mean"] > 0 and not r["pass"]


def test_sign_test_and_bmp_by_hand():
    cars = np.array([-0.03, -0.01, 0.02, -0.04, -0.02, 0.01])
    assert es.sign_test(cars) == pytest.approx(22 / 64, abs=1e-12)  # P(X >= 4), X ~ Bin(6, 0.5)
    v = np.array([0.0004, 0.0009, 0.0001, 0.0016, 0.0004, 0.0001])
    scar = cars / np.sqrt(v)
    t = np.sqrt(6) * scar.mean() / scar.std(ddof=1)
    r = es.bmp_test(cars, v)
    assert r["t_bmp"] == pytest.approx(t, rel=1e-12)
    from scipy import stats
    assert r["p_one_sided"] == pytest.approx(stats.t.cdf(t, 5), rel=1e-12)


def test_placebo_test_is_two_sided():
    rng = np.random.default_rng(4)
    cars = rng.normal(0, 0.02, 41)
    r = es.placebo_test(cars)
    from scipy import stats
    assert r["p_two_sided"] == pytest.approx(stats.wilcoxon(cars, method="exact").pvalue, rel=1e-12)
    assert r["significant"] == (r["p_two_sided"] < 0.10) and r["n"] == 41


# ---------------------------------------------------------------- robustness checks

def test_leave_one_out_drops_exactly_one():
    cars = dict(zip(["A1", "A2", "A3", "A4", "A5", "A6"], cars_with_positive_rank(2)))
    loo = es.leave_one_out(cars)
    assert list(loo["dropped"]) == list(cars) and (loo["n"] == 5).all()


def test_check_9_episodes():
    assert es.A_NO_BOJ_OVERLAP == ["A2", "A4", "A5"]


def test_event_day_shift_minus_one_reanchors():
    r_btc, r_spx = synthetic()
    a = es.event_result("X", DAYS[400], r_btc, r_spx, set())
    b = es.event_result("X", DAYS[399], r_btc, r_spx, set())
    assert b.anchor == DAYS[399] and a.cars != b.cars


# ---------------------------------------------------------------- guards and checks

def test_run_refused_before_mof_coverage_and_finalized_a6():
    with pytest.raises(RuntimeError, match="covers only through"):
        es.check_ready(date(2026, 6, 30), "A6 finalized", finalized=True)
    with pytest.raises(RuntimeError, match="not finalized"):
        es.check_ready(date(2026, 9, 30), "nothing logged", finalized=True)
    with pytest.raises(RuntimeError, match="not finalized"):
        es.check_ready(date(2026, 9, 30), "A6 finalized", finalized=False)
    es.check_ready(date(2026, 9, 30), "row: A6 finalized from MoF data", finalized=True)
    assert es.A6_FINALIZED is False  # as committed now


def test_mof_check_matches_and_rejects_mismatch():
    pair = "the US dollar (sold) the Japanese yen (bought)"
    mof = pd.DataFrame({"date": [date(2022, 9, 22), date(2022, 10, 21)], "pair": pair})
    es.check_mof(mof, {"A1": ["2022-09-22"], "A2": ["2022-10-21"]}, date(2022, 12, 31))
    with pytest.raises(RuntimeError, match="differ"):
        es.check_mof(mof, {"A1": ["2022-09-22"]}, date(2022, 12, 31))
    es.check_mof(mof, {"A1": ["2022-09-22"], "A2": ["2022-10-21"], "A6": ["2026-07-30"]}, date(2022, 12, 31))


def test_fomc_overlap_window():
    fomc = pd.DataFrame({"date": [DAYS[99].date(), DAYS[103].date()], "type": ["scheduled", "scheduled"]})
    out = es.fomc_overlaps({"X": DAYS[100]}, fomc, DAYS)
    assert list(out["fomc_date"]) == [DAYS[99].date()]  # day -1 in; day +3 out


def test_placebo_loader_takes_only_placebo_rows(tmp_path, monkeypatch):
    f = tmp_path / "placebo.csv"
    pd.DataFrame({"date": ["2019-01-23", "2019-04-25", "2022-10-28"], "placebo": [True, False, False]}).to_csv(f, index=False)
    monkeypatch.setattr(es, "PLACEBO", f)
    assert es.load_placebo() == ["2019-01-23"]


def test_first_run_guard(tmp_path, monkeypatch):
    m = tmp_path / "run_manifest_h1c.json"
    m.write_text("{}")
    monkeypatch.setattr(es, "RUN_MANIFEST", m)
    with pytest.raises(RuntimeError, match="already been produced"):
        es.main()


def test_main_refuses_now():
    """With the committed data (MoF through 2026-06-30) and A6 not finalized, the run must stop."""
    with pytest.raises(RuntimeError, match="covers only through|not finalized"):
        es.main()


def test_constants_match_preregistration():
    assert es.EST_WINDOW == (-260, -11) and es.WINDOWS["car_m1_p2"] == (-1, 2)
    assert es.H1C_ALPHA == 0.10 and es.CLUSTER_GAP == 10
    assert es.CATEGORY_A["A5"] == ["2026-04-30", "2026-05-04", "2026-05-06"]
    assert len(es.CATEGORY_B) == 9 and es.CATEGORY_C == ["2024-08-05"]
