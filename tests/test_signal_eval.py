"""Tests for src/signal_eval.py on synthetic data only. No H2 outcome is computed here.

Run from the repository root:  python -m pytest tests/test_signal_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import signal_eval as se  # noqa: E402

NY = se.NY


def flat_hourly(start: str, end: str, price: float = 100.0) -> pd.Series:
    """Candle closes indexed by close time (UTC), all equal to `price`."""
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.Series(price, index=idx, dtype=float)


def ny_close(day: str, hour: int) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hour:02d}:00", tz=NY).tz_convert("UTC")


# ---------------------------------------------------------------- target window boundaries

# Trading days spanning a weekend and the 2024-03-10 daylight-saving change.
DAYS = pd.bdate_range("2024-03-01", "2024-03-29")
T0, T10 = DAYS[0], DAYS[10]


def run_target(drop_at: pd.Timestamp | None, drop: float = 0.10) -> float:
    closes = flat_hourly("2024-02-28", "2024-04-05")
    if drop_at is not None:
        closes[drop_at] = 100.0 * (1 - drop)
    btc_close = pd.Series(100.0, index=DAYS)
    T, complete = se.target(DAYS, btc_close, closes)
    assert complete[T0]
    return T[T0]


def test_drop_exactly_at_1600_on_day_t_plus_10_counts():
    assert run_target(ny_close(T10.strftime("%Y-%m-%d"), 16)) == 1  # after the DST change


def test_drop_one_hour_after_window_end_does_not_count():
    assert run_target(ny_close(T10.strftime("%Y-%m-%d"), 17)) == 0


def test_reference_candle_closing_at_1600_on_day_t_does_not_count():
    assert run_target(ny_close(T0.strftime("%Y-%m-%d"), 16)) == 0


def test_first_hour_after_reference_counts_and_weekend_hours_count():
    assert run_target(ny_close(T0.strftime("%Y-%m-%d"), 17)) == 1
    assert run_target(pd.Timestamp("2024-03-02 15:00", tz="UTC")) == 1  # Saturday


def test_drop_just_under_ten_percent_does_not_count():
    assert run_target(ny_close(T10.strftime("%Y-%m-%d"), 12), drop=0.0999) == 0
    assert run_target(ny_close(T10.strftime("%Y-%m-%d"), 12), drop=0.10) == 1


def test_incomplete_windows_excluded():
    closes = flat_hourly("2024-02-28", "2024-04-05")
    T, complete = se.target(DAYS, pd.Series(100.0, index=DAYS), closes)
    assert complete.iloc[:-10].all() and not complete.iloc[-10:].any()  # no day t+10 in the sample
    assert T.iloc[-10:].isna().all()
    short = closes[closes.index < ny_close("2024-03-20", 12)]  # hourly data ends early
    T2, c2 = se.target(DAYS, pd.Series(100.0, index=DAYS), short)
    assert not c2[pd.Timestamp("2024-03-06")]  # its window ends 2024-03-20 16:00


def test_hourly_closes_shift_open_label_by_one_hour():
    h = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(
        ["2024-08-05 19:00", "2024-08-05 20:00"], tz="UTC", name="open_utc"))
    c = se.hourly_closes(h)
    assert list(c.index) == [pd.Timestamp("2024-08-05 20:00", tz="UTC"), pd.Timestamp("2024-08-05 21:00", tz="UTC")]


# ---------------------------------------------------------------- metrics on a hand-built series

def test_metrics_hand_computed():
    n = 60
    sig = np.zeros(n); T = np.zeros(n); complete = np.ones(n, dtype=bool)
    sig[[5, 12, 30, 45]] = 1          # episodes start at 5 (12 joins it), 30, 45
    T[[5, 6, 45, 50]] = 1              # target days: event at 5 (6 joins), event at 45 (50 joins)
    m = se.metrics(sig, T, complete, np.ones(n, dtype=bool))
    assert (m.episodes, m.complete_episodes, m.true_pos, m.false_pos) == (3, 3, 2, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.events == 2  # days 5-6 and 45-50 group into two events (45 and 50 are 5 days apart)
    assert m.preceded_events == 2 and m.recall == 1.0
    assert m.base_rate == pytest.approx(4 / 60)


def test_recall_same_day_counts_eleven_days_before_does_not():
    n = 40
    complete = np.ones(n, dtype=bool)
    T = np.zeros(n); T[20] = 1
    same = np.zeros(n); same[20] = 1
    early = np.zeros(n); early[9] = 1
    assert se.metrics(same, T, complete, np.ones(n, dtype=bool)).recall == 1.0
    assert se.metrics(early, T, complete, np.ones(n, dtype=bool)).recall == 0.0


def test_period_assignment_by_start_day_and_incomplete_starts_unscored():
    n = 40
    sig = np.zeros(n); sig[[8, 15]] = 1  # one episode starting at 8 (15 joins)
    T = np.zeros(n); T[8] = 1
    complete = np.ones(n, dtype=bool)
    first, second = np.arange(n) < 10, np.arange(n) >= 10
    assert se.metrics(sig, T, complete, first).episodes == 1
    assert se.metrics(sig, T, complete, second).episodes == 0
    complete[8] = False
    m = se.metrics(sig, T, complete, first)
    assert m.episodes == 1 and m.complete_episodes == 0 and np.isnan(m.precision)


# ---------------------------------------------------------------- bootstrap

def test_stationary_indices_mean_block_length_and_range():
    rng = np.random.default_rng(0)
    idx = np.concatenate([se.stationary_indices(500, rng) for _ in range(200)])
    assert idx.min() >= 0 and idx.max() < 500
    starts = 0
    for k in range(200):
        seq = idx[k * 500:(k + 1) * 500]
        starts += 1 + int((np.diff(seq) % 500 != 1).sum())
    assert 500 * 200 / starts == pytest.approx(20, rel=0.08)


def synthetic_period(n=300, seed=1):
    rng = np.random.default_rng(seed)
    T = (rng.random(n) < 0.1).astype(float)
    complete = np.ones(n, dtype=bool)
    sigs = {name: (rng.random(n) < 0.03).astype(float) for name in se.SIGNALS_EVALUATED}
    return sigs, T, complete


def test_bootstrap_is_reproducible_and_paired():
    sigs, T, complete = synthetic_period()
    a = se.bootstrap(sigs, T, complete, reps=50, seed=42)
    b = se.bootstrap(sigs, T, complete, reps=50, seed=42)
    pd.testing.assert_frame_equal(a, b)
    d = a["S_yen_precision"] - a["S_vix_precision"]
    pd.testing.assert_series_equal(d, a["diff_primary"], check_names=False)  # same draws for both


def test_undefined_replicates_reported_and_excluded():
    sigs, T, complete = synthetic_period()
    sigs["S_vix2"] = np.zeros_like(T)  # never fires: precision undefined in every replicate
    reps = se.bootstrap(sigs, T, complete, reps=40, seed=42)
    ci = se.intervals(reps).set_index("statistic")
    assert ci.loc["S_vix2_precision", "share_undefined"] == 1.0
    assert np.isnan(ci.loc["S_vix2_precision", "lower95"])
    assert ci.loc["diff_secondary", "share_undefined"] == 1.0
    defined = reps["S_yen_precision"].dropna()
    assert ci.loc["S_yen_precision", "lower95"] == pytest.approx(np.percentile(defined, 2.5))
    assert ci.loc["S_yen_precision", "share_undefined"] == pytest.approx(reps["S_yen_precision"].isna().mean())


# ---------------------------------------------------------------- verdict

def test_h2_margin_boundary():
    assert not se.h2_pass(0.499, 0.40, 0.1)   # 9.9 pp
    assert se.h2_pass(0.50, 0.40, 0.1)        # exactly 10.0 pp
    assert se.h2_pass(0.30, 0.20, 0.1)        # exactly 10.0 pp, other values


def test_h2_requires_precision_above_base_rate():
    assert not se.h2_pass(0.30, 0.10, 0.30)
    assert not se.h2_pass(0.30, 0.10, 0.35)
    assert not se.h2_pass(np.nan, 0.10, 0.05)


def test_combined_verdict_requires_both_comparisons():
    M = se.Metrics
    yen = M(10, 10, 5, 5, 0.5, 4, 2, 0.5, 0.1)
    good, bad = M(5, 5, 1, 4, 0.2, 4, 1, 0.25, 0.1), M(5, 5, 3, 2, 0.6, 4, 1, 0.25, 0.1)
    v = se.verdict_table({"S_yen": yen, "S_vix": good, "S_vix2": bad})
    assert list(v["pass"]) == [True, False, False]
    v2 = se.verdict_table({"S_yen": yen, "S_vix": good, "S_vix2": good})
    assert list(v2["pass"]) == [True, True, True]
    assert list(v["competitor_episodes"].iloc[:2]) == [5, 5]


def test_constants_match_preregistration():
    assert (se.DROP, se.WINDOW_DAYS, se.RECALL_DAYS) == (0.10, 10, 10)
    assert (se.BOOT_REPS, se.BLOCK_MEAN, se.SEED) == (2000, 20, 42)
    assert se.CI == (2.5, 97.5) and se.PRECISION_MARGIN == 0.10
    assert se.COMPARISONS == {"primary": "S_vix", "secondary": "S_vix2"}
