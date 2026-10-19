"""Tests for src/backtest.py on synthetic data only. No H3 outcome is computed here.

Run from the repository root:  python -m pytest tests/test_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import backtest as bt  # noqa: E402


def prices(n: int, r_btc=None, r_acwi=None, r_fxy=None, rf=None, month_end=()) -> bt.Prices:
    z = np.zeros(n)
    me = np.zeros(n, dtype=bool)
    me[list(month_end)] = True
    pick = lambda x: z.copy() if x is None else np.asarray(x, dtype=float)
    return bt.Prices(pick(r_acwi), pick(r_btc), pick(r_fxy), pick(rf), me)


def signal(n: int, days) -> np.ndarray:
    s = np.zeros(n)
    s[list(days)] = 1
    return s


# ---------------------------------------------------------------- windows (H3-B)

def test_signal_enters_at_t_plus_1_and_exits_after_hold():
    w = bt.windows_from_signal(signal(30, [3]))
    assert w == [(4, 14)]
    state = bt.state_from_windows(30, w)
    assert not state[3] and state[4:14].all() and not state[14]


def test_restart_moves_exit_including_signal_on_exit_day():
    assert bt.windows_from_signal(signal(40, [3, 8])) == [(4, 19)]
    assert bt.windows_from_signal(signal(40, [3, 14])) == [(4, 25)]   # 14 is the exit day: restart
    assert bt.windows_from_signal(signal(40, [3, 15])) == [(4, 14), (16, 26)]


def test_signals_on_last_two_days_not_executed_but_restart_open_window():
    n = 30
    assert bt.windows_from_signal(signal(n, [n - 2])) == []
    assert bt.windows_from_signal(signal(n, [n - 1])) == []
    assert bt.windows_from_signal(signal(n, [n - 3])) == [(n - 2, n - 2 + 10)]
    assert bt.windows_from_signal(signal(n, [n - 8, n - 2])) == [(n - 7, n - 1 + 10)]


def test_window_lengths_cut_at_period_end():
    assert bt.window_lengths(30, [(4, 14), (25, 35)]) == [10, 4]


# ---------------------------------------------------------------- simulation (H3-A, H3-C, H3-D)

def test_entry_day_return_at_full_weight_then_half():
    n = 8
    r_btc = np.zeros(n); r_btc[2] = 0.10; r_btc[3] = 0.10
    p = prices(n, r_btc=r_btc)
    r, state, w = bt.run_strategy(p, signal(n, [1]))
    assert w == [(2, 12)]
    cost = 0.0020 * 2 * 0.11                       # half of 0.22 BTC sold at double cost
    assert r[1] == pytest.approx((1.02 - cost) / 1.0 - 1, rel=1e-12)   # day 2 at 20% BTC, then entry cost
    assert r[2] == pytest.approx(0.011 / 1.02, rel=1e-12)              # day 3 at 10% BTC


def test_buy_and_hold_drifts_without_trades_between_month_ends():
    n = 6
    r_btc = np.full(n, 0.05); r_btc[0] = 0
    r, _, _ = bt.run_strategy(prices(n, r_btc=r_btc), None)
    v = 0.8 + 0.2 * 1.05 ** np.arange(n)
    np.testing.assert_allclose(r, v[1:] / v[:-1] - 1, rtol=1e-12)


def test_month_end_inside_window_rebalances_to_80_10_10():
    n = 20
    r_btc = np.full(n, 0.02); r_btc[0] = 0
    p = prices(n, r_btc=r_btc, month_end=[8])
    state = bt.state_from_windows(n, [(3, 13)])
    t = bt._targets({"ACWI": 0.7, "BTC": 0.2, "SLEEVE": 0.1}, True, True, True)
    assert t == pytest.approx({"ACWI": 0.8, "BTC": 0.1, "SLEEVE": 0.1})
    t2 = bt._targets({"ACWI": 0.7, "BTC": 0.2, "SLEEVE": 0.1}, False, True, True)
    assert t2 == pytest.approx({"ACWI": 0.8, "BTC": 0.2, "SLEEVE": 0.0})
    assert len(bt.simulate(p, state)) == n - 1


def test_exit_cost_is_normal_rate_on_sleeve_moved():
    n = 20
    p = prices(n)
    state = bt.state_from_windows(n, [(2, 5)])
    r = bt.simulate(p, state)
    entry_cost = 0.0020 * 2 * 0.10
    after_entry = 1 - entry_cost
    sleeve = 0.10 * after_entry
    exit_cost = 0.0020 * sleeve
    assert r[1] == pytest.approx(-entry_cost, rel=1e-12)
    assert r[4] == pytest.approx(-exit_cost / after_entry, rel=1e-12)
    assert (r[[0, 2, 3]] == 0).all()


def test_restart_is_not_a_trade():
    n = 40
    p = prices(n, r_btc=np.linspace(-0.01, 0.01, n))
    with_restart = bt.run_strategy(p, signal(n, [3, 8]))[0]      # one window 4..19 via a restart
    single = bt.simulate(p, bt.state_from_windows(n, [(4, 19)]))
    np.testing.assert_array_equal(with_restart, single)


def test_tbill_sleeve_earns_rf_and_hedge_uses_fxy_with_double_entry_cost():
    n = 10
    rf = np.full(n, 0.0001); r_fxy = np.full(n, 0.003)
    p = prices(n, rf=rf, r_fxy=r_fxy)
    state = bt.state_from_windows(n, [(2, 8)])
    r_tb = bt.simulate(p, state)
    r_fx = bt.simulate(p, state, hedge=True)
    entry_tb = 0.0020 * 2 * 0.10
    entry_fx = 0.0020 * 2 * 0.10 + 0.0010 * 2 * 0.10
    assert r_tb[1] == pytest.approx(-entry_tb, rel=1e-12)
    assert r_fx[1] == pytest.approx(-entry_fx, rel=1e-12)
    sleeve_tb, sleeve_fx = 0.10 * (1 - entry_tb), 0.10 * (1 - entry_fx)
    assert r_tb[2] == pytest.approx(sleeve_tb * 0.0001 / (1 - entry_tb), rel=1e-9)
    assert r_fx[2] == pytest.approx(sleeve_fx * 0.003 / (1 - entry_fx), rel=1e-9)


def test_no_trade_at_final_close():
    n = 10
    p = prices(n, month_end=[n - 1])
    state = np.zeros(n, dtype=bool); state[n - 1] = True  # would be an entry at the final close
    r = bt.simulate(p, state)
    assert (r == 0).all()


# ---------------------------------------------------------------- metrics (H3-H)

def test_metrics_formulas():
    r = np.array([0.01, -0.02, 0.015, 0.0, -0.005])
    rf = np.full(5, 0.0001)
    state = np.array([False, True, True, False, False, False])
    m = bt.metrics(r, rf, state, 1)
    ex = r - rf
    assert m["sharpe"] == pytest.approx(np.sqrt(252) * ex.mean() / ex.std(ddof=1))
    assert m["sortino"] == pytest.approx(np.sqrt(252) * ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0) ** 2)))
    assert m["ann_return"] == pytest.approx(np.prod(1 + r) ** (252 / 5) - 1)
    assert m["ann_vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    v = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], v]))[1:]
    assert m["max_drawdown"] == pytest.approx((v / peak - 1).min())
    assert m["calmar"] == pytest.approx(m["ann_return"] / abs(m["max_drawdown"]))
    assert m["pct_days_derisked"] == pytest.approx(2 / 5) and m["episodes"] == 1


# ---------------------------------------------------------------- random timing (H3-F)

def test_random_windows_non_overlapping_within_period_matching_lengths():
    rng = np.random.default_rng(42)
    lengths = [10, 23, 10, 5]
    n = 300
    for _ in range(200):
        w = bt.random_windows(lengths, n, rng)
        assert sorted(x - e for e, x in w) == sorted(lengths)
        assert all(e >= 1 and x <= n - 2 for e, x in w)
        assert all(w[i][1] < w[i + 1][0] for i in range(len(w) - 1))


def test_random_timing_reproducible_and_counts():
    n = 120
    p = prices(n, r_btc=np.random.default_rng(3).normal(0, 0.02, n), rf=np.full(n, 0.0001))
    a = bt.random_timing(p, [10, 12], n_sims=25, seed=42)
    b = bt.random_timing(p, [10, 12], n_sims=25, seed=42)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 25 and bt.N_RANDOM == 1000 and bt.SEED == 42


def test_random_timing_pays_its_own_costs():
    n = 120
    p = prices(n, rf=np.zeros(n))  # flat prices: any non-zero return is cost
    rng = np.random.default_rng(42)
    w = bt.random_windows([10], n, rng)
    r = bt.simulate(p, bt.state_from_windows(n, w))
    assert r.sum() < 0  # entry and exit costs charged on the simulation's own trades


# ---------------------------------------------------------------- bootstrap, verdict, grid

def test_bootstrap_paired_and_reproducible():
    rng = np.random.default_rng(1)
    r3, r2, rf = rng.normal(0, 0.01, 400), rng.normal(0, 0.01, 400), np.full(400, 0.0001)
    a = bt.sharpe_diff_bootstrap(r3, r2, rf, reps=60, seed=42)
    b = bt.sharpe_diff_bootstrap(r3, r2, rf, reps=60, seed=42)
    assert a == b
    assert a["point"] == pytest.approx(bt.sharpe(r3, rf) - bt.sharpe(r2, rf))
    assert a["lower95"] <= a["upper95"]
    assert bt.sharpe_diff_bootstrap(r3, r3, rf, reps=30)["upper95"] == pytest.approx(0, abs=1e-12)  # paired


def test_h3_pass_boundaries():
    random = np.arange(1, 101, dtype=float) / 100  # 90th percentile = 0.901
    p90 = np.percentile(random, 90)
    assert not bt.h3_pass(1.0, 1.0, random)          # equal Sharpe fails (strictly greater)
    assert bt.h3_pass(p90, 0.5, random)              # exactly the 90th percentile passes
    assert not bt.h3_pass(p90 - 1e-9, 0.5, random)
    assert bt.RANDOM_PCT == 90


def test_sensitivity_grid_cells_share_holding_period():
    n = 200
    rng = np.random.default_rng(5)
    p = prices(n, r_btc=rng.normal(0, 0.02, n), rf=np.full(n, 0.0001))
    sig = {c: signal(n, rng.choice(np.arange(5, 180), 4, replace=False))
           for c in ["S_yen", "S_vix", "S_vix2", "S_yen_p85", "S_yen_p95"]}
    rows = bt.sensitivity_grid(p, sig)
    assert len(rows) == 9
    assert {(r["yen_move_pct"], r["hold"]) for r in rows} == {(a, b) for a in (85, 90, 95) for b in (5, 10, 20)}
    cell = next(r for r in rows if r["yen_move_pct"] == 95 and r["hold"] == 20)
    rf = p.rf[1:]
    manual = bt.sharpe(bt.run_strategy(p, sig["S_yen_p95"], 20)[0], rf) - bt.sharpe(bt.run_strategy(p, sig["S_vix"], 20)[0], rf)
    assert cell["sharpe_diff_primary"] == pytest.approx(manual, rel=1e-12)


def test_constants_match_preregistration():
    assert (bt.W_ACWI, bt.W_BTC, bt.HOLD) == (0.80, 0.20, 10)
    assert bt.COST == {"ACWI": 0.0010, "BTC": 0.0020, "FXY": 0.0010, "TB": 0.0} and bt.ENTRY_MULT == 2
    assert (bt.BOOT_REPS, bt.BLOCK_MEAN, bt.CI) == (2000, 20, (2.5, 97.5))
    assert bt.COMPARISONS == {"primary": "S_vix", "secondary": "S_vix2"}


def test_first_run_guard(tmp_path, monkeypatch):
    m = tmp_path / "run_manifest_h3.json"
    m.write_text("{}")
    monkeypatch.setattr(bt, "RUN_MANIFEST", m)
    with pytest.raises(RuntimeError, match="already been produced"):
        bt.main()
