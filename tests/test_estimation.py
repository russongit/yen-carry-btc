"""Tests for estimate() and its helpers in src/local_projections.py, on synthetic data only.

No real-data coefficient is computed here.
Run from the repository root:  python -m pytest tests/test_estimation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import local_projections as lp  # noqa: E402


def synthetic_sample(gamma: float, h: int = 0, n: int = 1500, seed: int = 7) -> lp.Sample:
    """y = 0.1 r_yen + gamma S*r_yen + 0.002 S + noise, with AR(1) noise so HAC matters."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    r_yen = rng.normal(0, 0.005, n)
    S = (rng.random(n) < 0.15).astype(float)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.3 * e[t - 1] + rng.normal(0, 0.01)
    X = pd.DataFrame({"r_yen": r_yen, "S_x_ryen": S * r_yen, "S": S}, index=idx)
    y = pd.Series(0.1 * r_yen + gamma * S * r_yen + 0.002 * S + e, index=idx)
    spec = lp.Spec("synthetic", "test", "y", tuple(X.columns), h=h)
    return lp.Sample(spec, y, X)


# ---------------------------------------------------------------- 1. one-sided direction

def test_one_sided_p_is_small_for_planted_negative_gamma():
    res = lp.estimate(synthetic_sample(gamma=-1.5))
    p_less, p_greater = lp.one_sided_p(res, "S_x_ryen")
    assert res.params["S_x_ryen"] < 0
    assert p_less < 0.01 and p_greater > 0.99
    assert lp.verdict(res, "S_x_ryen")["pass"]


def test_one_sided_p_is_near_one_for_planted_positive_gamma():
    res = lp.estimate(synthetic_sample(gamma=+1.5))
    p_less, p_greater = lp.one_sided_p(res, "S_x_ryen")
    assert res.params["S_x_ryen"] > 0
    assert p_less > 0.99 and p_greater < 0.01
    assert not lp.verdict(res, "S_x_ryen")["pass"]


def test_one_sided_p_matches_half_the_two_sided_p():
    res = lp.estimate(synthetic_sample(gamma=-0.6))
    p_less, p_greater = lp.one_sided_p(res, "S_x_ryen")
    assert p_less + p_greater == pytest.approx(1.0, abs=1e-12)
    assert min(p_less, p_greater) == pytest.approx(res.pvalues["S_x_ryen"] / 2, rel=1e-10)


def test_verdict_requires_negative_sign_even_when_significant():
    res = lp.estimate(synthetic_sample(gamma=+1.5))
    v = lp.verdict(res, "S_x_ryen")
    assert v["coef"] > 0 and not v["pass"]


# ---------------------------------------------------------------- 2. beta_h + gamma_h bands

def test_sum_se_uses_covariance_term():
    res = lp.estimate(synthetic_sample(gamma=-1.0, h=5))
    v = res.cov_params()
    manual = np.sqrt(v.loc["r_yen", "r_yen"] + v.loc["S_x_ryen", "S_x_ryen"] + 2 * v.loc["r_yen", "S_x_ryen"])
    assert lp.sum_se(res, "r_yen", "S_x_ryen") == pytest.approx(manual, rel=1e-12)
    without_cov = np.sqrt(v.loc["r_yen", "r_yen"] + v.loc["S_x_ryen", "S_x_ryen"])
    assert abs(manual - without_cov) > 1e-6 * manual  # the covariance term actually matters here
    # Same result from statsmodels' own linear-combination test.
    t = res.t_test("r_yen + S_x_ryen = 0")
    assert lp.sum_se(res, "r_yen", "S_x_ryen") == pytest.approx(float(np.squeeze(t.sd)), rel=1e-10)


def test_lp_bands_are_90_percent_and_centered():
    fits = {f"H1b_h{h}": lp.estimate(synthetic_sample(gamma=-1.0, h=h, seed=h)) for h in lp.HORIZONS}
    bands = lp.lp_bands(fits)
    assert list(bands["h"]) == list(lp.HORIZONS)
    for _, r in bands.iterrows():
        res = fits[f"H1b_h{int(r['h'])}"]
        assert r["beta_plus_gamma"] == pytest.approx(res.params["r_yen"] + res.params["S_x_ryen"], rel=1e-12)
        assert r["upper90"] - r["beta_plus_gamma"] == pytest.approx(1.6448536269514722 * r["se"], rel=1e-9)
        assert r["beta_plus_gamma"] - r["lower90"] == pytest.approx(1.6448536269514722 * r["se"], rel=1e-9)


# ---------------------------------------------------------------- 3. HAC lag length

@pytest.mark.parametrize("h", [0, 1, 4, 5, 6, 10])
def test_maxlags_read_back_from_fit_is_max_5_h(h):
    res = lp.estimate(synthetic_sample(gamma=-1.0, h=h))
    assert lp.used_maxlags(res) == max(5, h)
    assert res.cov_type == "HAC" and res.use_t is False


def test_hac_differs_from_homoskedastic_with_autocorrelated_noise():
    sample = synthetic_sample(gamma=-1.0)
    res = lp.estimate(sample)
    import statsmodels.api as sm
    plain = sm.OLS(sample.y, sm.add_constant(sample.X)).fit()
    assert not np.allclose(res.bse, plain.bse)
    np.testing.assert_allclose(res.params, plain.params, rtol=1e-12)  # same point estimates


# ---------------------------------------------------------------- other estimation helpers

def test_coefficient_table_reports_s_days_vif_and_drops():
    sample = synthetic_sample(gamma=-1.0)
    res = lp.estimate(sample)
    t = lp.coefficient_table(sample, res, "full")
    assert set(t["term"]) == {"const", "r_yen", "S_x_ryen", "S"}
    assert (t["s_days"] == int(sample.X["S"].sum())).all()
    assert t.loc[t["term"] == "const", "vif"].isna().all()
    assert (t.loc[t["term"] != "const", "vif"] >= 1).all()
    assert (t["maxlags"] == 5).all() and (t["n"] == len(sample.y)).all()


def test_first_run_guard(tmp_path):
    manifest = tmp_path / "run_manifest.json"
    lp.check_first_run(manifest)  # no manifest: allowed
    manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="already been produced"):
        lp.check_first_run(manifest)


def test_verdict_map_matches_section_2():
    assert lp.VERDICTS == {"H1a": ("H1a", "S_x_ryen"), "H1b": ("H1b_h5", "S_x_ryen"), "H1d": ("H1d", "dJP2Y")}
    assert lp.ALPHA == 0.05
