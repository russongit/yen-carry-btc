"""Regression samples for PREREGISTRATION.md sections 8.2, 8.3 and the section 9 regression checks.

This module builds every regression's estimation sample (dependent variable, regressors,
rows) from data/processed/aligned_daily.csv and signals.csv, through one registry of
specifications. It fits nothing: estimation is a separate step, written only after the
samples and their tests are committed.

Every timing or threshold choice cites PREREGISTRATION.md or a DEVIATIONS.md entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "processed" / "aligned_daily.csv"
SIGNALS = ROOT / "data" / "processed" / "signals.csv"

# DEVIATIONS.md regressions decision E: pinned software and fit call.
STATSMODELS_VERSION = "0.15.0"
if statsmodels.__version__ != STATSMODELS_VERSION:
    raise RuntimeError(f"statsmodels {statsmodels.__version__} found; {STATSMODELS_VERSION} is pinned")


def fit_kwargs(h: int) -> dict:
    """Arguments for sm.OLS(y, X).fit(**fit_kwargs(h)), section 8 common rules and decision E."""
    return {"cov_type": "HAC",
            "cov_kwds": {"maxlags": max(5, h), "kernel": "bartlett", "use_correction": True},
            "use_t": False}


# Section 3 sample and periods; DEVIATIONS.md regressions decision D: full sample gives the verdicts.
SAMPLE_START, SAMPLE_END = pd.Timestamp("2018-01-02"), pd.Timestamp("2026-09-30")
TRAIN_END = pd.Timestamp("2022-12-30")
TEST_START = pd.Timestamp("2023-01-03")
POST2024 = pd.Timestamp("2024-01-11")  # section 2, H4: US spot BTC ETFs began trading
HORIZONS = range(1, 11)                # section 8.2: h = 1 to 10
H_TEST = 5                             # D5: primary horizon for H1b and H1d

# DEVIATIONS.md (extended fx_multi_day rule): rows flagged fx_multi_day are excluded from every
# regression in which a daily return built from the noon fixings appears as the dependent variable
# or a regressor, including interactions built from them.
FX_TERMS = {"r_yen", "r_dxyx", "r_eurusd", "r_dtwexbgs", "S_x_ryen", "post_S_ryen"}

# Regressor sets, section 8.2 and 8.3.
X_H1A = ("r_yen", "S_x_ryen", "S", "r_spx", "dVIX", "r_dxyx", "dUS2Y", "r_btc_noon_lag1")
X_H1B = ("r_yen", "S_x_ryen", "S", "r_spx", "dVIX", "r_dxyx", "dUS2Y", "r_btc_1700")
X_M1 = ("dJP2Y", "dUS2Y", "r_spx", "dVIX", "r_dxyx")
X_H1D = ("dJP2Y", "dUS2Y", "r_spx", "dVIX", "r_dxyx", "r_btc_1700")
X_H1D_H0 = ("dJP2Y", "dUS2Y", "r_spx", "dVIX", "r_dxyx", "r_btc_noon_lag1")
NOT_WINSORIZED = {"S", "post2024"}  # decision C: indicators are not winsorized


@dataclass(frozen=True)
class Spec:
    name: str
    group: str                       # analysis it belongs to, as named in DEVIATIONS.md (pre-analysis)
    y: str
    x: tuple[str, ...]
    h: int = 0                       # horizon; sets HAC maxlags (section 8 common rules)
    frequency: str = "daily"         # "weekly" only for robustness check 1
    extra_exclusions: tuple[str, ...] = ()  # e.g. dtw_multi_day for check 4 (decision F)
    drop_span: tuple[pd.Timestamp, pd.Timestamp] | None = None  # check 10
    winsorize: bool = False          # check 11
    verdict: bool = False            # one of the three tested specifications (decision D)

    @property
    def uses_fx_terms(self) -> bool:
        return bool(FX_TERMS & ({self.y} | set(self.x)))

    def uses(self, var: str) -> bool:
        return var == self.y or var in self.x


@dataclass
class Sample:
    spec: Spec
    y: pd.Series
    X: pd.DataFrame
    dropped_fx_multi_day: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.y.index


# ---------------------------------------------------------------- panels

def load() -> pd.DataFrame:
    a = pd.read_csv(ALIGNED, index_col="date", parse_dates=["date"])
    s = pd.read_csv(SIGNALS, index_col="date", parse_dates=["date"])
    return a.join(s[["S_yen", "S_yen_episode"]])


def daily_panel(a: pd.DataFrame) -> pd.DataFrame:
    """All daily variables used by any specification, on NYSE days in the analysis sample.

    Lags and leads are taken on the full aligned table before trimming, so 2018-01-02 has its lag
    and leads never reach past 2026-09-30, the last row of the aligned table (section 3).
    """
    p = pd.DataFrame(index=a.index)
    for src in ["r_yen", "r_dxyx", "r_eurusd", "r_dtwexbgs", "r_spx", "dVIX", "dUS2Y", "dUS10Y",
                "dJP2Y", "dJP2Y_lag1", "fx_multi_day", "dtw_multi_day"]:
        p[src] = a[src]
    p["S"] = a["S_yen"].astype(float)
    p["S_x_ryen"] = p["S"] * p["r_yen"]                      # section 8.2
    p["post2024"] = (p.index >= POST2024).astype(float)      # H4
    p["post_S_ryen"] = p["post2024"] * p["S_x_ryen"]
    p["dspread"] = p["dUS2Y"] - p["dJP2Y"]                   # check 8: change in US2Y - JP2Y, bp
    p["S_yen_episode"] = a["S_yen_episode"]
    for src, tag in [("BTC", "btc"), ("BTCB", "btcb")]:      # Bitstamp; Binance for check 2
        p[f"r_{tag}_noon"] = a[f"r_{tag}_noon"]              # section 5 rule 2: noon to noon
        p[f"r_{tag}_noon_lag1"] = a[f"r_{tag}_noon"].shift(1)
        p[f"r_{tag}_1700"] = a[f"r_{tag}_1700"]              # section 5 rule 3
        for h in HORIZONS:                                   # section 8.2: ln(BTC_1700_{t+h} / BTC_1700_t)
            p[f"lp_{tag}_h{h}"] = np.log(a[f"{src}_1700"].shift(-h) / a[f"{src}_1700"])
    p["fx_multi_day"] = p["fx_multi_day"].astype(bool)
    p["dtw_multi_day"] = p["dtw_multi_day"].astype(bool)
    return p.loc[SAMPLE_START:SAMPLE_END]


def weekly_panel(a: pd.DataFrame) -> pd.DataFrame:
    """Robustness check 1, DEVIATIONS.md regressions decision A.

    Weeks run Thursday to Wednesday; each is anchored at its last NYSE day (on or before Wednesday).
    FX uses the last fixing on or before the anchor (as in decision B). Weekly S = 1 if S_t = 1 on
    any trading day of the week.
    """
    week = a.index.to_period("W-WED")
    anchors = pd.Series(a.index, index=a.index).groupby(week).max()
    fx = a[["DEXJPUS", "DEXUSEU", "DEXUSUK", "DEXCAUS", "DEXSDUS", "DEXSZUS"]].ffill().loc[anchors.values]
    lv = a.loc[anchors.values]
    w = pd.DataFrame(index=pd.DatetimeIndex(anchors.values, name="date"))
    logfx = np.log(fx)
    w["r_yen"] = -logfx["DEXJPUS"].diff()
    dxyx = {"DEXUSEU": -0.576, "DEXUSUK": -0.119, "DEXCAUS": 0.091, "DEXSDUS": 0.042, "DEXSZUS": 0.036}
    w["r_dxyx"] = (1 / 0.864) * sum(wt * logfx[c].diff() for c, wt in dxyx.items())  # section 4
    w["r_spx"] = np.log(lv["SPX"]).diff()
    w["dVIX"] = lv["VIX"].diff()
    w["dUS2Y"] = lv["US2Y"].diff() * 100
    w["dJP2Y"] = lv["JP2Y"].diff() * 100
    w["r_btc_noon"] = np.log(lv["BTC_noon"]).diff()
    w["r_btc_noon_lag1"] = w["r_btc_noon"].shift(1)
    s_any = a["S_yen"].groupby(week).max()
    w["S"] = s_any.values
    w["S_x_ryen"] = w["S"] * w["r_yen"]
    return w.loc[SAMPLE_START:SAMPLE_END]


def winsorize(p: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Robustness check 11, decision C: each variable at its own training-period 1st and 99th
    percentiles; r_yen is winsorized before interactions are formed; indicators are not winsorized."""
    out = p.copy()
    for c in cols:
        if c in NOT_WINSORIZED or c in ("S_x_ryen", "post_S_ryen"):
            continue
        lo, hi = winsor_bounds(p, c)
        out[c] = p[c].clip(lo, hi)
    # Interactions are rebuilt from the winsorized r_yen, never winsorized themselves.
    r_yen = out["r_yen"] if "r_yen" in cols else p["r_yen"].clip(*winsor_bounds(p, "r_yen"))
    out["S_x_ryen"] = out["S"] * r_yen
    out["post_S_ryen"] = out["post2024"] * out["S_x_ryen"]
    return out


def winsor_bounds(p: pd.DataFrame, col: str) -> tuple[float, float]:
    """Training-period 1st and 99th percentiles (section 9, check 11)."""
    train = (p.index >= SAMPLE_START) & (p.index <= TRAIN_END)
    lo, hi = p.loc[train, col].quantile([0.01, 0.99])
    return float(lo), float(hi)


# ---------------------------------------------------------------- registry

def registry(p: pd.DataFrame) -> list[Spec]:
    """Every regression named in DEVIATIONS.md (pre-analysis entry, item 3) plus robustness check 1."""
    specs = [
        Spec("H1a", "H1a", "r_btc_noon", X_H1A, verdict=True),
        Spec("H4", "H4", "r_btc_noon", X_H1A + ("post_S_ryen",)),
        Spec("M1", "M1", "r_yen", X_M1),
        Spec("H1d", "H1d", f"lp_btc_h{H_TEST}", X_H1D, h=H_TEST, verdict=True),
        Spec("H1d_h0", "H1d", "r_btc_noon", X_H1D_H0),
        Spec("decomp_H1a", "decomposition", "r_btc_noon", X_H1A + ("dJP2Y",)),
        Spec(f"decomp_H1b_h{H_TEST}", "decomposition", f"lp_btc_h{H_TEST}", X_H1B + ("dJP2Y",), h=H_TEST),
    ]
    specs += [Spec(f"H1b_h{h}", "H1b", f"lp_btc_h{h}", X_H1B, h=h, verdict=(h == H_TEST)) for h in HORIZONS]

    base = list(specs)  # the registered base regressions (DEVIATIONS.md, robustness-check scope)
    by = {s.name: s for s in base}

    def swap(s: Spec, old: str, new: str) -> tuple[str, ...]:
        return tuple(new if v == old else v for v in s.x)

    def btc_vars(s: Spec) -> bool:
        return any(v.startswith(("r_btc_", "lp_btc_")) for v in (s.y, *s.x))

    # Check 1: weekly h = 0 regressions in 8.2 and 8.3 (decision A); outside the fx_multi_day rule.
    specs += [Spec("rc1_H1a_weekly", "rc1", "r_btc_noon", X_H1A, frequency="weekly"),
              Spec("rc1_M1_weekly", "rc1", "r_yen", X_M1, frequency="weekly"),
              Spec("rc1_H1d_h0_weekly", "rc1", "r_btc_noon", X_H1D_H0, frequency="weekly")]
    # Substitution checks 2, 3, 4, 5 and 11 apply to every base regression containing the variable they change.
    for s in base:
        if btc_vars(s):  # check 2: Binance BTC in place of Bitstamp
            y = s.y.replace("lp_btc_", "lp_btcb_").replace("r_btc_", "r_btcb_")
            x = tuple(v.replace("r_btc_", "r_btcb_") for v in s.x)
            specs.append(Spec(f"rc2_{s.name}", "rc2", y, x, h=s.h))
        if s.uses("r_dxyx"):  # checks 3 and 4: dollar control; check 4 also drops dtw_multi_day (decision F)
            specs.append(Spec(f"rc3_{s.name}", "rc3", s.y, swap(s, "r_dxyx", "r_eurusd"), h=s.h))
            specs.append(Spec(f"rc4_{s.name}", "rc4", s.y, swap(s, "r_dxyx", "r_dtwexbgs"), h=s.h,
                              extra_exclusions=("dtw_multi_day",)))
        if s.uses("dUS2Y"):  # check 5: US 10Y in place of US 2Y
            specs.append(Spec(f"rc5_{s.name}", "rc5", s.y, swap(s, "dUS2Y", "dUS10Y"), h=s.h))
        specs.append(Spec(f"rc11_{s.name}", "rc11", s.y, s.x, h=s.h, winsorize=True))  # check 11 (decision C)
    # Checks 6, 7, 8 and 10 follow section 9's stated targets.
    for name in ("H1a", f"H1b_h{H_TEST}"):  # check 6: dJP2Y added as a control in H1a and H1b
        s = by[name]
        specs.append(Spec(f"rc6_{s.name}", "rc6", s.y, s.x + ("dJP2Y",), h=s.h))
    h1d = by["H1d"]
    # Check 7: H1d with dJP2Y lagged one day (rule 6). Check 8: H1d with dspread.
    specs.append(Spec("rc7_H1d", "rc7", h1d.y, swap(h1d, "dJP2Y", "dJP2Y_lag1"), h=h1d.h))
    specs.append(Spec("rc8_H1d", "rc8", h1d.y,
                      tuple(v for v in h1d.x if v not in ("dJP2Y", "dUS2Y")) + ("dspread",), h=h1d.h))
    # Check 10: H1a leaving out each S_yen episode, first to last signal day (approved plan, B).
    for ep, days in p.dropna(subset=["S_yen_episode"]).groupby("S_yen_episode").groups.items():
        specs.append(Spec(f"rc10_H1a_ep{int(ep)}", "rc10", "r_btc_noon", X_H1A,
                          drop_span=(days.min(), days.max())))
    return specs


# ---------------------------------------------------------------- samples

def subsample_mask(index: pd.DatetimeIndex, subsample: str) -> np.ndarray:
    """Decision D: full sample decides; training and test are descriptive."""
    if subsample == "full":
        return np.ones(len(index), dtype=bool)
    if subsample == "train":
        return np.asarray(index <= TRAIN_END)
    if subsample == "test":
        return np.asarray(index >= TEST_START)
    raise ValueError(subsample)


def make_sample(spec: Spec, daily: pd.DataFrame, weekly: pd.DataFrame, subsample: str = "full",
                _apply_fx_rule: bool = True) -> Sample:
    """One estimation sample. _apply_fx_rule exists only so tests can measure the rows it drops."""
    p = weekly if spec.frequency == "weekly" else daily
    cols = [spec.y, *spec.x]
    if spec.winsorize:
        p = winsorize(p, cols)
    rows = pd.Series(subsample_mask(p.index, subsample), index=p.index)
    if spec.drop_span is not None:
        rows &= ~((p.index >= spec.drop_span[0]) & (p.index <= spec.drop_span[1]))
    for flag in spec.extra_exclusions:
        rows &= ~p[flag]
    dropped = pd.DatetimeIndex([])
    if spec.frequency == "daily" and spec.uses_fx_terms and _apply_fx_rule:
        complete = rows & p[cols].notna().all(axis=1)
        dropped = p.index[complete & p["fx_multi_day"]]
        rows &= ~p["fx_multi_day"]
    data = p.loc[rows, cols].dropna()
    return Sample(spec, data[spec.y], data[list(spec.x)], dropped)


def build_samples(subsample: str = "full", data: pd.DataFrame | None = None) -> dict[str, Sample]:
    a = load() if data is None else data
    daily, weekly = daily_panel(a), weekly_panel(a)
    return {s.name: make_sample(s, daily, weekly, subsample) for s in registry(daily)}
