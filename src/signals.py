"""Signal construction, PREREGISTRATION.md section 6.

Reads data/processed/aligned_daily.csv and computes the yen stress signal, the
frequency-matched VIX and equity signals, and episode numbers for each. Writes
data/processed/signals.csv and outputs/tables/calibration.csv.

No event study, regression, signal evaluation, or backtest is run here, and
only training-period frequencies are printed.

Usage:
    python src/signals.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "processed" / "aligned_daily.csv"
OUT = ROOT / "data" / "processed" / "signals.csv"
CALIBRATION = ROOT / "outputs" / "tables" / "calibration.csv"

# Section 3: signals are reported from the analysis start; training period for calibration (D4).
ANALYSIS_START = pd.Timestamp("2018-01-02")
TRAIN_START, TRAIN_END = pd.Timestamp("2018-01-02"), pd.Timestamp("2022-12-30")

MOVE_DAYS = 5          # section 6, condition 1: 5-day yen return
VOL_DAYS = 20          # section 6, condition 2: 20-day realized volatility
VOL_MIN_OBS = 18       # DEVIATIONS.md decision C: at least 18 of the 20 returns
ANNUALIZE = np.sqrt(252)  # section 6: annualized by sqrt(252)
MOVE_PCT = 90          # section 6, condition 1: above its expanding 90th percentile
VOL_PCT = 80           # section 6, condition 2: above its expanding 80th percentile
SENSITIVITY_MOVE_PCTS = (85, 95)  # section 8.5 sensitivity grid
EQ_DAYS = 5            # section 6: S&P 500 5-day log return
Q_GRID = range(1, 100)  # section 6: grid of whole percentiles
EPISODE_GAP = 10       # section 6: signal days within 10 trading days share an episode


class SignalError(RuntimeError):
    pass


def expanding_pct(s: pd.Series, pct: float) -> pd.Series:
    """Section 6: expanding percentile from the first available value through t inclusive.

    Linear interpolation, the same definition as numpy.percentile's default. Blank values are
    skipped, so the window holds every available value up to and including day t.
    """
    return s.expanding(min_periods=1).quantile(pct / 100, interpolation="linear")


def yen_move(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Section 6 condition 1: -ln(USDJPY_t / USDJPY_{t-5}) on NYSE days, noon fixings.

    DEVIATIONS.md decision B: with no fixing on day t-5, use the last fixing on or before t-5,
    and flag the day. Blank when day t has no fixing (decision A handles the signal).
    """
    usdjpy = d["DEXJPUS"]
    base = usdjpy.ffill().shift(MOVE_DAYS)  # forward-fill only to find the reference fixing
    long_window = usdjpy.shift(MOVE_DAYS).isna() & base.notna()
    return -np.log(usdjpy / base), long_window


def yen_vol(d: pd.DataFrame) -> pd.Series:
    """Section 6 condition 2 with DEVIATIONS.md decision C.

    Sample standard deviation (ddof = 1) of daily r_yen over the last 20 NYSE days, annualized.
    Returns flagged fx_multi_day are treated as missing; at least 18 of the 20 must be available.
    Blank until 20 daily returns exist (approved burn-in rule).
    """
    r = d["r_yen"].where(~d["fx_multi_day"].astype(bool))
    vol = r.rolling(VOL_DAYS, min_periods=VOL_MIN_OBS).std(ddof=1) * ANNUALIZE
    enough_history = d["r_yen"].notna().cumsum() >= VOL_DAYS
    return vol.where(enough_history)


def yen_signal(move: pd.Series, move_pct: pd.Series, vol: pd.Series, vol_pct: pd.Series,
               unavailable: pd.Series) -> pd.Series:
    """Section 6: S_yen = 1 when both conditions hold at noon New York on day t.

    DEVIATIONS.md decision A and its extension: 0 on days with no fixing or with fewer than 18
    usable returns in the volatility window. Blank before the analysis start
    (approved burn-in rule).
    """
    s = ((move > move_pct) & (vol > vol_pct)).astype("Int8")
    s[unavailable] = 0
    return s.where(s.index >= ANALYSIS_START)


def episodes(signal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Section 6: signal days within 10 trading days of the previous signal day share an episode.

    Returns (episode number on signal days, episode-start flag). Episode start = first signal day.
    """
    on = signal.fillna(0).astype(bool).to_numpy()
    pos = np.flatnonzero(on)
    new = np.diff(pos, prepend=-EPISODE_GAP - 1) > EPISODE_GAP
    ep = pd.Series(pd.NA, index=signal.index, dtype="Int32")
    ep.iloc[pos] = np.cumsum(new)
    start = pd.Series(False, index=signal.index)
    start.iloc[pos[new]] = True
    return ep, start


def competing(d: pd.DataFrame, q_v: int, q_e: int) -> pd.DataFrame:
    """Section 6 competing signals at percentiles q_v (VIX, above) and q_e (S&P 500 5-day return, below)."""
    spx5 = np.log(d["SPX"] / d["SPX"].shift(EQ_DAYS))
    out = pd.DataFrame(index=d.index)
    out["spx_ret5"] = spx5
    out["vix_pct"] = expanding_pct(d["VIX"], q_v)
    out["spx_ret5_pct"] = expanding_pct(spx5, q_e)
    late = out.index >= ANALYSIS_START
    out["S_vix"] = (d["VIX"] > out["vix_pct"]).astype("Int8").where(late)
    out["S_eq"] = (spx5 < out["spx_ret5_pct"]).astype("Int8").where(late)
    return out


def calibrate(d: pd.DataFrame, s_yen: pd.Series) -> tuple[int, int, pd.DataFrame]:
    """Section 6 and D4: choose q_V and q_E so each signal's training-period frequency is closest
    to the yen signal's. Ties go to the higher q_V and the lower q_E (DEVIATIONS.md decision D).

    Only training-period days are counted, and every expanding percentile on those days uses data
    through that day only, so no test-period data enters the choice.
    """
    train = (d.index >= TRAIN_START) & (d.index <= TRAIN_END)
    target = float(s_yen[train].astype(float).mean())
    spx5 = np.log(d["SPX"] / d["SPX"].shift(EQ_DAYS))
    rows = []
    for q in Q_GRID:
        f_v = float((d["VIX"] > expanding_pct(d["VIX"], q))[train].mean())
        f_e = float((spx5 < expanding_pct(spx5, q))[train].mean())
        rows += [("vix", q, f_v), ("equity", q, f_e)]
    grid = pd.DataFrame(rows, columns=["signal", "q", "training_freq"])
    grid["yen_training_freq"] = target
    grid["abs_diff"] = (grid["training_freq"] - target).abs()

    def pick(name: str, prefer_high: bool) -> int:
        g = grid[grid["signal"] == name]
        best = g[np.isclose(g["abs_diff"], g["abs_diff"].min(), rtol=0, atol=1e-12)]
        return int(best["q"].max() if prefer_high else best["q"].min())

    q_v, q_e = pick("vix", True), pick("equity", False)
    grid["chosen"] = ((grid["signal"] == "vix") & (grid["q"] == q_v)) | ((grid["signal"] == "equity") & (grid["q"] == q_e))
    return q_v, q_e, grid


def compute(d: pd.DataFrame, q_v: int | None = None, q_e: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """All section 6 signals. If q_v and q_e are given, calibration is skipped (used by tests)."""
    out = pd.DataFrame(index=d.index)
    out["yen_move5"], out["yen_move5_long_window"] = yen_move(d)
    out["yen_vol20"] = yen_vol(d)
    # DEVIATIONS.md decision A (no fixing on day t) and its extension (volatility window with fewer
    # than 18 usable returns): the signal cannot be computed, so it does not fire.
    late = out.index >= ANALYSIS_START
    out["signal_unavailable"] = late & (d["fx_missing"].astype(bool) | out["yen_vol20"].isna())
    out["yen_vol20_pct80"] = expanding_pct(out["yen_vol20"], VOL_PCT)
    for p in (MOVE_PCT, *SENSITIVITY_MOVE_PCTS):
        out[f"yen_move5_pct{p}"] = expanding_pct(out["yen_move5"], p)
        name = "S_yen" if p == MOVE_PCT else f"S_yen_p{p}"
        out[name] = yen_signal(out["yen_move5"], out[f"yen_move5_pct{p}"], out["yen_vol20"],
                               out["yen_vol20_pct80"], out["signal_unavailable"])

    # Any other blank input from the analysis start is not covered by decision A and stops the run.
    gaps = late & ~out["signal_unavailable"] & out["yen_move5"].isna()
    if gaps.any():
        raise SignalError(f"Yen signal inputs blank on days not covered by decision A: {list(out.index[gaps].date)[:5]}")

    grid = None
    if q_v is None or q_e is None:
        q_v, q_e, grid = calibrate(d, out["S_yen"])
    out = out.join(competing(d, q_v, q_e))
    out["q_v"], out["q_e"] = q_v, q_e

    for name in ["S_yen", "S_yen_p85", "S_yen_p95", "S_vix", "S_eq"]:
        out[f"{name}_episode"], out[f"{name}_episode_start"] = episodes(out[name])
    return out, grid


def main() -> None:
    d = pd.read_csv(ALIGNED, index_col="date", parse_dates=["date"])
    out, grid = compute(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, date_format="%Y-%m-%d")
    grid.to_csv(CALIBRATION, index=False, float_format="%.6f")

    train = (out.index >= TRAIN_START) & (out.index <= TRAIN_END)
    chosen = grid[grid["chosen"]].set_index("signal")
    print(f"Training period {TRAIN_START.date()} to {TRAIN_END.date()} ({int(train.sum())} trading days)")
    print(f"  S_yen fires on {out.loc[train, 'S_yen'].astype(float).mean():.4f} of training days")
    for sig, q in [("vix", out['q_v'].iloc[0]), ("equity", out['q_e'].iloc[0])]:
        print(f"  {sig}: q = {q}, training frequency {chosen.loc[sig, 'training_freq']:.4f}")
    print(f"Flags: signal_unavailable {int(out['signal_unavailable'].sum())} days, "
          f"yen_move5_long_window {int(out.loc[out.index >= ANALYSIS_START, 'yen_move5_long_window'].sum())} days "
          f"(from {ANALYSIS_START.date()})")
    print(f"Wrote {OUT.relative_to(ROOT)} and {CALIBRATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
