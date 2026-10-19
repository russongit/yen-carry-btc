"""Signal evaluation for H2, PREREGISTRATION.md section 8.4.

Compares the yen signal with the VIX signal on episode-level precision in the test period,
under the primary comparison (S_vix, section 6) and the secondary comparison (S_vix2,
DEVIATIONS.md). The equity signals are reported as descriptive only (decision H2-G).

Every timing or threshold choice cites PREREGISTRATION.md or a DEVIATIONS.md entry.

Usage (one run only; guarded by outputs/run_manifest_h2.json):
    python src/signal_eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "processed" / "aligned_daily.csv"
SIGNALS = ROOT / "data" / "processed" / "signals.csv"
HOURLY = ROOT / "data" / "processed" / "btc_hourly.csv"
OUT_TABLES = ROOT / "outputs" / "tables"
RUN_MANIFEST = ROOT / "outputs" / "run_manifest_h2.json"

NY = "America/New_York"
DROP = 0.10            # section 8.4: BTC falls at least 10% below BTC_close_t
WINDOW_DAYS = 10       # section 8.4: through 16:00 on trading day t + 10
EPISODE_GAP = 10       # section 6: episodes; decision H2-B: target events use the same rule
RECALL_DAYS = 10       # section 8.4: preceded by a signal episode within 10 trading days
TRAIN = (pd.Timestamp("2018-01-02"), pd.Timestamp("2022-12-30"))  # section 3
TEST = (pd.Timestamp("2023-01-03"), pd.Timestamp("2026-09-30"))
PRECISION_MARGIN = 0.10  # section 2, H2: at least 10 percentage points
BOOT_REPS, BLOCK_MEAN, SEED = 2000, 20, 42  # section 8.4
CI = (2.5, 97.5)       # decision H2-E: 95% percentile intervals
SIGNALS_EVALUATED = ["S_yen", "S_vix", "S_vix2", "S_eq", "S_eq2"]
COMPARISONS = {"primary": "S_vix", "secondary": "S_vix2"}  # DEVIATIONS.md, secondary H2/H3 comparison


# ---------------------------------------------------------------- target event

def hourly_closes(hourly: pd.DataFrame) -> pd.Series:
    """Candle closes indexed by the time the candle closes (open time + 1 hour), decision H2-A."""
    s = hourly["close"].astype(float)
    s.index = pd.DatetimeIndex(hourly.index) + pd.Timedelta(hours=1)
    return s.sort_index()


def target(days: pd.DatetimeIndex, btc_close: pd.Series, closes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Section 8.4 and decision H2-A: T_t = 1 if any candle close after 16:00 New York on day t, up to
    and including 16:00 on trading day t+10, is at least 10% below BTC_close_t.

    Returns (T, complete). complete is False when day t+10 is not in `days`, the window runs past the
    last hourly close, or BTC_close_t is missing (decision H2-C); T is blank there.
    """
    start = (days + pd.Timedelta(hours=16)).tz_localize(NY).tz_convert("UTC")
    t_vals, complete = [], []
    last = closes.index.max()
    for i, d in enumerate(days):
        if i + WINDOW_DAYS >= len(days) or pd.isna(btc_close.get(d, np.nan)):
            t_vals.append(np.nan); complete.append(False)
            continue
        end = start[i + WINDOW_DAYS]
        if end > last:
            t_vals.append(np.nan); complete.append(False)
            continue
        window = closes[(closes.index > start[i]) & (closes.index <= end)]
        t_vals.append(float(window.min() <= (1 - DROP) * btc_close[d]))
        complete.append(True)
    return pd.Series(t_vals, index=days), pd.Series(complete, index=days)


# ---------------------------------------------------------------- metrics

def episode_starts(on: np.ndarray, gap: int = EPISODE_GAP) -> np.ndarray:
    """Positions of episode starts: a day with on = 1 more than `gap` positions after the previous one."""
    pos = np.flatnonzero(on)
    return pos[np.diff(pos, prepend=-gap - 1) > gap]


@dataclass
class Metrics:
    episodes: int
    complete_episodes: int
    true_pos: int
    false_pos: int
    precision: float
    events: int
    preceded_events: int
    recall: float
    base_rate: float


def metrics(signal: np.ndarray, T: np.ndarray, complete: np.ndarray, in_period: np.ndarray) -> Metrics:
    """Section 8.4 metrics on one sequence of trading days.

    signal, T and complete are aligned day arrays; in_period marks the days of the period being scored.
    Episodes are formed on the whole sequence and assigned to the period by start day (decision H2-D).
    Target events are grouped from complete T = 1 days in the period by the 10-day rule (decision H2-B).
    """
    on = np.nan_to_num(signal, nan=0.0) > 0
    starts = episode_starts(on)
    starts_in = starts[in_period[starts]]
    scored = starts_in[complete[starts_in]]
    tp = int(np.nansum(T[scored] == 1))
    precision = tp / len(scored) if len(scored) else np.nan

    target_days = np.flatnonzero(in_period & complete & (np.nan_to_num(T, nan=0.0) == 1))
    events = episode_starts(np.isin(np.arange(len(T)), target_days))
    preceded = sum(bool(((starts >= e - RECALL_DAYS) & (starts <= e)).any()) for e in events)
    recall = preceded / len(events) if len(events) else np.nan

    base_days = in_period & complete
    base_rate = float(np.nanmean(T[base_days])) if base_days.any() else np.nan
    return Metrics(len(starts_in), len(scored), tp, len(scored) - tp, precision,
                   len(events), int(preceded), recall, base_rate)


# ---------------------------------------------------------------- bootstrap

def stationary_indices(n: int, rng: np.random.Generator, mean_block: int = BLOCK_MEAN) -> np.ndarray:
    """Circular stationary bootstrap (Politis and Romano): each step starts a new block with
    probability 1 / mean_block, otherwise continues to the next day (decision H2-E)."""
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < 1 / mean_block
    jumps = rng.integers(n, size=n)
    for k in range(1, n):
        idx[k] = jumps[k] if new_block[k] else (idx[k - 1] + 1) % n
    return idx


def bootstrap(day_signals: dict[str, np.ndarray], T: np.ndarray, complete: np.ndarray,
              reps: int = BOOT_REPS, seed: int = SEED) -> pd.DataFrame:
    """Paired replicates within one period: every signal and T share each replicate's draws."""
    rng = np.random.default_rng(seed)
    n = len(T)
    everywhere = np.ones(n, dtype=bool)
    rows = []
    for r in range(reps):
        idx = stationary_indices(n, rng)
        row = {"rep": r}
        for name, sig in day_signals.items():
            m = metrics(sig[idx], T[idx], complete[idx], everywhere)
            row[f"{name}_precision"], row[f"{name}_recall"] = m.precision, m.recall
            row["base_rate"] = m.base_rate
        for label, comp in COMPARISONS.items():
            row[f"diff_{label}"] = row["S_yen_precision"] - row[f"{comp}_precision"]  # NaN if either undefined
        rows.append(row)
    return pd.DataFrame(rows)


def intervals(reps: pd.DataFrame) -> pd.DataFrame:
    """95% percentile intervals over defined replicates, with the share undefined (decision H2-E)."""
    out = []
    for col in reps.columns.drop("rep"):
        v = reps[col].to_numpy(dtype=float)
        defined = v[~np.isnan(v)]
        lo, hi = (np.percentile(defined, CI) if len(defined) else (np.nan, np.nan))
        out.append({"statistic": col, "lower95": lo, "upper95": hi,
                    "share_undefined": 1 - len(defined) / len(v), "replicates": len(v)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- verdict

def h2_pass(prec_yen: float, prec_comp: float, base_rate: float) -> bool:
    """Section 2, H2: Precision(yen) - Precision(VIX) >= 10 pp AND Precision(yen) > base rate.
    Rounded to 12 decimals so a difference of exactly 0.10 is not lost to floating point."""
    if np.isnan(prec_yen) or np.isnan(prec_comp):
        return False
    return bool(round(prec_yen - prec_comp, 12) >= PRECISION_MARGIN and prec_yen > base_rate)


def verdict_table(test_metrics: dict[str, Metrics]) -> pd.DataFrame:
    """Primary and secondary comparisons; H2 passes only if it passes under both
    (DEVIATIONS.md, pre-analysis entry, item 1). Competitor episode counts are reported beside."""
    y = test_metrics["S_yen"]
    rows = []
    for label, comp in COMPARISONS.items():
        c = test_metrics[comp]
        rows.append({"comparison": label, "competitor": comp, "precision_yen": y.precision,
                     "precision_competitor": c.precision, "difference": y.precision - c.precision,
                     "base_rate": y.base_rate, "yen_episodes": y.complete_episodes,
                     "competitor_episodes": c.complete_episodes,
                     "pass": h2_pass(y.precision, c.precision, y.base_rate)})
    v = pd.DataFrame(rows)
    combined = {"comparison": "combined (both must pass)", "pass": bool(v["pass"].all())}
    return pd.concat([v, pd.DataFrame([combined])], ignore_index=True)


# ---------------------------------------------------------------- run

def load():
    a = pd.read_csv(ALIGNED, index_col="date", parse_dates=["date"])
    s = pd.read_csv(SIGNALS, index_col="date", parse_dates=["date"])
    h = pd.read_csv(HOURLY, index_col="open_utc", parse_dates=["open_utc"])
    return a, s, h


def period_mask(days: pd.DatetimeIndex, period: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    return np.asarray((days >= period[0]) & (days <= period[1]))


def main() -> None:
    import local_projections as lp  # shared guard: refuse to run with uncommitted code
    if RUN_MANIFEST.exists():
        raise RuntimeError(f"{RUN_MANIFEST.name} exists: H2 results have already been produced. "
                           "Log a deviation (results seen) before any rerun.")
    commit = lp.run_commit()
    a, s, h = load()
    days = a.loc[TRAIN[0]:TEST[1]].index
    T, complete = target(days, a["BTC_close"], hourly_closes(h))
    sig = {n: s.loc[days, n].to_numpy(dtype=float) for n in SIGNALS_EVALUATED}
    Tn, cn = T.to_numpy(), complete.to_numpy()

    rows, test_metrics, boots = [], {}, []
    for pname, period in [("train", TRAIN), ("test", TEST)]:
        mask = period_mask(days, period)
        for name in SIGNALS_EVALUATED:
            m = metrics(sig[name], Tn, cn, mask)
            rows.append({"period": pname, "signal": name, **m.__dict__,
                         "incomplete_days_excluded": int((mask & ~cn).sum())})
            if pname == "test":
                test_metrics[name] = m
        reps = bootstrap({n: v[mask] for n, v in sig.items()}, Tn[mask], cn[mask])
        boots.append(intervals(reps).assign(period=pname))

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_TABLES / "h2_metrics.csv", index=False)
    pd.concat(boots, ignore_index=True).to_csv(OUT_TABLES / "h2_bootstrap.csv", index=False)
    verdict = verdict_table(test_metrics)
    verdict.to_csv(OUT_TABLES / "h2_verdict.csv", index=False)
    RUN_MANIFEST.write_text(json.dumps({
        "run_commit": commit, "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "numpy": np.__version__, "pandas": pd.__version__, "bootstrap": {"reps": BOOT_REPS, "mean_block": BLOCK_MEAN,
                                                                          "seed": SEED},
        "outputs": ["tables/h2_metrics.csv", "tables/h2_bootstrap.csv", "tables/h2_verdict.csv"]}, indent=2))
    print(verdict.to_string(index=False))
    print(f"Run commit {commit}; outputs listed in {RUN_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
