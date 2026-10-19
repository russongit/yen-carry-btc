"""Backtest for H3, PREREGISTRATION.md section 8.5.

Portfolio of 80% ACWI and 20% BTC, rebalanced at month-end. De-risking strategies cut the
BTC sleeve by half on a signal. Compares the yen strategy (3) with the VIX strategy (2, primary;
2', secondary per DEVIATIONS.md) and with random timing (4), after costs.

Every timing or threshold choice cites PREREGISTRATION.md or a DEVIATIONS.md entry (H3-A to H3-K).

Usage (one run only; guarded by outputs/run_manifest_h3.json):
    python src/backtest.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_eval import stationary_indices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "processed" / "aligned_daily.csv"
SIGNALS = ROOT / "data" / "processed" / "signals.csv"
OUT_TABLES = ROOT / "outputs" / "tables"
RUN_MANIFEST = ROOT / "outputs" / "run_manifest_h3.json"

TRAIN = (pd.Timestamp("2018-01-02"), pd.Timestamp("2022-12-30"))  # section 3
TEST = (pd.Timestamp("2023-01-03"), pd.Timestamp("2026-09-30"))   # H3-E: verdict period
W_ACWI, W_BTC = 0.80, 0.20         # section 8.5: 80% ACWI, 20% BTC
HOLD = 10                          # section 8.5: hold 10 trading days
COST = {"ACWI": 0.0010, "BTC": 0.0020, "FXY": 0.0010, "TB": 0.0}  # section 8.5 and H3-C, per side
ENTRY_MULT = 2                     # section 8.5: doubled on de-risking entry trades
N_RANDOM, SEED = 1000, 42          # section 8.5, strategy 4
BOOT_REPS, BLOCK_MEAN = 2000, 20   # section 8.5 inference, H3-G
CI = (2.5, 97.5)                   # H3-G: 95% percentile intervals
RANDOM_PCT = 90                    # section 2, H3: at least the 90th percentile of random timing
ANNUAL = 252
GRID_PCTS = {85: "S_yen_p85", 90: "S_yen", 95: "S_yen_p95"}  # section 8.5 sensitivity grid
GRID_HOLDS = (5, 10, 20)
COMPARISONS = {"primary": "S_vix", "secondary": "S_vix2"}     # DEVIATIONS.md, secondary comparison
MAX_TRIES = 100_000


# ---------------------------------------------------------------- windows (H3-B)

def windows_from_signal(signal: np.ndarray, hold: int = HOLD) -> list[tuple[int, int]]:
    """(entry close, exit close) index pairs within one period.

    A signal on day t enters at the close of t+1 and exits at the close of t+1+hold. A signal on
    any day from a window's entry day through its exit day restarts the count: the exit moves to
    (t+1)+hold, with no trade. The exit may lie past the period end; such a window is cut there.
    """
    n = len(signal)
    out: list[list[int]] = []
    for t in np.flatnonzero(np.nan_to_num(signal, nan=0.0) > 0):
        entry = t + 1
        if out and t <= out[-1][1]:  # inside the open window (entry day through exit day): restart, no trade
            out[-1][1] = max(out[-1][1], entry + hold)
        elif entry <= n - 2:  # no trade at the period's final close
            out.append([entry, entry + hold])
    return [(e, x) for e, x in out]


def state_from_windows(n: int, windows: list[tuple[int, int]]) -> np.ndarray:
    """de-risked[k] is True when the portfolio held after the close of day k is de-risked."""
    s = np.zeros(n, dtype=bool)
    for e, x in windows:
        s[e:min(x, n)] = True
    return s


def window_lengths(n: int, windows: list[tuple[int, int]]) -> list[int]:
    """Number of de-risked closes in each window, after cutting at the period end."""
    return [min(x, n - 1) - e for e, x in windows]


# ---------------------------------------------------------------- portfolio simulation

@dataclass
class Prices:
    """Daily simple returns on the period's trading days (index 0 has no return) and month-ends."""
    r_acwi: np.ndarray
    r_btc: np.ndarray
    r_fxy: np.ndarray
    rf: np.ndarray
    month_end: np.ndarray


def _targets(values: dict, derisked_now: bool, was_derisked: bool, month_end: bool) -> dict:
    """Holdings after the close's trades, before costs (H3-A)."""
    total = sum(values.values())
    if month_end:  # month-end rebalance to 80/20, or 80/10/10 inside a window
        if derisked_now:
            return {"ACWI": W_ACWI * total, "BTC": W_BTC / 2 * total, "SLEEVE": W_BTC / 2 * total}
        return {"ACWI": W_ACWI * total, "BTC": W_BTC * total, "SLEEVE": 0.0}
    t = dict(values)
    if derisked_now and not was_derisked:  # entry: sell half the BTC held into the sleeve
        t["BTC"], t["SLEEVE"] = values["BTC"] / 2, values["SLEEVE"] + values["BTC"] / 2
    elif was_derisked and not derisked_now:  # exit: the whole sleeve back into BTC
        t["BTC"], t["SLEEVE"] = values["BTC"] + values["SLEEVE"], 0.0
    return t


def simulate(p: Prices, derisked: np.ndarray, hedge: bool = False) -> np.ndarray:
    """Daily portfolio returns (length n - 1) after costs (H3-A to H3-D).

    Holdings move with returns each day; trades happen at the close. The sleeve is T-bills
    earning rf_t, or FXY in the hedge variant. No trade is made at the final close.
    """
    sleeve_asset = "FXY" if hedge else "TB"
    r_sleeve = p.r_fxy if hedge else p.rf
    n = len(derisked)
    v = {"ACWI": W_ACWI, "BTC": W_BTC, "SLEEVE": 0.0}
    out = np.empty(n - 1)
    prev_total = 1.0
    for k in range(1, n):
        v = {"ACWI": v["ACWI"] * (1 + p.r_acwi[k]), "BTC": v["BTC"] * (1 + p.r_btc[k]),
             "SLEEVE": v["SLEEVE"] * (1 + r_sleeve[k])}
        if k < n - 1:
            now, before = bool(derisked[k]), bool(derisked[k - 1])
            target = _targets(v, now, before, bool(p.month_end[k]))
            entry = now and not before
            cost = 0.0
            for asset, rate_key in (("ACWI", "ACWI"), ("BTC", "BTC"), ("SLEEVE", sleeve_asset)):
                rate = COST[rate_key] * (ENTRY_MULT if entry and asset != "ACWI" else 1)
                cost += rate * abs(target[asset] - v[asset])
            total = sum(target.values())
            v = {a: x * (total - cost) / total for a, x in target.items()}
        total_now = sum(v.values())
        out[k - 1] = total_now / prev_total - 1
        prev_total = total_now
    return out


# ---------------------------------------------------------------- metrics (H3-H)

def metrics(r: np.ndarray, rf: np.ndarray, derisked: np.ndarray, n_windows: int) -> dict:
    excess = r - rf
    sd = excess.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    value = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    mdd = float((value / peak - 1).min())
    ann_ret = float(np.prod(1 + r) ** (ANNUAL / len(r)) - 1)
    return {"ann_return": ann_ret, "ann_vol": float(r.std(ddof=1) * np.sqrt(ANNUAL)),
            "sharpe": float(np.sqrt(ANNUAL) * excess.mean() / sd) if sd > 0 else np.nan,
            "sortino": float(np.sqrt(ANNUAL) * excess.mean() / downside) if downside > 0 else np.nan,
            "max_drawdown": mdd, "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
            "pct_days_derisked": float(np.mean(derisked[:-1])), "episodes": n_windows}


def sharpe(r: np.ndarray, rf: np.ndarray) -> float:
    excess = r - rf
    return float(np.sqrt(ANNUAL) * excess.mean() / excess.std(ddof=1))


# ---------------------------------------------------------------- random timing (H3-F)

def random_windows(lengths: list[int], n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Non-overlapping windows with the given lengths at uniform random entry closes from the second
    close (a period starts undisturbed at 80/20, H3-E) so each window exits before the final close,
    by rejection sampling."""
    if not lengths:
        return []
    for _ in range(MAX_TRIES):
        entries = [int(rng.integers(1, n - 1 - L)) for L in lengths]
        spans = sorted((e, e + L) for e, L in zip(entries, lengths))
        if all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1)):
            return spans
    raise RuntimeError("Could not place non-overlapping random windows")


def random_timing(p: Prices, lengths: list[int], n_sims: int = N_RANDOM, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(p.r_acwi)
    out = np.empty(n_sims)
    for i in range(n_sims):
        w = random_windows(lengths, n, rng)
        out[i] = sharpe(simulate(p, state_from_windows(n, w)), p.rf[1:])
    return out


# ---------------------------------------------------------------- bootstrap (H3-G)

def sharpe_diff_bootstrap(r3: np.ndarray, r2: np.ndarray, rf: np.ndarray,
                          reps: int = BOOT_REPS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(r3)
    diffs = np.empty(reps)
    for b in range(reps):
        idx = stationary_indices(n, rng, BLOCK_MEAN)
        diffs[b] = sharpe(r3[idx], rf[idx]) - sharpe(r2[idx], rf[idx])
    lo, hi = np.percentile(diffs, CI)
    return {"point": sharpe(r3, rf) - sharpe(r2, rf), "lower95": float(lo), "upper95": float(hi), "replicates": reps}


# ---------------------------------------------------------------- verdict (section 2)

def h3_pass(sharpe_yen: float, sharpe_vix: float, random_sharpes: np.ndarray) -> bool:
    """Sharpe(yen) > Sharpe(VIX) AND Sharpe(yen) >= 90th percentile of random timing."""
    return bool(sharpe_yen > sharpe_vix and sharpe_yen >= np.percentile(random_sharpes, RANDOM_PCT))


# ---------------------------------------------------------------- run

def prices_for(a: pd.DataFrame, days: pd.DatetimeIndex, all_days: pd.DatetimeIndex) -> Prices:
    sub = a.loc[days]
    month = all_days.to_period("M")
    last_of_month = pd.Series(all_days, index=all_days).groupby(month).transform("max")
    me = np.asarray(days.isin(all_days[all_days == last_of_month.values]))
    ret = lambda c: np.concatenate([[0.0], sub[c].to_numpy()[1:] / sub[c].to_numpy()[:-1] - 1])
    return Prices(ret("ACWI"), ret("BTC_close"), ret("FXY"), sub["rf"].to_numpy(), me)


def run_strategy(p: Prices, signal: np.ndarray | None, hold: int = HOLD, hedge: bool = False):
    n = len(p.r_acwi)
    w = [] if signal is None else windows_from_signal(signal, hold)
    state = state_from_windows(n, w)
    return simulate(p, state, hedge), state, w


def sensitivity_grid(p: Prices, sig: dict[str, np.ndarray]) -> list[dict]:
    """Section 8.5 grid, H3-I: in each cell the yen variant and strategy 2 (and 2') share the
    cell's holding period. Reported in full; never used to choose parameters."""
    rf = p.rf[1:]
    rows = []
    for pct, col in GRID_PCTS.items():
        for hold in GRID_HOLDS:
            r_yen = run_strategy(p, sig[col], hold)[0]
            row = {"yen_move_pct": pct, "hold": hold}
            for label, comp in COMPARISONS.items():
                row[f"sharpe_diff_{label}"] = sharpe(r_yen, rf) - sharpe(run_strategy(p, sig[comp], hold)[0], rf)
            rows.append(row)
    return rows


def main() -> None:
    import local_projections as lp  # shared guard: refuse to run with uncommitted code
    if RUN_MANIFEST.exists():
        raise RuntimeError(f"{RUN_MANIFEST.name} exists: H3 results have already been produced. "
                           "Log a deviation (results seen) before any rerun.")
    commit = lp.run_commit()
    a = pd.read_csv(ALIGNED, index_col="date", parse_dates=["date"])
    s = pd.read_csv(SIGNALS, index_col="date", parse_dates=["date"])
    all_days = a.index

    metric_rows, grid_rows, boot_rows, verdict_rows, random_out = [], [], [], [], None
    for pname, (start, end) in [("train", TRAIN), ("test", TEST)]:
        days = all_days[(all_days >= start) & (all_days <= end)]
        p = prices_for(a, days, all_days)
        rf = p.rf[1:]
        sig = {c: s.loc[days, c].to_numpy(dtype=float) for c in ["S_yen", "S_vix", "S_vix2", "S_yen_p85", "S_yen_p95"]}
        runs = {"1_buy_and_hold": run_strategy(p, None),
                "2_vix": run_strategy(p, sig["S_vix"]),
                "2prime_vix2": run_strategy(p, sig["S_vix2"]),
                "3_yen": run_strategy(p, sig["S_yen"])}
        for name in ["2_vix", "2prime_vix2", "3_yen"]:  # H3-J: hedge variant, descriptive
            src = {"2_vix": "S_vix", "2prime_vix2": "S_vix2", "3_yen": "S_yen"}[name]
            runs[f"{name}_fxy_hedge"] = run_strategy(p, sig[src], hedge=True)
        for name, (r, state, w) in runs.items():
            metric_rows.append({"period": pname, "strategy": name, **metrics(r, rf, state, len(w))})

        grid_rows += [{"period": pname, **row} for row in sensitivity_grid(p, sig)]

        r3 = runs["3_yen"][0]
        for label, name in [("primary", "2_vix"), ("secondary", "2prime_vix2")]:
            boot_rows.append({"period": pname, "comparison": label,
                              **sharpe_diff_bootstrap(r3, runs[name][0], rf)})

        if pname == "test":  # strategy 4 and the verdict (H3-E, H3-F)
            lengths = window_lengths(len(days), runs["3_yen"][2])
            random_out = random_timing(p, lengths)
            s3 = sharpe(r3, rf)
            for label, name in [("primary", "2_vix"), ("secondary", "2prime_vix2")]:
                verdict_rows.append({"comparison": label, "sharpe_yen": s3, "sharpe_competitor": sharpe(runs[name][0], rf),
                                     "random_p90": float(np.percentile(random_out, RANDOM_PCT)),
                                     "yen_episodes": len(runs["3_yen"][2]), "competitor_episodes": len(runs[name][2]),
                                     "pass": h3_pass(s3, sharpe(runs[name][0], rf), random_out)})

    verdict = pd.DataFrame(verdict_rows)
    verdict = pd.concat([verdict, pd.DataFrame([{"comparison": "combined (both must pass)",
                                                  "pass": bool(verdict["pass"].all())}])], ignore_index=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(OUT_TABLES / "h3_metrics.csv", index=False)
    pd.DataFrame({"sim": range(len(random_out)), "sharpe": random_out}).to_csv(OUT_TABLES / "h3_random_timing.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT_TABLES / "h3_bootstrap.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT_TABLES / "h3_sensitivity.csv", index=False)
    verdict.to_csv(OUT_TABLES / "h3_verdict.csv", index=False)
    RUN_MANIFEST.write_text(json.dumps({
        "run_commit": commit, "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "numpy": np.__version__, "pandas": pd.__version__,
        "random_timing": {"sims": N_RANDOM, "seed": SEED}, "bootstrap": {"reps": BOOT_REPS, "mean_block": BLOCK_MEAN, "seed": SEED},
        "outputs": ["tables/h3_metrics.csv", "tables/h3_random_timing.csv", "tables/h3_bootstrap.csv",
                    "tables/h3_sensitivity.csv", "tables/h3_verdict.csv"]}, indent=2))
    print(verdict.to_string(index=False))
    print(f"Run commit {commit}; outputs listed in {RUN_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
