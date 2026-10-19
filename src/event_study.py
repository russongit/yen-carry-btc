"""Event study for H1c, PREREGISTRATION.md sections 7 and 8.1.

Category A (MoF yen-buying interventions) is the primary test; categories B (BoJ tightening) and
C (2024-08-05) are descriptive; category D (no-change BoJ meetings) is the placebo. Robustness
checks 9, 10 and 13 (section 9).

Every timing or threshold choice cites PREREGISTRATION.md or a DEVIATIONS.md entry
(event-study decisions A to J, placebo rule).

Usage (one run only, and only after A6 is finalized; guarded by outputs/run_manifest_h1c.json):
    python src/event_study.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingestion as ing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "processed" / "aligned_daily.csv"
PLACEBO = ROOT / "outputs" / "tables" / "placebo_dates.csv"
DEVIATIONS = ROOT / "DEVIATIONS.md"
OUT_TABLES = ROOT / "outputs" / "tables"
RUN_MANIFEST = ROOT / "outputs" / "run_manifest_h1c.json"

# Section 7, category A: event days per episode (all USD sold, JPY bought).
CATEGORY_A = {
    "A1": ["2022-09-22"],
    "A2": ["2022-10-21", "2022-10-24"],
    "A3": ["2024-04-29", "2024-05-01"],
    "A4": ["2024-07-11", "2024-07-12"],
    "A5": ["2026-04-30", "2026-05-04", "2026-05-06"],
    "A6": ["2026-07-30"],  # provisional start (section 7); set from MoF daily data before the run
}
# Section 7 rule: H1c runs only after the July-September 2026 MoF release. Set to True, together
# with the final A6 days and any new category A episodes, in the same commit as the DEVIATIONS.md
# entry that finalizes them (event-study decision F).
A6_FINALIZED = False
A6_FINAL_MARKER = "A6 finalized"   # text the finalizing DEVIATIONS.md entry must contain
MOF_COVERAGE_REQUIRED = date(2026, 9, 30)
A_NO_BOJ_OVERLAP = ["A2", "A4", "A5"]  # section 7 disclosure; robustness check 9

# Section 7, categories B and C.
CATEGORY_B = ["2022-12-20", "2023-07-28", "2023-10-31", "2024-03-19", "2024-07-31", "2025-01-24",
              "2025-12-19", "2026-06-16", "2026-09-18"]
CATEGORY_C = ["2024-08-05"]

CLUSTER_GAP = 10          # section 7: events within 10 trading days merge into one episode
EST_WINDOW = (-260, -11)  # section 8.1: estimation window, trading days before the event
EXCLUDE_WINDOW = (-1, 10) # event-study decision C: category A event windows dropped from estimation
MIN_EST_DAYS = 200        # event-study decision C
WINDOWS = {"car_m1_p2": (-1, 2), "car_m1_p5": (-1, 5), "car_m1_p10": (-1, 10)}  # section 8.1
PRIMARY = "car_m1_p2"
H1C_ALPHA = 0.10          # section 2: one-sided Wilcoxon p < 0.10
PLACEBO_ALPHA = 0.10      # event-study decision I
FOMC_WINDOW = (-1, 2)     # event-study decision J


# ---------------------------------------------------------------- event days and episodes

def to_trading_day(d: str | date, days: pd.DatetimeIndex) -> tuple[pd.Timestamp, bool]:
    """Section 7 and event-study decision A: the NYSE day with that date, else the next NYSE day (flagged)."""
    ts = pd.Timestamp(d)
    pos = days.searchsorted(ts)
    if pos >= len(days):
        raise ValueError(f"{d} is after the last trading day")
    return days[pos], days[pos] != ts


def cluster(event_days: list[pd.Timestamp], days: pd.DatetimeIndex, gap: int = CLUSTER_GAP) -> list[list[pd.Timestamp]]:
    """Section 7: events within `gap` trading days of the previous one merge; each episode is dated by its first."""
    out: list[list[pd.Timestamp]] = []
    for d in sorted(event_days):
        if out and days.get_loc(d) - days.get_loc(out[-1][-1]) <= gap:
            out[-1].append(d)
        else:
            out.append([d])
    return out


# ---------------------------------------------------------------- market model and CARs

@dataclass
class EventResult:
    name: str
    anchor: pd.Timestamp
    n_est: int
    flag_short_estimation: bool
    alpha: float
    beta: float
    resid_var: float
    cars: dict
    raw: dict
    car_var_primary: float  # forecast-error variance of the primary CAR (event-study decision E)


def estimation_positions(anchor_pos: int, exclude: set[int]) -> np.ndarray:
    """Section 8.1 window [-260, -11] around the anchor, minus excluded positions (decision C)."""
    pos = np.arange(anchor_pos + EST_WINDOW[0], anchor_pos + EST_WINDOW[1] + 1)
    pos = pos[pos >= 0]
    return np.array([p for p in pos if p not in exclude], dtype=int)


def exclusion_positions(anchors: list[int], n: int) -> set[int]:
    """Event-study decision C: every category A event window [-1, +10]."""
    return {p for a in anchors for p in range(a + EXCLUDE_WINDOW[0], a + EXCLUDE_WINDOW[1] + 1) if 0 <= p < n}


def event_result(name: str, anchor: pd.Timestamp, r_btc: pd.Series, r_spx: pd.Series, exclude: set[int]) -> EventResult:
    """Market model AR_t = r_btc_t - (a + b r_spx_t) (section 8.1), log returns (decision B)."""
    days = r_btc.index
    a_pos = days.get_loc(anchor)
    est = estimation_positions(a_pos, exclude)
    y, x = r_btc.to_numpy(), r_spx.to_numpy()
    ok = est[~np.isnan(y[est]) & ~np.isnan(x[est])]
    X = np.column_stack([np.ones(len(ok)), x[ok]])
    coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid = y[ok] - X @ coef
    s2 = float(resid @ resid / (len(ok) - 2))
    xm, sxx = x[ok].mean(), float(((x[ok] - x[ok].mean()) ** 2).sum())
    cars, raw = {}, {}
    car_var = np.nan
    for wname, (lo, hi) in WINDOWS.items():
        w = np.arange(a_pos + lo, a_pos + hi + 1)
        ar = y[w] - (coef[0] + coef[1] * x[w])
        cars[wname], raw[wname.replace("car", "raw")] = float(ar.sum()), float(y[w].sum())
        if wname == PRIMARY:  # Patell-adjusted forecast-error variance of the CAR (decision E)
            L = len(w)
            car_var = s2 * (L + L ** 2 / len(ok) + (x[w].sum() - L * xm) ** 2 / sxx)
    return EventResult(name, anchor, len(ok), len(ok) < MIN_EST_DAYS, float(coef[0]), float(coef[1]), s2,
                       cars, raw, float(car_var))


# ---------------------------------------------------------------- tests

def h1c_test(cars: np.ndarray) -> dict:
    """Section 8.1 and section 2: exact one-sided Wilcoxon (less) on CAR[-1, +2]; H1c passes if
    mean < 0, median < 0 and p < 0.10."""
    w = stats.wilcoxon(cars, alternative="less", method="exact", zero_method="wilcox")
    mean, median = float(np.mean(cars)), float(np.median(cars))
    return {"n": len(cars), "mean": mean, "median": median, "wilcoxon_stat": float(w.statistic),
            "p_one_sided": float(w.pvalue), "pass": bool(mean < 0 and median < 0 and w.pvalue < H1C_ALPHA)}


def sign_test(cars: np.ndarray) -> float:
    """Section 8.1, reported only: one-sided sign test (more negative CARs than chance)."""
    nonzero = cars[cars != 0]
    return float(stats.binomtest(int((nonzero < 0).sum()), len(nonzero), 0.5, alternative="greater").pvalue)


def bmp_test(cars: np.ndarray, car_vars: np.ndarray) -> dict:
    """Event-study decision E: standardized CARs, t = sqrt(N) mean / sd, one-sided less, t(N - 1)."""
    scar = cars / np.sqrt(car_vars)
    n = len(scar)
    t = float(np.sqrt(n) * scar.mean() / scar.std(ddof=1))
    return {"t_bmp": t, "p_one_sided": float(stats.t.cdf(t, n - 1))}


def placebo_test(cars: np.ndarray) -> dict:
    """Event-study decision I: two-sided exact Wilcoxon on placebo CAR[-1, +2]; p < 0.10 is significant."""
    w = stats.wilcoxon(cars, alternative="two-sided", method="exact", zero_method="wilcox")
    return {"n": len(cars), "mean": float(np.mean(cars)), "median": float(np.median(cars)),
            "p_two_sided": float(w.pvalue), "significant": bool(w.pvalue < PLACEBO_ALPHA)}


def leave_one_out(cars: dict[str, float]) -> pd.DataFrame:
    """Robustness check 10: H1c re-run dropping each episode in turn."""
    rows = []
    for drop in cars:
        rest = np.array([v for k, v in cars.items() if k != drop])
        rows.append({"dropped": drop, **h1c_test(rest)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- guards and checks

def check_mof(mof: pd.DataFrame, category_a: dict[str, list[str]], coverage_end: date) -> None:
    """Event-study decision G: the category A days must equal MoF's yen-buying days in the sample."""
    buy = mof[mof["pair"].str.contains("yen \\(bought\\)", regex=True)]
    in_sample = {str(d) for d in buy["date"] if date(2018, 1, 2) <= d <= min(coverage_end, date(2026, 9, 30))}
    listed = {d for days in category_a.values() for d in days if pd.Timestamp(d).date() <= coverage_end}
    if in_sample != listed:
        raise RuntimeError(f"Category A days differ from MoF data: only MoF {sorted(in_sample - listed)}, "
                           f"only section 7 {sorted(listed - in_sample)}")


def check_ready(coverage_end: date, deviations_text: str, finalized: bool = None) -> None:
    """Event-study decision F: no run before MoF covers 2026-09-30 and A6 is finalized in DEVIATIONS.md."""
    finalized = A6_FINALIZED if finalized is None else finalized
    if coverage_end < MOF_COVERAGE_REQUIRED:
        raise RuntimeError(f"MoF data covers only through {coverage_end}; H1c needs {MOF_COVERAGE_REQUIRED}")
    if not finalized or A6_FINAL_MARKER not in deviations_text:
        raise RuntimeError("A6 is not finalized: set A6_FINALIZED with the final days and log "
                           f"'{A6_FINAL_MARKER}' in DEVIATIONS.md first")


def fomc_overlaps(anchors: dict[str, pd.Timestamp], fomc: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Event-study decision J: FOMC decision days inside each [-1, +2] window; disclosure only."""
    rows = []
    for name, a in anchors.items():
        p = days.get_loc(a)
        lo, hi = days[p + FOMC_WINDOW[0]], days[p + FOMC_WINDOW[1]]
        for _, f in fomc.iterrows():
            fd = pd.Timestamp(f["date"])
            if lo <= fd <= hi:
                rows.append({"episode": name, "anchor": a.date(), "fomc_date": fd.date(), "fomc_type": f["type"]})
    return pd.DataFrame(rows, columns=["episode", "anchor", "fomc_date", "fomc_type"])


def load_placebo() -> list[str]:
    """Category D: meetings marked placebo in outputs/tables/placebo_dates.csv (DEVIATIONS.md placebo rule)."""
    p = pd.read_csv(PLACEBO)
    return [str(d) for d in p.loc[p["placebo"].astype(bool), "date"]]


# ---------------------------------------------------------------- run

def episodes_for(dates_by_name: dict[str, list[str]], days: pd.DatetimeIndex) -> tuple[dict, list]:
    """Anchor (first event day) per named episode, with any non-trading-day shifts flagged (decision A)."""
    anchors, shifted = {}, []
    for name, ds in dates_by_name.items():
        mapped = [to_trading_day(d, days) for d in ds]
        shifted += [(name, d) for d, (_, s) in zip(ds, mapped) if s]
        anchors[name] = min(m for m, _ in mapped)
    return anchors, shifted


def run_events(anchors: dict[str, pd.Timestamp], a: pd.DataFrame, exclude: set[int]) -> dict[str, EventResult]:
    return {n: event_result(n, t, a["r_btc_close"], a["r_spx"], exclude) for n, t in anchors.items()}


def main() -> None:
    import local_projections as lp  # shared guard: refuse to run with uncommitted code
    if RUN_MANIFEST.exists():
        raise RuntimeError(f"{RUN_MANIFEST.name} exists: H1c results have already been produced. "
                           "Log a deviation (results seen) before any rerun.")
    mof_file = sorted((ing.RAW / "mof_intervention").glob("foreign_exchange_intervention_operations_*.csv"))[-1]
    mof, coverage_end = ing.parse_mof_intervention(mof_file)
    check_ready(coverage_end, DEVIATIONS.read_text(encoding="utf-8"))
    check_mof(mof, CATEGORY_A, coverage_end)
    commit = lp.run_commit()

    a = pd.read_csv(ALIGNED, index_col="date", parse_dates=["date"])
    days = a.index
    a_anchors, a_shifted = episodes_for(CATEGORY_A, days)
    exclude = exclusion_positions([days.get_loc(t) for t in a_anchors.values()], len(days))

    def table(results: dict[str, EventResult], category: str) -> pd.DataFrame:
        return pd.DataFrame([{"category": category, "episode": r.name, "anchor": r.anchor.date(), "n_est": r.n_est,
                              "flag_short_estimation": r.flag_short_estimation, "alpha": r.alpha, "beta": r.beta,
                              **r.cars, **r.raw} for r in results.values()])

    res_a = run_events(a_anchors, a, exclude)
    b_anchors, b_shifted = episodes_for({f"B{i + 1}": [d] for i, d in enumerate(CATEGORY_B)}, days)
    c_anchors, _ = episodes_for({"C1": CATEGORY_C}, days)
    d_anchors, d_shifted = episodes_for({f"D_{d}": [d] for d in load_placebo()}, days)
    res_b, res_c, res_d = (run_events(x, a, exclude) for x in (b_anchors, c_anchors, d_anchors))

    cars_a = {n: r.cars[PRIMARY] for n, r in res_a.items()}
    car_arr = np.array(list(cars_a.values()))
    verdict = {**h1c_test(car_arr), "sign_test_p": sign_test(car_arr),
               **{f"bmp_{k}": v for k, v in bmp_test(car_arr, np.array([r.car_var_primary for r in res_a.values()])).items()}}

    # Robustness 13: every category A event day shifted to t-1, estimation re-anchored.
    shifted_anchors = {n: days[days.get_loc(t) - 1] for n, t in a_anchors.items()}
    exclude13 = exclusion_positions([days.get_loc(t) for t in shifted_anchors.values()], len(days))
    res_13 = run_events(shifted_anchors, a, exclude13)
    robust = [{"check": "rc9_no_boj_overlap", **h1c_test(np.array([cars_a[n] for n in A_NO_BOJ_OVERLAP]))},
              {"check": "rc13_event_day_t_minus_1", **h1c_test(np.array([r.cars[PRIMARY] for r in res_13.values()]))}]
    loo = leave_one_out(cars_a).assign(check="rc10_leave_one_out")

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.concat([table(res_a, "A"), table(res_b, "B"), table(res_c, "C"), table(res_d, "D")],
              ignore_index=True).to_csv(OUT_TABLES / "event_cars.csv", index=False)
    pd.DataFrame([verdict]).to_csv(OUT_TABLES / "h1c_verdict.csv", index=False)
    pd.concat([pd.DataFrame(robust), loo], ignore_index=True).to_csv(OUT_TABLES / "event_robustness.csv", index=False)
    pd.DataFrame([placebo_test(np.array([r.cars[PRIMARY] for r in res_d.values()]))]).to_csv(
        OUT_TABLES / "placebo_results.csv", index=False)
    fomc_overlaps({**a_anchors, **b_anchors}, ing.parse_fomc(), days).to_csv(OUT_TABLES / "fomc_overlaps.csv", index=False)
    RUN_MANIFEST.write_text(json.dumps({
        "run_commit": commit, "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mof_file": mof_file.name, "mof_coverage_end": str(coverage_end), "scipy": scipy.__version__,
        "shifted_event_days": [list(map(str, x)) for x in a_shifted + b_shifted + d_shifted],
        "outputs": ["tables/event_cars.csv", "tables/h1c_verdict.csv", "tables/event_robustness.csv",
                    "tables/placebo_results.csv", "tables/fomc_overlaps.csv"]}, indent=2))
    print(pd.DataFrame([verdict]).to_string(index=False))
    print(f"Run commit {commit}; outputs listed in {RUN_MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
