"""Compile the category D placebo list, PREREGISTRATION.md section 7 and DEVIATIONS.md event-study H.

Section 7: placebo dates are BoJ Monetary Policy Meeting decision dates from 2018-01-02 to
2026-09-30 whose released documents contain no change to the money market operations guideline,
YCC parameters, or policy framework; meetings inside a category A event window are excluded.

Each meeting is classified below by hand from its statement, with the reason. The script attaches a
sentence of evidence from the raw statement, checks that no "no change" meeting contains a
"decided to <action>" sentence, and writes outputs/tables/placebo_dates.csv for review.

Classes:
  no_change        guideline, YCC parameters and framework unchanged; no other decision
  other_action     guideline, YCC parameters and framework unchanged, but another policy decision
                   (facility, forward guidance, asset purchases or sales); excluded from the placebo
                   (DEVIATIONS.md, category D placebo rule: "policy framework" includes these)
  other_change     change to the guideline, YCC parameters or framework that is not tightening
  tightening_B     category B decision (section 7)
"""
from __future__ import annotations

import html
import io
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingestion as ing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "raw" / "boj_statements"
OUT = ROOT / "outputs" / "tables" / "placebo_dates.csv"
SAMPLE = (date(2018, 1, 2), date(2026, 9, 30))

# Section 7, category A episodes (A6 provisional until the July-September 2026 MoF data): event days.
CATEGORY_A = {"A1": "2022-09-22", "A2": "2022-10-21", "A3": "2024-04-29", "A4": "2024-07-11",
              "A5": "2026-04-30", "A6 (provisional start)": "2026-07-30"}
A_WINDOW = (-1, 10)  # DEVIATIONS.md event-study C: category A event windows [-1, +10]

NC, OA, OC, B = "no_change", "other_action", "other_change", "tightening_B"
# date: (class, reason, evidence regex)
CLASSIFICATION = {
    "2018-01-23": (NC, "", None), "2018-03-09": (NC, "", None), "2018-04-27": (NC, "", None),
    "2018-06-15": (NC, "", None),
    "2018-07-31": (OC, "Strengthening the Framework: long-term yields allowed to move with the price target; forward guidance introduced",
                   r"move upward and downward|forward guidance"),
    "2018-09-19": (NC, "", None), "2018-10-31": (NC, "", None), "2018-12-20": (NC, "", None),
    "2019-01-23": (NC, "", None), "2019-03-15": (NC, "", None),
    "2019-04-25": (OA, "Forward guidance clarified: low rates at least through around spring 2020", r"at least through around spring 2020"),
    "2019-06-20": (NC, "", None), "2019-07-30": (NC, "", None), "2019-09-19": (NC, "", None),
    "2019-10-31": (OA, "New forward guidance for policy rates", r"new forward guidance"),
    "2019-12-19": (NC, "", None), "2020-01-21": (NC, "", None),
    "2020-03-16": (OA, "COVID-19 easing: ETF and J-REIT purchase limits raised, special funds-supplying operations; rate and YCC target unchanged",
                   r"(?:12 trillion|special funds-supplying|actively purchase)"),
    "2020-04-27": (OC, "JGB purchases without an upper limit (guideline for market operations changed)", r"without setting an upper limit"),
    "2020-05-22": (OA, "Unscheduled meeting: new fund-provisioning measure for small and medium-sized firms", r"new fund-provisioning measure|fund-provisioning measure"),
    "2020-06-16": (NC, "", None), "2020-07-15": (NC, "", None), "2020-09-17": (NC, "", None),
    "2020-10-29": (NC, "", None),
    "2020-12-18": (OA, "Special COVID-19 program extended by 6 months; assessment of policy measures announced", r"extend the duration of the Special Program"),
    "2021-01-21": (NC, "", None),
    "2021-03-19": (OC, "Assessment: YCC band clarified, fixed-rate purchase operations for consecutive days, ETF purchase changes, interest scheme",
                   r"fluctuate|consecutive days"),
    "2021-04-27": (NC, "", None),
    "2021-06-18": (OA, "Special COVID-19 program extended; climate-change fund-provisioning measure announced", r"to extend the duration of the Special Program"),
    "2021-07-16": (OA, "Preliminary outline of climate-change fund-provisioning measure", r"climate change"),
    "2021-09-22": (NC, "", None), "2021-10-28": (NC, "", None),
    "2021-12-17": (OA, "Special COVID-19 program partly extended, partly ended", r"decided to extend the Special Program"),
    "2022-01-18": (NC, "", None), "2022-03-18": (NC, "", None),
    "2022-04-28": (OC, "YCC conduct changed: fixed-rate purchase operations at 0.25 percent every business day (judgment: operational, target and band unchanged)",
                   r"every business day"),
    "2022-06-17": (NC, "", None), "2022-07-21": (NC, "", None),
    "2022-09-22": (OA, "Special funds-supplying operations phased out", r"phase out"),
    "2022-10-28": (NC, "", None),
    "2022-12-20": (B, "Category B (section 7)", r"modify the conduct of yield curve control"),
    "2023-01-18": (OA, "Funds-supplying operations against pooled collateral enhanced", r"pooled collateral"),
    "2023-03-10": (NC, "", None),
    "2023-04-28": (OA, "Broad-perspective review launched; forward guidance on policy rates removed", r"broad-perspective review"),
    "2023-06-16": (NC, "", None),
    "2023-07-28": (B, "Category B (section 7)", r"conduct yield curve control with greater flexibility"),
    "2023-09-22": (NC, "", None),
    "2023-10-31": (B, "Category B (section 7)", r"further increase the flexibility"),
    "2023-12-19": (NC, "", None), "2024-01-23": (NC, "", None),
    "2024-03-19": (B, "Category B (section 7)", r"monetary policy framework"),
    "2024-04-26": (NC, "", None),
    "2024-06-14": (OA, "Decision to reduce JGB purchases, with a plan due in July", r"reduce its purchase amount of JGBs"),
    "2024-07-31": (B, "Category B (section 7)", r"guideline for money market operations"),
    "2024-09-20": (NC, "", None), "2024-10-31": (NC, "", None),
    "2024-12-19": (NC, "Review of monetary policy published; no policy change", None),
    "2025-01-24": (B, "Category B (section 7)", r"guideline for money market operations"),
    "2025-03-19": (NC, "", None), "2025-05-01": (NC, "", None),
    "2025-06-17": (OA, "Interim assessment: plan for JGB purchase reductions from April 2026", r"plan to reduce the amount of its monthly outright purchases"),
    "2025-07-31": (NC, "", None),
    "2025-09-19": (OA, "Decision to sell ETF and J-REIT holdings", r"sell these assets|sell its holdings"),
    "2025-10-30": (NC, "", None),
    "2025-12-19": (B, "Category B (section 7)", r"guideline for money market operations"),
    "2026-01-23": (NC, "", None), "2026-03-19": (NC, "", None), "2026-04-28": (NC, "", None),
    "2026-06-16": (B, "Category B (section 7)", r"guideline for money market operations"),
    "2026-07-31": (NC, "", None),
    "2026-09-18": (B, "Category B (section 7)", r"guideline for money market operations"),
}
NO_CHANGE_EVIDENCE = r"(?:negative interest rate of minus 0\.1 percent|encourage the uncollateralized overnight call rate to remain at around [\d.]+(?: to [\d.]+)? percent)"
ACTION_CHECK = r"decided (?:to|that it would) (?:introduce|change|extend|make|conduct|discontinue|sell|reduce|modify|phase|abolish|terminate|enhance|strengthen)"


def statement_text(href: str) -> str:
    doc = FOLDER / "docs" / href.rsplit("/", 1)[1]
    raw = doc.read_bytes()
    if doc.suffix == ".pdf":
        from pypdf import PdfReader
        t = " ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages)
    else:
        s = raw.decode("utf-8", errors="replace")
        i = s.find("<main")
        t = html.unescape(re.sub(r"<[^>]+>", " ", s[i:s.find("</main>")] if i >= 0 else s))
    return re.sub(r"\s+", " ", t)


def sentence(text: str, pattern: str) -> str:
    m = re.search(r"[^.]*(?:" + pattern + r")[^.]*\.?", text, re.IGNORECASE)
    return m.group(0).strip()[:300] if m else ""


def a_episode(day: date, trading_days: pd.DatetimeIndex) -> str:
    """Category A episode whose [-1, +10] window contains the day, if any."""
    for name, d in CATEGORY_A.items():
        pos = trading_days.get_loc(pd.Timestamp(d))
        lo, hi = trading_days[pos + A_WINDOW[0]], trading_days[pos + A_WINDOW[1]]
        if lo <= pd.Timestamp(day) <= hi:
            return name
    return ""


def main() -> None:
    trading_days = pd.read_csv(ROOT / "data" / "processed" / "aligned_daily.csv", usecols=["date"],
                               parse_dates=["date"])["date"]
    trading_days = pd.DatetimeIndex(trading_days)
    meetings: dict[date, list[dict]] = {}
    for year in ing.BOJ_YEARS:
        for r in ing.parse_boj_index((FOLDER / f"index_{year}.htm").read_bytes()):
            if r["title"].startswith("(Reference)") or not r["href"] or not SAMPLE[0] <= r["date"] <= SAMPLE[1]:
                continue
            meetings.setdefault(r["date"], []).append(r)

    missing = sorted({str(d) for d in meetings} - set(CLASSIFICATION))
    extra = sorted(set(CLASSIFICATION) - {str(d) for d in meetings})
    if missing or extra:
        raise SystemExit(f"Classification table out of step with the BoJ index: missing {missing}, extra {extra}")

    rows = []
    for d in sorted(meetings):
        cls, reason, pat = CLASSIFICATION[str(d)]
        text = " ".join(statement_text(r["href"]) for r in meetings[d])
        evidence = sentence(text, pat) if pat else sentence(text, NO_CHANGE_EVIDENCE)
        actions = sentence(text, ACTION_CHECK)
        ep = a_episode(d, trading_days)
        rows.append({
            "date": d, "title": " | ".join(r["title"] for r in meetings[d]), "class": cls, "reason": reason,
            "in_category_A_window": ep, "placebo": cls == NC and not ep,  # DEVIATIONS.md placebo rule
            "flag_for_review": cls == OA or "judgment" in reason or (cls == NC and bool(actions)),
            "evidence": evidence, "decided_to_sentence": actions,
            "source": " | ".join(r["href"] for r in meetings[d]),
        })
    out = pd.DataFrame(rows)
    if (out["evidence"] == "").any():
        raise SystemExit(f"No evidence sentence found for {list(out.loc[out['evidence'] == '', 'date'])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby("class").size().to_string())
    print(f"placebo: {int(out['placebo'].sum())} meetings; "
          f"flagged for review: {int(out['flag_for_review'].sum())}; inside a category A window: "
          f"{int((out['in_category_A_window'] != '').sum())}")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
