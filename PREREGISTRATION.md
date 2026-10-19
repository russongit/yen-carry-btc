# Pre-Registration: Yen Carry Unwind Transmission to Bitcoin

**Status:** FROZEN, version 1.0
**Freeze date:** 2026-10-19 (Asia/Singapore)
**Freeze record:** git tag `prereg-v1.0`. The commit hash is recorded in the repository history and in the memo appendix; it cannot be written inside the file it identifies.
**Author:** Russ

Rule: this file is never edited after the freeze commit. Every later change goes in `DEVIATIONS.md` (template in section 11), stating whether it was made before or after seeing affected results.

---

## 0. Pre-freeze revision history

| Version | Changes |
|---|---|
| v1 (draft) | Initial hypotheses, data, signals, events, analysis plan |
| v2 (draft) | Dollar control changed to DXY ex-yen; US 2Y added as a control; Japan 2Y added as a tested driver (H1d); mechanism check M1; rate-channel decomposition |
| v1.0 (frozen) | H1c test changed from sign test to Wilcoxon signed-rank; pass verdicts tied to raw p-values with Holm reporting rule; every "Verify" item checked against primary sources; predictive windows moved to start at 17:00 New York (after US yield publication); MoF daily intervention dates added; BoJ meeting overlaps disclosed; category B scope and placebo rule defined; Coinbase fallback and A6 dating rules pre-specified; DXY ex-yen formula tested on real data |

---

## 1. Design decisions (confirmed at freeze)

These are design decisions, not regression variables.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Identification strategy | MoF yen-buying intervention episodes are the primary event set (H1c) | Interventions are policy-driven yen appreciation shocks, not caused by global risk-off. They are the cleanest available test of a yen-specific channel. |
| D2 | Primary signal components | FX move + FX volatility only; COT excluded from primary signal | Fewer degrees of freedom; COT is weekly and release dates shift |
| D3 | Dollar control | DXY ex-yen: DXY's formula rebuilt from its five non-yen currencies, weights rescaled to sum to 1 | Raw DXY includes the yen (13.6% weight), so it would mechanically absorb part of the effect being tested |
| D4 | Competing-signal calibration | VIX and equity thresholds set so their firing frequency in the training period equals the yen signal's | Unequal firing rates make precision comparisons meaningless |
| D5 | Primary horizon for H1b and H1d | h = 5 trading days | One pre-specified horizon avoids testing 10 horizons and picking the best |
| D6 | Backtest de-risking scope | BTC sleeve only, cut by half, into T-bills | The thesis concerns BTC; cutting equity too would mix in a generic risk-off effect |
| D7 | Sample end date | 2026-09-30 | A fixed end prevents the sample from moving during the work |
| D8 | Role of US 2Y yield | Control in every regression (ΔUS2Y), no directional hypothesis | US rate shocks move both USD/JPY and BTC. A US yield fall cuts carry incentive (BTC down) but eases dollar liquidity (BTC up), so the sign is ambiguous. |
| D9 | Role of Japan 2Y yield | Tested driver of carry unwinds (H1d), not a control in H1a or H1b | A rise in Japanese yields cuts carry incentive without directly changing US dollar liquidity, so it is the cleaner test of the carry-incentive channel. Controlling for it in H1a and H1b would remove rate-driven unwinds from the variation the yen coefficient is estimated on. |

---

## 2. Hypotheses and pass criteria

**Sign conventions used throughout:**
- `r_yen = −Δln(USDJPY)`. Positive `r_yen` means the yen appreciated.
- `r_dxyx` positive means the dollar strengthened against the non-yen basket.
- Yield changes (`ΔUS2Y`, `ΔJP2Y`) are in basis points. Positive means yields rose.

### Primary hypotheses

| ID | Hypothesis | Test | Pass criterion |
|---|---|---|---|
| H1a | In stress regimes, yen appreciation coincides with larger BTC declines, after controls | h = 0 regression (8.2), coefficient γ₀ on `S × r_yen` | γ₀ < 0, one-sided HAC p < 0.05 |
| H1b | In stress regimes, yen appreciation predicts BTC declines over the next 5 days, after controls | h = 5 local projection (8.2), coefficient γ₅ | γ₅ < 0, one-sided HAC p < 0.05 |
| H1c | BTC falls after MoF yen-buying intervention episodes | Event study on category A episodes (7, 8.1), CAR[−1,+2] | Mean CAR < 0, median CAR < 0, and exact one-sided Wilcoxon signed-rank p < 0.10 |
| H1d | Rising Japanese 2Y yields (lower carry incentive) predict BTC declines over the next 5 days, after controls including US 2Y yields | Carry-incentive regression (8.3), coefficient φ on `ΔJP2Y` | φ < 0, one-sided HAC p < 0.05 |
| H2 | The yen signal predicts BTC drawdowns better than a frequency-matched VIX signal | Episode-level precision, test period (8.4) | Precision(yen) − Precision(VIX) ≥ 10 percentage points AND Precision(yen) > unconditional base rate |
| H3 | De-risking on the yen signal beats VIX timing and random timing after costs | Backtest, test period (8.5) | Sharpe(yen) > Sharpe(VIX) AND Sharpe(yen) ≥ 90th percentile of the random-timing distribution |

**Why Wilcoxon for H1c:** a sign test discards magnitudes, so with 6 episodes one small positive CAR (for example +0.5% caused by an unrelated crypto event) would fail H1c even if the other five were large declines. The Wilcoxon signed-rank test weights each episode by the rank of its magnitude. Exact one-sided p-values with n = 6, where W+ is the sum of ranks of positive CARs:

| Outcome | W+ | p |
|---|---|---|
| All 6 negative | 0 | 0.016 |
| 5 negative, the positive has the smallest magnitude | 1 | 0.031 |
| 5 negative, the positive has the 2nd smallest magnitude | 2 | 0.047 |
| 5 negative, the positive has the 3rd smallest magnitude, or 4 negative with the two smallest positive | 3 | 0.078 |
| 5 negative, the positive has the 4th smallest magnitude | 4 | 0.109 (fails) |
| 5 negative, the positive has the largest magnitude | 6 | 0.219 (fails) |

H1c therefore tolerates a small contrary episode but not a large one.

**Multiple testing:** pass verdicts use each test's raw p-value against the threshold in the table. The memo also reports Holm-adjusted p-values across H1a, H1b, H1c, and H1d and states which results survive at a family-wise α of 0.10. H2 and H3 are criterion-based and reported with bootstrap confidence intervals.

**Interpretation rules, fixed in advance:**
- H1a, H1b, H1c, H1d all fail: no evidence of a yen-specific or rate-driven channel. Memo thesis: the carry-to-crypto narrative is generic risk-off.
- H1c passes, H1a and H1b fail: the channel appears on policy shocks but not in the general stress regime.
- H1d passes, H1a and H1b fail: rate-driven carry pressure reaches BTC, but daily yen moves are a poor real-time measure of it.
- H1a or H1b passes, H1d fails: the yen channel exists but is not driven by rate differentials; positioning and intervention are the more likely drivers.
- H1 passes, H2 and H3 fail: the channel exists but is not tradable with this signal.
- All pass: the signal has incremental value. The memo must still state the episode count beside every result.

### Secondary hypothesis (reported, not used for the thesis verdict)

| ID | Hypothesis | Test |
|---|---|---|
| H4 | The yen-BTC stress relationship changed after US spot BTC ETFs began trading (2024-01-11) | Add `Post2024 × S × r_yen` to the h = 0 regression |

---

## 3. Sample and calendar

| Item | Value |
|---|---|
| Burn-in start (for expanding percentiles) | 2015-01-02 |
| Analysis start | 2018-01-02 |
| Analysis end | 2026-09-30 |
| Training period | 2018-01-02 to 2022-12-30 |
| Test period | 2023-01-03 to 2026-09-30 |
| Daily index | NYSE trading days |

The training period is used only to calibrate competing-signal frequencies (D4) and for in-sample reporting. All thresholds are expanding-window, so no parameters are fitted on the test period.

---

## 4. Data sources and variable definitions

### Market series

| Variable | Definition | Source | Verification |
|---|---|---|---|
| `USDJPY` | Noon New York buying rate, JPY per USD | FRED `DEXJPUS` | Verified: FRED notes state noon buying rates in New York City; units JPY per USD |
| `EURUSD` | Noon New York buying rate, USD per EUR | FRED `DEXUSEU` | Verified: FRED units USD per EUR; noon New York |
| `GBPUSD` | Noon New York buying rate, USD per GBP | FRED `DEXUSUK` | Verified: H.10 quotes GBP in USD per unit |
| `USDCAD` | Noon New York buying rate, CAD per USD | FRED `DEXCAUS` | Verified: H.10 quotes CAD per USD |
| `USDSEK` | Noon New York buying rate, SEK per USD | FRED `DEXSDUS` | Verified: H.10 quotes SEK per USD |
| `USDCHF` | Noon New York buying rate, CHF per USD | FRED `DEXSZUS` | Verified: H.10 quotes CHF per USD |
| `BTC_noon`, `BTC_close`, `BTC_1700` | BTC-USD price at 12:00, 16:00, and 17:00 New York time | Coinbase BTC-USD hourly candles | Pre-specified fallback below |
| `SPX` | S&P 500 close | yfinance `^GSPC` | |
| `VIX` | VIX close | yfinance `^VIX` | |
| `ACWI` | Adjusted close (total return) | yfinance `ACWI` | Portfolio sleeve only |
| `FXY` | Adjusted close | yfinance `FXY` | Hedge variant only |
| `DGS3MO` | 3-month Treasury constant maturity yield | FRED `DGS3MO` | |
| `US2Y` | 2-year Treasury constant maturity yield | FRED `DGS2` | Verified: H.15 daily update posted at 4:15 p.m. ET, based on closing market bid yields (timing rule 8) |
| `US10Y` | 10-year Treasury constant maturity yield | FRED `DGS10` | Robustness only |
| `JP2Y` | 2-year JGB constant maturity yield | MoF `jgbcme_all.csv` (historical) + `jgbcme.csv` (current) | Verified structure: one title line, header `Date,1Y,2Y,...,40Y`, dates `YYYY/M/D`, missing values `-`. Parse with `skiprows=1, na_values="-"`. |
| `COT_net` | Leveraged funds (long − short) / open interest, JPY futures | CFTC Traders in Financial Futures report | Verified: Japanese Yen, Chicago Mercantile Exchange, CFTC Code #097741. Exploratory only (D2). |
| Robustness BTC | BTC/USDT hourly | Binance public API | |
| Robustness dollar | Broad dollar index | FRED `DTWEXBGS` | |

**BTC source fallback (pre-specified):** if Coinbase BTC-USD hourly candles do not cover 2017-01-01 to 2026-09-30 with no gap longer than 6 hours, Bitstamp BTC/USD hourly candles (via CCXT) become the primary BTC source. This is decided at ingestion, before any analysis, and logged in `DEVIATIONS.md`.

**JGB current-month file:** `jgbcme.csv` is checked at ingestion for the same structure as the historical file. Any difference is handled in the parser and logged.

### Derived variables

| Variable | Definition |
|---|---|
| `r_yen` | `−ln(USDJPY_t / USDJPY_{t−1})` |
| `r_dxyx` | `(1/0.864) × (−0.576 Δln EURUSD − 0.119 Δln GBPUSD + 0.091 Δln USDCAD + 0.042 Δln USDSEK + 0.036 Δln USDCHF)`. Rescaled weights: EUR −0.6667, GBP −0.1377, CAD +0.1053, SEK +0.0486, CHF +0.0417. |
| `r_eurusd` | `Δln EURUSD` (robustness) |
| `r_spx` | `ln(SPX_t / SPX_{t−1})` |
| `ΔVIX` | `VIX_t − VIX_{t−1}`, in points |
| `ΔUS2Y` | `US2Y_t − US2Y_{t−1}`, in basis points |
| `ΔJP2Y` | `JP2Y_t − JP2Y_{t−1}`, in basis points, Tokyo date t (timing rules 6 and 7) |
| `r_btc_t` | `ln(BTC_1700_t / BTC_1700_{t−1})` in predictive regressions; `ln(BTC_noon_t / BTC_noon_{t−1})` in h = 0 regressions |
| `rf` | `DGS3MO / 100 / 252` per trading day |
| `spread` | `US2Y − JP2Y` level, lagged one trading day. Exploratory regime variable only. |

**DXY ex-yen verification (completed before freeze):** `tests/verify_dxyx_formula.py` rebuilds full DXY (constant 50.14348112, ICE exponents) from Federal Reserve H.10 noon rates for 2024-08-05 to 2024-08-09. The rebuilt value falls inside the actual ICE DXY daily low-high range on all five days. Flipping the sign of any single exponent breaks the match (level errors of 1.2% to 74.3%), which confirms every quote direction. The script also confirms that the ex-yen weights sum to 1 after rescaling and that the return formula equals the log change of the geometric ex-yen index.

**Data integrity rules:**
- Raw downloads saved unmodified to `data/raw/` with download date and SHA-256 hash logged in `data/raw/MANIFEST.csv`.
- Spot-check at least five dates per series against the source website, including 2024-08-05, before any analysis.
- If a source fails or its format is unclear, stop and log it. No substitute series without a deviation entry, except the pre-specified BTC fallback.

---

## 5. Timing and alignment rules

**Admissibility standard for regressors:** a regressor is admissible if every market price it is computed from traded before the dependent window starts. Trading rules (signals and backtests) additionally require that the value was published before the trade.

1. **FX timing:** all FRED exchange rates are noon New York rates. The prices were observable in the market at noon, so using them at noon t is not look-ahead, even though FRED publishes them with a delay.
2. **Contemporaneous regressions (h = 0):** BTC return measured noon-to-noon (`BTC_noon`) to match FX timing. Equity, VIX, and US yield changes are end-of-day values, a documented offset that is not corrected.
3. **Predictive regressions (h ≥ 1, including H1d):** the dependent variable is `ln(BTC_1700_{t+h} / BTC_1700_t)`, where `BTC_1700` is the hourly price at 17:00 New York (local time, so daylight saving is handled by time-zone conversion). Every US regressor for day t is published by 16:15 New York, so all regressors precede the window.
4. **Weekends:** BTC moves over weekends are part of the following Monday's return. Nothing is dropped.
5. **Missing FX days:** the Federal Reserve observes some holidays when NYSE is open (e.g., Columbus Day, Veterans Day). On those days FX returns are missing. The next available FX return spans both days and is flagged. No forward-filling of returns, because FX trades on those days even though no fixing is published.
6. **JGB timing:** MoF yields are 3 p.m. Tokyo values, which precede the New York open on the same calendar date. `ΔJP2Y` for NYSE day t uses the Tokyo close of the same calendar date, which meets the admissibility standard. MoF publishes the value at 9:30 a.m. Tokyo the next business day, which is after 17:00 New York, so robustness check 7 in section 9 lags `ΔJP2Y` one day. Any trading rule uses `ΔJP2Y` lagged one trading day.
7. **Japanese holidays when NYSE is open:** the JGB market is closed, so `ΔJP2Y = 0` for that day (no new information) and the day is flagged. The next Tokyo trading day's change is measured from the last available close. This differs from rule 5 because JGBs do not trade on those days.
8. **US yield timing:** the H.15 daily update containing `DGS2` is posted at 4:15 p.m. ET, and the yields are based on closing market bid yields from quotations obtained by the Federal Reserve Bank of New York. The clock time of those quotes could not be verified from accessible official sources. Rule 3 removes any dependence on it: predictive windows start at 17:00 New York, after publication. `ΔUS2Y` is not lagged, because lagging would misalign the control with the same-day rate shock it is meant to absorb.
9. **COT timing (exploratory use only):** usable from the first trading day after the actual release date, not Tuesday + 3, because releases can be delayed.

---

## 6. Signal definitions

### Primary yen stress signal

`S_t = 1` when both conditions hold at noon New York on day t:
1. **Yen move:** 5-day yen return `−ln(USDJPY_t / USDJPY_{t−5})` above its expanding 90th percentile
2. **Yen volatility:** 20-day realized volatility of daily `r_yen`, annualized by √252, above its expanding 80th percentile

Expanding windows run from 2015-01-02 through t inclusive.

### Competing signals

- **VIX signal:** VIX close above its expanding percentile q_V
- **Equity signal:** S&P 500 5-day log return below its expanding percentile q_E

q_V and q_E are chosen from a grid of whole percentiles so that each signal's firing frequency over the training period is closest to the yen signal's training-period frequency (D4). Chosen values are recorded in `outputs/tables/calibration.csv` before any test-period result is computed.

### Episode clustering

Signal days within 10 trading days of the previous signal day belong to the same episode. Episode start = first signal day. H2 and H3 use episodes, not days, to avoid counting one stress period many times.

---

## 7. Event set

**Event day rule:** t = the Japan calendar date of the event, mapped to the NYSE trading day with that date. Because exact intervention timestamps are unpublished, all event windows start at t − 1.

**Clustering:** events of the same category within 10 trading days merge into one episode, dated by the first event.

### A. MoF yen-buying interventions (primary, H1c)

All dates through June 2026 are from MoF's official historical intervention data and quarterly releases. All are US dollar sold, Japanese yen bought.

| Episode | Event days | Source |
|---|---|---|
| A1 | 2022-09-22 | MoF historical data |
| A2 | 2022-10-21, 2022-10-24 | MoF historical data; MoF quarterly release Oct to Dec 2022 |
| A3 | 2024-04-29, 2024-05-01 | MoF historical data; MoF quarterly release Apr to Jun 2024 |
| A4 | 2024-07-11, 2024-07-12 | MoF historical data; MoF quarterly release Jul to Sep 2024 |
| A5 | 2026-04-30, 2026-05-04, 2026-05-06 | MoF historical data; MoF quarterly release Apr to Jun 2026 |
| A6 | Start 2026-07-30 | MoF monthly release confirms ¥15,399.3 billion of intervention between 2026-07-30 and 2026-08-26 without daily dates; start date from multiple reports that MoF intervened on July 30 and 31, the latter jointly with the US Treasury |

MoF's historical data show no other intervention days from 2018 through June 2026. MoF's monthly release for 2026-08-27 to 2026-09-28 reports zero intervention.

**Rules for days not yet published (fixed now, applied mechanically):**
- When MoF publishes the daily breakdown for July to September 2026, the A6 start date is set to the first intervention day in that data. Intervention days within 10 trading days of the previous A6 day extend A6; any other day in the sample forms a new category A episode. Both changes are logged in `DEVIATIONS.md`.
- The same rule applies to 2026-09-29 and 2026-09-30, which no MoF release covers as of the freeze date.
- H1c is computed after the July to September 2026 quarterly release is available. Any earlier run is labeled provisional.

**Overlaps with BoJ meeting days (disclosed in advance):**

| Episode | Overlap |
|---|---|
| A1 | BoJ decision on 2022-09-22 (no policy change) falls on event day t |
| A3 | BoJ decision on 2024-04-26 (no policy change) falls on t − 1 |
| A6 | BoJ decision on 2026-07-31 (no policy change) falls inside the event window |

Robustness check 9 in section 9 reports H1c results for the remaining episodes descriptively. Overlaps with FOMC decision days are identified from the Federal Reserve calendar at analysis time and disclosed the same way.

### B. BoJ policy tightening decisions (secondary event set)

**Scope:** BoJ decisions from 2022 onward that raised the policy rate or relaxed yield curve control, i.e., the normalization cycle relevant to the carry thesis. YCC adjustments made during the easing period before 2022 are out of scope and are excluded from the placebo set.

| Date | Decision | Source |
|---|---|---|
| 2022-12-20 | 10-year yield fluctuation range widened from about ±0.25 to about ±0.5 percentage points | BoJ statement |
| 2023-07-28 | YCC conducted with greater flexibility; range bounds became references, not rigid limits | BoJ statement |
| 2023-10-31 | 1.0% upper bound for 10-year yields became a reference | BoJ statement |
| 2024-03-19 | Changes in the monetary policy framework (end of negative interest rate policy) | BoJ statement |
| 2024-07-31 | Change in the guideline for money market operations (rate raised to around 0.25%) | BoJ statement; level from news reports |
| 2025-01-24 | Change in the guideline for money market operations (rate raised to 0.5%) | BoJ statement |
| 2025-12-19 | Change in the guideline for money market operations (rate raised to around 0.75%) | BoJ statement; level from Jiji Press |
| 2026-06-16 | Change in the guideline for money market operations (rate raised to 1.0%) | BoJ statement; level from news reports |
| 2026-09-18 | Change in the guideline for money market operations (rate raised to 1.25%) | BoJ statement; level from news reports |

### C. Market-identified unwind

| Date | Event | Source |
|---|---|---|
| 2024-08-05 (Monday) | Global selloff; Nikkei 225 fell about 12.4% | Associated Press report |

### D. Placebo set

All BoJ Monetary Policy Meeting decision dates from 2018-01-02 to 2026-09-30 whose released documents contain no change to the money market operations guideline, YCC parameters, or policy framework. Meeting dates that fall inside any category A event window are excluded. The list is compiled from BoJ statement pages before any analysis and saved to `outputs/tables/placebo_dates.csv`. Expected CAR ≈ 0. A significant placebo result means the event-study method is picking up something other than policy news.

---

## 8. Analysis specifications

**Common rules for every regression in 8.2 and 8.3:**
- Estimator: OLS. Standard errors: Newey-West HAC, `maxlags = max(5, h)`. Default homoskedastic errors are never reported.
- Report variance inflation factors for all regressors. If any VIF exceeds 10, report it and keep the specification unchanged; do not drop regressors after seeing results.
- Estimate on the full sample, the training period, and the test period.

### 8.1 Event study (H1c and descriptive)

- Abnormal return: `AR_t = r_btc_t − (a + b · r_spx_t)` using 16:00 New York BTC prices, with a and b estimated over trading days [−260, −11] before each event.
- Windows: CAR[−1, +2] (primary), CAR[−1, +5], CAR[−1, +10].
- Report every episode individually, plus raw (unadjusted) BTC returns beside abnormal returns.
- H1c test: `scipy.stats.wilcoxon(car, alternative="less", method="exact", zero_method="wilcox")` on CAR[−1, +2] across category A episodes, plus mean and median CAR.
- Also reported, not used for the verdict: one-sided sign test and the Boehmer, Musumeci and Poulsen (1991) standardized cross-sectional test.

### 8.2 Yen-channel regressions (H1a, H1b)

**h = 0 (H1a):**

`r_btc_t = α + β₁ r_yen_t + γ₀ (S_t × r_yen_t) + θ S_t + δ₁ r_spx_t + δ₂ ΔVIX_t + δ₃ r_dxyx_t + δ₄ ΔUS2Y_t + δ₅ r_btc_{t−1} + ε_t`

(BTC returns noon-to-noon, rule 2.)

**Local projections, h = 1 to 10 (H1b uses h = 5):**

`ln(BTC_1700_{t+h} / BTC_1700_t) = α_h + β_h r_yen_t + γ_h (S_t × r_yen_t) + θ_h S_t + δ_h' X_t + ε_{t+h}`

with `X_t` = `r_spx_t`, `ΔVIX_t`, `r_dxyx_t`, `ΔUS2Y_t`, `r_btc_t`.

- `S_t` enters on its own so that γ does not absorb the average stress-day return.
- Report the number of days with `S_t = 1` beside every coefficient involving S.
- Chart: β_h + γ_h with 90% confidence bands for h = 1 to 10. Horizons other than h = 5 are descriptive.

**Rate-channel decomposition (descriptive, not a hypothesis test):** re-estimate the h = 0 and h = 5 specifications adding `ΔJP2Y_t`. Report γ with and without it. The change in γ shows how much of the yen effect is associated with same-day Japanese rate moves. This is a descriptive comparison, not a causal mediation estimate.

### 8.3 Carry-incentive regressions (H1d and mechanism check)

**Mechanism check M1 (reported, not a hypothesis test):** do yield changes move the yen in the direction the carry story requires?

`r_yen_t = a + κ ΔJP2Y_t + λ ΔUS2Y_t + δ₁ r_spx_t + δ₂ ΔVIX_t + δ₃ r_dxyx_t + u_t`

Expected: κ > 0 (higher Japanese yields strengthen the yen) and λ < 0 (lower US yields strengthen the yen). If κ is indistinguishable from zero, the memo must say that daily Japanese rate moves do not visibly drive the yen, and H1d is interpreted accordingly.

**H1d (h = 5):**

`ln(BTC_1700_{t+5} / BTC_1700_t) = α + φ ΔJP2Y_t + ψ ΔUS2Y_t + δ' X_t + ε_{t+5}`

with `X_t` = `r_spx_t`, `ΔVIX_t`, `r_dxyx_t`, `r_btc_t`.

- φ is the coefficient of interest. H1d passes if φ < 0 with one-sided HAC p < 0.05.
- ψ is reported with no directional hypothesis (D8).
- The h = 0 version (noon-to-noon BTC, same regressors) is reported as descriptive.
- Japanese 2Y yields moved very little for much of the period when the BoJ policy rate was negative (until March 2024). Report the standard deviation of `ΔJP2Y` by calendar year beside the result so readers can see where the identifying variation sits.

### 8.4 Signal evaluation (H2)

- **Target event:** BTC falls at least 10% below `BTC_close_t` at any hourly observation between 16:00 on day t and 16:00 on trading day t + 10 (hourly prices capture weekend moves).
- **Per signal, per period:** number of episodes, precision (share of episodes followed by the target event), recall (share of target events preceded by a signal episode within 10 trading days), false positives, and the unconditional base rate.
- **Confidence intervals:** stationary block bootstrap, mean block length 20 days, 2,000 replications, seed 42. Expect wide intervals given the episode count; report them regardless.

### 8.5 Backtest (H3)

| Item | Specification |
|---|---|
| Portfolio | 80% ACWI, 20% BTC, rebalanced on the last NYSE trading day of each month |
| Prices | Daily at 16:00 New York |
| Strategy 1 | Buy-and-hold with monthly rebalance |
| Strategy 2 | VIX signal de-risking (frequency-matched) |
| Strategy 3 | Yen signal de-risking |
| Strategy 4 | Random timing: 1,000 simulations placing the same number of non-overlapping de-risking windows of the same length at random start dates within the same period, seed 42 |
| De-risking rule | Signal at t → at the 16:00 close of t + 1, cut the BTC sleeve by half (20% → 10%) into T-bills earning `rf`. Hold 10 trading days. A new signal during the window restarts the 10-day count. |
| Hedge variant | Same rule, freed weight into FXY instead of T-bills |
| Costs | 20 bps per side for BTC, 10 bps per side for ETFs; doubled on de-risking entry trades |
| Metrics | Annualized return, annualized volatility, Sharpe (with time-varying `rf`), Sortino, max drawdown, Calmar, percentage of days de-risked, number of episodes |
| Inference | Stationary block bootstrap CI for Sharpe(3) − Sharpe(2), same settings as 8.4 |

**Sensitivity grid (reported in full, not used to choose parameters):** yen-move percentile {85, 90, 95} × holding period {5, 10, 20} trading days. Report the Sharpe difference vs strategy 2 in every cell.

---

## 9. Pre-listed robustness checks

1. Weekly (Wednesday-to-Wednesday) returns for the h = 0 regressions in 8.2 and 8.3
2. Binance BTC/USDT instead of the primary BTC source
3. EUR/USD instead of DXY ex-yen as the dollar control
4. `DTWEXBGS` instead of DXY ex-yen as the dollar control
5. US 10Y instead of US 2Y in all regressions
6. `ΔJP2Y` added as a control in H1a and H1b (also tests whether JGB-driven global bond moves confound γ)
7. H1d with `ΔJP2Y` lagged one trading day (publication-time standard, rule 6)
8. H1d with `Δspread` in place of the two separate yield changes
9. H1c CARs reported for episodes without BoJ meeting overlaps (A2, A4, A5), descriptive only
10. Leave-one-episode-out: re-estimate H1a and H1c dropping each episode in turn; report the range
11. Returns winsorized at the 1st and 99th percentiles of the training period
12. Execution at the 16:00 close of t instead of t + 1
13. Event day shifted to t − 1 for all category A events

Any analysis not listed in sections 8 and 9 is labeled exploratory in the memo.

---

## 10. Exploratory analyses (labeled as such, no hypothesis tests)

- COT leveraged-fund JPY positioning as a conditioning variable
- BTC perpetual funding rates and open interest around category A episodes, if history covers them
- Combined yen AND VIX signal
- Regime split by `spread` (compressing vs widening over the prior 60 trading days)
- Carry-to-risk ratio (`spread` / 20-day realized yen volatility) as a conditioning variable, since carry attractiveness depends on the rate differential per unit of FX risk, not the differential alone

---

## 11. Deviation log template (`DEVIATIONS.md`)

| Date | Section | Change | Reason | Made before or after seeing affected results? |
|---|---|---|---|---|

---

## 12. Sources

**Primary sources**
- MoF, Foreign Exchange Intervention Operations (historical data, quarterly and monthly releases): https://www.mof.go.jp/english/policy/international_policy/reference/feio/index.html
- MoF, JGB interest rate data: https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/index.htm
- MoF, JGB interest rate Q&A (3 p.m. close, 9:30 a.m. next-day release): https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/qa.htm
- BoJ, Statements on Monetary Policy (2022 to 2026 index pages): https://www.boj.or.jp/en/mopo/mpmdeci/state_2022/index.htm (and `state_2023` to `state_2026`)
- Federal Reserve, H.10 Foreign Exchange Rates, release dated 2024-08-12: https://www.federalreserve.gov/releases/h10/20240812/
- Federal Reserve, H.15 Selected Interest Rates: https://www.federalreserve.gov/releases/h15/
- FRED, DEXJPUS series notes: https://fred.stlouisfed.org/series/DEXJPUS
- CFTC, Traders in Financial Futures report, 2024-07-09: https://www.cftc.gov/files/dea/cotarchives/2024/futures/financial_lf070924.htm

**Secondary sources**
- Wikipedia, U.S. Dollar Index (formula, verified empirically by the test script): https://en.wikipedia.org/wiki/U.S._Dollar_Index
- Yahoo Finance, DX-Y.NYB daily history (test data): https://finance.yahoo.com/quote/DX-Y.NYB/history/
- PNC, Yen Intervention Q&A (MoF intervened July 30 and 31, 2026): https://www.pnc.com/content/dam/pnc-com/pdf/aboutpnc/EconomicReports/EconomicUpdates/2026/Yen_Intervention_Q_A.pdf
- Jiji Press via Adnkronos, December 2025 rate level: https://english.adnkronos.com/2026/01/23/boj-keeps-policy-rate-unchanged-ups-fy-2026-growth-outlook/
- BabyPips, June 2026 rate level: https://babypips.com/news/headline-boj-rate-hike-june-2026-1-percent-31-year-high
- Free Press Journal, September 2026 rate level: https://www.freepressjournal.in/business/bank-of-japan-raises-interest-rate-to-125-highest-level-in-three-decades
- Associated Press via AccessWDUN, Nikkei drop on 2024-08-05: https://accesswdun.com/print/2024/8/1256059
