# Data Spot-Check Record

Date of check: 2026-10-19 (Asia/Singapore). Performed independently of the
ingestion code, against the original sources, before any analysis.

Dates checked: 2018-01-02, 2020-03-16, 2022-09-22, 2024-08-05, 2026-07-31.
Values compared: the raw downloaded values in data/raw/ (unadjusted Close for
Yahoo series).

## Result

101 of 105 values match their source exactly. The 4 differences are
documented revisions (DTWEXBGS, below).

| Series | Values | Source checked against | Result |
|---|---|---|---|
| DEXJPUS, DEXUSEU, DEXUSUK, DEXCAUS, DEXSDUS, DEXSZUS | 30 | Federal Reserve H.10 weekly releases dated 2018-01-08, 2020-03-23, 2022-09-26, 2024-08-12, 2026-08-03 | All exact |
| DGS2, DGS3MO, DGS10 | 15 | FRED series data (fredgraph CSV) | All exact |
| DTWEXBGS | 5 | Federal Reserve H.10 weekly releases (as published) | 1 exact, 4 differ (see below) |
| JP2Y | 5 | MoF jgbcme_all.csv, 2Y column | All exact; 2018-01-02 uses 2018-01-04 (Tokyo closed 2 and 3 January) |
| ^GSPC, ^VIX, ACWI, FXY (Close) | 20 | Yahoo Finance historical data pages | All exact |
| BTC Bitstamp hourly closes | 15 | Bitstamp public OHLC API (btcusd, step 3600) | All exact; all candles had nonzero volume |
| CFTC TFF Japanese yen 097741 (open interest, leveraged long, leveraged short) | 15 (5 reports) | CFTC TFF archive pages for as-of 2018-01-02, 2020-03-10, 2022-09-20, 2024-07-30, 2026-07-28 | All exact |

Total: 105 values. 101 match exactly; 4 are the DTWEXBGS revisions below.

## DTWEXBGS differences (revisions, not download errors)

| Date | H.10 as published | FRED (downloaded) |
|---|---|---|
| 2018-01-02 | 118.6053 (base Jan 1997 = 100) | 109.6445 (base Jan 2006 = 100) |
| 2020-03-16 | 121.2512 | 120.9352 |
| 2022-09-22 | 126.3149 | 126.1667 |
| 2024-08-05 | 123.7919 | 123.5787 |
| 2026-07-31 | 119.7034 | 119.7034 |

FRED holds the current revised vintage of the broad dollar index; H.10
releases show values as originally published. Exchange rates are not
revised, which is why all 30 FX values match. Affects robustness check 4
only. Logged in DEVIATIONS.md.

## Notes

- BTC candles are labeled by UTC open time. 16:00 New York = 20:00 UTC in
  summer time (e.g., 2024-08-05) and 21:00 UTC in winter time (2018-01-02).
- The COT table in the check assigned reports by as-of date; alignment.py
  correctly uses release dates (section 5, rule 9).
