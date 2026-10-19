# Deviations from Pre-Registration

Every change to the frozen `PREREGISTRATION.md` (tag `prereg-v1.0`) is logged here. The pre-registration file itself is never edited.

| Date | Section | Change | Reason | Made before or after seeing affected results? |
|---|---|---|---|---|
| 2026-10-19 | 4 (BTC source fallback) | Primary BTC source switched from Coinbase BTC-USD to Bitstamp BTC/USD hourly candles (via CCXT), as pre-specified. | Coinbase hourly candles cover 85,387 of 85,440 hours from 2017-01-01 00:00 to 2026-10-01 00:00 UTC; the longest gap is 15 hours, starting 2018-08-10 01:00 UTC (confirmed by a second request to the Coinbase API), which exceeds the 6-hour limit. Bitstamp covers all 85,440 hours. Coinbase and Binance data are kept in `data/raw/`; Binance remains the robustness series. | Before. Decided at ingestion; no returns, signals, or analysis had been computed. |
| 2026-10-19 | 4 (data note, robustness dollar `DTWEXBGS`) | None; data note only. | DTWEXBGS is a revised vintage; values differ from originally published H.10 figures by up to 0.3% after rebasing; affects robustness check 4 only; no results seen. | Before. No results seen. |
| 2026-10-19 | 5 (timing rules; US yields) | New rule for US Treasury yields (`DGS2`, `DGS3MO`, `DGS10`) on days the Treasury market is closed while NYSE is open: the yield change is 0, the day is flagged `ust_holiday`, the next day's change spans both days, and the `rf` level carries over. | Section 5 covers FX (rule 5) and JGBs (rule 7) on such days but not US Treasuries. The cash Treasury market is closed on those days, so there is no new information; this follows the logic of rule 7. | Before. No results seen. |
