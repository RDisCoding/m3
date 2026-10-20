# trader

Paper-trading research system for NSE equities: market data, strategies, risk, paper broker and metrics, built to test whether a rule-based swing-trading system has an edge on ₹10,000 of virtual capital.

**Status:**
- **Phase 1 (data layer): complete.** All 500 Nifty 500 stocks plus the Nifty 50 and Nifty 500 indices are stored from 2015, with a daily update, a trading calendar, point-in-time index membership and corporate-action price corrections.
- **Phase 2 (building blocks): complete.** A market profile holds every exchange-specific rule. Also built: the core trading types, the Indian cost model, the price panel, synthetic test markets and the indicator library.
- **Audit (2026-10-20): complete.** Everything above was reviewed and checked against NSE's official daily files; see [docs/AUDIT-2026-10-20.md](docs/AUDIT-2026-10-20.md) for findings, fixes and the architecture diagram.
- **Next:** portfolio accounting, the paper broker's fill rules, position sizing and the risk engine.

## Setup

```powershell
cd D:\eta\trader
uv sync                          # creates .venv with Python 3.11 and all dependencies
uv run trader data auth-check    # verifies the Upstox token with one read-only quote call
uv run trader data backfill      # first run only: full history since 2015 (~6 minutes)
uv run trader data official-prices   # first run only: NSE's official daily files since 2015 (~1 hour, resumable)
uv run trader data corporate-actions
uv run trader data refresh       # the daily update; run after 15:30 IST
```

### Upstox token

The system uses the Upstox **analytics token**. It is read-only, lasts one year and covers market-data APIs only; it cannot place orders.

The token is read from the first of these that defines `UPSTOX_ACCESS_TOKEN`:

1. the process environment,
2. `D:\eta\trader\.env`,
3. `D:\eta\.env` (where it currently lives).

Both `.env` files are outside version control and `.gitignore` excludes `.env*`. The token is never printed: `auth-check` shows only its location, length and expiry date, and every log handler redacts it.

## Commands

Run any of them as `uv run trader data ...`.

| Command | What it does |
|---|---|
| `refresh` | Daily update: recent history for every stock, today's bars after the close, today's index list, new NSE files and corporate actions, corrections, quality checks |
| `backfill [--force] [--only A,B]` | Full history since 2015 for the universe and both indices; resumable |
| `universe` | Records today's Nifty 500 list in the point-in-time membership table |
| `calendar` | Today's status, latest and next session, upcoming holidays, sessions per year |
| `corporate-actions [--no-fetch]` | Downloads NSE corporate actions and recomputes all price corrections |
| `official-prices [--start] [--end]` | Downloads NSE's official daily price files (bhavcopy); resumable, one request per second |
| `status` | Coverage, freshness, correction counts and recent runs |
| `panel-check [--adjustment ...]` | Loads the full 500-stock panel as the backtester will, times it, checks its structure |
| `verify [--raw]` | Quality checks on the corrected series strategies read; `--raw` checks the vendor's files as stored |
| `auth-check`, `instruments`, `map`, `fetch`, `show`, `smoke` | Step-1 tools: token, instrument master, symbol mapping, ad-hoc fetch, file inspection, end-to-end check |

Outside the `data` group:

| Command | What it does |
|---|---|
| `uv run trader market show` | Prints the active market profile |
| `uv run trader features check` | Builds every indicator for all stocks, times it, and proves on random real dates that no indicator uses future data |

## How the data layer works

**Storage.** Bars are Parquet files, one per instrument, holding Upstox's data exactly as delivered. NSE's official daily files (bhavcopy, every session since 2015) are kept under `data/official/`, with each stock's history extracted to one table that follows renames and ISIN changes. Relational data lives in `data/trader.sqlite`: index membership, per-file status, holidays, corporate actions, correction factors, the run log and quality findings.

**Daily update** (`refresh`):
1. Re-fetches each stock's last 10 stored sessions from Upstox history. If any of those closes moved by more than 0.5%, Upstox has re-adjusted the stock for a new split or bonus, so the full history is fetched again.
2. After 15:30 IST on a trading day, today's candles for all stocks come from the batch quote endpoint, in two requests. They are stored as provisional (`source = upstox_live`), and the next day's history confirms and relabels them.
3. Records today's Nifty 500 list, downloads any missing NSE daily files, fetches NSE corporate actions for the recent and upcoming window, recomputes corrections, and runs quality checks on the corrected series.
4. Reruns are safe: unchanged data never rewrites a file. A failed full re-download leaves the file as it was, and a crash rolls back the database.

**Calendar.** Past trading days are the days Nifty 50 actually traded, which includes special sessions such as Diwali Muhurat trading. Future days use the weekday rule minus Upstox's official holiday list for the year.

**Membership.** The table records each stock's start and end dates in the Nifty 500. Today's list was the first load, so membership before 2026-10-20 assumes today's members were always members. That is survivorship bias: stocks that dropped out of the index or delisted are missing from backtests before that date. Membership is accurate from 2026-10-20 onward.

**Price corrections.** Strategies read `LocalDataFeed`, which applies corrections when data is loaded and never modifies stored files. Each corporate action is measured exactly: the vendor's adjustment at the ex-date is the ratio of its prices to NSE's on the days before and on the ex-date.

| Event | Upstox data | Correction |
|---|---|---|
| Split, bonus | Adjusted, except two 2015 bonuses | Applied only where missing: MINDACORP and GODREJIND 2015 |
| Demerger | Not adjusted (ITC 2025), partly adjusted (RELIANCE 2023) or over-adjusted (NMDC 2022, SCI 2023) | Price made continuous at NSE's ex-date opening price (set in a special price-discovery session); all 20 corrected |
| Scheme of arrangement | Same, often a demerger under another name | Same method when the ex-date move exceeds 3%; 14 of 20 corrected, e.g. CG Power 2016, IIFL 2019, Adani Enterprises 2015 |
| Rights issue | Adjusted for some, not others | Theoretical ex-rights price, applied only where the vendor did not; 20 of 37 |
| Dividend | Usually not adjusted; for some stocks subtracted from all earlier prices (IOC, Coal India, BPCL, GAIL, Hero, OFSS) | Vendor subtractions undone; dividends are included only in the `total_return` view and are credited as cash by the engine |

**Official prices.** Upstox rounds old split-adjusted prices for about 30 stocks: TATAINVEST's ₹494.45 is stored as 49.0 after a 1:10 split, an error of up to 2.7% a day. It also has a few wrong days, such as its close on the 2025-02-01 Budget Saturday. So on every day NSE published a file, the feed uses NSE's official open, high, low, close and volume, multiplied by the stock's adjustment ratio for that stretch between corporate actions. The ratio is a median, so rounding cancels out, and it is taken next to the following corporate action, so the series stays continuous. Days without an NSE file keep the vendor's values. Result: 0 of 497 stocks disagree with NSE's prices in a way no corporate action explains (56 in the raw vendor data).

The feed also:
- drops junk from before a stock's listing, such as MAZDOCK's zero-price bars from 2017 and DELHIVERY's odd trades in 2016;
- drops PRIVISCL's placeholder bars before 2020-08-21 (a flat ₹0.15 for five years);
- repairs ADANIENT's stale ex-date opening price of 2015-06-03;
- repairs a 32-bit volume overflow in the vendor data (IDEA on 2024-08-30).

For each event, `adjustment_factors` records the method used and the evidence.

Views: `none` returns stored prices. `economic`, the default, applies the corrections above. `total_return` also applies dividends. Every view adds `unadj_close`, the price as actually traded that day (NSE's close), and `traded_value` (NSE's turnover).

**Sessions.** Each day is classified from the whole market's data for that day only, so a backtest and the live view agree. Days most stocks lack in the vendor data (4, all 2015-2018) and short sessions are left out of the panel. The short sessions are the 11 Muhurat sessions, NSE's two 2024 disaster-recovery drills and the 2017-07-10 outage; typical activity on them is under 25% of normal, against over 50% on every regular day.

## Building blocks for the trading engine (Phase 2)

**Market profile** (`src/trader/market/`). Everything that differs between exchanges lives in one object, selected by the `TRADER_MARKET` setting (default `IN_NSE`):
- currency, time zone and session hours,
- calendar and regime index, benchmarks, risk-free rate and trading days per year,
- share rules (whole or fractional), tick sizes, the daily price-band rule, and the cost model.

The engine reads the profile once at start-up, so it adds no measurable time; calls through it are within a fraction of a microsecond of direct calls and give identical results. Moving to another market means adding a profile, not editing the engine. See `docs/us-market-backup-plan.md`.

**Indian costs.** Zerodha delivery charges are reproduced to the paisa: STT, exchange charge, SEBI fee, stamp duty, GST, and the flat ₹15.34 DP charge on the first sale of a stock each day.

**Core types** (`src/trader/core/types.py`). Signals, opportunities, proposals, risk verdicts, orders, fills, positions, trades and equity points are plain slotted dataclasses, which keeps them cheap to create in the backtest loop.

**Panel** (`src/trader/core/panel.py`). Aligned daily fields for all stocks plus the indices, on the market's sessions (minus the sparse and short sessions above). Slices are independent copies, so changing a slice can never corrupt the original. Prices are adjusted; each `Bar` carries `traded_scale` to convert them to the prices actually traded that day, which costs, ticks, price bands and share counts need.

**Synthetic markets** (`src/trader/data/synthetic.py`). A random walk with no edge, a market with planted momentum, and one with planted post-breakout drift. Later phases use them to check that strategies find nothing in noise and do find a planted edge.

**Indicators** (`src/trader/features/`). 31 indicators, computed for all 500 stocks at once in about 4-5 seconds: moving averages, RSI, MACD, ATR, ADX, Donchian channels, returns, 12-1 momentum, volatility, relative strength, volume z-score, distance from the 52-week high, typical traded value and cross-sectional ranks. Market-trend features come from the regime index. Guarantees, each tested:
- A value on day t is identical whether the data ends on day t or continues past it.
- Changing later data never changes earlier values.
- The 600-session window used in live trading matches full history within one part in a million.
- Stocks with gaps are computed on their own bars.
- Equals TA-Lib (to about 1e-14) from each stock's first value: averages are seeded the way TA-Lib seeds them. TA-Lib is only a test dependency.
- The look-ahead checker covers the market-trend features too, and is itself tested on a planted leak.
- Strategies receive copies, so editing what they get cannot change the shared store.

## Data layout

```
data/                                    (git-ignored, re-creatable)
  trader.sqlite                          membership, bar status, holidays, corporate actions, factors, runs, quality
  raw/instruments/NSE_YYYYMMDD.json.gz   daily copy of the Upstox instrument master
  raw/universe/ind_nifty500list_*.csv    NSE Indices constituent lists
  ref/instruments_NSE.parquet            parsed master (NSE_EQ + NSE_INDEX)
  ref/nifty500_mapping.csv               constituents mapped to instrument keys
  bars/daily/{SYMBOL}.parquet            one file per stock
  bars/index/{INDEX}.parquet             NIFTY_50.parquet, NIFTY_500.parquet
  official/bhavcopy/YYYY/YYYYMMDD.parquet  NSE's official daily file, equity series
  official/by_stock.parquet              each stored stock's official history (renames followed)
  logs/trader.log
```

## Verified data facts

- **Endpoints.** Daily history: `GET /v3/historical-candle/{key}/days/1/{to}/{from}`, at most 10 years per request, newest first. Today's candles for up to 500 instruments per call: `GET /v3/market-quote/ohlc?interval=1d`; above 500 the API returns `UDAPI100043`. Holidays: `GET /v2/market/holidays`. All work with the analytics token.
- **Today's bar.** History stops at the previous session. The provisional candle from the quote endpoint matched the final history exactly when tested on 2026-10-05 and 2026-10-19.
- **Upstox corporate-actions endpoint** (`GET /v2/fundamentals/{isin}/corporate-actions`). It returns only about the last 12 months of dividends, bonuses, splits and rights issues. It omits demergers, even recent ones such as Vedanta 2026-04-30 and HEG 2026-09-07. It gives no issue price for rights issues. It is not usable for backtest corrections, so NSE's feed is the source.
- **NSE corporate actions** come from `nseindia.com/api/corporates-corporateActions` after a cookie handshake. Date-range queries return full lists; per-symbol queries are capped at 20 rows.
- **NSE bhavcopy.** The old format runs until 2024-07-05, the UDiFF format from 2024-07-08. Upstox's close equals NSE's official close on normal days.
- **Close at the high.** On some days in 2026, 9-15% of stocks close exactly at their high. NSE's files show the same, so this is real, not a feed artifact.
- **Network.** One of the two Cloudflare addresses behind `api.upstox.com` does not accept connections from this network. A 3-second connect timeout makes the client fail over quickly.

## Tests

```powershell
uv run pytest             # 229 offline tests, using recorded real responses in tests/fixtures
uv run pytest -m network  # 5 live checks: Upstox history, batch quotes, holidays, instrument master, NSE feed
```
