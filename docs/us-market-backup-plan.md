# Backup Plan: Switching the System to US Equities

**Status:** draft, not started. Written 2026-10-20 as a fallback in case the Indian system fails its evaluation. No code exists for it yet.
**Main plan:** `C:\Users\rudra\.claude\plans\this-is-the-plan-composed-flamingo.md` (Indian market).

## 1. Purpose and when to use it

The prototype is being built for NSE stocks. This plan describes how to point the same system at US stocks if the Indian version does not perform. It is a backup, not a replacement. The Indian track continues unless one of the triggers below fires.

**Activate this plan if, after the Indian walk-forward backtest (main plan Phase 5) or three months of paper trading, any of these hold:**

- **No edge.** Net expectancy per trade is zero or negative after costs on the validation period, or profit factor is below 1.1.
- **Costs dominate.** Trading costs take more than half of gross profit.
- **The ₹10,000 book can't trade.** More than a third of good candidates are rejected because whole shares don't fit the position limits, or the system makes fewer than 10 trades a year.
- **Too much risk.** Maximum drawdown exceeds 25%.
- **No better than luck.** The strategies fail to beat the random-entry baseline.

**Cheaper option worth considering earlier:** once the trading core exists, run the same frozen strategies on US data as a comparison. That needs only the US data layer, about 4-5 days of work, and gives evidence before the Indian track has to fail.

## 2. Why the US could work better, and why it might not

| Factor | India today | US | Effect on this system |
|---|---|---|---|
| Cost per round trip on a small position | About 1.0-1.8% on ₹1,000-2,000, mostly the flat ₹15.34 DP charge | About 0.04% on $20: $0 commission, tiny SEC and FINRA fees on sells, spread of a few hundredths of a percent | **25-40x cheaper.** The single biggest reason for this plan |
| Whole shares | Required, so stocks above about ₹1,500 are excluded | Fractional shares at most brokers, from $1 | Any S&P 500 stock is usable, and position sizes can be exact |
| Capital | ₹10,000 | About $104 at roughly ₹96 per dollar | Workable only because of fractional shares |
| Survivorship-free history | Not available cheaply; current constituents only | Available, e.g. Norgate, with delisted stocks and daily index membership | Backtests can be honest |
| Corporate-action quality | Vendor misses or botches demergers and rights issues; our own correction layer was needed | Major vendors adjust splits, dividends and spin-offs well; still to be verified | Less custom work |
| Historical news | None free, so the LLM layer cannot be backtested | Alpaca/Benzinga news back to 2015 | The news layer can be tested on history |
| Market efficiency | Less efficient; documented momentum edge, e.g. Nifty200 Momentum 30 | Very efficient large caps; momentum still documented but weaker since the 2000s | Edges are harder to find |
| Benchmark to beat | Nifty 50 / Nifty 500 | S&P 500 index fund, historically hard to beat | Higher bar |
| Trading hours in IST | 09:15-15:30 | 19:00-01:30 (US summer time), 20:00-02:30 (US winter time) | Daily job runs in the early morning IST; no human needed |
| Currency | Rupees | Dollars | Results must be shown in both USD and INR; rupee moves add risk and noise |
| Real money later | Simple | Money must leave India under LRS, with remittance costs and foreign-tax paperwork | Paper trading is unaffected; real money gets harder (section 10) |

**Honest summary:** the US removes the cost and share-size problems that most threaten the Indian version, and it allows proper backtests. In exchange, edges are harder to find and real-money logistics are worse. Paper trading in the US is easy.

## 3. What changes and what stays

The data layer was built with a market-neutral core, so most of it carries over.

| Area | Current module | US change |
|---|---|---|
| Parquet bar storage, idempotent upserts | `storage/parquet.py` | Reuse as is |
| SQLite tables, run log | `storage/db.py` | Reuse; add a `market` column or a separate database file per market |
| Rate limiter, retry logic | `data/providers/ratelimit.py` | Reuse |
| Price provider | `data/providers/upstox.py` | **New** `data/providers/alpaca.py` (or Norgate); same `MarketDataProvider` protocol |
| Instrument master, symbol mapping | `data/instruments.py` | **New**: Alpaca assets list (symbol, exchange, tradable, fractionable). US tickers get reused, e.g. old FB vs META, so store bars under a **permanent ID** (Norgate asset ID, FIGI or CIK), not the ticker |
| Universe membership | `data/universe.py` | **New source**: daily point-in-time S&P 500 membership (Norgate) or free monthly snapshots (hanshof/sp500_constituents on GitHub). Same membership table |
| Trading calendar | `data/calendar.py` | Reuse the class. Observed sessions come from SPY bars; future sessions from Alpaca's calendar endpoint or the `exchange_calendars` NYSE calendar; half-day sessions included |
| Corporate actions | `data/corporate_actions.py` | Drop the NSE subject parser. Keep the per-event verification idea: check vendor-adjusted prices for gaps on split, dividend and spin-off dates from Alpaca's corporate-actions feed |
| Quality checks, fragment trimming, volume repair | `data/quality.py` | Reuse |
| Data feed for strategies | `data/feed.py` | Reuse; add `currency` and the market's calendar index (SPY) |
| Daily refresh | `data/bars.py` | Reuse the logic. Today's bars come from Alpaca's multi-symbol snapshot endpoint after the US close |
| CLI | `trader/cli/` | Add a `--market us` option or a `trader-us` command group |
| Settings | `settings.py` | Add `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` (paper keys) in `.env`; market time zone `America/New_York` |
| Trading core (not built yet) | main plan Phases 2-5 | Cost model, share rules, price bands, regime index and benchmarks come from a per-market profile (section 12) |
| LLM news layer (not built yet) | main plan Phase 7 | Sources change to Alpaca/Benzinga news and SEC EDGAR filings; prompts lose India-specific wording |

## 4. Data sources

| Option | Cost | History | Delisted stocks | Point-in-time index membership | Role |
|---|---|---|---|---|---|
| **Alpaca** (free Basic plan) | $0 | 7+ years of daily bars; real-time IEX, other exchanges 15 minutes delayed | Unverified | No | Daily bars, today's prices, calendar, corporate actions, news, and a free paper-trading account |
| **Norgate Data** Platinum | $630/year or $346.50 for 6 months | Decades | Yes | Yes, daily, through its Windows Python plugin | **Backtest source**: the standard retail choice for honest US backtests |
| Massive (formerly Polygon.io) | $0 (5 calls/min, 2 years); $29/month for 5 years; $79/month for 10 years | Plan-dependent | Unverified | No | Alternative to Alpaca for bars |
| Tiingo | Free tier; paid around $10-30/month | Long, adjusted for splits and dividends | Unverified | No | Cheap alternative or cross-check |
| hanshof/sp500_constituents (GitHub) | Free | S&P 500 membership snapshots since 1996 | n/a | Monthly snapshots | Free substitute for Norgate's membership data |
| SEC EDGAR API | Free | All filings | n/a | n/a | Company filings (8-K, 10-Q) for the news layer |

**Recommendation:**
1. Use **Alpaca** for the daily pipeline, today's prices, the calendar, news and paper trading. It is free and also gives a broker-grade paper account to mirror trades against.
2. Use a **6-month Norgate Platinum subscription**, $346.50, for the research and backtest phase. It is the only listed option confirmed to include delisted stocks and daily index membership. Without it, US backtests repeat the survivorship bias the Indian version already has.
3. Keep **Tiingo or Massive** as a cross-check source, the role Yahoo played in the Indian verification.

## 5. US market rules to encode

| Rule | Detail (as researched 2026-10-20) | Where it goes |
|---|---|---|
| Commission | $0 at Alpaca and most US brokers | Cost model |
| SEC Section 31 fee | $20.60 per $1 million of sales, sells only, since 2026-04-04 | Cost model |
| FINRA trading activity fee | $0.000195 per share sold, capped at $9.79 per trade. One source reports it is temporarily $0 from Oct to Dec 2026; verify | Cost model |
| Spread and slippage | A few hundredths of a percent for S&P 500 stocks; model 2-5 basis points per side | Slippage model |
| Fractional shares | Most S&P 500 stocks, minimum order about $1; check each stock's `fractionable` flag | Position sizing: round to 0.0001 shares, not down to whole shares |
| Settlement | T+1, same as India | Portfolio cash logic already handles it |
| Price limits | No daily price bands like NSE's 5/10/20%. Instead there are 5-minute trading pauses (limit up/limit down) after sharp moves | Broker fill model: remove the circuit-lock rule; stops can still gap through |
| Day-trading rule | The $25,000 pattern-day-trader rule was abolished from 2026-06-04 and replaced by intraday margin rules; brokers may phase in until Oct 2027 | Not a constraint for a multi-day swing system; note it in the risk engine |
| Cash-account rules | Selling stock bought with unsettled money before settlement is a "good faith violation" | Risk engine: only spend settled cash, as the Indian design already does |
| Tick size | $0.01 for prices of $1 or more | Order rounding |
| Hours | 09:30-16:00 New York time; early 13:00 closes on some holiday eves | Calendar and scheduler |

## 6. Universe, benchmarks and calendar

- **Universe:** S&P 500 constituents on each date, plus a liquidity floor. The price cap disappears because of fractional shares. The S&P 1500 is a later option for more candidates.
- **Regime index:** SPY or the S&P 500 index in place of Nifty 50, using the same 200-day trend and volatility rules. VIX can be added as a volatility signal.
- **Benchmarks:** SPY buy-and-hold with dividends, RSP (equal-weight S&P 500), cash at the 3-month T-bill rate from FRED, and the random-entry baseline.
- **Calendar:** observed SPY sessions for history; Alpaca's calendar or `exchange_calendars` for the future; half days marked.

## 7. Strategy and risk adjustments

- **Same four strategies to start:** momentum, trend, breakout and mean reversion, under the same research protocol: tune on early years, validate once, test once.
- **Mean reversion becomes viable.** It was expected to fail in India because each round trip cost about 1.5%. At about 0.04% in the US, short holding periods of 2-10 days can be tested properly.
- **Position sizing:** risk-based sizing stays, with fractional quantities. A minimum order of $5-10 avoids dust positions. The 2% maximum-cost rejection rule will almost never fire.
- **Risk rules unchanged:** same limits on positions, exposure, sector and drawdown. Sectors come from GICS classifications in Norgate or Alpaca data.
- **Long-only stays.** US shorting needs a margin account and borrowing stock; out of scope for the prototype.
- **Currency:** the book runs in dollars. The dashboard shows results in USD and in INR at that day's rate, so currency gains are not mistaken for skill.

## 8. News and LLM layer

- **Sources:** Alpaca's Benzinga news feed (REST history from 2015, live over WebSocket) and SEC EDGAR filings, especially 8-K event filings. EDGAR's fair-access policy allows up to 10 requests per second with a declared User-Agent.
- **Big difference from India:** historical news exists, so the LLM's assessments can be scored on past data before they are allowed to act.
- **Cost guardrail:** assessing about 20 candidate stocks per day over 10 years is roughly 50,000 calls, about $500 at $0.01 each, or about $250 with the half-price Batches API. Start with one or two test years.

## 9. Scheduling and time zones

- US close is 01:30 IST in US summer time and 02:30 IST in winter. The daily job runs at about **06:30 IST**, safely after the close and after vendor data settles.
- New orders execute at the next US open, 19:00 or 20:00 IST, through the paper broker. No human needs to be awake.
- Store all session dates in New York time and convert only for display. US daylight-saving changes, in March and November, shift the IST times by an hour; the scheduler must use New York time, not fixed IST times.

## 10. Real money later: constraints for an Indian resident

Paper trading avoids all of this. It matters only for main-plan Stage 3 and beyond. These are research notes, not tax advice; confirm with a chartered accountant before moving real money.

- **LRS:** up to $250,000 per financial year may be sent abroad. ₹10,000 a month is far below this.
- **TCS:** 20% tax collected at source on remittances above ₹10 lakh per financial year. ₹1.2 lakh a year stays under the threshold, and TCS is adjustable against income tax anyway.
- **Remittance cost is the real problem at this size.** Bank wire fees plus currency markup can take several percent of a ₹10,000 transfer. Send quarterly, or use a low-cost route, or the cost advantage over India disappears.
- **Taxes:** US dividends lose 25% to US withholding after filing form W-8BEN; India taxes them again at slab rates with a foreign tax credit (Form 67). Gains are taxed only in India: slab rate if held 24 months or less, 12.5% beyond. Foreign holdings must be reported in Schedule FA.
- **Broker eligibility:** sources conflict on whether Alpaca opens live accounts for Indian residents; verify directly. Interactive Brokers is the usual API-capable alternative. NSE IFSC receipts on GIFT City are another LRS route worth comparing.

## 11. Implementation phases (if activated)

Assumes the Indian trading core (main plan Phases 2-5) already exists.

| Phase | Work | Effort | Done when |
|---|---|---|---|
| U0 | Decision; Alpaca paper account and keys in `.env`; buy Norgate if approved | 0.5 day | Keys pass an auth check; Norgate plugin returns S&P 500 membership |
| U1 | Market profile refactor, if not already done (section 12) | 1-2 days | Indian backtests reproduce identical results through the profile |
| U2 | US data layer: provider, asset master with permanent IDs, point-in-time S&P 500 membership, backfill, calendar, corporate-action verification, refresh | 4-5 days | All S&P 500 members since 2015, delisted included, stored; quality errors zero; panel loads under 5 seconds; refresh is idempotent; vendor adjustments cross-checked against a second source |
| U3 | Trading-core adaptation: US cost model, fractional sizing, no price bands, SPY regime and benchmarks | 1-2 days | Cost unit tests match hand calculations; sizing tests cover fractional shares |
| U4 | Backtests: frozen Indian parameters first, then re-tuning under the research protocol; side-by-side comparison with India | 2-3 days | Report with costs, drawdowns and baselines for both markets |
| U5 | News layer on US sources; historical scoring of LLM assessments on one or two years | 2 days | Gating report shows whether assessments predict anything |
| U6 | Scheduling after the US close (New York time aware); USD and INR in the dashboard | 1 day | Scheduled run completes; dashboard shows both currencies |
| U7 | Live paper trading, optionally mirrored on Alpaca's paper account | ongoing | Daily decisions recorded for at least 3 months |

**Total: about 12-16 working days.**

## 12. Low-cost preparation while the Indian core is built (optional)

One choice in the next phase decides whether a US switch takes two weeks or three: keeping India-specific rules out of the trading core's code. Instead of hard-coding them, the core would read a small **market profile**:

- currency, time zone and session hours,
- calendar source and regime index (Nifty 50 / SPY),
- cost model (Zerodha delivery / US),
- share rules (whole shares / fractional, minimum order),
- price-band rule (NSE bands / none) and tick size,
- benchmarks and risk-free rate.

This costs perhaps half a day during Phase 2. It is only a design choice, with no US code. It needs your approval because it slightly changes how the next phase is built.

## 13. Risks

- **The edge may not transfer.** US large caps are more efficient; lower costs do not guarantee profit. Mitigation: the same honest research protocol and random-entry baseline.
- **Data cost creep.** Norgate plus any paid feed is a recurring cost. Mitigation: 6-month subscription during research only; Alpaca's free tier for daily running.
- **Ticker reuse and symbol changes** silently corrupt history if bars are stored by ticker. Mitigation: permanent IDs from day one.
- **Currency.** A rupee move of more than 10% in a year (₹86 to ₹96 per dollar from mid-2025 to mid-2026) can swamp strategy returns measured in INR. Mitigation: judge the strategy in USD; report INR separately.
- **Real-money friction** (section 10) could make the US unattractive for live trading even if paper results are good.
- **Vendor eligibility.** Alpaca live accounts for Indian residents are unconfirmed. Mitigation: paper trading doesn't depend on it; Interactive Brokers is the fallback.

## 14. Verify before activating

1. Alpaca: live-account eligibility for Indian residents; depth of free daily history; whether delisted symbols are served; current rate limits; calendar and corporate-action endpoints.
2. Norgate: current Platinum price and trial; Python plugin works on this machine.
3. FINRA trading activity fee status for the period of trading, and the SEC fee rate for the fiscal year.
4. Broker implementation of the new intraday margin rules (replacing pattern-day-trader limits).
5. Dividend withholding rate under the India-US tax treaty (sources disagree, 25% vs 15%), TCS threshold, and remittance costs at the chosen bank or provider.
6. Current dollar-rupee rate for converting the ₹10,000 starting capital.

## Sources

- Alpaca market data plans: https://docs.alpaca.markets/us/docs/about-market-data-api, https://alpaca.markets/data
- Alpaca historical news: https://docs.alpaca.markets/us/docs/historical-news-data
- Alpaca non-US accounts: https://alpaca.markets/learn/live-trading-account-non-us, https://alpaca.markets/support/tag/non-us
- Norgate packages: https://norgatedata.com/stockmarketpackages.php, https://norgatedata.com/data-package-faq.php
- Massive (Polygon) pricing summary: https://qveris.ai/guides/polygon-pricing-optimized
- Tiingo coverage notes: https://www.quantstart.com/articles/evaluating-data-coverage-with-tiingo/
- S&P 500 historical membership snapshots: https://github.com/hanshof/sp500_constituents
- Pattern-day-trader rule change: https://www.wilmerhale.com/en/insights/client-alerts/20260423-sec-approves-amendments-to-finra-rule-4210-replacing-day-trading-margin-requirements-with-a-modernized-intraday-margin-standard
- SEC Section 31 fee FY2026: https://www.finra.org/rules-guidance/notices/information-notice-20260317
- FINRA trading activity fee: https://dollarsahead.com/finra-trading-activity-fee/
- Tax on US stocks for Indian investors: https://www.motilaloswal.com/learning-centre/2026/8/tax-on-us-stocks-for-indian-investors-a-complete-guide-to-capital-gains-tcs-dtaa-and-ftc
- Investing in US stocks from India: https://vested.blog/posts/how-to-invest-in-us-stocks-from-india-2026
- Dollar-rupee rate: https://pluang.com/en/tools/currency-converter/usd-inr
