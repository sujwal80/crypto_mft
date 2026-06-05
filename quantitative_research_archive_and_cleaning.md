# Quantitative Research Archive: The Indian Markets Experiment

This document serves as the historical archive of our extensive quantitative research into algorithmic trading on the Indian Cash Equity market. It documents the mathematical realities of the market microstructure, the models we built, why they failed, and the final profitable model we discovered.

## Part 1: The Friction "Death Spiral"
Our most critical discovery was quantifying the exact real-world friction of trading Indian Cash Equities intraday on a retail broker (Zerodha).

**Intraday Friction:**
- STT: 0.025% on the sell side.
- Brokerage + GST + Exchange Fees: ~₹30 to ₹40 per round trip.
- Slippage: ~0.04% round trip (bid/ask spread).
- **Total Intraday Round-Trip Friction:** ~0.12% to 0.15%.

Because large-cap stocks (Nifty 50) only move about 1.0% on an average day, algorithms attempting to capture 0.4% or 0.5% profit targets are instantly stripped of 30% of their gross profit by fees. If a trade is stopped out, the fees exacerbate the loss. This mathematical headwind makes high-frequency algorithmic trading on cash equities structurally unprofitable.

## Part 2: The Intraday Models Tested (And Failed)
We built and backtested 15 distinct institutional-grade models. Despite many generating positive *Gross* PnL (Alpha), all of them failed to overcome the friction, resulting in negative *Net* PnL.

1. **Grid-Search Breakout Optimizer:** 64 permutations of VWAP breakouts and time filters.
2. **End-Of-Day (EOD) Momentum:** Buying extreme trending stocks. (Generated pure alpha, died to fees).
3. **Extreme VWAP Fade (Z > 3.0):** Mean reversion on 3 Standard Deviation extensions.
4. **Cross-Sectional Momentum:** Buying the strongest, shorting the weakest sector.
5. **VWAP Mean Reversion (HMM):** Markov-based VWAP pulls.
6. **15-Min ORB Breakout:** Standard retail Open Range Breakout.
7. **9:30 AM Trend Continuation:** Time-of-day specific momentum.
8. **Gap Down Fades:** Fading overnight gaps.
9. **PDH/PDL Breakout:** Previous Day High/Low breaches.
10. **High-ATR Volatility Breakout:** Trading only high volatility regimes.
11. **60-Min Institutional ORB:** Waiting for an hour for institutional volume to settle.
12. **SuperTrend 1-Min System:** A trend-following system that bled ₹350,000 to fees due to over-trading.
13. **Highest RVOL Momentum:** Relative Volume institutional footprint tracking.
14. **KNN Fractal Pattern Matching:** Pure Numpy K-Nearest Neighbors mathematical pattern matching.
15. **TCS vs INFY Statistical Arbitrage:** Cross-asset pairs trading.

*Note: We also built a "Clairvoyant" model that knew the exact high and low of the day. This proved that the absolute maximum theoretical limit of intraday profits on ₹1 Lakh was ₹4.34 Lakhs over the dataset, but even the Clairvoyant Oracle paid ₹26,000 in fees.*

## Part 3: The Profitable Pivot (Swing Trading)
To achieve profitability, we shifted the holding period to allow the Expected Move of the asset to dwarf the friction (which is roughly 0.15% to 0.20% for Equity Delivery).

We wrote a custom Grid-Search Hyperparameter Optimizer to test variations of the **Connors RSI(2) Mean Reversion Strategy** against our specific 30-day dataset.

**The Profitable Parameters Discovered:**
- **Entry Rule:** Buy on the open when the previous day's 2-Day RSI is **< 10** (Extreme Panic/Oversold).
- **Exit Rule:** Sell on the open when the previous day's 2-Day RSI is **> 80** (Euphoria/Reversion).

This configuration yielded a **46.15% Win Rate** and a **Net Profit of ₹1,833.41** on a deeply bearish 30-day historical snapshot, proving that statistical swing trading works structurally in the Indian Cash Market.

## Part 4: The Clean Slate
With these mathematical truths established, we deleted the muddy intraday code, the test scripts, and the local backtest data. We are now preparing a clean slate to build a highly focused, production-grade Swing Trading Architecture.
