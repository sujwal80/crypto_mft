# Indian Intraday System

A highly optimized, asynchronous, event-driven quantitative model trading NIFTY futures on the National Stock Exchange (NSE) of India, driven by Options Gamma (GEX) macro-boundaries and executed via high-frequency micro-structural order flow indicators.

---

## Core Architectural Layers

```
indian_intraday_system/
├── README.md                  # Operational Manual
├── config.py                  # API credentials, parameters, and NSE tick formatting
├── main.py                    # Async orchestrator linking live feeds to routers
├── .env                       # Sterile environment profile toggle file
├── Dockerfile                 # Lightweight reproducible container builder
├── docker-compose.yml         # Local data lake volume mount manager
├── deploy.sh                  # Automated 1-click deployment trigger
├── layer_1_data/
│   ├── truedata_ws.py         # TrueData WebSocket client with Dynamic ATM Option Window
│   └── historical_replay.py   # Segregated historical daily Bhavcopy tick replay player
├── layer_2_macro/
│   ├── vanilla_bs.py          # Vectorized Greeks & Newton-Raphson bisection solvers
│   └── gex_mapper.py          # zero-gamma flip mapper & Put/Call walls locator
├── layer_3_micro/
│   ├── cvd_engine.py          # Cumulative Volume Delta (taker buying aggression)
│   └── basis_tracker.py       # Real-time Spot-to-Future premium tracking
├── layer_4_execution/
│   ├── base_router.py         # Abstract execution interface
│   ├── shadow_router.py       # Paper trading broker with exact Indian friction fees (₹90)
│   ├── dhan_router.py         # Production router with pre-trade margins & terminal Stop-Losses
│   ├── time_manager.py        # Clock manager tracking IST regimes (Lock, MR, Momentum)
│   └── state_machine.py       # Event-driven async Swing State Machine (States 0 to 3)
├── backtest/
│   ├── bhavcopy_loader.py     # Daily EOD derivatives Bhavcopy ingestion engine
│   ├── replay_engine.py       # 1-minute simulated tick player utilizing Brownian Bridges
│   ├── backtest_runner.py     # Historical backtest simulator routing through ShadowRouter
│   └── run_friday_night_test.py # Math verification and breakout simulation script
└── utils/
    └── data_recorder.py       # Asynchronous non-blocking live tick data recorder
```

---

## Key Upgraded Features

### 1. Dynamic Option Strike Window (`layer_1_data/truedata_ws.py`)
*   **The Problem**: Subscribing to all 300+ weekly option contracts on Thursday expeires saturates internet bandwidth and slows down CPU deserialization, leading to price lags.
*   **The Solution**: TrueData WebSocket automatically tracks the cash spot price and dynamically maintains a subscription window of **ATM +/- 5 strikes** (11 active strikes, 22 contracts). As the spot price moves, it automatically sends `unsubscribe` requests for far out-of-the-money strikes and `subscribe` requests for new near-the-money strikes, dropping network overhead by over **90%**!

### 2. Asynchronous Data Recorder (`utils/data_recorder.py`)
*   **No Local Data solution**: If you do not have any local historical data saved on your AWS server, the bot **automatically populates your local data lake for free in the background** while you are paper trading in real-time!
*   **Non-Blocking I/O**: Incoming ticks are buffered in memory and written to daily CSV files using a separate Python thread pool, ensuring **zero execution latency impact** on the primary trading thread.

### 3. Statistical Basis Tracker (`layer_3_micro/basis_tracker.py`)
*   Tracks the spread: $\text{Basis} = \text{Future Price} - \text{Spot Price}$.
*   Computes rolling means and standard deviations of basis spreads to output a **Basis Z-score**.
*   **0DTE Expiry Decay**: On Thursdays, the tracker automatically contracts its standard deviation thresholds toward 0 as 3:30 PM approaches, preventing false anomaly blocks as Futures and Spot converge.

### 4. NSE Tick Size Precision (`config.py`)
*   The NSE requires all orders (Futures & Options) in price increments of **0.05 paise**. Price values with decimal fractions are instantly rejected by the Dhan API.
*   `config.py` provides a fast rounding utility utilized globally before order submission:
    ```python
    def round_to_nse_tick(price: float) -> float:
        return round(price * 20) / 20
    ```

### 5. Time-Based Swing State Machine (`layer_4_execution/state_machine.py`)
The machine locks execution states dynamically based on exact Indian Standard Time (IST) regimes:
*   **State 0: LOCKED (9:15 - 9:45 AM)**: No execution. Processes EOD Bhavcopy option chain walls and calculates Zero-Gamma levels.
*   **State 1: MEAN-REVERSION (9:45 AM - 1:30 PM)**: Hunts option walls rejections. If spot price hits the Put Wall or Call Wall AND CVD order-flow shows rejections, places limit reversion orders targeting Zero-Gamma convergence.
*   **State 2: MOMENTUM / 0DTE BREAKOUT (1:30 PM - 3:15 PM)**: Momentum squeeze. If spot price breaches a major wall AND CVD taker buyer/seller aggression surges AND the basis tracker Z-score confirms institutional futures buying, rides the dealer short-covering breakout.
*   **State 3: KILL (3:15 PM)**: Active liquidation. Squaring off all open positions at market, canceling pending orders, and halting.

---

## Verification & Testing

Verify that the entire system compiles, resolves package imports correctly, and passes mathematical and integration checks:

```bash
# Run the comprehensive test suite
PYTHONPATH=. ./venv/bin/pytest -v indian_intraday_system/tests/
```

### Expected Output:
```
indian_intraday_system/tests/test_system.py::test_nse_tick_formatting PASSED
indian_intraday_system/tests/test_system.py::test_vanilla_bs_with_bisection_patch PASSED
indian_intraday_system/tests/test_system.py::test_basis_tracker PASSED
indian_intraday_system/tests/test_system.py::test_dynamic_option_window PASSED
indian_intraday_system/tests/test_system.py::test_indian_intraday_system_momentum_breakout PASSED
indian_intraday_system/tests/test_system.py::test_live_data_recorder PASSED
```

---

## Offline Bhavcopy Backtesting (No API Keys Required)

The system includes the fully integrated, historical daily derivatives Bhavcopy loader and 1-minute intraday simulated tick player (powered by Brownian Bridges and dynamic Black-Scholes Implied Volatility solvers):

```bash
# Run EOD Friday Night Math breakout backtest
PYTHONPATH=. ./venv/bin/python indian_intraday_system/backtest/run_friday_night_test.py
```

---

## Live-Simulation Replay Mode (Intraday Historical Sandbox)

You can run the **actual live automated trading orchestrator (`main.py`)** against **any historical market date** (like last Friday) to test your entire pipeline as if you were trading live on that day!

The system will automatically set a virtual clock, arm at virtual 9:45 AM, evaluate GEX walls and CVD order flow, execute paper trades, and exit cleanly at virtual 3:15 PM at high-speed (375 minutes executed in 3.7 seconds):

1. Configure your `.env` file:
```bash
SYSTEM_ENVIRONMENT="SIMULATION"
HISTORICAL_REPLAY_DATE="2026-05-22"  # Target date
REPLAY_SPEED_FACTOR="0.01"           # Speed multiplier
```
2. Launch the main bot:
```bash
PYTHONPATH=. python3 indian_intraday_system/main.py
```

---

## Run Live / Paper Strategy (Local Terminal)

```bash
PYTHONPATH=. python3 indian_intraday_system/main.py
```

---

## Docker Deployment Strategies (AWS Mumbai)

For unattended production or forward paper trading on AWS EC2 Mumbai (`ap-south-1`), we run the system inside an isolated Docker container:
1.  **Environment Isolation**: The container matches the Python 3.9-slim image with system-level GCC compilers, ensuring packages like NumPy/SciPy execute identically on AWS as they do on your local laptop.
2.  **Host Mount Data Lake**: Mounts `./datasets` as a host volume, saving raw trade ticks directly to your EC2 SSD so that they survive container updates and restarts.
3.  **Auto-Restarts**: Includes `restart: always` in `docker-compose.yml` to re-arm connection loops if AWS suffers network Drops.

### 1. Deploy in 1-Click:
Run the automated deployment script on your AWS EC2 instance:
```bash
cd indian_intraday_system/
./deploy.sh
```

### 2. Monitor Container:
```bash
# Tail real-time dashboard stdout logs
docker logs -f nse_gex_bot

# Check running state
docker ps -a | grep nse_gex_bot
```

### 3. Emergency Stop:
```bash
docker stop nse_gex_bot
```

