# Mathematical Quant Trading Sandbox: Strategy Factory & Comparisons

Welcome to the Enterprise Cryptocurrency MFT Quant Trading Sandbox! This sandbox is built around the **Strategy Factory Design Pattern**, decoupling the orchestrator from individual mathematical model logic.

---

## 🏗️ System Architecture & Strategy Factory Pattern

Instead of having individual prediction parameters and hardcoded routing logic inside the central model classes, this project uses the **Strategy Pattern** combined with an **Abstract Factory**:

1.  **Standardized Contract (`BaseAlphaStrategy`)**: Enforces a single predictive interface signature:
    ```python
    def predict(self, features: np.ndarray) -> float:
    ```
2.  **Decoupled Strategies**:
    - Each individual strategy class (`OrnsteinUhlenbeckAlpha`, `KalmanFilterAlpha`, etc.) inherits from `BaseAlphaStrategy` and unpacks exactly the features it requires from the unified 6-dimensional microstructural feature vector.
3.  **Dynamic Registry (`AlphaStrategyFactory`)**:
    - Maintains an active lookup registry of all strategies.
    - Instantiates strategies dynamically on startup based on configuration strings (e.g., `OU`, `KALMAN`, `OFI`, `ML`, `HEURISTIC`).
4.  **Dynamic Orchestrator (`AlphaModel`)**:
    - Decoupled from strategy implementation details. Simply requests the factory to instantiate the active strategy, and delegates execution directly:
    ```python
    return self.active_strategy.predict(features)
    ```

---

## 🔢 Active Strategies & Mathematical Models

### 1. Ornstein-Uhlenbeck Mean-Reversion (`OU`)
*   **File**: `intelligence/ou_alpha.py`
*   **Concept**: Models price spreads as a continuous-time stochastic spring. If prices stretch too far (Z-score exceeds $\pm 1.2$ and stays under a volatility circuit-breaker limit of $2.5$), the model estimates the optimal speed and scale of the reversion snapback force.
*   **Mathematical Formula**:
    
    $$dx_t = \theta (\mu - x_t) dt + \sigma dW_t$$
    
    Where $\theta$ is the pull speed, $\mu$ is the long-term mean, and $dW_t$ is Wiener noise.

---

### 2. State-Space Kalman Filter (`KALMAN`)
*   **File**: `intelligence/kalman_alpha.py`
*   **Concept**: A recursive state-space filter that separates underlying "fair price" trends from bid-ask microstructural spreads and order book noise.
*   **Implementation**: Predicts the clean state, computes error covariance, and adjusts the estimate using a dynamically calculated Kalman Gain as new L2 tick drifts arrive. A trade signal is generated when unweighted market price diverges from the filtered fair-price estimate by $> 0.15\%$.

---

### 3. Order Flow Imbalance (`OFI`)
*   **File**: `intelligence/ofi_alpha.py`
*   **Concept**: High-frequency order book microstructure momentum tracker.
*   **Implementation**: Measures the net accumulation of order sizes added or removed at the top-of-book (Best Bid/Best Ask) to forecast immediate buying or selling pressure.

---

### 4. Tabular Machine Learning (`ML`)
*   **File**: `intelligence/ml_alpha.py`
*   **Concept**: Tabular supervised machine learning model using **LightGBM** (Gradient Boosted Decision Trees).
*   **Implementation**: Predicts the 10-tick forward log return based on L2 microstructural features. Dynamically loads its parameters from `weights.lgb` at runtime.

---

## 🚀 Step-by-Step Operational Guide

Before running, make sure you have your python environment configured and dependencies installed:
```bash
python3 -m pip install -r requirements.txt
```

### Phase 1: Compare Mathematical Models
To compare the three classical math models and the baseline statistical fallback side-by-side, run the competition harness:
```bash
python3 run_competition.py
```
This executes a tick-by-tick historical simulator over a **15,000-tick series** (containing both quiet channels and violent breakouts) and outputs a competitive league table showing **P&L ($), Net Return (%), Max Drawdown, Win Rate, and Total Fees Paid**.

---

### Phase 2: Train the Machine Learning Model (To Do Later)
Whenever you decide to train the LightGBM model, execute the modular training script:
```bash
python3 train_ml_model.py
```
**What the script does internally**:
1. Generates a large historical tick dataset (25,000 synthetic ticks).
2. Feeds ticks to the `FeatureStore` to build the tabular 6-dimensional feature array.
3. Formulates labels by calculating the **10-tick forward log return**: $\log(P_{t+10} / P_t)$.
4. Splits data (80% training, 20% test).
5. Trains the Gradient-Boosted Decision Trees with early stopping.
6. Outputs the optimized model parameters to **`weights.lgb`** in your root workspace folder.

---

### Phase 3: Run 5-Way Competitions
Once `weights.lgb` is saved in your root folder, **`run_competition.py` will automatically detect it!**

Run the competition harness again:
```bash
python3 run_competition.py
```
It will now dynamically include the **`ML`** strategy, running it side-by-side with `OU`, `KALMAN`, `OFI`, and `HEURISTIC` to give you a full 5-way league table comparison of all mathematical quantitative trading strategies!
