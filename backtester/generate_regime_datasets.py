import os
import sys
import time
import json
import numpy as np
from typing import List

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_path)

from core.schemas import InternalTick

# Master Regime Configuration Parameters (Calibrated to real-world Binance L2 LOB baseline)
REGIME_CONFIGS = {
    "downtrend": {
        "drift": -0.008,           # Moderate downward drift per tick (bps)
        "volatility": 0.65,        # Price step standard deviation (matches 0.084 bps baseline)
        "momentum": 0.155,         # lag-1 return autocorrelation (+0.1559)
        "spread_range": (0.01, 0.02),
        "obi_skew": -0.15          # Skewed towards sell pressure
    },
    "extremely_downtrend": {
        "drift": -0.06,            # Severe crash downward breakout per tick (bps)
        "volatility": 2.5,         # Elevated panic volatility (3x higher)
        "momentum": 0.26,          # Strong downward momentum correlation
        "spread_range": (0.03, 0.08), # Widened spreads due to panic illiquidity
        "obi_skew": -0.40          # Extreme seller sweep pressure
    },
    "sideways": {
        "drift": 0.0,              # Zero systemic price drift (ranging channel)
        "volatility": 0.60,
        "momentum": 0.120,         # Moderate return autocorrelation
        "spread_range": (0.01, 0.015),
        "obi_skew": 0.0            # Balanced Order Book Imbalance
    },
    "sideways_downtrend": {
        "drift": -0.003,           # Slow negative channel drift
        "volatility": 0.62,
        "momentum": 0.140,
        "spread_range": (0.01, 0.018),
        "obi_skew": -0.08
    },
    "sideways_uptrend": {
        "drift": 0.003,            # Slow positive channel drift
        "volatility": 0.62,
        "momentum": 0.140,
        "spread_range": (0.01, 0.018),
        "obi_skew": 0.08
    },
    "extremely_uptrend": {
        "drift": 0.06,             # Severe upward squeeze breakout per tick
        "volatility": 2.2,         # Elevated breakout squeeze volatility
        "momentum": 0.24,          # Strong positive breakout momentum correlation
        "spread_range": (0.02, 0.06),
        "obi_skew": 0.40           # Extreme buyer sweep pressure
    }
}

def generate_regime_ticks(regime_name: str, num_ticks: int = 300000) -> List[InternalTick]:
    """
    Generates 5 hours (approx 300K ticks) of statistically calibrated 
    Level 2 order book depth ticks for a given macro market regime.
    """
    cfg = REGIME_CONFIGS[regime_name]
    
    ticks = []
    base_price = 77500.0
    np.random.seed(42)
    
    prev_step = 0.0
    momentum = cfg["momentum"]
    drift = cfg["drift"]
    vol = cfg["volatility"]
    skew = cfg["obi_skew"]
    spread_min, spread_max = cfg["spread_range"]

    print(f"Synthesizing {num_ticks} ticks for regime: [{regime_name.upper()}]...")
    start_time = time.time()

    for i in range(num_ticks):
        # Step price using AR(1) model for momentum
        noise = np.random.normal(loc=drift, scale=vol)
        price_step = (momentum * prev_step) + ((1.0 - momentum) * noise)
        base_price += price_step
        prev_step = price_step

        # Spread simulation
        spread = np.random.uniform(spread_min, spread_max)
        bid = base_price - (spread / 2.0)
        ask = base_price + (spread / 2.0)

        # Order flow imbalance depth simulation
        # base balance + skew factor
        b_base = np.random.uniform(0.5, 2.5)
        a_base = np.random.uniform(0.5, 2.5)
        
        if skew > 0:
            bid_size = b_base * (1.0 + skew * 2)
            ask_size = a_base * (1.0 - skew * 0.5)
        elif skew < 0:
            bid_size = b_base * (1.0 - abs(skew) * 0.5)
            ask_size = a_base * (1.0 + abs(skew) * 2)
        else:
            bid_size = b_base
            ask_size = a_base

        tick = InternalTick(
            symbol="BTCUSDT",
            exchange="BINANCE",
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp_ns=int((time.time() - (num_ticks - i) * 0.1) * 1e9) # 100ms spacing
        )
        ticks.append(tick)

    print(f"Completed [{regime_name.upper()}] synthesis in {time.time() - start_time:.2f}s.")
    return ticks

def main():
    print("=================================================================================")
    print("🔬 MASTER HIGH-FIDELITY REGIME DATA SYNTHESIS ENGINE")
    print("=================================================================================")

    # Create directory for regime datasets
    regimes_dir = os.path.join(workspace_path, "datasets")
    os.makedirs(regimes_dir, exist_ok=True)
    print(f"Target output folder: {regimes_dir}\n")

    # 5 Hours is approx 180,000 ticks at 100ms spacing (we will generate 180K to keep file sizes at a highly optimized ~25MB each)
    NUM_TICKS = 180000 

    for regime in REGIME_CONFIGS.keys():
        ticks = generate_regime_ticks(regime, num_ticks=NUM_TICKS)
        
        output_file = os.path.join(regimes_dir, f"synthetic_market_data_{regime}.log")
        print(f"Writing ticks to JSONL log: {output_file} ...")
        
        start_write = time.time()
        with open(output_file, "w") as f:
            for tick in ticks:
                f.write(tick.model_dump_json() + "\n")
        print(f"Written successfully in {time.time() - start_write:.2f}s. File size: {os.path.getsize(output_file)/(1024*1024):.2f} MB\n")

    print("=================================================================================")
    print("👑 SYNTHESIS SCRIPT COMPLETED SUCCESSFULLY. 6 CALIBRATED REGIMES STORED.")
    print("=================================================================================")

if __name__ == "__main__":
    main()
