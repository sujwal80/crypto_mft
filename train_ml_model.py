import os
import sys
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_Trainer")

from core.schemas import InternalTick
from perception.feature_store import FeatureStore
from backtester.run_backtest import generate_synthetic_market_data

lgb = None
try:
    import lightgbm as lgb
except (ImportError, OSError) as e:
    logger.warning(f"LightGBM library failed to import: {e}. NumPy Ridge fallback training will be used.")

def build_tabular_dataset(ticks, window_size: int = 1000, forward_ticks_label: int = 10):
    """
    Iterates ticks, processes them through FeatureStore, and generates features and forward return labels.
    """
    feature_store = FeatureStore(window_size=window_size)
    
    X_list = []
    prices = []
    tick_indices = []

    logger.info("Step 1: Extracting microstructural features from tick series...")
    for idx, tick in enumerate(ticks):
        features = feature_store.process_tick(tick)
        if features is not None:
            X_list.append(features)
            # Store current mid-market price for labelling
            mid_price = (tick.bid + tick.ask) / 2.0
            prices.append(mid_price)
            tick_indices.append(idx)

    if len(X_list) < 100:
        logger.error("Insufficient ticks to generate features.")
        return None, None

    X = np.array(X_list)
    y_list = []
    valid_indices = []

    logger.info(f"Step 2: Labelling dataset using {forward_ticks_label}-tick forward returns...")
    # We need future prices to compute labels, so we only label samples that have future ticks
    num_samples = len(prices)
    for i in range(num_samples):
        # Find price in the future corresponding to tick index + forward_ticks_label
        current_tick_idx = tick_indices[i]
        target_tick_idx = current_tick_idx + forward_ticks_label
        
        # Check if we have that future tick in our dataset
        if target_tick_idx < len(ticks):
            # Calculate log return over forward window
            future_tick = ticks[target_tick_idx]
            future_mid = (future_tick.bid + future_tick.ask) / 2.0
            current_mid = prices[i]
            
            log_return = np.log(future_mid / current_mid)
            y_list.append(log_return)
            valid_indices.append(i)

    X_final = X[valid_indices]
    y_final = np.array(y_list)

    logger.info(f"Dataset compiled successfully. Features shape: {X_final.shape} | Labels shape: {y_final.shape}")
    return X_final, y_final

def main():
    logger.info("=======================================================================")
    # 1. Generate large synthetic dataset for model training (25,000 ticks)
    NUM_TICKS = 25000
    ticks = generate_synthetic_market_data(num_ticks=NUM_TICKS)
    
    # 2. Extract L2 Features and generate forward return labels
    X, y = build_tabular_dataset(ticks, window_size=1000, forward_ticks_label=10)
    if X is None or y is None:
        logger.error("Failed to build training dataset. Halting pipeline.")
        sys.exit(1)

    # 3. Train/Test Split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 4. Train NumPy Ridge Regression Fallback Model
    logger.info("Step 2b: Initiating NumPy Ridge Regression fallback training...")
    X_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    lambda_val = 1.0
    I = np.eye(X_bias.shape[1])
    I[-1, -1] = 0.0
    numpy_weights = np.linalg.solve(X_bias.T.dot(X_bias) + lambda_val * I, X_bias.T.dot(y_train))
    OUTPUT_NP_WEIGHTS = "weights.npy"
    np.save(OUTPUT_NP_WEIGHTS, numpy_weights)
    logger.info(f"NumPy Ridge weights saved to: {os.path.abspath(OUTPUT_NP_WEIGHTS)}")

    # 5. Train LightGBM Booster Model if available
    if lgb is not None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'verbose': -1
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        logger.info("Step 3: Initiating LightGBM booster training...")
        try:
            gbm = lgb.train(
                params,
                train_data,
                num_boost_round=150,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(period=25)]
            )
            OUTPUT_WEIGHTS = "weights.lgb"
            gbm.save_model(OUTPUT_WEIGHTS)
            logger.info(f"👑 LIGHTGBM TRAINING COMPLETE. Weights saved to: {os.path.abspath(OUTPUT_WEIGHTS)}")
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}. Proceeding with NumPy-only weights.")
    else:
        logger.info("⚠️ LightGBM not available. Skipping LightGBM training rounds.")

    logger.info("=======================================================================")
    logger.info("👑 MODEL TRAINING COMPLETE.")
    logger.info("=======================================================================")

if __name__ == "__main__":
    main()
