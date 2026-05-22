import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../3_alpha_micro")))
from kalman_filter import MicroPriceKalmanFilter

def test_kalman_filter_smoothing():
    """Verify Kalman filter correctly smooths out white noise on a constant price path."""
    kf = MicroPriceKalmanFilter(process_noise=1e-5, measurement_noise=1e-2)
    
    true_price = 60000.0
    np.random.seed(42)
    noise = np.random.normal(0.0, 0.1, 100) # Noise standard deviation of 0.1
    
    noisy_prices = true_price + noise
    filtered_prices = []
    
    for p in noisy_prices:
        filtered = kf.filter_tick(p)
        filtered_prices.append(filtered)
        
    # The variance of the filtered prices should be significantly lower than the noisy prices
    variance_noisy = np.var(noisy_prices)
    variance_filtered = np.var(filtered_prices[10:]) # Skip initial warmup steps
    
    assert variance_filtered < variance_noisy * 0.5
    
    # The final estimate should be very close to the true price
    np.testing.assert_allclose(filtered_prices[-1], true_price, atol=0.05)
