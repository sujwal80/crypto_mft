import sys
import os
import pytest

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))
from leung_calibrator import LeungThresholdCalibrator

def test_leung_calibrator_convergence():
    """Verify that the HJB solver converges successfully under standard parameters."""
    calibrator = LeungThresholdCalibrator(round_trip_fee=0.0020)
    
    # Inputs: Lambda = 0.1, Sigma = 0.02 (standard hourly spread metrics)
    xe_z, xs_z, sl_z = calibrator.calibrate_optimal_thresholds(lambda_mr=0.1, sigma=0.02)
    
    # Verify outputs are valid, non-nan float numbers
    assert xe_z is not None
    assert xs_z is not None
    assert sl_z is not None
    
    # Verify boundaries satisfy statistical constraints
    assert xe_z < 0.0          # Entry must be below mean (buy spread)
    assert xs_z > xe_z          # Exit must be above/near entry
    assert sl_z < xe_z          # Stop loss must be below entry
    
    # Verify boundaries are clipped to standard safety zones
    assert -4.0 <= xe_z <= -1.5
    assert -0.2 <= xs_z <= 0.5
    assert -6.0 <= sl_z <= -3.0

def test_leung_calibrator_parameter_adaptability():
    """Verify that the HJB solver adapts dynamically to high-frequency vs. low-frequency regimes."""
    calibrator = LeungThresholdCalibrator(round_trip_fee=0.0020)
    
    # State A: High-Speed Mean Reversion (Lambda = 0.5, Sigma = 0.01) -> Boundaries should tighten
    xe_high, xs_high, _ = calibrator.calibrate_optimal_thresholds(lambda_mr=0.5, sigma=0.01)
    
    # State B: Slow-Speed Mean Reversion (Lambda = 0.01, Sigma = 0.04) -> Boundaries should expand
    xe_slow, xs_slow, _ = calibrator.calibrate_optimal_thresholds(lambda_mr=0.01, sigma=0.04)
    
    # Verify that the slow-reacting, high-volatility spread forces the optimal entry to widen 
    # to safely clear the 0.2% round-trip transaction cost hurdle
    assert abs(xe_slow) >= abs(xe_high)
