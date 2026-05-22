import sys
import os
import pytest
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../3_alpha_micro")))
from welford_stats import WelfordRollingStats

def test_welford_stats_correctness():
    """Verify Welford's O(1) stats match raw numpy results on arbitrary data streams."""
    welford = WelfordRollingStats(window_size=5)
    
    stream = [2.5, 4.0, 1.5, 6.0, 8.0, 3.5, 9.0, 11.0]
    
    # Feed first 5 items to fill the window
    for x in stream[:5]:
        mean, std, z = welford.update(x)
        
    # Window at this point holds: [2.5, 4.0, 1.5, 6.0, 8.0]
    expected_mean = np.mean(stream[:5])
    expected_std = np.std(stream[:5], ddof=1)
    
    np.testing.assert_allclose(welford.mean, expected_mean, rtol=1e-7)
    np.testing.assert_allclose(std, expected_std, rtol=1e-7)
    
    # Feed next item, shifting the window to [4.0, 1.5, 6.0, 8.0, 3.5]
    mean, std, z = welford.update(stream[5])
    
    active_window = stream[1:6]
    expected_mean = np.mean(active_window)
    expected_std = np.std(active_window, ddof=1)
    expected_z = (stream[5] - expected_mean) / expected_std
    
    np.testing.assert_allclose(welford.mean, expected_mean, rtol=1e-7)
    np.testing.assert_allclose(std, expected_std, rtol=1e-7)
    np.testing.assert_allclose(z, expected_z, rtol=1e-7)
