import sys
import os
import pytest
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))
from hmm_regime import GaussianHMMClassifier

def test_hmm_range_classification():
    """Verify that the HMM correctly classifies low-volatility sideways consolidations as State 0 (Range)."""
    hmm = GaussianHMMClassifier()
    
    # Feed 30 bars of perfect ranging observations (Z = 0.5, Trend = 0.00001)
    for _ in range(30):
        state = hmm.classify_tick(z_score=0.5, spread_trend=0.00001)
        
    # Should resolve to State 0
    assert state == 0
    assert hmm.alpha[0] > hmm.alpha[1]
    assert hmm.alpha[0] > hmm.alpha[2]

def test_hmm_trend_transition():
    """Verify that the HMM transitions dynamically to State 1 (Trend) when spread momentum breaks out."""
    hmm = GaussianHMMClassifier()
    
    # Warm up in Range
    for _ in range(25):
        hmm.classify_tick(z_score=0.5, spread_trend=0.00001)
        
    # Inject a strong trending breakout sequence (Z = 2.2, Trend = 0.00045)
    for _ in range(5):
        state = hmm.classify_tick(z_score=2.2, spread_trend=0.00045)
        
    # Should transition to State 1
    assert state == 1
    assert hmm.alpha[1] > hmm.alpha[0]

def test_hmm_decoupling_transition():
    """Verify that the HMM transitions dynamically to State 2 (Decoupling) under extreme volatility shocks."""
    hmm = GaussianHMMClassifier()
    
    # Warm up in Range
    for _ in range(25):
        hmm.classify_tick(z_score=0.5, spread_trend=0.00001)
        
    # Inject an extreme decoupling liquidation shock (Z = 4.5, Trend = 0.00150)
    for _ in range(3):
        state = hmm.classify_tick(z_score=4.5, spread_trend=0.00150)
        
    # Should transition to State 2 (catastrophic decouple)
    assert state == 2
    assert hmm.alpha[2] > hmm.alpha[0]
