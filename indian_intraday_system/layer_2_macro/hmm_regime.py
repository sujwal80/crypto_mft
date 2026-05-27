import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger("HMMRegime")

class GaussianHMMClassifier:
    """
    Recursive 3-State Gaussian Hidden Markov Model (HMM) Intraday Regime Classifier.
    States: 0 = Sideways Range, 1 = Trend Breakout, 2 = Decoupling / Liquidation.
    
    Executes a high-speed recursive Forward Algorithm using multi-variable 
    Gaussian emission probability densities in pure NumPy.
    """
    def __init__(self):
        # 1. Transition Probability Matrix A (Calibrated to 5-minute intraday cycles)
        # Row represents state at t-1, Column represents state at t
        self.A = np.array([
            [0.95, 0.04, 0.01],  # State 0 (Range): 95% persistence
            [0.03, 0.94, 0.03],  # State 1 (Trend): 94% persistence
            [0.08, 0.02, 0.90]   # State 2 (Decouple): 90% persistence
        ])
        
        # 2. Multivariable Emission Parameters (Observation Vector: [abs(Z-score), abs(Trend_slope)])
        self.means = [
            np.array([0.60, 0.00001]),  # State 0 (Range): Low Z, Flat trend
            np.array([2.20, 0.00025]),  # State 1 (Trend): High Z, Steep trend
            np.array([4.20, 0.00095])   # State 2 (Decouple): Extreme Z, Extreme trend
        ]
        
        # Covariances (Sigma) for each state
        self.covs = [
            np.array([[0.10, 0.0], [0.0, 1e-9]]),  # State 0
            np.array([[0.40, 0.0], [0.0, 1e-7]]),  # State 1
            np.array([[1.50, 0.0], [0.0, 1e-6]])   # State 2
        ]
        
        # Precompute inverses and determinants
        self.inv_covs = [np.linalg.inv(cov) for cov in self.covs]
        self.det_covs = [np.linalg.det(cov) for cov in self.covs]
        
        # Recursive Forward Variables (Alpha)
        self.alpha = np.array([0.34, 0.33, 0.33])
        self.warmup_samples = 12
        self.samples_count = 0

    def classify_tick(self, z_score: float, spread_trend: float) -> int:
        self.samples_count += 1
        obs = np.array([abs(z_score), abs(spread_trend)])
        
        # 1. Calculate emission densities
        emissions = np.zeros(3)
        for i in range(3):
            diff = obs - self.means[i]
            exponent = -0.5 * np.dot(diff, np.dot(self.inv_covs[i], diff))
            exponent = max(min(exponent, 100.0), -100.0)
            denominator = 2.0 * np.pi * np.sqrt(self.det_covs[i])
            emissions[i] = np.exp(exponent) / (denominator + 1e-9)
            
        if self.samples_count < self.warmup_samples:
            return 0
            
        # 2. Recursive Forward Update
        alpha_pred = np.dot(self.alpha, self.A)
        alpha_new = alpha_pred * emissions
        
        # 3. Normalize to prevent floating underflow
        alpha_sum = np.sum(alpha_new)
        if alpha_sum > 0.0:
            self.alpha = alpha_new / alpha_sum
        else:
            self.alpha = np.array([0.34, 0.33, 0.33])
            
        return int(np.argmax(self.alpha))
