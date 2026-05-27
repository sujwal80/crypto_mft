import numpy as np
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger("HMMRegime")

class GaussianHMMClassifier:
    """
    Recursive 3-State Gaussian Hidden Markov Model (HMM) Regime Classifier.
    States: 0 = Sideways Range, 1 = Trend Breakout, 2 = Decoupling / Liquidation.
    
    Executes a high-speed recursive Forward Algorithm using multi-variable 
    Gaussian emission probability densities in pure NumPy.
    """
    def __init__(self):
        # 1. Transition Probability Matrix A (State persistence calibrated to hourly cycles)
        # Row represents state at t-1, Column represents state at t
        self.A = np.array([
            [0.92, 0.06, 0.02],  # State 0 (Range): 92% persistence
            [0.05, 0.90, 0.05],  # State 1 (Trend): 90% persistence
            [0.10, 0.05, 0.85]   # State 2 (Decouple): 85% persistence
        ])
        
        # 2. Multivariable Emission Parameters (Observation Vector: [abs(Z-score), abs(Trend_slope)])
        # Means for each state
        self.means = [
            np.array([0.70, 0.00005]),  # State 0 (Range): Low Z, Flat trend
            np.array([2.00, 0.00035]),  # State 1 (Trend): High Z, Steep trend
            np.array([4.00, 0.00120])   # State 2 (Decouple): Extreme Z, Extreme trend
        ]
        
        # Covariance matrices (Sigma) for each state (assumed diagonal for independence)
        self.covs = [
            np.array([[0.15, 0.0], [0.0, 1e-8]]),  # State 0
            np.array([[0.50, 0.0], [0.0, 1e-6]]),  # State 1
            np.array([[2.00, 0.0], [0.0, 1e-5]])   # State 2
        ]
        
        # Precompute inverse and determinants of covariances for high-speed evaluation
        self.inv_covs = [np.linalg.inv(cov) for cov in self.covs]
        self.det_covs = [np.linalg.det(cov) for cov in self.covs]
        
        # 3. Recursive Forward Variables (Alpha) initialized to uniform prior
        self.alpha = np.array([0.34, 0.33, 0.33])
        self.warmup_samples = 24
        self.samples_count = 0

    def classify_tick(self, z_score: float, spread_trend: float) -> int:
        """
        Ingests observations recursively, runs the Forward update, 
        and returns the predicted active state (0, 1, or 2).
        """
        self.samples_count += 1
        obs = np.array([abs(z_score), abs(spread_trend)])
        
        # 1. Calculate emission probabilities b_i(Obs) for each state
        emissions = np.zeros(3)
        for i in range(3):
            diff = obs - self.means[i]
            exponent = -0.5 * np.dot(diff, np.dot(self.inv_covs[i], diff))
            # Bounded exponent to prevent exp math underflow
            exponent = max(min(exponent, 100.0), -100.0)
            
            denominator = 2.0 * np.pi * np.sqrt(self.det_covs[i])
            emissions[i] = np.exp(exponent) / (denominator + 1e-8)
            
        if self.samples_count < self.warmup_samples:
            # Warm-up phase: return Range State as safe default
            return 0
            
        # 2. Forward Step Update: Alpha_t(i) = [ Sum_j Alpha_{t-1}(j) * A_ji ] * b_i(Obs)
        alpha_pred = np.dot(self.alpha, self.A)
        alpha_new = alpha_pred * emissions
        
        # 3. Normalize Alpha vector to prevent floating-point underflow
        alpha_sum = np.sum(alpha_new)
        if alpha_sum > 0.0:
            self.alpha = alpha_new / alpha_sum
        else:
            # Reset to prior if numerical instability occurs
            self.alpha = np.array([0.34, 0.33, 0.33])
            
        # 4. Return Argmax (State with highest posterior probability)
        active_state = int(np.argmax(self.alpha))
        return active_state
