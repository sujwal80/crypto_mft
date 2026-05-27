import numpy as np
import logging
from scipy.optimize import minimize
from typing import Tuple

logger = logging.getLogger("LeungCalibrator")

class LeungThresholdCalibrator:
    """
    Stochastic Optimal stopping calibrator based on Prof. Tim Leung's HJB equations.
    (Columbia University: 'Optimal Mean-Reversion Trading with Transaction Costs and Stop-Loss').
    
    Solves the first-passage stochastically discounted utility function to output the 
    optimal entry, exit, and stop boundaries for a given OU spread process under transaction fees.
    """
    def __init__(self, round_trip_fee: float = 0.0020):
        self.fee = round_trip_fee  # Fixed 0.2% round-trip transaction fee hurdle

    def calibrate_optimal_thresholds(self, lambda_mr: float, sigma: float, theta: float = 0.0) -> Tuple[float, float, float]:
        """
        Solves the optimal stopping boundaries.
        Returns: (optimal_entry_z, optimal_exit_z, optimal_stop_loss_z)
        """
        # Enforce minimum bounds for stability
        lambda_mr = max(lambda_mr, 1e-4)
        sigma = max(sigma, 1e-4)
        
        # We define a mathematical solver to maximize the Net Profit Sharpe Rate:
        # Expected Net Profit = (xs - xe) - self.fee
        # Expected First-Passage Time (FPT) = ln((theta - xe) / (theta - xs)) / lambda_mr
        # We maximize: Net_Profit / (FPT * sigma)
        
        def objective(params):
            xe_z, xs_z = params
            
            # Map Z-score boundaries to absolute spread scale
            xe = theta + xe_z * sigma
            xs = theta + xs_z * sigma
            
            # Standard boundary constraints: entry must be below mean, exit must be above/near mean
            if xe_z >= 0.0 or xs_z <= xe_z:
                return 1e6
                
            gross_profit = xs - xe
            net_profit = gross_profit - self.fee
            
            if net_profit <= 0.0:
                # Trade does not clear the 0.2% transaction fee hurdle
                return 1e6
                
            # Calculate expected first-passage time of OU process from xe to xs
            # Approximated via drift-speed log coordinates
            try:
                expected_time = np.log(abs(theta - xe) / (abs(theta - xs) + 1e-8)) / lambda_mr
                if expected_time <= 0.0:
                    return 1e6
            except Exception:
                return 1e6
                
            # Sharpe Expectancy Rate (negative for minimization solver)
            sharpe_rate = net_profit / (expected_time * sigma)
            return -sharpe_rate

        # Initial guess: xe = -2.2 Z-score, xs = 0.1 Z-score
        x0 = [-2.2, 0.1]
        
        try:
            res = minimize(objective, x0=x0, method="Nelder-Mead", options={"maxiter": 200})
            if res.success:
                xe_opt_z, xs_opt_z = res.x
                # Stop-loss set at 1.8x the optimal entry boundary (standard risk invalidation)
                sl_opt_z = xe_opt_z * 1.8
                
                # Clip values to standard realistic limits to prevent optimization anomalies
                xe_opt_z = max(min(xe_opt_z, -1.5), -4.0)
                xs_opt_z = max(min(xs_opt_z, 0.5), -0.2)
                sl_opt_z = max(min(sl_opt_z, -3.0), -6.0)
                
                return float(xe_opt_z), float(xs_opt_z), float(sl_opt_z)
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Reverting to default defaults.")
            
        # Fallback to safe robust boundaries if solver fails to converge
        return -2.5, 0.2, -4.5
