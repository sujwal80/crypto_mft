import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class WelfordRollingStats:
    """
    Microsecond-speed, numerically stable, O(1) rolling window mean & variance computer.
    Uses Welford's algorithm with precise add/remove mechanics to completely eliminate 
    memory leaks and O(N) CPU scaling issues associated with raw array windowing.
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque()
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0 # Sum of squares of differences from the current mean

    def update(self, x: float) -> tuple:
        """
        Appends a new value to the rolling window and pops the oldest value if window is full.
        Calculates rolling stats in O(1) operations.
        
        Returns:
            tuple: (rolling_mean, rolling_std_dev, z_score)
        """
        x = float(x)
        
        if self.n < self.window_size:
            # Window is not full yet, just add the new item
            self.n += 1
            self.values.append(x)
            
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)
        else:
            # Window is full, pop the oldest item and add the new item
            old_x = self.values.popleft()
            self.values.append(x)
            
            old_mean = self.mean
            # Update mean
            self.mean += (x - old_x) / self.n
            # Update sum of squares exactly and stably
            self.m2 += (x - old_x) * ((x - self.mean) + (old_x - old_mean))
            
        # Calculate variance and standard deviation
        # Standard sample variance (unbiased)
        variance = self.m2 / (self.n - 1) if self.n > 1 else 0.0
        # Numerical safety cap to prevent square root of negative due to floating point precision
        variance = max(0.0, variance)
        std_dev = np.sqrt(variance)
        
        # Compute Z-score
        z_score = (x - self.mean) / (std_dev + 1e-8) if std_dev > 1e-8 else 0.0
        
        return self.mean, std_dev, z_score
