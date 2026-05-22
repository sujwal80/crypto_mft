class PortfolioOptimizer:
    """
    Calculates target sizing weights based on expected returns (predictions) and volatility,
    implementing a scaled Kelly Criterion mechanism.
    """
    def __init__(self, leverage_limit: float = 1.0, kelly_fraction: float = 0.5):
        self.leverage_limit = leverage_limit
        self.kelly_fraction = kelly_fraction

    def calculate_target_weight(self, alpha_forecast: float) -> float:
        """
        Computes the target portfolio weight.
        
        Args:
            alpha_forecast: Estimated expected return forecast (directional prediction percentage).
            
        Returns:
            float: Bounded target weight, positive for buy, negative for sell.
        """
        if alpha_forecast == 0.0:
            return 0.0

        # Scaled Kelly Criterion sizing
        scaling_factor = 250.0 * self.kelly_fraction
        target_weight = alpha_forecast * scaling_factor

        # Clamp between [-leverage_limit, leverage_limit]
        target_weight = max(-self.leverage_limit, min(self.leverage_limit, target_weight))
        return target_weight
