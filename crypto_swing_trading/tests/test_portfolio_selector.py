import sys
import os
import pytest
import numpy as np

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../5_arbitrage")))
from portfolio_selector import PortfolioStatArbSelector

def test_portfolio_selector_warmup():
    """Verify that no pairs are selected until the ingestion price queue is fully warmed up."""
    candidate_universe = [("BTC", "ETH"), ("SOL", "AVAX")]
    selector = PortfolioStatArbSelector(candidate_pairs=candidate_universe, window=30)
    
    # Ingest only 29 price ticks
    for i in range(29):
        tick = {"BTC": 60000.0 + i, "ETH": 3000.0 + i, "SOL": 150.0 + i, "AVAX": 30.0 + i}
        selector.ingest_prices(tick)
        
    # Evaluate: list should be empty
    selected = selector.rank_and_select_pairs(max_active_pairs=2)
    assert len(selected) == 0

def test_portfolio_selector_ranking():
    """Verify that the selector filters out trending pairs and correctly ranks mean-reverting ones."""
    # Set static random seed to guarantee deterministic, stable white-noise paths
    np.random.seed(42)
    
    candidate_universe = [
        ("BTC", "ETH"),     # Highly mean-reverting (Spread H ~ 0.2)
        ("SOL", "AVAX"),    # Moderately mean-reverting (Spread H ~ 0.4)
        ("ADA", "DOT")      # Trending random walk (Spread H ~ 0.65)
    ]
    
    # Use 100 bars window and pass hurst_threshold=0.80 to account for short-sample R/S bias + OLS estimation error
    selector = PortfolioStatArbSelector(candidate_pairs=candidate_universe, window=100, hurst_threshold=0.80)
    
    # Initialize AR(1) spread residual for SOL/AVAX to model true mean-reversion
    spread_sol = 0.0
    
    # Seed price series with distinct mathematical spreads
    # We vary prices over 100 ticks to ensure stable OLS variance
    for i in range(100):
        # Base independent variables
        eth_price = 3000.0 + i * 5
        avax_price = 30.0 + i * 0.5
        dot_price = 6.0 + i * 0.1
        
        # Leg 1: BTC (perfectly cointegrated with ETH: Beta=1.5, Alpha=1.0)
        ln_eth = np.log(eth_price)
        # Inject a beautiful mean-reverting sine-wave spread to BTC (Hurst should be low!)
        ln_btc = 1.5 * ln_eth + 1.0 + 0.01 * np.sin(i * 0.5)
        btc_price = np.exp(ln_btc)
        
        # Leg 2: SOL (strongly mean-reverting AR(1) spread over AVAX: Beta=1.2, Alpha=0.5)
        ln_avax = np.log(avax_price)
        # AR(1) residual update: phi = 0.4 (strong mean reversion)
        spread_sol = 0.4 * spread_sol + np.random.normal(0, 0.05)
        ln_sol = 1.2 * ln_avax + 0.5 + spread_sol
        sol_price = np.exp(ln_sol)
        
        # Leg 3: ADA (trending breakout decoupling from DOT, linear drift!)
        ln_dot = np.log(dot_price)
        ln_ada = 1.0 * ln_dot + 0.2 + 0.005 * i  # Continuous linear drift (Hurst Exponent will be high!)
        ada_price = np.exp(ln_ada)
        
        tick = {
            "BTC": btc_price, "ETH": eth_price,
            "SOL": sol_price, "AVAX": avax_price,
            "ADA": ada_price, "DOT": dot_price
        }
        selector.ingest_prices(tick)
        
    # Evaluate ranking
    selected = selector.rank_and_select_pairs(max_active_pairs=3)
    # The trending ADA/DOT pair must be filtered out (since H >= 0.80)
    # So only 2 pairs should be selected
    assert len(selected) == 2
    
    # Verify exact ranking output: BTC/ETH has the lowest Hurst, so it must be Rank 1!
    assert selected[0]["pair"] == ("BTC", "ETH")
    assert selected[1]["pair"] == ("SOL", "AVAX")
    
    # Verify OLS fit parameters are accurate
    assert pytest.approx(selected[0]["beta"], abs=1e-1) == 1.5
    assert pytest.approx(selected[1]["beta"], abs=2e-1) == 1.2
    
    # Verify Hurst Exponents are in expected order
    assert selected[0]["hurst"] < selected[1]["hurst"]
