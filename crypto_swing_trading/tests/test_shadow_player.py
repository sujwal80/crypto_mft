import sys
import os
import pytest
import asyncio

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run_shadow_paper_trading import HistoricTickPlayer

@pytest.mark.asyncio
async def test_shadow_player_init():
    """Verify HistoricTickPlayer initializes correctly with state structures."""
    file_path = "/Users/singhujwal/crypto_mft/datasets/synthetic_market_data_extremely_downtrend.log"
    
    player = HistoricTickPlayer(
        file_path=file_path,
        speedup_factor=1000.0, # Accelerated speed
        symbol="BTCUSDT"
    )
    
    assert player.symbol == "BTCUSDT"
    assert player.speedup == 1000.0
    assert player.state_machine.state == 0
    assert player.state_machine.in_position == False
