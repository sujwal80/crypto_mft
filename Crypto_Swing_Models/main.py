import asyncio
import logging
import sys
import os
import signal
import numpy as np

# Configure loggers for production standards
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gex_micro_system.log")
    ]
)

logger = logging.getLogger("Orchestrator")

# Setup import paths to find all subsystems
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "1_data_layer")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "2_alpha_macro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "3_alpha_micro")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "4_execution")))

from binance_ws import BinanceWebSocketClient
from deribit_ws import DeribitWebSocketClient
from gex_mapper import GexMapper
from state_machine import GexMicroStateMachine

class QuantSystemOrchestrator:
    """
    Master Orchestrator of the GEX-Micro State Machine.
    Spawns all high-speed async feeds, maps out options dealer hedging profiles,
    updates the Master State Machine, and manages capital risk gates dynamically.
    """
    def __init__(self, symbol: str = "BTCUSDT", mode: str = "SHADOW"):
        self.symbol = symbol
        self.mode = mode
        self.running = False
        
        # Queues for high-speed async inter-task communications
        self.binance_depth_queue = asyncio.Queue(maxsize=10000)
        self.binance_trade_queue = asyncio.Queue(maxsize=10000)
        self.deribit_trade_queue = asyncio.Queue(maxsize=10000)
        
        # Feed Clients
        self.binance_ws = BinanceWebSocketClient(
            symbol=self.symbol,
            depth_queue=self.binance_depth_queue,
            trade_queue=self.binance_trade_queue
        )
        self.deribit_ws = DeribitWebSocketClient(
            currency="BTC",
            trade_queue=self.deribit_trade_queue,
            chain_update_interval=10.0
        )
        
        # Logic Blocks
        self.mapper = GexMapper(model_type="COIN_MARGINED")
        self.state_machine = GexMicroStateMachine(symbol=self.symbol, mode=self.mode)
        
        self.tasks = []

    async def start(self):
        """Spawns all concurrent worker tasks."""
        self.running = True
        logger.info("Initializing GEX-Micro State Machine system...")
        
        # 1. Start Feeds
        await self.binance_ws.start()
        await self.deribit_ws.start()
        
        # 2. Spawn Worker Tasks
        self.tasks = [
            asyncio.create_task(self._process_binance_depth_feed()),
            asyncio.create_task(self._process_binance_trades_feed()),
            asyncio.create_task(self._process_deribit_trades_feed()),
            asyncio.create_task(self._macro_gex_mapping_loop())
        ]
        
        logger.info("All high-speed processing pipelines spawned successfully.")

    async def stop(self):
        """Graceful system shutdown."""
        logger.info("Initiating graceful shutdown sequence...")
        self.running = False
        
        # Stop WS feeds
        await self.binance_ws.stop()
        await self.deribit_ws.stop()
        
        # Cancel worker tasks
        for task in self.tasks:
            task.cancel()
            
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        logger.info("System stopped. All capital risk gates locked.")

    async def _process_binance_depth_feed(self):
        """High-speed L2 Order Book Depth Consumer."""
        while self.running:
            try:
                data = await self.binance_depth_queue.get()
                
                # Extract best bid/ask quantities
                bids = data.get("b", [])
                asks = data.get("a", [])
                
                if bids and asks:
                    # Format: [price, quantity]
                    best_bid_qty = float(bids[0][1])
                    best_ask_qty = float(asks[0][1])
                    
                    # Extract mid price
                    best_bid_price = float(bids[0][0])
                    best_ask_price = float(asks[0][0])
                    mid_price = (best_bid_price + best_ask_price) / 2.0
                    
                    # Feed tick to Master State Machine
                    await self.state_machine.process_market_tick(
                        mid_price=mid_price,
                        bid_qty=best_bid_qty,
                        ask_qty=best_ask_qty,
                        is_trade=False
                    )
                    
                self.binance_depth_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Depth processing error: {e}")
                await asyncio.sleep(0.1)

    async def _process_binance_trades_feed(self):
        """High-speed Aggregated Futures Trade Consumer (updates CVD Engine)."""
        while self.running:
            try:
                data = await self.binance_trade_queue.get()
                
                price = float(data.get("p", 0.0))
                qty = float(data.get("q", 0.0))
                # Binance is_buyer_maker=True means taker was seller (aggressive sell)
                is_buyer_maker = bool(data.get("m", False))
                
                # Feed trade to state machine to update CVD engine
                await self.state_machine.process_market_tick(
                    mid_price=price,
                    bid_qty=10.0, # Dummy L2 placeholder during trade events
                    ask_qty=10.0,
                    is_trade=True,
                    trade_price=price,
                    trade_qty=qty,
                    is_buyer_maker=is_buyer_maker
                )
                
                self.binance_trade_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Futures trade processing error: {e}")
                await asyncio.sleep(0.1)

    async def _process_deribit_trades_feed(self):
        """Option Taker Trade Classifier."""
        while self.running:
            try:
                trade = await self.deribit_trade_queue.get()
                # Tick Rule classifier receives public option trade
                # Translates it into estimated dealer positioning updates
                # (e.g., updates OptionsTickRule internal positions)
                # Wait, the state machine will handle direct or mapper-driven positioning.
                logger.debug(f"Option Trade Received: Strike {trade['strike']} | Side: {trade['type']} | Qty: {trade['amount']}")
                self.deribit_trade_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deribit trade processing error: {e}")
                await asyncio.sleep(0.1)

    async def _macro_gex_mapping_loop(self):
        """
        Quant Macro GEX Mapping Loop.
        Periodically recalculates GEX curves and updates state machine GEX boundaries.
        """
        # Allow feeds to warm up for 5 seconds initially
        await asyncio.sleep(5.0)
        
        while self.running:
            try:
                chain = self.deribit_ws.options_chain
                index_price = self.deribit_ws.index_price
                
                if chain and index_price:
                    strikes = np.array(list(chain.keys()), dtype=np.float64)
                    call_oi = np.array([v["call_oi"] for v in chain.values()], dtype=np.float64)
                    put_oi = np.array([v["put_oi"] for v in chain.values()], dtype=np.float64)
                    # Average Call & Put volatilities
                    sigmas = np.array([(v["call_iv"] + v["put_iv"])/2.0 for v in chain.values()], dtype=np.float64)
                    # Clean sigmas: fallback to 40% if 0.0
                    sigmas = np.where(sigmas <= 0.01, 0.40, sigmas)
                    
                    # Calculate dealer GEX profile
                    gex_profile = self.mapper.calculate_gex_profile(
                        spot_price=index_price,
                        strikes=strikes,
                        call_oi=call_oi,
                        put_oi=put_oi,
                        sigmas=sigmas,
                        t=7.0/365.0, # Assumed rolling 7-day expiry
                        r=0.05,
                        q=0.00
                    )
                    
                    # Find walls
                    mapping = self.mapper.map_structural_hedging(gex_profile)
                    
                    # Locate closest key wall:
                    # If spot is close to put wall, select put wall as target
                    # If spot is close to call wall, select call wall as target
                    target_strike = None
                    target_gex = 0.0
                    
                    if mapping["support_walls"] and mapping["squeeze_walls"]:
                        put_wall_strike = mapping["support_walls"][0][0]
                        call_wall_strike = mapping["squeeze_walls"][0][0]
                        
                        dist_put = abs(index_price - put_wall_strike)
                        dist_call = abs(index_price - call_wall_strike)
                        
                        if dist_put < dist_call:
                            target_strike = put_wall_strike
                            target_gex = mapping["support_walls"][0][1]
                        else:
                            target_strike = call_wall_strike
                            target_gex = mapping["squeeze_walls"][0][1]
                            
                    elif mapping["support_walls"]:
                        target_strike = mapping["support_walls"][0][0]
                        target_gex = mapping["support_walls"][0][1]
                    elif mapping["squeeze_walls"]:
                        target_strike = mapping["squeeze_walls"][0][0]
                        target_gex = mapping["squeeze_walls"][0][1]
                        
                    if target_strike:
                        self.state_machine.update_gex_profile(
                            key_strike=target_strike,
                            gex_value=target_gex,
                            deribit_index=index_price
                        )
                        
            except Exception as e:
                logger.error(f"Macro GEX mapping loop error: {e}")
                
            await asyncio.sleep(10.0) # Run mapping calculation every 10 seconds

async def main():
    orchestrator = QuantSystemOrchestrator(symbol="BTCUSDT", mode="SHADOW")
    
    # Handle signal terminations
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(orchestrator.stop()))
        except NotImplementedError:
            # Signal handlers not supported on some OS platforms/environments
            pass
            
    try:
        await orchestrator.start()
        while orchestrator.running:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Manual system interrupt received.")
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(main())
