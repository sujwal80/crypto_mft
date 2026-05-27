import asyncio
import json
import logging
import websockets
import aiohttp
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DeribitWebSocketClient:
    """
    Production-grade, asynchronous WebSocket client for Deribit Options Chain and Trades.
    Utilizes single-request book summaries to fetch strikes, volatilities, and Open Interest 
    in a highly rate-limit friendly manner, combined with real-time trades feed.
    """
    def __init__(self, currency: str = "BTC", trade_queue: Optional[asyncio.Queue] = None, chain_update_interval: float = 10.0):
        self.currency = currency.upper()
        self.trade_queue = trade_queue or asyncio.Queue()
        self.chain_update_interval = chain_update_interval
        self.ws_url = "wss://www.deribit.com/ws/api/v2"
        self.running = False
        self.options_chain = {}  # Live Options Chain cache: {strike: {call_oi, put_oi, sigma, etc.}}
        self.index_price = None
        self.tasks = []
        self.session = None
        
    async def start(self):
        """Starts the asynchronous ingestion loops."""
        self.running = True
        self.session = aiohttp.ClientSession()
        self.tasks = [
            asyncio.create_task(self._connect_trades_loop()),
            asyncio.create_task(self._poll_options_chain_loop())
        ]
        logger.info(f"Deribit WS client started for {self.currency}")

    async def stop(self):
        """Stops the ingestion loops gracefully."""
        self.running = False
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.session:
            await self.session.close()
        logger.info("Deribit WS client stopped")

    async def _connect_trades_loop(self):
        """WebSocket trades subscription loop with auto-reconnect."""
        backoff = 1.0
        max_backoff = 60.0
        
        while self.running:
            try:
                logger.info(f"Connecting to Deribit WebSocket: {self.ws_url}")
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("Deribit WebSocket connected successfully")
                    backoff = 1.0
                    
                    # Subscribe to raw options trades
                    subscription_msg = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "public/subscribe",
                        "params": {
                            "channels": [f"trades.option.{self.currency}.raw"]
                        }
                    }
                    await ws.send(json.dumps(subscription_msg))
                    
                    while self.running:
                        message = await ws.recv()
                        data = json.loads(message)
                        
                        # Filter for trade notification events
                        if "params" in data and "data" in data["params"]:
                            trades = data["params"]["data"]
                            for trade in trades:
                                instrument = trade["instrument_name"]
                                # Parse instrument components (e.g. BTC-22MAY26-65000-C)
                                parts = instrument.split("-")
                                if len(parts) == 4:
                                    strike = float(parts[2])
                                    opt_type = parts[3]  # C or P
                                    
                                    trade_event = {
                                        "strike": strike,
                                        "type": "CALL" if opt_type == "C" else "PUT",
                                        "price": float(trade["price"]),  # Price in coin (BTC)
                                        "amount": float(trade["amount"]), # Contract quantity
                                        "direction": trade["direction"], # buy or sell
                                        "timestamp": trade["timestamp"]
                                    }
                                    
                                    try:
                                        self.trade_queue.put_nowait(trade_event)
                                    except asyncio.QueueFull:
                                        self.trade_queue.get_nowait()
                                        self.trade_queue.put_nowait(trade_event)
                                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deribit Trades WS connection error: {e}. Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)

    async def _poll_options_chain_loop(self):
        """Periodically queries Deribit book summary to maintain a live, high-fidelity Options Chain cache."""
        while self.running:
            try:
                # Request options book summary via JSON-RPC over HTTP (highly robust)
                url = f"https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={self.currency}&kind=option"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        res_data = await response.json()
                        if "result" in res_data:
                            summaries = res_data["result"]
                            
                            temp_chain = {}
                            latest_index = None
                            
                            for summary in summaries:
                                instrument = summary["instrument_name"]
                                parts = instrument.split("-")
                                if len(parts) == 4:
                                    strike = float(parts[2])
                                    opt_type = parts[3]  # C or P
                                    oi = float(summary.get("open_interest", 0.0))
                                    iv = float(summary.get("mark_iv", 0.0)) / 100.0 # Convert % to decimal
                                    underlying_price = float(summary.get("underlying_price", 0.0))
                                    
                                    if underlying_price > 0:
                                        latest_index = underlying_price
                                        
                                    if strike not in temp_chain:
                                        temp_chain[strike] = {
                                            "call_oi": 0.0, "put_oi": 0.0,
                                            "call_iv": 0.0, "put_iv": 0.0,
                                            "underlying_price": underlying_price
                                        }
                                        
                                    if opt_type == "C":
                                        temp_chain[strike]["call_oi"] = oi
                                        temp_chain[strike]["call_iv"] = iv
                                    else:
                                        temp_chain[strike]["put_oi"] = oi
                                        temp_chain[strike]["put_iv"] = iv
                                        
                            if temp_chain:
                                self.options_chain = temp_chain
                                if latest_index:
                                    self.index_price = latest_index
                                logger.debug(f"Deribit Options Chain cache updated with {len(temp_chain)} strikes")
                                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deribit Options Chain poll error: {e}")
                
            await asyncio.sleep(self.chain_update_interval)
