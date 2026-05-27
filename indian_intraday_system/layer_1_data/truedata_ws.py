"""TrueData WebSocket API Client managing indices, futures, and dynamic options chains."""

import asyncio
import json
import struct
import time
from typing import Callable, Dict, List, Set
import websockets
from indian_intraday_system.config import (
    DYNAMIC_OPTION_STRIKE_COUNT,
    HEARTBEAT_TIMEOUT_MS,
    TRUEDATA_PASSWORD,
    TRUEDATA_USERNAME,
    TRUEDATA_WS_URL,
)


class TrueDataWSClient:
    """Manages live stream loops with dynamic option chain strike subscription maps."""

    def __init__(self):
        self.url = TRUEDATA_WS_URL
        self.username = TRUEDATA_USERNAME
        self.password = TRUEDATA_PASSWORD

        self.websocket = None
        self.is_connected = False
        self.running = False
        self.last_tick_time = time.time()

        self.callbacks: List[Callable[[Dict], None]] = []
        self.emergency_callback: Callable[[], None] = None

        # Track active option strikes to prevent bandwidth overload
        self.subscribed_option_symbols: Set[str] = set()
        self.latest_atm_strike = None

        self._heartbeat_task = None
        self._simulated_feed_task = None
        self._listen_task = None

    def register_callback(self, cb: Callable[[Dict], None]):
        self.callbacks.append(cb)

    def register_emergency_callback(self, cb: Callable[[], None]):
        self.emergency_callback = cb

    async def connect(self):
        """Establishes WebSocket connections, falling back to synthetic playback if placeholders exist."""
        import os
        env = os.getenv("SYSTEM_ENVIRONMENT", "PAPER")
        
        if env == "SIMULATION":
            from indian_intraday_system.layer_1_data.historical_replay import HistoricalReplayFeed
            print("[TrueDataWS] SIMULATION Profile detected. Launching segregated Historical Replay...")
            self.running = True
            self.is_connected = True
            self.replay_feed = HistoricalReplayFeed(
                callbacks=self.callbacks,
                window_update_callback=self.update_dynamic_option_window
            )
            replay_date = os.getenv("HISTORICAL_REPLAY_DATE", "2026-05-22")
            self._replay_task = asyncio.create_task(self.replay_feed.start(replay_date))
            return

        if self.username == "placeholder_user" or "placeholder" in self.url:
            print("[TrueDataWS] Placeholder creds. Starting High-Fidelity Simulated Feed...")
            self.running = True
            self.is_connected = True
            self._simulated_feed_task = asyncio.create_task(self._run_simulated_feed())
            self._heartbeat_task = asyncio.create_task(self._monitor_heartbeat())
            return

        print(f"[TrueDataWS] Connecting to real-time feed at {self.url}...")
        self.running = True

        while self.running:
            try:
                self.websocket = await websockets.connect(self.url)
                self.is_connected = True
                self.last_tick_time = time.time()

                # Submit authentication payload
                auth_payload = {
                    "action": "login",
                    "username": self.username,
                    "password": self.password,
                }
                await self.websocket.send(json.dumps(auth_payload))

                # Receive login response
                response = await self.websocket.recv()
                print(f"[TrueDataWS] Login Response: {response}")

                # Start background tasks
                self._heartbeat_task = asyncio.create_task(self._monitor_heartbeat())
                self._listen_task = asyncio.create_task(self._listen_loop())
                break
            except Exception as e:
                print(f"[TrueDataWS] Connection failed: {e}. Retrying in 5 seconds...")
                self.is_connected = False
                await asyncio.sleep(5.0)

    async def subscribe_futures(self, symbol: str):
        """Subscribes to index future contract."""
        if not self.is_connected:
            print("[TrueDataWS] Cannot subscribe, not connected.")
            return

        if self.websocket is not None:
            payload = {"action": "subscribe", "symbols": [symbol]}
            await self.websocket.send(json.dumps(payload))
            print(f"[TrueDataWS] Subscription sent for {symbol}")
        else:
            print(f"[TrueDataWS] [Replay/Sim] Mock Subscribed to index future: {symbol}")

    async def _listen_loop(self):
        """Asynchronously listens for raw payloads and deserializes them."""
        while self.running and self.is_connected:
            try:
                message = await self.websocket.recv()
                self.last_tick_time = time.time()

                # Deserialize JSON ticks from TrueData feed
                data = json.loads(message)

                # Update dynamic option window if a trade tick has a spot reference
                if data.get("type") == "trade" and "spot_reference" in data:
                    self.update_dynamic_option_window(data["spot_reference"])

                # Dispatch ticks to strategy callbacks
                for cb in self.callbacks:
                    try:
                        cb(data)
                    except Exception as cb_err:
                        print(f"[TrueDataWS] Callback error: {cb_err}")

            except websockets.exceptions.ConnectionClosed:
                print("[TrueDataWS] WebSocket closed by remote host. Reconnecting...")
                self.is_connected = False
                await self.connect()
                break
            except Exception as e:
                print(f"[TrueDataWS] Deserialization error in listen loop: {e}")
                await asyncio.sleep(1.0)

    async def _monitor_heartbeat(self):
        timeout_sec = HEARTBEAT_TIMEOUT_MS / 1000.0
        while self.running:
            await asyncio.sleep(0.5)
            elapsed = time.time() - self.last_tick_time
            if elapsed > timeout_sec:
                print(f"[TrueDataWS] CRITICAL DATA LOSS: No feed ticks received in {elapsed:.2f}s!")
                if self.emergency_callback:
                    try:
                        self.emergency_callback()
                    except Exception as err:
                        print(f"[TrueDataWS] Error in emergency callback: {err}")
                self.running = False
                self.is_connected = False
                break

    def update_dynamic_option_window(self, spot: float):
        """Dynamic Options Window: Auto-subscribes to ATM +/- 5 strikes as spot shifts."""
        strike_step = 50
        atm_strike = round(spot / strike_step) * strike_step

        if self.latest_atm_strike == atm_strike:
            return

        self.latest_atm_strike = atm_strike
        
        # Generate strikes surrounding the ATM
        target_strikes = [
            atm_strike + (offset * strike_step)
            for offset in range(-DYNAMIC_OPTION_STRIKE_COUNT, DYNAMIC_OPTION_STRIKE_COUNT + 1)
        ]

        target_symbols = set()
        for strike in target_strikes:
            target_symbols.add(f"NIFTY26MAY{strike}CE")
            target_symbols.add(f"NIFTY26MAY{strike}PE")

        to_subscribe = target_symbols - self.subscribed_option_symbols
        to_unsubscribe = self.subscribed_option_symbols - target_symbols

        if to_unsubscribe:
            self._unsubscribe_option_symbols(list(to_unsubscribe))
        if to_subscribe:
            self._subscribe_option_symbols(list(to_subscribe))

        self.subscribed_option_symbols = target_symbols

    def _subscribe_option_symbols(self, symbols: List[str]):
        if self.websocket is not None and self.is_connected:
            payload = {"action": "subscribe", "symbols": symbols}
            asyncio.create_task(self.websocket.send(json.dumps(payload)))

    def _unsubscribe_option_symbols(self, symbols: List[str]):
        if self.websocket is not None and self.is_connected:
            payload = {"action": "unsubscribe", "symbols": symbols}
            asyncio.create_task(self.websocket.send(json.dumps(payload)))

    async def disconnect(self):
        print("[TrueDataWS] Shutting down live ws connections...")
        self.running = False
        self.is_connected = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._simulated_feed_task:
            self._simulated_feed_task.cancel()
        if self._listen_task:
            self._listen_task.cancel()

        if hasattr(self, 'replay_feed') and self.replay_feed:
            self.replay_feed.stop()
            if hasattr(self, '_replay_task') and self._replay_task:
                self._replay_task.cancel()

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        print("[TrueDataWS] Disconnected.")

    async def _run_simulated_feed(self):
        """Generates live simulated index spot, futures premium, and options chain tick payloads."""
        import random

        base_spot = 22000.0
        tick_count = 0

        while self.running:
            await asyncio.sleep(0.1)
            tick_count += 1
            self.last_tick_time = time.time()

            base_spot += random.normalvariate(0, 0.8)
            self.update_dynamic_option_window(base_spot)

            futures_tick = {
                "symbol": "NIFTY_FUT",
                "type": "trade",
                "price": base_spot + 15.0,
                "spot_reference": base_spot,
                "volume": float(random.randint(25, 150)),
                "bid": base_spot + 14.5,
                "ask": base_spot + 15.5,
                "timestamp": time.time(),
            }
            for cb in self.callbacks:
                cb(futures_tick)

            if tick_count % 10 == 0:
                strikes_list = [
                    self.latest_atm_strike + (offset * 50)
                    for offset in range(-DYNAMIC_OPTION_STRIKE_COUNT, DYNAMIC_OPTION_STRIKE_COUNT + 1)
                ]
                
                option_chain_tick = {
                    "symbol": "NIFTY_OPTION_CHAIN",
                    "type": "chain",
                    "spot": base_spot,
                    "strikes": strikes_list * 2,
                    "ivs": [0.15] * len(strikes_list) * 2,
                    "open_interests": [35000] * len(strikes_list) * 2,
                    "option_types": (["C"] * len(strikes_list)) + (["P"] * len(strikes_list)),
                    "expiries_days": [6] * len(strikes_list) * 2,
                    "timestamp": time.time(),
                }
                for cb in self.callbacks:
                    cb(option_chain_tick)


