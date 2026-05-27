"""Asynchronous Orchestrator and Main entry point for indian_intraday_system.

Links TrueData WebSocket streams, GEX Mapping, CVD aggregation, and Basis tracking
directly to the Swing State Machine execution router.
"""

import asyncio
import os
import sys
from indian_intraday_system import config
from indian_intraday_system.layer_1_data.truedata_ws import TrueDataWSClient
from indian_intraday_system.layer_2_macro.gex_mapper import map_gex_levels
from indian_intraday_system.layer_3_micro.basis_tracker import BasisTracker
from indian_intraday_system.layer_3_micro.cvd_engine import CVDEngine
from indian_intraday_system.layer_4_execution.dhan_router import DhanRouter
from indian_intraday_system.layer_4_execution.shadow_router import ShadowRouter
from indian_intraday_system.layer_4_execution.state_machine import SwingStateMachine
from indian_intraday_system.utils.data_recorder import LiveDataRecorder


class GexMicroSystem:
    """Main System Coordinator linking live pipelines to quantitative executors."""

    def __init__(self):
        print("==============================================================================")
        print("                  NSE INTRADAY GEX-MICRO SYSTEM: ENGAGED                     ")
        print("==============================================================================")
        print(f"[Boot] Profile Environment: {os.getenv('SYSTEM_ENVIRONMENT', 'PAPER')}")
        print(f"[Boot] Target Broker Class: {os.getenv('BROKER_ROUTER_CLASS', 'SHADOW_ROUTER')}")

        # 1. Senses & Trackers
        self.data_client = TrueDataWSClient()
        self.cvd_engine = CVDEngine()
        self.basis_tracker = BasisTracker()
        self.recorder = LiveDataRecorder()

        # 2. Router Layer
        if config.USE_SHADOW_MODE:
            print("[System] Active Router: SHADOW / PAPER TRADING (Exact Friction / Slippage)")
            self.router = ShadowRouter(starting_capital=150000.0)
        else:
            print("[System] Active Router: PRODUCTION (Live Dhan HQ Broker API)")
            self.router = DhanRouter()

        # 3. Time & State Machine
        self.state_machine = SwingStateMachine(
            router=self.router,
            cvd_engine=self.cvd_engine,
        )

        self.gex_state = {}
        self.latest_future_price = 22000.0
        self.running = True

        # Register WebSocket Handlers
        self.data_client.register_callback(self.on_market_update)
        self.data_client.register_emergency_callback(self.on_critical_disconnect)

    def on_critical_disconnect(self):
        """Fail-safe trigger called by Heartbeat Monitor upon WebSocket silence."""
        print("[System] CRITICAL: TrueData feed silence > 3000ms. Disengaging System!")
        self.running = False
        square_off_details = self.router.emergency_square_off()
        print(f"[System] Emergency positions flattened: {square_off_details}")
        print("[System] Shutdown sequence complete.")
        sys.exit(1)

    def on_market_update(self, tick: dict):
        """Asynchronous event callback processing live market packets."""
        if not self.running:
            return

        # Record every incoming tick asynchronously in the background (non-blocking)
        asyncio.create_task(self.recorder.record_tick(tick))

        # Register Virtual Clock for Historical Simulation Replay Mode
        if "timestamp" in tick and isinstance(tick["timestamp"], str):
            try:
                from datetime import datetime
                if len(tick["timestamp"]) == 19:
                    dt = datetime.strptime(tick["timestamp"], "%Y-%m-%dT%H:%M:%S")
                    dt = self.state_machine.time_manager.tz.localize(dt)
                    self.state_machine.time_manager.set_virtual_time(dt)
            except Exception:
                pass

        tick_type = tick.get("type")

        if tick_type == "trade":
            # A. high-frequency trade data (Spot & Future prices)
            fut_price = tick["price"]
            spot_price = tick["spot_reference"]
            volume = tick["volume"]
            bid = tick.get("bid")
            ask = tick.get("ask")

            self.latest_future_price = fut_price

            # Update CVD buyer aggression metrics
            self.cvd_engine.process_trade(fut_price, volume, bid, ask)

            # Update Spot-to-Future Basis spread tracking
            self.basis_tracker.add_tick(future_price=fut_price, spot_price=spot_price)

            # If paper trading, process active limit fills
            if config.USE_SHADOW_MODE and isinstance(self.router, ShadowRouter):
                self.router.process_feed_tick(symbol="NIFTY_FUT", last_price=fut_price)

        elif tick_type == "chain":
            # B. Low-frequency options chain snapshot updates
            spot = tick["spot"]
            self.gex_state = map_gex_levels(
                spot=spot,
                strikes=tick["strikes"],
                expiries_days=tick["expiries_days"],
                ivs=tick["ivs"],
                open_interests=tick["open_interests"],
                option_types=tick["option_types"],
            )

            # Fetch Basis tracking statistics
            basis_stats = self.basis_tracker.get_basis_stats()
            basis_z = basis_stats["z_score"]
            basis_mean = basis_stats["mean"]

            # Print Live Telemetry Dashboard
            print(
                f"[Dashboard] [{self.state_machine.get_state()}] Spot: {spot:.2f} | "
                f"ZeroG: {self.gex_state['zero_gamma']:.2f} | Wall(C/P): {self.gex_state['call_wall']:.2f}/{self.gex_state['put_wall']:.2f} | "
                f"CVD Ratio: {self.cvd_engine.get_taker_ratio():.3f} | Basis Z: {basis_z:.2f} (Mean: ₹{basis_mean:.2f})"
            )

            # C. Pass parameters to State Machine to evaluate entry/exit triggers!
            self.state_machine.evaluate_signals(
                spot=spot,
                future_price=self.latest_future_price,
                gex_levels=self.gex_state,
                basis_z_score=basis_z,
                lot_size=config.LOT_SIZE_NIFTY,
            )

    async def start(self):
        """Main event loop scheduler."""
        print("[System] Connect feed streams...")
        self.recorder.start()  # Start live tick recording
        await self.data_client.connect()
        await self.data_client.subscribe_futures("NIFTY_FUT")

        sleep_interval = 0.001 if os.getenv("SYSTEM_ENVIRONMENT") == "SIMULATION" else 1.0
        while self.running:
            # Time-regime check
            self.state_machine.update_state()
            
            if self.state_machine.get_state() == "KILL":
                print("[System] Kill schedule active. Flat all and halt.")
                self.running = False
                break
                
            # If in simulation and data loader completed streaming, stop orchestrator safely
            if os.getenv("SYSTEM_ENVIRONMENT") == "SIMULATION" and not self.data_client.running:
                print("[System] Simulation stream completed. Stopping orchestrator.")
                self.running = False
                break

            await asyncio.sleep(sleep_interval)

        # Graceful disconnect
        await self.data_client.disconnect()
        await self.recorder.stop()  # Complete recording and flush remaining buffers
        print("==============================================================================")
        print("                    SYSTEM CONVOLUTION COMPLETE. NET ACCOUNT:                 ")
        funds = self.router.get_funds()
        for k, v in funds.items():
            print(f"  {k.upper()}: {v}")
        print("==============================================================================")


if __name__ == "__main__":
    system = GexMicroSystem()
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        print("[System] Interrupted by user. Squaring all positions.")
        system.router.emergency_square_off()
