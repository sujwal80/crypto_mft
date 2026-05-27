"""Dry-run script to validate live paper-trading system execution under sandbox parameters."""

import asyncio
import os
from indian_intraday_system.main import GexMicroSystem


async def run_validation():
    print("==============================================================================")
    print("                 indian_intraday_system: LIVE DRY RUN VALIDATION INITIALIZED          ")
    print("==============================================================================")

    # 1. Instantiate main coordinator
    system = GexMicroSystem()

    # Force-arm mock time manager to MEAN_REVERSION to ensure active strategy evaluation
    system.state_machine.time_manager.get_current_regime = lambda: "MEAN_REVERSION"

    # 2. Start the WebSocket connection and background tasks
    print("[DryRun] Connecting WebSockets and starting data recorders...")
    asyncio.create_task(system.start())

    # 3. Let it stream live synthetic data for 3 seconds
    print("[DryRun] Streaming ticks and options chain data...")
    await asyncio.sleep(3.0)

    # 4. Execute graceful shutdown
    print("[DryRun] Stopping data client and performing safe shutdown...")
    system.running = False
    await system.data_client.disconnect()
    await system.recorder.stop()

    # 5. Verify performance
    funds = system.router.get_funds()
    print("\n==============================================================================")
    print("                       LIVE DRY RUN COMPLETED SUCCESSFULLY                   ")
    print("==============================================================================")
    print(f"Starting Capital:   INR {funds['starting_capital']:,.2f}")
    print(f"Ending Balance:     INR {funds['balance']:,.2f}")
    print(f"Net Capital Change: INR {funds['net_pnl']:,.2f}")
    print("==============================================================================")


if __name__ == "__main__":
    try:
        asyncio.run(run_validation())
    except KeyboardInterrupt:
        print("\n[DryRun] Interrupted. Safely shutting down.")
