import asyncio
import os
import sys
import time
import json
import logging
from typing import Optional

# Add workspace to path
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(workspace_path)

from ingestion.binance_adapter import BinanceCryptoAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Market_Recorder")

async def record_indefinitely(symbol: str, output_filename: str):
    """
    Connects to the live Binance L2 WebSocket stream and continuously records 
    all ticks (bids/asks and exact sizes) to a JSONL file. Handles network drops 
    and reconnects automatically.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_dir, output_filename)
    
    # Setup Binance WS endpoints
    wss_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth5@100ms"
    rest_url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}&limit=5"
    
    # Leverage the production-grade Binance adapter with built-in watchdog & auto-reconnect
    adapter = BinanceCryptoAdapter(symbol=symbol.upper(), wss_url=wss_url, rest_url=rest_url)
    queue = asyncio.Queue()
    
    start_time = time.time()
    
    logger.info("==================================================================")
    logger.info("🎙️ PRODUCTION-GRADE REAL-TIME L2 MARKET DATA RECORDER")
    logger.info("==================================================================")
    logger.info(f"  - Asset Symbol    : {symbol.upper()}")
    logger.info(f"  - Target File     : {output_path}")
    logger.info(f"  - Connection Type : Live Binance WebSocket (@depth5@100ms)")
    logger.info("  - Status          : Recording Continuously (Press Ctrl+C to stop)")
    logger.info("==================================================================")
    
    # Start the streaming client task in the background
    stream_task = asyncio.create_task(adapter.connect_and_stream(queue))
    
    tick_count = 0
    last_log_time = time.time()
    
    try:
        # Open file in append-only mode
        with open(output_path, "a") as f:
            while True:
                try:
                    # Retrieve next L2 tick from queue
                    tick = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # Write complete high-fidelity InternalTick to disk
                    f.write(tick.model_dump_json() + "\n")
                    f.flush() # Immediate flush to avoid data loss in aborts
                    
                    tick_count += 1
                    
                    # Log progress statistics every 10 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 10.0:
                        elapsed = current_time - start_time
                        # Format human-readable elapsed time
                        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                        mid_price = (tick.bid + tick.ask) / 2.0
                        logger.info(
                            f"Elapsed: {elapsed_str} | "
                            f"Recorded Ticks: {tick_count:,} | "
                            f"Avg Rate: {tick_count / elapsed:.1f} ticks/sec | "
                            f"Current Mid: ${mid_price:,.2f}"
                        )
                        last_log_time = current_time
                        
                    queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Normal timeout if queue is empty briefly
                    continue
                    
    except asyncio.CancelledError:
        logger.warning("Recording task cancelled cleanly.")
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Closing threads safely...")
    except Exception as e:
        logger.error(f"🛑 Critical Error in recorder loop: {e}")
    finally:
        # Clean shutdown
        logger.info("Closing WebSocket connections and cancelling adapter threads...")
        stream_task.cancel()
        await adapter.close()
        logger.info(f"✅ Recorder closed. Logged {tick_count:,} ticks to: {output_path}")

def main():
    # Load configuration or fallbacks
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    
    # Allow target file naming override from CLI arguments
    output_filename = "real_market_data_live.log"
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        if not output_filename.endswith((".log", ".jsonl", ".json")):
            output_filename += ".log"
            
    try:
        asyncio.run(record_indefinitely(symbol=SYMBOL, output_filename=output_filename))
    except KeyboardInterrupt:
        logger.info("Active recorder aborted cleanly.")

if __name__ == "__main__":
    main()
