import asyncio
import logging
import os
import sys
import time
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add workspace to path
workspace_path = "/Users/singhujwal/crypto_mft"
sys.path.append(workspace_path)

from core.schemas import InternalTick
from ingestion.binance_adapter import BinanceCryptoAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MarketDataRecorder")

OUTPUT_FILE = os.path.join(workspace_path, "real_market_data_10m.jsonl")
RECORDING_DURATION_SECS = 600  # 10 minutes

async def recorder_consumer(queue: asyncio.Queue, start_time: float):
    """Consumes ticks from the queue and writes them to the output file."""
    logger.info(f"Recorder consumer started. Writing to: {OUTPUT_FILE}")
    
    tick_count = 0
    last_report_time = time.time()
    
    # Open file in append mode, ensuring it starts fresh
    with open(OUTPUT_FILE, "w") as f:
        pass

    try:
        while True:
            tick: InternalTick = await queue.get()
            tick_count += 1
            
            # Write tick as JSON line
            with open(OUTPUT_FILE, "a") as f:
                f.write(tick.model_dump_json() + "\n")
                
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Periodic reporting every 10 seconds
            if current_time - last_report_time >= 10.0:
                percent_done = (elapsed / RECORDING_DURATION_SECS) * 100.0
                logger.info(
                    f"Progress: {percent_done:.1f}% | Elapsed: {elapsed:.1f}s / {RECORDING_DURATION_SECS}s | "
                    f"Ticks Captured: {tick_count} | Bid: {tick.bid:.2f} | Ask: {tick.ask:.2f}"
                )
                last_report_time = current_time
                
            # Check if 10 minutes have elapsed
            if elapsed >= RECORDING_DURATION_SECS:
                logger.info(f"Target recording duration reached ({RECORDING_DURATION_SECS} seconds). Stopping consumer...")
                break
                
            queue.task_done()
    except asyncio.CancelledError:
        logger.info("Consumer task cancelled.")
    except Exception as e:
        logger.error(f"Error in recorder consumer: {e}")

async def main():
    logger.info("=================================================================================")
    logger.info("📥 REAL MARKET DATA RECORDER - BINANCE WEBSOCKET")
    logger.info("=================================================================================")
    
    SYMBOL = "BTCUSDT"
    BINANCE_WSS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth5@100ms"
    BINANCE_REST_URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=5"
    
    message_queue = asyncio.Queue()
    
    # Initialize Adapter
    binance_adapter = BinanceCryptoAdapter(symbol=SYMBOL, wss_url=BINANCE_WSS_URL, rest_url=BINANCE_REST_URL)
    
    start_time = time.time()
    
    # Create task for ingestion streaming
    streaming_task = asyncio.create_task(binance_adapter.connect_and_stream(message_queue))
    # Create task for writing files
    consumer_task = asyncio.create_task(recorder_consumer(message_queue, start_time))
    
    try:
        # Wait for consumer to finish (automatically stops after 10 mins)
        await consumer_task
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected. Initiating graceful shutdown...")
    finally:
        logger.info("Stopping adapter and cleaning up tasks...")
        await binance_adapter.close()
        
        # Cancel streaming task if running
        if not streaming_task.done():
            streaming_task.cancel()
            try:
                await streaming_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Data recording session complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted cleanly.")
