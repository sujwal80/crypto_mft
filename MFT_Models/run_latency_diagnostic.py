import asyncio
import time
import json
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Latency_Diagnostics")

try:
    import aiohttp
    import websockets
except ImportError as e:
    logger.critical(f"Required library missing: {e}. Run 'pip install aiohttp websockets' first.")
    sys.exit(1)

# Configuration for profiling
SYMBOL = "BTCUSDT"
BINANCE_WSS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth5@100ms"
BINANCE_PING_URL = "https://api.binance.com/api/v3/ping"
BINANCE_DEPTH_URL = f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=5"

async def profile_ingestion_latency_and_loss(duration_seconds: float = 10.0):
    """
    Connects to Binance L2 depth stream (100ms intervals) to measure:
    1. Ingestion latency: local arrival time minus exchange event timestamp (E).
    2. Throughput / Packet Loss: actual packet arrival count vs expected (10 packets/sec).
    """
    logger.info(f"Step 1: Connecting to Binance WebSocket depth stream for {SYMBOL}...")
    logger.info(f"Profiling data rate and latency for {duration_seconds} seconds...")
    
    latencies_ms = []
    arrival_intervals_ms = []
    last_arrival_time = None
    packet_count = 0
    
    start_test_time = time.time()
    
    try:
        async with websockets.connect(BINANCE_WSS_URL) as ws:
            while time.time() - start_test_time < duration_seconds:
                try:
                    # Wait for message with timeout to detect starvation
                    raw_message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    arrival_time = time.time()
                    packet_count += 1
                    
                    # Interval tracking
                    if last_arrival_time is not None:
                        interval = (arrival_time - last_arrival_time) * 1000.0
                        arrival_intervals_ms.append(interval)
                    last_arrival_time = arrival_time
                    
                    data = json.loads(raw_message)
                    event_time_ms = data.get("E") # Exchange event timestamp (E) in ms
                    
                    if event_time_ms:
                        # Ingestion Latency = Local Arrival Time - Exchange Event Time
                        ingestion_latency_ms = (arrival_time * 1000.0) - event_time_ms
                        latencies_ms.append(ingestion_latency_ms)
                        
                except asyncio.TimeoutError:
                    logger.warning("⚠️ WebSocket packet starvation warning: No data received for 2.0 seconds.")
                except Exception as e:
                    logger.error(f"Error parsing frame: {e}")
                    
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        return None

    actual_duration = time.time() - start_test_time
    
    # Compute metrics
    if not latencies_ms:
        logger.error("No packets received. Latency diagnostic failed.")
        return None
        
    avg_latency = np.mean(latencies_ms)
    median_latency = np.median(latencies_ms)
    p99_latency = np.percentile(latencies_ms, 99)
    std_latency = np.std(latencies_ms)
    
    avg_interval = np.mean(arrival_intervals_ms) if arrival_intervals_ms else 0.0
    
    # In 100ms stream, we expect exactly 10 packets per second
    expected_packets = int(actual_duration * 10)
    packet_loss = max(0.0, 1.0 - (packet_count / expected_packets)) if expected_packets > 0 else 0.0
    
    return {
        "actual_duration_s": actual_duration,
        "packet_count": packet_count,
        "expected_packets": expected_packets,
        "packet_loss_pct": packet_loss * 100.0,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "p99_latency_ms": p99_latency,
        "std_latency_ms": std_latency,
        "avg_interval_ms": avg_interval
    }

async def profile_execution_latency(runs: int = 5):
    """
    Measures API round-trip execution latency (RTT) by pinging Binance public endpoints.
    1. /api/v3/ping: System latency ping (very lightweight).
    2. /api/v3/depth: Public market data REST fetch (heavier, matches order placement payload size).
    """
    logger.info(f"Step 2: Profiling API execution round-trip RTT over {runs} runs...")
    
    ping_rtts = []
    depth_rtts = []
    
    async with aiohttp.ClientSession() as session:
        for i in range(runs):
            # RTT 1: Light-weight system ping
            t0 = time.time()
            try:
                async with session.get(BINANCE_PING_URL) as resp:
                    if resp.status == 200:
                        rtt = (time.time() - t0) * 1000.0
                        ping_rtts.append(rtt)
            except Exception as e:
                logger.error(f"Ping failed: {e}")
                
            # RTT 2: Heavy order book depth query (simulates order placement payload transit)
            t0 = time.time()
            try:
                async with session.get(BINANCE_DEPTH_URL) as resp:
                    if resp.status == 200:
                        rtt = (time.time() - t0) * 1000.0
                        depth_rtts.append(rtt)
            except Exception as e:
                logger.error(f"Depth fetch failed: {e}")
                
            await asyncio.sleep(0.2) # Brief gap between pings
            
    return {
        "avg_ping_rtt_ms": np.mean(ping_rtts) if ping_rtts else 0.0,
        "p99_ping_rtt_ms": np.percentile(ping_rtts, 99) if ping_rtts else 0.0,
        "avg_depth_rtt_ms": np.mean(depth_rtts) if depth_rtts else 0.0,
        "p99_depth_rtt_ms": np.percentile(depth_rtts, 99) if depth_rtts else 0.0
    }

async def main():
    print("=================================================================================")
    print("🏎️  ENTERPRISE CRYPTO MFT - LATENCY & INGESTION DIAGNOSTIC UTILITY")
    print("=================================================================================")
    
    # Profile Ingestion Latency & Packet Loss
    ingest_metrics = await profile_ingestion_latency_and_loss(duration_seconds=10.0)
    
    # Profile Execution API Round-Trip RTT
    exec_metrics = await profile_execution_latency(runs=5)
    
    if not ingest_metrics or not exec_metrics:
        print("\n❌ Latency profiling failed. Check network connection.")
        print("=================================================================================")
        return

    print("\n=================================================================================")
    print("📊 INGESTION LATENCY & THROUGHPUT REPORT:")
    print("=================================================================================")
    print(f"Active Data Stream     : {BINANCE_WSS_URL}")
    print(f"Test Duration          : {ingest_metrics['actual_duration_s']:.2f} seconds")
    print(f"Packets Received       : {ingest_metrics['packet_count']} / Expected: {ingest_metrics['expected_packets']}")
    print(f"Calculated Packet Loss : {ingest_metrics['packet_loss_pct']:.2f}%")
    print(f"Average Packet Interval: {ingest_metrics['avg_interval_ms']:.2f} ms (Expected: 100.00 ms)")
    print("-" * 81)
    print(f"Average Ingestion Lat  : {ingest_metrics['avg_latency_ms']:.2f} ms")
    print(f"Median Ingestion Lat   : {ingest_metrics['median_latency_ms']:.2f} ms")
    print(f"99th Percentile Lat    : {ingest_metrics['p99_latency_ms']:.2f} ms")
    print(f"Latency Jitter (StdDev): {ingest_metrics['std_latency_ms']:.2f} ms")
    
    print("\n=================================================================================")
    print("⚡ API EXECUTION ROUND-TRIP LATENCY (RTT) REPORT:")
    print("=================================================================================")
    print(f"Lightweight Ping RTT   : Avg {exec_metrics['avg_ping_rtt_ms']:.2f} ms | 99th {exec_metrics['p99_ping_rtt_ms']:.2f} ms")
    print(f"Order-Depth Query RTT  : Avg {exec_metrics['avg_depth_rtt_ms']:.2f} ms | 99th {exec_metrics['p99_depth_rtt_ms']:.2f} ms")
    print("=================================================================================")
    print("💡 DIAGNOSTIC SUMMARY:")
    
    total_rtt = exec_metrics['avg_depth_rtt_ms']
    ingest_lat = ingest_metrics['avg_latency_ms']
    
    if total_rtt < 50.0 and ingest_lat < 20.0:
        print("🟢 EXCELLENT: Connection has sub-50ms execution RTT and sub-20ms socket latency. Ready for live trading.")
    elif total_rtt < 120.0 and ingest_lat < 50.0:
        print("🟡 MODERATE: Standard residential/VPN network connection. Expect moderate latency slippage on market wicks.")
    else:
        print("🔴 HIGH LATENCY WARNING: Execution RTT exceeds 120ms or socket delay exceeds 50ms. High risk of order starvation!")
    print("=================================================================================")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Diagnostics halted by user.")
