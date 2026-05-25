import os
import sys
import requests
from datetime import datetime, timedelta

def download_file(url, output_path):
    """Downloads a file with streaming to prevent memory leaks on large datasets."""
    if os.path.exists(output_path):
        print(f"   [✓] File already exists locally: {os.path.basename(output_path)}")
        return True
        
    print(f"   [↓] Downloading: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, headers=headers)
        if response.status_code == 404:
            print(f"   [✗] Data not found (404): {os.path.basename(output_path)}")
            return False
            
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"   [✓] Successfully saved: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"   [✗] Download failed: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    symbol = "BTCUSDT"
    days_to_download = 5
    
    print("================================================================================")
    print("🚀 BINANCE VISION - PUBLIC HISTORICAL DATA DOWNLOADER")
    print("   Target Asset: USD-M Futures | BTCUSDT")
    print("================================================================================")
    
    # Target local folders
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    depth_dir = os.path.join(base_dir, "raw_depth")
    trades_dir = os.path.join(base_dir, "raw_trades")
    
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(trades_dir, exist_ok=True)
    
    # Calculate dates (yesterday backwards)
    dates = []
    for i in range(1, days_to_download + 1):
        date_str = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        dates.append(date_str)
        
    print(f"📅 Fetching raw zip archives for the last {days_to_download} days: {', '.join(dates)}\n")
    
    for date_str in dates:
        print(f"📦 Processing date: {date_str}...")
        
        # 1. L2 Book Depth Update Snapshots (100ms updates)
        depth_filename = f"{symbol}-bookDepth-{date_str}.zip"
        depth_url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{depth_filename}"
        depth_path = os.path.join(depth_dir, depth_filename)
        download_file(depth_url, depth_path)
        
        # 2. Aggregated Trades Sweeps (AggTrades)
        trades_filename = f"{symbol}-aggTrades-{date_str}.zip"
        trades_url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{trades_filename}"
        trades_path = os.path.join(trades_dir, trades_filename)
        download_file(trades_url, trades_path)
        
        print("-" * 80)
        
    print("\n🎉 All downloads finished successfully!")
    print(f"📂 Raw L2 Bids/Asks Updates ZIPs saved to: {depth_dir}")
    print(f"📂 Raw AggTrades ZIPs saved to: {trades_dir}")
    print("================================================================================")

if __name__ == "__main__":
    main()
