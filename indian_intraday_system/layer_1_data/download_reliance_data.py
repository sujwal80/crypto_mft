import json
import os
import urllib.request
import time
from typing import Dict, Optional

def fetch_yahoo_data(symbol: str, interval: str, date_range: str) -> Optional[Dict]:
    """
    Fetches raw chart data from Yahoo Finance public REST API.
    Zero-dependency implementation using built-in urllib.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={date_range}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        print(f"🌐 Querying Yahoo: Symbol={symbol} | Interval={interval} | Range={date_range}...")
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"❌ Failed to fetch {interval} data: {e}")
        return None

def parse_and_save_chart(payload: Dict, output_path: str) -> bool:
    """
    Parses Yahoo chart payload into a standardized database format and saves it.
    """
    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        
        closes = quote["close"]
        opens = quote["open"]
        highs = quote["high"]
        lows = quote["low"]
        volumes = quote["volume"]
        
        aligned_bars = []
        for i in range(len(timestamps)):
            ts = timestamps[i]
            close = closes[i]
            
            # Filter out nulls (market holidays / pricing gaps)
            if ts is None or close is None:
                continue
                
            date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))
            aligned_bars.append({
                "timestamp": int(ts),
                "date": date_str,
                "open": float(opens[i]) if opens[i] is not None else float(close),
                "high": float(highs[i]) if highs[i] is not None else float(close),
                "low": float(lows[i]) if lows[i] is not None else float(close),
                "close": float(close),
                "volume": int(volumes[i]) if volumes[i] is not None else 0
            })
            
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(aligned_bars, f, indent=4)
            
        print(f"   -> Saved {len(aligned_bars):,} bars to {output_path}")
        print(f"   -> Timeline: {aligned_bars[0]['date']} to {aligned_bars[-1]['date']}\n")
        return True
    except Exception as e:
        print(f"❌ Failed to parse and save chart: {e}")
        return False

def main():
    symbol = "RELIANCE.NS"
    data_dir = "/Users/singhujwal/crypto_mft/indian_intraday_system/data"
    
    # Define target price resolutions & historical ranges
    resolutions = [
        {"interval": "1d", "range": "5y", "name": "daily_5year.json"},      # Daily (60 Months)
        {"interval": "1h", "range": "2y", "name": "hourly_2year.json"},     # Hourly (24 Months - Max limit)
        {"interval": "5m", "range": "60d", "name": "5min_60days.json"},    # 5-Minute (60 Days - Max limit)
        {"interval": "1m", "range": "7d", "name": "1min_7days.json"}     # 1-Minute (7 Days - Max limit)
    ]
    
    print("=================================================================================")
    print(f"🚀 STARTING HISTORICAL DATA INGESTION FOR {symbol}")
    print("=================================================================================\n")
    
    success_count = 0
    for res in resolutions:
        payload = fetch_yahoo_data(symbol, interval=res["interval"], date_range=res["range"])
        if payload:
            output_file = os.path.join(data_dir, res["name"])
            if parse_and_save_chart(payload, output_file):
                success_count += 1
        time.sleep(1.5)  # Rate-limiting delay between queries
        
    print("=================================================================================")
    print(f"✅ DATA COMPILATION COMPLETE! {success_count}/{len(resolutions)} RESOLUTIONS GENERATED.")
    print(f"   -> Target Folder: {data_dir}")
    print("=================================================================================")

if __name__ == "__main__":
    main()
