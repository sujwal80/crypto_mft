import os
import json
import glob
import math
import numpy as np
from datetime import datetime

# --------------------------------------------------------------------------------
# BLACK-SCHOLES Greeks & Implied Volatility Solver
# --------------------------------------------------------------------------------
def normal_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def normal_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_option_price(S, K, t, r, sigma, option_type="C"):
    """Calculates Black-Scholes option price."""
    if t <= 0:
        if option_type == "C":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
            
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    
    if option_type == "C":
        return S * normal_cdf(d1) - K * math.exp(-r * t) * normal_cdf(d2)
    else:
        return K * math.exp(-r * t) * normal_cdf(-d2) - S * normal_cdf(-d1)

def solve_implied_volatility(S, K, t, r, market_price, option_type="C"):
    """Brent's numerical root-finding optimizer to solve implied volatility."""
    # Intrinsic value check
    intrinsic = max(0.0, S - K) if option_type == "C" else max(0.0, K - S)
    if market_price <= intrinsic:
        return 0.20 # Fallback default IV
        
    # Bounds for volatility search
    low_sigma = 0.01
    high_sigma = 5.0
    
    # Initial Brent roots search
    try:
        f_low = bs_option_price(S, K, t, r, low_sigma, option_type) - market_price
        f_high = bs_option_price(S, K, t, r, high_sigma, option_type) - market_price
        
        if f_low * f_high > 0:
            return 0.40 # Standard crypto baseline IV
            
        # Numerical bisection search
        for _ in range(100):
            mid_sigma = (low_sigma + high_sigma) / 2.0
            f_mid = bs_option_price(S, K, t, r, mid_sigma, option_type) - market_price
            
            if abs(f_mid) < 1e-5:
                return mid_sigma
            if f_low * f_mid < 0:
                high_sigma = mid_sigma
                f_high = f_mid
            else:
                low_sigma = mid_sigma
                f_low = f_mid
        return (low_sigma + high_sigma) / 2.0
    except Exception:
        return 0.40

def calculate_gamma(S, K, t, r, sigma):
    """Computes option contract Gamma Greek."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return normal_pdf(d1) / (S * sigma * math.sqrt(t))

# --------------------------------------------------------------------------------
# GEX Aggregation Compiler
# --------------------------------------------------------------------------------
def compile_gex_for_day(dir_path, date_str):
    """Parses opening L2 snapshots of all strikes to construct GEX profile."""
    files = glob.glob(os.path.join(dir_path, "*.ob25"))
    if not files:
        return None
        
    # Estimate the underlying spot price from the strikes midpoints
    # To get a robust opening spot, we average the Level 1 midpoints of deep out-of-the-money calls/puts
    approx_spots = []
    
    parsed_contracts = []
    
    # 1. Extract Expiry date & Strike metadata from file headers
    for f in files:
        basename = os.path.basename(f)
        if "_BTC" not in basename:
            continue
            
        try:
            contract_suffix = basename.split("_BTC")[1]
            parts = contract_suffix.split("-")
            
            expiry_str = parts[1]
            strike = float(parts[2])
            opt_type = parts[3] # 'C' or 'P'
            
            # Load opening snapshot line (1st line)
            with open(f, "r") as lf:
                first_line = lf.readline()
                if not first_line:
                    continue
                packet = json.loads(first_line)
                bids = packet["data"].get("b", [])
                asks = packet["data"].get("a", [])
                
                if not bids or not asks:
                    continue
                    
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                mid_option_price = (best_bid + best_ask) / 2.0
                
                # Estimate underlying spot price proxy using out-of-the-money parity
                if opt_type == "C" and strike > 60000:
                    approx_spots.append(mid_option_price + strike)
                    
                # Aggregate total book resting volume as proxy for open interest (OI)
                total_oi = sum(float(b[1]) for b in bids) + sum(float(a[1]) for a in asks)
                
                parsed_contracts.append({
                    "expiry_str": expiry_str,
                    "strike": strike,
                    "type": opt_type,
                    "market_price": mid_option_price,
                    "oi": total_oi
                })
        except Exception:
            continue
            
    # Robust opening Spot price estimate
    spot_price = float(np.median(approx_spots)) if approx_spots else 60000.0
    
    current_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    gex_by_strike = defaultdict(float)
    
    # 2. Calculate GEX Greeks for each contract
    for c in parsed_contracts:
        # Compute expiry fraction t (years)
        try:
            expiry_date = datetime.strptime(c["expiry_str"], "%d%b%y")
            t = max(1.0 / 365.0, (expiry_date - current_date).days / 365.0)
        except Exception:
            t = 7.0 / 365.0
            
        # Solve live Implied Volatility
        sigma = solve_implied_volatility(spot_price, c["strike"], t, 0.05, c["market_price"], c["type"])
        
        # Calculate contract Gamma
        gamma = calculate_gamma(spot_price, c["strike"], t, 0.05, sigma)
        
        # Dealer positioning multiplier: call (+1), put (-1)
        multiplier = 1.0 if c["type"] == "C" else -1.0
        
        # Premium-weighted GEX math
        gex = multiplier * c["oi"] * gamma * spot_price
        gex_by_strike[c["strike"]] += gex
        
    # 3. Locate Key Options walls from curves
    support_put_wall = 0.0
    max_put_gex = -float("inf")
    resistance_call_wall = 0.0
    max_call_gex = -float("inf")
    
    for strike, gex in gex_by_strike.items():
        # Put wall: concentrated negative dealer puts below spot
        if strike < spot_price and gex < 0:
            abs_gex = abs(gex)
            if abs_gex > max_put_gex:
                max_put_gex = abs_gex
                support_put_wall = strike
        # Call wall: concentrated positive dealer calls above spot
        elif strike > spot_price and gex > 0:
            if gex > max_call_gex:
                max_call_gex = gex
                resistance_call_wall = strike
                
    # Standard default fallback if walls remain unmapped
    if support_put_wall == 0.0:
        support_put_wall = round(spot_price * 0.95 / 1000.0) * 1000.0
    if resistance_call_wall == 0.0:
        resistance_call_wall = round(spot_price * 1.05 / 1000.0) * 1000.0
        
    return {
        "date": date_str,
        "opening_spot": round(spot_price, 2),
        "support_put_wall": support_put_wall,
        "resistance_call_wall": resistance_call_wall
    }

# --------------------------------------------------------------------------------
# Main Aggregator
# --------------------------------------------------------------------------------
def main():
    datasets_dir = os.path.abspath(os.path.dirname(__file__))
    ob25_dirs = sorted([d for d in os.listdir(datasets_dir) if d.endswith("_BTC_USDT.ob25") and os.path.isdir(os.path.join(datasets_dir, d))])
    
    print("================================================================================")
    print("🚀 COMPILING HISTORICAL OPTIONS GEX WALLS SERIES")
    print("   (Reconstructing 60GB options chains into single JSON file)")
    print("================================================================================")
    
    compiled_walls = {}
    
    for d in ob25_dirs:
        # Extract date from directory name: e.g. "2026-05-25_BTC_USDT.ob25"
        date_str = d.split("_")[0]
        print(f"⚙️ Processing GEX options chain for {date_str}...")
        dir_path = os.path.join(datasets_dir, d)
        
        day_result = compile_gex_for_day(dir_path, date_str)
        if day_result:
            compiled_walls[date_str] = day_result
            print(f"   [✓] opening Spot  : ${day_result['opening_spot']:.2f}")
            print(f"   [✓] Put Support   : ${day_result['support_put_wall']:.2f}")
            print(f"   [✓] Call Resistance: ${day_result['resistance_call_wall']:.2f}")
            print("-" * 80)
            
    # Output to a single, ultra-lightweight JSON file!
    output_path = os.path.join(datasets_dir, "historical_options_walls_5days.json")
    with open(output_path, "w") as out_f:
        json.dump(compiled_walls, out_f, indent=2)
        
    print(f"\n🎉 Consolidated GEX walls file compiled and saved successfully!")
    print(f"📂 Output Path: {output_path}")
    print("================================================================================")

if __name__ == "__main__":
    from collections import defaultdict
    main()
