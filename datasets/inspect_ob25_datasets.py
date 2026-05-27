import os
import glob
from collections import defaultdict

def inspect_dataset():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    ob25_dirs = sorted([d for d in os.listdir(base_dir) if d.endswith("_BTC_USDT.ob25") and os.path.isdir(os.path.join(base_dir, d))])
    
    print("================================================================================")
    print("🔍 QUANTITATIVE .OB25 OPTIONS DATASET INVENTORY AUDIT")
    print("================================================================================")
    
    for d in ob25_dirs:
        dir_path = os.path.join(base_dir, d)
        files = glob.glob(os.path.join(dir_path, "*.ob25"))
        total_files = len(files)
        
        # Calculate aggregate folder size
        total_size_bytes = sum(os.path.getsize(f) for f in files)
        total_size_mb = total_size_bytes / (1024.0 * 1024.0)
        
        # Group by Expiry and Strike/Option Type
        expiries = defaultdict(int)
        option_types = defaultdict(int)
        strikes = []
        
        for f in files:
            basename = os.path.basename(f)
            try:
                if "_BTC" in basename:
                    # Extract suffix: e.g. "-12JUN26-62000-C-USDT.ob25"
                    contract_suffix = basename.split("_BTC")[1]
                    parts = contract_suffix.split("-")
                    
                    # parts: ['', '12JUN26', '62000', 'C', 'USDT.ob25']
                    expiry = parts[1]
                    strike = float(parts[2])
                    opt_type = parts[3] # 'C' or 'P'
                    
                    expiries[expiry] += 1
                    option_types[opt_type] += 1
                    strikes.append(strike)
            except Exception:
                continue
                
        min_strike = min(strikes) if strikes else 0.0
        max_strike = max(strikes) if strikes else 0.0
        unique_expiries = sorted(list(expiries.keys()))
        
        print(f"📂 Directory: {d}")
        print(f"   • Total Contracts Tracked : {total_files}")
        print(f"   • Total Aggregate Size    : {total_size_mb:.2f} MB")
        print(f"   • Unique Expiry Dates     : {', '.join(unique_expiries)}")
        print(f"   • Strike Price Range      : ${min_strike:,.0f} to ${max_strike:,.0f}")
        print(f"   • Call vs Put Breakdown   : Calls={option_types['C']} | Puts={option_types['P']}")
        print("-" * 80)
        
    print("🎉 Audit finished successfully!")
    print("================================================================================")

if __name__ == "__main__":
    inspect_dataset()
