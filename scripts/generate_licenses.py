#!/usr/bin/env python3
"""
AI Document Assistant - Commercial License Key Batch Generator
Use this script to generate authentic license keys to populate into Gumroad / LemonSqueezy license inventory.

Usage:
    python generate_licenses.py --count 50 --tier STD --output gumroad_standard_keys.txt
    python generate_licenses.py --count 20 --tier PRO --output gumroad_pro_keys.txt
"""

import sys
import os
import argparse
import secrets

# Add backend to sys.path to import license module
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BACKEND_DIR)

from core.licensing import generate_checksum, verify_license_key

def generate_key(tier: str = "STD") -> str:
    """Generate a single authentic license key."""
    tier = tier.upper()
    seed = secrets.token_hex(4).upper() # 8 hex characters e.g. 'A4F2C910'
    checksum = generate_checksum(tier, seed)
    key = f"AIDA-{tier}-{seed}-{checksum}"
    
    # Self-validation check
    check = verify_license_key(key)
    assert check["valid"], f"Generated invalid key: {key}"
    return key

def main():
    parser = argparse.ArgumentParser(description="Generate commercial license keys for AI Document Assistant.")
    parser.add_argument("--count", type=int, default=10, help="Number of license keys to generate (default: 10)")
    parser.add_argument("--tier", choices=["STD", "PRO", "ENT"], default="STD", help="License tier: STD (Standard), PRO (Professional), ENT (Enterprise)")
    parser.add_argument("--output", type=str, default=None, help="Optional output text file path")

    args = parser.parse_args()

    keys = [generate_key(args.tier) for _ in range(args.count)]

    print(f"\n=======================================================")
    print(f" Generated {len(keys)} {args.tier} Commercial License Keys")
    print(f" Ready to import into Gumroad / LemonSqueezy / Stripe")
    print(f"=======================================================\n")

    for i, k in enumerate(keys, 1):
        print(f"  {i:02d}. {k}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for k in keys:
                f.write(f"{k}\n")
        print(f"\n[Success] Keys exported to: {os.path.abspath(args.output)}")
    else:
        print("\nTip: Run with --output keys.txt to save directly to a file.")

if __name__ == "__main__":
    main()
