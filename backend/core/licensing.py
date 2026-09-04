import os
import json
import hashlib
from typing import Dict, Any, Optional
from core.config import settings

LICENSE_SECRET = "AIDA_OFFLINE_COMMERCIAL_SECRET_2026_V98Z"
LICENSE_FILE_PATH = os.path.join(settings.DATA_DIR, "license.json")

def generate_checksum(tier: str, seed: str) -> str:
    """Generate deterministic HMAC-like checksum for license verification."""
    raw = f"{tier}:{seed}:{LICENSE_SECRET}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:6].upper()

def verify_license_key(key: str) -> Dict[str, Any]:
    """
    Validates an offline license key format:
    Format: AIDA-[TIER]-[SEED]-[CHECKSUM]
    Example: AIDA-STD-9F3B2A1C-7E4A19
    """
    if not key or not isinstance(key, str):
        return {"valid": False, "tier": None, "message": "License key required."}
    
    clean_key = key.strip().upper().replace(" ", "")
    parts = clean_key.split("-")
    
    if len(parts) != 4 or parts[0] != "AIDA":
        return {"valid": False, "tier": None, "message": "Invalid license key format (Expected AIDA-XXXX-XXXX-XXXX)."}
    
    tier, seed, checksum = parts[1], parts[2], parts[3]
    
    if tier not in ["STD", "PRO", "ENT"]:
        return {"valid": False, "tier": None, "message": "Unrecognized license tier."}
    
    expected_checksum = generate_checksum(tier, seed)
    if checksum != expected_checksum:
        return {"valid": False, "tier": None, "message": "License signature mismatch or corrupted key."}
    
    tier_names = {
        "STD": "Standard Edition (Lifetime)",
        "PRO": "Professional Edition (Lifetime)",
        "ENT": "Enterprise Commercial License"
    }
    
    return {
        "valid": True,
        "tier": tier,
        "tier_name": tier_names.get(tier, "Commercial"),
        "key": clean_key,
        "message": f"Successfully activated {tier_names.get(tier)}!"
    }

TRIAL_USAGE_FILE_PATH = os.path.join(settings.DATA_DIR, "trial_usage.json")
MAX_TRIAL_DOCS = 1
MAX_TRIAL_QUERIES = 3

def get_trial_usage() -> Dict[str, int]:
    """Retrieve number of queries used during trial evaluation."""
    if os.path.exists(TRIAL_USAGE_FILE_PATH):
        try:
            with open(TRIAL_USAGE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"queries_used": int(data.get("queries_used", 0))}
        except Exception:
            pass
    return {"queries_used": 0}

def record_trial_query() -> bool:
    """
    Increment trial query counter. Returns False if limit exceeded.
    """
    license_status = get_current_license()
    if license_status.get("is_licensed"):
        return True # Unlimited for licensed users

    usage = get_trial_usage()
    if usage["queries_used"] >= MAX_TRIAL_QUERIES:
        return False

    usage["queries_used"] += 1
    os.makedirs(os.path.dirname(TRIAL_USAGE_FILE_PATH), exist_ok=True)
    try:
        with open(TRIAL_USAGE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2)
    except Exception:
        pass
    return True

def can_upload_doc(current_doc_count: int) -> bool:
    """Enforces 1 document trial limit and blocks uploads if trial is expired."""
    license_status = get_current_license()
    if license_status.get("is_licensed"):
        return True
    if license_status.get("is_trial_locked"):
        return False
    return current_doc_count < MAX_TRIAL_DOCS

def get_current_license() -> Dict[str, Any]:
    """Reads the stored license status or returns trial defaults."""
    trial_info = get_trial_usage()
    queries_used = trial_info.get("queries_used", 0)

    if os.path.exists(LICENSE_FILE_PATH):
        try:
            with open(LICENSE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                key = data.get("license_key", "")
                result = verify_license_key(key)
                if result["valid"]:
                    return {
                        "is_licensed": True,
                        "tier": result["tier"],
                        "tier_name": result["tier_name"],
                        "license_key": key[:9] + "..." + key[-4:],
                        "registered_to": data.get("registered_to", "Registered Owner"),
                        "trial_queries_used": queries_used,
                        "trial_queries_max": MAX_TRIAL_QUERIES,
                        "trial_docs_max": MAX_TRIAL_DOCS
                    }
        except Exception:
            pass
            
    return {
        "is_licensed": False,
        "tier": "TRIAL",
        "tier_name": "Evaluation Trial",
        "license_key": None,
        "registered_to": None,
        "trial_queries_used": queries_used,
        "trial_queries_max": MAX_TRIAL_QUERIES,
        "trial_queries_remaining": max(0, MAX_TRIAL_QUERIES - queries_used),
        "trial_docs_max": MAX_TRIAL_DOCS,
        "is_trial_locked": queries_used >= MAX_TRIAL_QUERIES
    }

def save_license(key: str, registered_to: str = "Verified Customer") -> Dict[str, Any]:
    """Validates and persists a license key."""
    verification = verify_license_key(key)
    if not verification["valid"]:
        return verification
        
    os.makedirs(os.path.dirname(LICENSE_FILE_PATH), exist_ok=True)
    with open(LICENSE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "license_key": verification["key"],
            "tier": verification["tier"],
            "registered_to": registered_to
        }, f, indent=2)
        
    return verification
