from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
from core.licensing import get_current_license, save_license, verify_license_key, LICENSE_FILE_PATH
import os

router = APIRouter(prefix="/license", tags=["Licensing"])

class ActivateRequest(BaseModel):
    license_key: str
    registered_to: Optional[str] = "Verified Owner"

@router.get("/status")
def get_license_status():
    """Retrieve current offline license activation status."""
    return get_current_license()

@router.post("/activate")
def activate_software(payload: ActivateRequest):
    """Activate software offline using a cryptographic commercial license key."""
    result = save_license(payload.license_key, payload.registered_to or "Verified Owner")
    if not result.get("valid"):
        raise HTTPException(status_code=400, detail=result.get("message", "Invalid license key."))
    return result

@router.post("/deactivate")
def deactivate_software():
    """Remove current license and return to trial evaluation mode."""
    if os.path.exists(LICENSE_FILE_PATH):
        try:
            os.remove(LICENSE_FILE_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove license: {str(e)}")
    return {"message": "Software deactivated. Reverted to Evaluation Trial."}
