from typing import Optional
from fastapi import HTTPException, Header, Request
from core.config import settings

async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
    return response

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key and x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized API Request: Invalid Security Token")
