"""
SMISIA — Seguridad API
Middleware de API key.
"""

import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Verifica API key en header X-API-Key."""

    def __init__(self, app, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key or os.getenv("SMISIA_API_KEY", "smisia-dev-key-2026")

    async def dispatch(self, request: Request, call_next):
        # Permitir docs y health sin auth
        path = request.url.path
        if path in ["/docs", "/openapi.json", "/redoc", "/metrics", "/"]:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if api_key != self.api_key:
            raise HTTPException(
                status_code=401,
                detail="API key inválida o faltante. Use header X-API-Key.",
            )

        return await call_next(request)
