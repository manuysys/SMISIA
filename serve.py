"""
SMISIA — Servidor FastAPI
Script para levantar la API.
"""
import sys
import os
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config  # noqa: E402


def main():
    config = load_config()
    api_cfg = config.get("api", {})

    uvicorn.run(
        "src.api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
