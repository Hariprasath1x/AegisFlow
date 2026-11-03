"""Main FastAPI application entry point for AegisFlow transaction ingestion."""

from __future__ import annotations

import json
import sys
from fastapi import FastAPI
from loguru import logger

from ingestion.api.routes import router


def _json_formatter(record: dict) -> str:
    """Format log records as structured JSON."""
    log_data = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "line": record["line"],
    }
    if record.get("extra"):
        log_data["extra"] = record["extra"]
    return json.dumps(log_data) + "\n"


# Configure Loguru structured JSON logging to stdout
logger.remove()
logger.add(sys.stdout, format=_json_formatter, level="INFO")

# Initialize FastAPI application
app = FastAPI(
    title="AegisFlow ingestion",
    version="0.2.0",
)

# Include API routes
app.include_router(router)
