"""Ingestion settings from environment variables (loads repo .env without overriding env)."""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Populate unset environment variables from repo-root .env if present."""
    env_path = Path(__file__).resolve().parents[4] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://aegisflow:aegisflow@localhost:5432/aegisflow"
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)

SERVICE_KEY = os.getenv(
    "SERVICE_KEY",
    "dev-service-key"
)

