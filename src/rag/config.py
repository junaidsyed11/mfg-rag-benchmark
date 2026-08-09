import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]  # repo root


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else _ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_api_key(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing environment variable: {name}\n"
            f"Copy .env.example to .env and fill in your keys."
        )
    return value
