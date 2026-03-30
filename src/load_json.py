import json
import logging
from pathlib import Path
from typing import Any


def load_json(
        json_path: str |
        Path, logger: logging.Logger
) -> dict[str, Any]:
    path = Path(json_path)
    logger.info(f"Loading parameters from: '{path}'")

    if not path.exists():
        logger.error(f"JSON file does not exist: '{path}'")
        raise FileNotFoundError

    with path.open() as f:
        d = json.load(f)

    logger.info(f"JSON succesfully loaded from: '{path}'")

    return d
