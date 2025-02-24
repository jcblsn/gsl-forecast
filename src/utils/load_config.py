import json
import os
from typing import Dict


class ConfigurationError(Exception):
    pass


def load_configuration(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        raise ConfigurationError(f"Config file not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {str(e)}") from e

    # validate required paths
    if "database" not in config or "path" not in config["database"]:
        raise ConfigurationError("Missing required config: database/path")

    if (
        "storage" not in config
        or "local" not in config["storage"]
        or "path" not in config["storage"]["local"]
    ):
        raise ConfigurationError("Missing required config: storage/local/path")

    return config
