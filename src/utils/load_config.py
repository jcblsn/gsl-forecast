import json
import os
from typing import Dict


class ConfigurationError(Exception):
    pass


def load_configuration(file_path: str = None) -> Dict:
    if file_path is None:
        file_path = os.path.join(os.getcwd(), "config", "config.json")

    if not os.path.exists(file_path):
        raise ConfigurationError(f"Config file not found: {file_path}")

    try:
        with open(file_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file: {str(e)}") from e

    return config
