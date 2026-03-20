import json
import os
from typing import Optional


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, "config", "config.json")
    with open(config_path) as f:
        return json.load(f)
