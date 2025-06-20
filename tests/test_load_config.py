import json
import os

import pytest

from src.utils.load_config import ConfigurationError, load_configuration


def test_load_valid_config(tmp_path):
    config_data = {
        "database": {"path": "/path/to/db"},
        "storage": {"local": {"path": "/path/to/storage"}},
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    result = load_configuration(os.path.join(tmp_path, "config.json"))
    assert result == config_data


def test_missing_config_file():
    with pytest.raises(ConfigurationError):
        load_configuration("nonexistent.json")
