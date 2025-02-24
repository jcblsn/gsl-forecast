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
    with pytest.raises(ConfigurationError) as exc:
        load_configuration("nonexistent.json")
    assert "Config file not found" in str(exc.value)


def test_invalid_json_config(tmp_path):
    config_file = tmp_path / "bad_config.json"
    config_file.write_text("{ invalid json")

    with pytest.raises(ConfigurationError) as exc:
        load_configuration(os.path.join(tmp_path, "bad_config.json"))
    assert "Invalid JSON" in str(exc.value)
