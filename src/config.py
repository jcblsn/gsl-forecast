import json
import os

DEFAULTS = {
    "forecasting": {
        "train_start": None,
        "horizon": 12,
        "experiment_db": "forecast_experiments.db",
        "output_dir": "./outputs",
        "headline_model": None,
        "issue_status": "experimental",
        "forecast_version": "prototype-v0",
        "cv": {"split": "development"},
    },
}


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, "config", "config.json")
    with open(config_path) as f:
        config = json.load(f)
    fc = {**DEFAULTS["forecasting"], **config.get("forecasting", {})}
    fc["cv"] = {**DEFAULTS["forecasting"]["cv"], **fc.get("cv", {})}
    config["forecasting"] = fc
    return config
