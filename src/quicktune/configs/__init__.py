from typing import Optional
import os
import yaml


def get_qt_config(
    data_path: str, budget: int, config_path: Optional[str] = None
) -> dict:
    if config_path is None:
        _default_config = "cv/classification/default.yaml"
        config_path = os.path.join(os.path.dirname(__file__), _default_config)
    else:
        assert os.path.exists(
            config_path
        ), f"Configuration file {config_path} does not exist."

    with open(config_path, "r") as f:
        qt_config = yaml.safe_load(f)

    qt_config["experiment"]["data_path"] = data_path
    qt_config["experiment"]["budget"] = budget

    return qt_config
