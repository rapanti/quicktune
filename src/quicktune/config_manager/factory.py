import os

from .manager import ConfigManager


def get_config_manager(config: dict) -> ConfigManager:
    path = config.get("path")
    configspace = config.get("configspace")

    if path is not None:
        assert isinstance(path, str), f"Invalid path: {path}"
        match path:
            case "*mtlbm*":
                file_path = os.path.dirname(__file__)
                config_path = os.path.join(file_path, ".configs", "mtlbm.json")
                return ConfigManager.from_json_file(config_path)
            case _:
                assert os.path.isfile(path), f"File not found: {path}"
                return ConfigManager.from_json_file(path)

    elif configspace is not None:
        return ConfigManager(configspace)

    else:
        raise ValueError("No path or configspace specified in config")
