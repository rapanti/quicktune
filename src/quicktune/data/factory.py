import os
from .metaset import MetaSet
from ..config_manager import ConfigManager

def get_metaset(config: dict, config_manager: ConfigManager) -> MetaSet:
    root = config.get("root")
    assert root is not None, "No root specified in config"

    match root:
        case "*mtlbm*":
            file_path = os.path.dirname(__file__)
            default_root = os.path.join(file_path, ".mtlbm")
            config["root"] = default_root
            return MetaSet(**config, config_manager=config_manager)
        case _:
            assert os.path.isdir(root), f"Directory not found: {root}"
            return MetaSet(**config, config_manager=config_manager)