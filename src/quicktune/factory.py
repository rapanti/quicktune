from typing import Dict

from .config_manager import get_config_manager
from .data import get_metaset
from .optimizers import get_qt_optimizer


def setup_quicktune(config: Dict):
    
    config_manager_cfg: dict = config["configuration-manager"]
    config_manager = get_config_manager(config_manager_cfg)

    metaset_cfg: dict = config["metaset"]
    metaset = get_metaset(metaset_cfg, config_manager)

    optimizer_cfg: dict = config["optimizer"]
    optimizer = get_qt_optimizer(optimizer_cfg, config_manager, metaset)

    return optimizer
