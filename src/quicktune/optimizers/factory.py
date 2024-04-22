from ..config_manager import ConfigManager
from ..data import MetaSet
from .quickoptimizer import QuickTuneOptimizer
from .surrogates import get_surrogate


def get_qt_optimizer(
    config: dict,
    config_manager: ConfigManager,
    metaset: MetaSet,
) -> QuickTuneOptimizer:
    """
    Returns a QuickTuneOptimizer object based on the provided configuration.

    Args:
        config (dict): The configuration dictionary.
        config_manager (ConfigManager): The configuration manager object.
        metaset (MetaSet): The metaset object.

    Returns:
        QuickTuneOptimizer: The QuickTuneOptimizer object.

    """
    surrogate_config: dict = config["surrogate"]
    surrogate = get_surrogate(surrogate_config, metaset)
    num_configs = config["num_configs"]
    optimizer = QuickTuneOptimizer(surrogate, config_manager, metaset, num_configs)
    return optimizer
