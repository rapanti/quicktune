from ..configuration_manager import ConfigurationManager
from ..data import MetaSet
from .quicktune import QuickTuneOptimizer
from .surrogates import get_surrogate


def get_qt_optimizer(
    config: dict,
    config_manager: ConfigurationManager,
    metaset: MetaSet,
) -> QuickTuneOptimizer:
    surrogate_config: dict = config["surrogate"]
    surrogate = get_surrogate(surrogate_config, metaset)
    num_configs = config["num_configs"]
    optimizer = QuickTuneOptimizer(surrogate, config_manager, metaset, num_configs)
    return optimizer
