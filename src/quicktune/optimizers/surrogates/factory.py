import os

import torch

from quicktune.data import MetaSet
from quicktune.optimizers.surrogates.surrogate import Surrogate

from ..meta import CostMetaTrainer, PerfMetaTrainer
from .dyhpo import DyHPO


def get_surrogate(config: dict, metaset: MetaSet) -> Surrogate:
    """
    Get the surrogate model based on the provided configuration and meta dataset.

    Args:
        config (dict): The configuration for the surrogate model.
        metaset (MetaSet): The meta dataset used for training the surrogate model.

    Returns:
        Surrogate: The instantiated surrogate model.

    """
    if config["feature_extractor"].get("in_features") == "auto":
        num_hps = metaset.get_num_hps()
        config["feature_extractor"]["in_features"] = num_hps
        config["cost_predictor"]["in_features"] = num_hps

    surrogate = DyHPO(config)

    if config.get("meta-train", False):
        meta_train_config = config["meta-train-config"]
        surrogate = PerfMetaTrainer(meta_train_config).train(surrogate, metaset)
        surrogate.cost_predictor = CostMetaTrainer(meta_train_config).train(
            surrogate.cost_predictor, metaset
        )

    elif config.get("load_from_pretrained", False):
        path = config["pretrained_path"]
        if path == "*mtlbm*":
            path = os.path.join("pretrained", "mtlbm", metaset.version, "dyhpo.pt")
        state_dict = torch.load(path, map_location="cpu")
        msg = surrogate.load_state_dict(state_dict)
        print(f"Loaded model from {path} with message: {msg}")

    return surrogate
