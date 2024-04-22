import torch

from quicktune.data import MetaSet

from .dyhpo import DyHPO
from .meta import CostMetaTrainer, PerfMetaTrainer


def get_surrogate(config: dict, metaset: MetaSet) -> torch.nn.Module:
    if config.get("in_features") is None:
        num_hps = metaset.get_num_hps()
        config["feature_extractor"]["in_features"] = num_hps
        config["cost_predictor"]["in_features"] = num_hps

    surrogate = DyHPO(config)

    if config.get("meta-train", False):
        meta_train_config = config["meta-train-config"]
        surrogate = PerfMetaTrainer(meta_train_config).train(surrogate, metaset)
        surrogate.cost_predictor = CostMetaTrainer().train(
            surrogate.cost_predictor, metaset
        )

    if config.get("load_from_pretrained", False):
        path = config["pretrained_path"]
        state_dict = torch.load(path, map_location="cpu")
        msg = surrogate.load_state_dict(state_dict)
        print(f"Loaded model from {path} with message: {msg}")
        print(f"Loaded model from {path} with message: {msg}")
        print(f"Loaded model from {path} with message: {msg}")

    return surrogate
