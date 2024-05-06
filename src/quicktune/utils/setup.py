import os

from ConfigSpace.read_and_write import json as cs_json

from quicktune.configuration import ConfigManager, MetaSpace
from quicktune.data import MetaSet
from quicktune.optimizers.quick import QuickOptimizer
from quicktune.optimizers.surrogates.dyhpo import DyHPO

PRTRND = "../pretrained/mtlbm"
MTLBM = "../data/.mtlbm"


def from_pretrained(path_or_name: str, num_configs: int = 128):
    """
    Load a pretrained optimizer from .

    Parameters
    ----------
    path_or_name : str
        The path to the pretrained model or the name of the pretrained model.
    num_configs : int, default = 128
        The number of candidate configurations to generate.

    Returns
    -------
    QuickOptimizer
    """
    if path_or_name.startswith("*mtlbm"):
        _, version = path_or_name.split("/")
        file_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(file_path, "..", "pretrained", "mtlbm", "config.json")
        model_path = os.path.join(file_path, PRTRND, "surrogate", version)
        metaset_path = os.path.join(file_path, MTLBM)
    else:
        raise ValueError(f"Invalid path or name: {path_or_name}")

    config = cs_json.read(open(config_path, "r").read())
    space = MetaSpace(config)
    metaset = MetaSet(metaset_path, version, space)
    manager = ConfigManager(space, metaset)
    dyhpo = DyHPO.from_pretrained(model_path)

    optimizer = QuickOptimizer(dyhpo, manager, num_configs)
    return optimizer
