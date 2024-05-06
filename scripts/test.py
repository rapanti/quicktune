from ConfigSpace.read_and_write import json as cs_json

from quicktune import QuickTuner
from quicktune.data.metaset import MetaSet
from quicktune.optimizers.quick import QuickOptimizer
from quicktune.optimizers.surrogates.dyhpo import DyHPO
# from quicktune.tuners import QuickTuner
from quicktune.finetune.finetune_wrapper import eval_finetune_conf
from quicktune.configuration.manager import ConfigManager
from quicktune.configuration.space import MetaSpace

if __name__ == "__main__":
    dyhpo = DyHPO.from_pretrained("/home/evilknivl/projects/quicktune/pretrained/micro")

    config_ = cs_json.read(
        open(
            "/home/evilknivl/projects/quicktune/src/quicktune/configuration/.configs/mtlbm.json",
            "r",
        ).read()
    )
    space = MetaSpace(config_)
    metaset = MetaSet(
        "/home/evilknivl/projects/quicktune/src/quicktune/data/.mtlbm", "micro", space
    )
    manager = ConfigManager(space, metaset)

    optimizer = QuickOptimizer(dyhpo, manager, 128)
    qt = QuickTuner(optimizer, eval_finetune_conf)
    qt.fit("/home/evilknivl/projects/datasets/102flowers", time_limit=3600)
