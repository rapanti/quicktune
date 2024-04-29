import os

import torch
from quicktune.optimizers.surrogates.dyhpo import DyHPO
from quicktune.config_manager import ConfigManager
from quicktune.data import MetaSet
from quicktune.optimizers.meta import CostMetaTrainer, PerfMetaTrainer

config = {
    "feature_extractor": {
        "in_features": 71,
        "enc_slice_ranges": [24],
        "enc_num_layers": 2,
    },
    "cost_predictor": {"in_features": 71},
}


if __name__ == "__main__":
    version = "mini"
    cs_path = "/home/evilknivl/projects/quicktune/src/quicktune/config_manager/.configs/mtlbm.json"
    cm = ConfigManager.from_json_file(cs_path)
    meta_path = "/home/evilknivl/projects/quicktune/src/quicktune/data/.mtlbm"
    metaset = MetaSet(meta_path, version, cm)
    dyhpo = DyHPO(config)

    dyhpo = PerfMetaTrainer(cache_dir=version, use_scheduler=True).train(dyhpo, metaset)
    dyhpo.cost_predictor = CostMetaTrainer(cache_dir=version).train(
        dyhpo.cost_predictor, metaset
    )
    path = os.path.join(version, "dyhpo.pt")
    torch.save(dyhpo.state_dict(), path)
