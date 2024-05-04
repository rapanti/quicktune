import os
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json
from quicktune.configuration.space import MetaSpace
from quicktune.optimizers.surrogates.dyhpo import DyHPO
from quicktune.data.metaset import MetaSet
from quicktune.optimizers.surrogates.metatrain import meta_train_surrogate

config = {
    "extractor": {
        "in_features": 71,
        "enc_slice_ranges": [45],
        "enc_num_layers": 2,
    },
    "predictor": {"in_features": 71},
}

fe_cfg = {
    "in_features": 71,
    "enc_slice_ranges": [45],
    "enc_num_layers": 2,
}

cp_cfg = {
    "in_features": 71,
}

if __name__ == "__main__":
    cspath = "/home/evilknivl/projects/quicktune/src/quicktune/configuration/.configs/mtlbm.json"
    cs: ConfigurationSpace = cs_json.read(open(cspath, "r").read())
    space = MetaSpace(cs)
    version = "extended"
    meta_path = "/home/evilknivl/projects/quicktune/src/quicktune/data/.mtlbm"
    metaset = MetaSet(meta_path, version, space)
    # metaset.save_standardization_csv(

    #         "/home/evilknivl/projects/quicktune/pretrained/micro",
    # )
    dyhpo = DyHPO(
        fe_cfg,
        cp_cfg,
    )

    dyhpo = meta_train_surrogate(dyhpo, metaset, metric="perf")
    dyhpo = meta_train_surrogate(dyhpo, metaset, metric="cost")

    dyhpo.save_checkpoint(os.path.join(version, "dyhpo.pth"))
