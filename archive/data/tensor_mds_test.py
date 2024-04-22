import datetime
import os

# from quicktune.bo.quicktune import QuickTune
from quicktune.data.tensormetadataset import TensorMetaDataset
from quicktune.meta_train.cost_metatrain import cost_meta_train
from quicktune.meta_train.perf_mt_test import perf_meta_train
from quicktune.optimizers.quickoptimizer import QuickTuneOptimizer
from quicktune.optimizers.surrogates import DyHPO
from quicktune.tools.searchspace import SearchSpace

config = {
    "meta-dataset": {
        "root": "/home/evilknivl/projects/edit-qt-mds/mtlbm-metadataset",
        "version": "mini",
        "standardize_num_args": True,
        "model_args_first": True,
        "load_only_dataset_descriptors": True,
    },
    "surrogate_config": {
        "feature_extractor": {
            "hidden_dim": 128,
            "output_dim": 128,
            "in_curves_dim": 1,
            "out_curves_dim": 128,
            "in_metafeatures_dim": 4,
            "out_metafeatures_dim": 16,
            "encoder_num_layers": 2,
            "encoder_dim_ranges": (24, 69),
        },
        "cost_predictor": {
            "hidden_dim": 128,
            "output_dim": 1,
            "in_curves_dim": 1,
            "out_curves_dim": 128,
            "in_metafeatures_dim": 4,
            "out_metafeatures_dim": 16,
            "encoder_num_layers": 2,
            "encoder_dim_ranges": (24, 69),
        },
        "include_metafeatures": True,
        "cost_aware": True,
        "lr": 1e-3,
    },
}

if __name__ == "__main__":
    date_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    output_path = f"experiments/{date_time}"
    os.makedirs(output_path, exist_ok=True)

    md = TensorMetaDataset(
        root="/home/evilknivl/projects/edit-qt-mds/mtlbm-metadataset",
        version= "mini",
        curve= "eval_top1",
    )
    print(md.get_hyperparameters_names())

    dyhpo = DyHPO(config["surrogate_config"], cost_aware=True)
    metatrain_config = {
        "lr": 1e-3,
        "train_iter": 1000,
        "val_iter": 50,
        "val_freq": 100,
        "batch_size": 32,
        "with_scheduler": True,
    }
    dyhpo = perf_meta_train(dyhpo, md, metatrain_config)
    dyhpo = cost_meta_train(dyhpo, md, metatrain_config)

    ss = SearchSpace("v6")
    task_info = {
        "dataset": "imagenette",
        "train_split": "train",
        "val_split": "val",
        "num_samples": 3200,
        "num_classes": 10,
        "image_size": 320,
        "num_channels": 3,
    }

    hp_names = md.get_hyperparameters_names()
    candidates = md.get_hp_candidates()
    num_hps = md.get_num_hyperparameters()

    qto = QuickTuneOptimizer(
        surrogate=dyhpo,
        hp_candidates=candidates,
        hp_names=hp_names,
        log_indicator=[False for _ in range(num_hps)],
    )

    data_path = "/home/evilknivl/projects/datasets/imagenetest"
    # bo = QuickTune(
    #     task_info=task_info,
    #     optimizer=qto,
    #     searchspace=ss,
    #     metadataset=md,
    #     data_path=data_path,
    #     output_dir=output_path,
    # )

    # total_budget = 4 * 60 * 60
    # bo.run(total_budget)
