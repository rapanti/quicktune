import datetime
import logging
import os

import torch
from quicktune import (
    BO,
    DyHPO,
    FeatureExtractor,
    MetaDataset,
    QuickTuneOptimizer,
    SearchSpace,
)

if __name__ == "__main__":
    date_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    output_path = f".../experiments/{date_time}"
    os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(
        filename=f"{output_path}/exp.log",
        level=logging.INFO,
    )

    metadataset_path = "/path/to/qt_metadataset/"
    metadataset = MetaDataset(version="mini", path=metadataset_path)
    metadataset.set_dataset_name(metadataset.get_datasets()[0])

    feature_extractor = FeatureExtractor(
        input_dim_hps=70,
        output_dim=32,
        input_dim_curves=1,
        hidden_dim=32,
        output_dim_metafeatures=4,
        input_dim_metafeatures=4,
        encoder_dim_ranges=[(0, 23), (23, 69)],
    ).cuda()

    dyhpo = DyHPO(
        feature_extractor=feature_extractor,
        device=torch.device("cuda"),
        output_dim=32,
        dataset_name="imagenette",
        output_path=output_path,
    )
    opt_path = "/path/to/optimizer/mini.pth.tar"
    dyhpo.load_checkpoint(checkpoint_file=opt_path)

    hp_candidates = metadataset.get_hyperparameters_candidates().values.tolist()
    log_indicator = [False for _ in range(len(hp_candidates))]
    qto = QuickTuneOptimizer(
        model=dyhpo,
        dataset_name="imagenette",
        hp_candidates=hp_candidates,
        log_indicator=log_indicator,
        hp_names=metadataset.get_hyperparameters_names(),
        acqf_fc="ei",
        max_benchmark_epochs=50,
        minimization=False,
        explore_factor=0,
        learning_rate=1e-4,
        device="cuda",
        output_path=output_path,
    )

    ss = SearchSpace("v6")

    task_info = {
        "dataset": "imagenette",
        "train_split": "train",
        "val_split": "val",
        "num_samples": 9469,
        "num_classes": 10,
        "image_size": 320,
        "num_channels": 3,
    }

    data_path = "/path/to/imagenette2-320"
    bo = BO(
        task_info=task_info,
        optimizer=qto,
        searchspace=ss,
        metadataset=metadataset,
        data_path=data_path,
        output_dir=output_path,
    )

    total_budget = 4 * 60 * 60
    bo.run(total_budget)
