import datetime
import logging
import os

from quicktune import BO, MetaDataset, SearchSpace
from quicktune.optimizers.utils import create_qt_optimizer

if __name__ == "__main__":
    date_time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    output_path = f"path/to/experiments/{date_time}"
    os.makedirs(output_path, exist_ok=True)

    logging.basicConfig(
        filename=f"{output_path}/exp.log",
        level=logging.INFO,
    )

    metadataset_path = "path/to/qt_metadataset/"
    metadataset = MetaDataset(version="mini", path=metadataset_path)
    metadataset.set_dataset_name(metadataset.get_datasets()[0])

    qto = create_qt_optimizer(
        metadataset=metadataset,
        dataset_name=metadataset.dataset_name,
        device="cuda",
        acqf_fc="ei",
        minimization=False,
        explore_factor=0,
        learning_rate=1e-4,
        cost_aware=True,
        meta_train=True,
    )
    qto.dataset_name = "imagenette"
    qto.output_path = output_path
    
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

    data_path = "path/to/imagenette2-320"
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
