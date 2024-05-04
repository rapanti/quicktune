import json
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch

from quicktune.configuration.space import MetaSpace


NUM_HP = [
    "bss_reg",
    "clip_grad",
    "cotuning_reg",
    "cutmix",
    "decay_rate",
    "delta_reg",
    "drop",
    "layer_decay",
    "lr",
    "mixup",
    "mixup_prob",
    "momentum",
    "pct_to_freeze",
    "smoothing",
    "sp_reg",
    "warmup_lr",
    "weight_decay",
]


class MetaSet:
    def __init__(
        self,
        root: str,
        version: str,
        space: MetaSpace,
        standardize_num_hps: bool = True,
        load_only_dataset_descriptors=True,
        return_tensor: bool = True,
        sort_hp: bool = True,
    ):
        self.root = root
        self.version = version
        self.space = space
        self.standardize_num_hps = standardize_num_hps
        self.load_only_dataset_descriptors = load_only_dataset_descriptors
        self.return_tensor = return_tensor
        self.sort_hp = sort_hp

        self.path = os.path.join(self.root, self.version)

        self.configs_df = self._load_table()
        self.curve_metrics = ["perf", "cost"]
        self.curves = self._load_curves()
        self.metafeatures = self._load_meta()
        self.datasets, self.ds_to_exp_ids = self._get_info()

    def _load_table(self) -> pd.DataFrame:
        path = os.path.join(self.path, "table.csv")
        df = pd.read_csv(path, index_col=0)

        if self.standardize_num_hps:
            self.num_hp_mean = df[NUM_HP].mean()
            self.num_hp_std = df[NUM_HP].std()
            df[NUM_HP] = df[NUM_HP] - self.num_hp_mean / self.num_hp_std

        if self.sort_hp:
            encoding = self.space.get_hot_encoding()

            df = df.reindex(encoding, axis=1)

        df = df.astype(float)
        df.fillna(-1, inplace=True)
        return df

    def _load_curves(self):
        path = os.path.join(self.path, "curves")
        curves = {}
        for curve in self.curve_metrics:
            data = json.load(open(os.path.join(path, f"{curve}.json")))
            curves[curve] = data
        return curves

    def _load_meta(self) -> dict:
        path = os.path.join(self.path, "meta")
        if self.load_only_dataset_descriptors:
            path = os.path.join(path, "descriptors.json")
        else:
            path = os.path.join(path, "hessians.json")
        metafeatures = json.load(open(path))
        return metafeatures

    def _get_info(self) -> Tuple[List[str], dict[str, List[int]]]:
        datasets = list(self.metafeatures.keys())
        ds_exp_ids = {}
        curve = self.curve_metrics[0]
        for dataset in datasets:
            exp_ids = list(self.curves[curve][dataset].keys())
            exp_ids = list(map(int, exp_ids))
            ds_exp_ids[dataset] = exp_ids
        return datasets, ds_exp_ids

    def len_data(self):
        return len(self.configs_df)

    def get_batch(
        self,
        batch_size: int,
        metric: str = "perf",
        dataset: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        if dataset is None:
            dataset = random.choice(self.datasets)
        assert dataset in self.datasets, f"{dataset} not found in the MetaSet"
        assert metric in self.curve_metrics, f"{metric} not found in the MetaSet"

        exp_ids = random.sample(self.ds_to_exp_ids[dataset], batch_size)

        curve, target, budget = [], [], []

        for idx in exp_ids:
            crv = self.curves[metric][dataset][str(idx)]
            bdgt = random.randint(1, len(crv))
            crv = crv[:bdgt]
            trgt = crv[-1]
            crv = crv[:-1]
            crv = crv + [0] * (50 - len(crv))
            curve.append(crv)
            target.append(trgt)
            budget.append(bdgt)

        config = self.configs_df.loc[exp_ids].values
        metaftr = self.metafeatures[dataset]
        metafeat = [metaftr for _ in range(batch_size)]

        if self.return_tensor:
            config = torch.tensor(config, dtype=torch.float32)
            curve = torch.tensor(curve, dtype=torch.float32) / 100
            target = torch.tensor(target, dtype=torch.float32) / 100
            budget = torch.tensor(budget, dtype=torch.float32) / 50
            metafeat = torch.tensor(metafeat, dtype=torch.float32) / 10000

        batch = {
            "config": config,
            "curve": curve,
            "budget": budget,
            "target": target,
            "metafeat": metafeat,
        }
        return batch

    def get_hyperparameters_names(self) -> List[str]:
        return list(self.configs_df.columns)

    def get_num_hps(self) -> int:
        return len(self.configs_df.columns)

    def get_cat_models(self) -> int:
        models = [col for col in self.configs_df.columns if col.startswith("cat:model")]
        return len(models)

    def get_num_datasets(self) -> int:
        return len(self.datasets)

    def get_datasets(self) -> List[str]:
        return self.datasets

    def get_hp_candidates(self):
        return self.configs_df.values

    def save_standardization(self, path):
        norm = {
            "mean": self.num_hp_mean.to_dict(),
            "std": self.num_hp_std.to_dict(),
        }
        output_path = os.path.join(path, "standardization.json")
        with open(output_path, "w") as writer:
            writer.write(json.dumps(norm, indent=2, sort_keys=True))

    def save_standardization_csv(self, path):
        norm = {
            "mean": self.num_hp_mean.to_dict(),
            "std": self.num_hp_std.to_dict(),
        }
        output_path = os.path.join(path, "standardization.csv")
        pd.DataFrame(norm).to_csv(output_path)
