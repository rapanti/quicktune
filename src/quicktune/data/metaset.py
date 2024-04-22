from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch

from ..configuration_manager import ConfigurationManager

NUM_HPS_TO_STD = [
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
        standardize_num_hps: bool = True,
        model_args_first: bool = True,
        load_only_dataset_descriptors=True,
        return_tensor: bool = True,
    ):
        self.root = root
        self.version = version
        self.standardize_num_hps = standardize_num_hps
        self.model_args_first = model_args_first
        self.load_only_dataset_descriptors = load_only_dataset_descriptors
        self.return_tensor = return_tensor

        self.path = os.path.join(self.root, self.version)

        self.configs_df = self._load_data()
        self.curve_names = ["eval_top1", "total_time"]
        self.curves = self._load_curves()
        self.metafeatures = self._load_metafeatures()
        self.datasets, self.ds_to_exp_ids = self._get_info()

    def _load_data(self) -> pd.DataFrame:
        path = os.path.join(self.path, "args", "preprocessed_args_table.csv")
        df = pd.read_csv(path, index_col=0)

        if self.standardize_num_hps:
            # TODO: implement this
            # NUM_ARGS_TO_STD = self.searchspace.get_num_args_to_std()
            self.num_args_mean = df[NUM_HPS_TO_STD].mean()
            self.num_args_std = df[NUM_HPS_TO_STD].std()
            df[NUM_HPS_TO_STD] = (
                df[NUM_HPS_TO_STD] - self.num_args_mean / self.num_args_std
            )

        if self.model_args_first:
            model_args = [col for col in df.columns if col.startswith("cat=model")]
            others = [col for col in df.columns if col not in model_args]
            df = df[model_args + others]

        df = df.astype(float)
        df = df.fillna(-1)
        return df

    def _load_curves(self) -> dict:
        path = os.path.join(self.path, "curves")
        curves = {}
        for curve in self.curve_names:
            data = json.load(open(os.path.join(path, f"{curve}.json")))
            curves[curve] = data
        return curves

    def _load_metafeatures(self) -> dict:
        path = os.path.join(self.path, "meta-features")
        if self.load_only_dataset_descriptors:
            path = os.path.join(path, "descriptors.json")
        else:
            path = os.path.join(path, "hessians.json")
        metafeatures = json.load(open(path))
        return metafeatures

    def _get_info(self) -> Tuple[List[str], dict[str, List[int]]]:
        datasets = list(self.metafeatures.keys())
        ds_exp_ids = {}
        curve = self.curve_names[0]
        for dataset in datasets:
            exp_ids = list(self.curves[curve][dataset].keys())
            exp_ids = list(map(int, exp_ids))
            ds_exp_ids[dataset] = exp_ids
        return datasets, ds_exp_ids

    def len_data(self):
        return len(self.configs_df)

    def get_batch(
        self, batch_size: int, curve: str = "eval_top1", dataset: Optional[str] = None
    ) -> dict[str, torch.Tensor]:
        if dataset is None:
            dataset = random.choice(self.datasets)
        else:
            assert dataset in self.datasets, f"{dataset} not found in the metadataset"
        assert curve in self.curve_names, f"{curve} not found in the metadataset"

        exp_ids = random.sample(self.ds_to_exp_ids[dataset], batch_size)

        def _get_curves():
            _targets = []
            _budgets = []
            _curves = []
            for exp in exp_ids:
                _curve = self.curves[curve][dataset][str(exp)]
                _budget = random.randint(1, len(_curve))
                _curve = _curve[:_budget]
                target = _curve[-1]
                _curve = _curve[:-1]
                _curve = _curve + [0] * (50 - len(_curve))
                _budgets.append(_budget)
                _curves.append(_curve)
                _targets.append(target)
            return _curves, _targets, _budgets

        args = self.configs_df.loc[exp_ids].values
        curves, targets, budgets = _get_curves()
        metafeatures = self.metafeatures[dataset]
        metafeatures = [metafeatures for _ in range(batch_size)]

        if self.return_tensor:
            args = torch.tensor(args, dtype=torch.float32)
            curves = torch.tensor(curves, dtype=torch.float32) / 100
            targets = torch.tensor(targets, dtype=torch.float32) / 100
            budgets = torch.tensor(budgets, dtype=torch.float32) / 50
            metafeatures = torch.tensor(metafeatures, dtype=torch.float32) / 10000

        out = {
            "args": args,
            "curves": curves,
            "targets": targets,
            "budgets": budgets,
            "metafeatures": metafeatures,
        }
        return out

    def get_hyperparameters_names(self) -> List[str]:
        return list(self.configs_df.columns)

    def get_num_hps(self) -> int:
        return len(self.configs_df.columns)

    def get_cat_models(self) -> int:
        models = [col for col in self.configs_df.columns if col.startswith("cat_model")]
        return len(models)

    def get_num_datasets(self) -> int:
        return len(self.datasets)

    def get_datasets(self) -> List[str]:
        return self.datasets

    def get_hp_candidates(self):
        return self.configs_df.values


if __name__ == "__main__":
    config = {
        "meta-dataset": {
            "root": "/home/evilknivl/projects/edit-qt-mds/mtlbm-metadataset",
            "version": "mini",
            "standardize_num_args": True,
            "model_args_first": True,
            "load_only_dataset_descriptors": True,
        }
    }
    md = MetaSet(**config["meta-dataset"])
