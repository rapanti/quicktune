import json
from pathlib import Path
import random
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


NUM_ARGS_TO_STD = {
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
}


class TensorMetaDataset(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        version: str,
        curve: str,
        standardize_num_args: bool = True,
        model_args_first: bool = True,
        load_only_dataset_descriptors=True,
        return_tensor: bool = True,
    ):
        super().__init__()
        self.root = root
        self.version = version
        self.curve = curve
        self.standardize_num_args = standardize_num_args
        self.model_args_first = model_args_first
        self.load_only_dataset_descriptors = load_only_dataset_descriptors
        self.return_tensor = return_tensor

        self.path = Path(root) / version

        self.args_df = self._load_data()
        self.curve_names = ["eval_top1", "total_time"]
        self.curves = self._load_curves()
        self.metafeatures = self._load_metafeatures()
        self.datasets, self.ds_to_exp_ids = self._get_info()

    def _load_data(self) -> pd.DataFrame:
        path = self.path / "args" / "processed_args_table.csv"
        df = pd.read_csv(path, index_col=0)

        if self.standardize_num_args:
            mean = df.mean()
            std = df.std()
            for col in df.columns:
                if col in NUM_ARGS_TO_STD:
                    df[col] = (df[col] - mean[col]) / std[col]

        if self.model_args_first:
            model_args = [col for col in df.columns if col.startswith("cat__model")]
            others = [col for col in df.columns if col not in model_args]
            df = df[model_args + others]

        df = df.astype(float)
        df = df.fillna(-1)
        return df

    def _load_curves(self) -> dict:
        path = self.path / "curves"
        curves = {}
        for curve in self.curve_names:
            data = json.load(open(path / f"{curve}.json"))
            curves[curve] = data
        return curves

    def _load_metafeatures(self) -> dict:
        path = self.path / "meta-features"
        if self.load_only_dataset_descriptors:
            path /= "descriptors.json"
        else:
            path /= "hessians.json"
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
        return len(self.args_df)
    
    def __len__(self):
        return len(self.args_df)
    
    def __getitem__(self, idx):        
        dataset = random.choice(self.datasets)
        exp_ids = random.sample(self.ds_to_exp_ids[dataset], 1)

        def _get_curves():
            _targets = []
            _budgets = []
            _curves = []
            for exp in exp_ids:
                _curve = self.curves[self.curve][dataset][str(exp)]
                _budget = random.randint(1, len(_curve))
                _curve = _curve[:_budget]
                target = _curve[-1]
                _curve = _curve[:-1]
                _curve = _curve + [0] * (50 - len(_curve))
                _budgets.append(_budget)
                _curves.append(_curve)
                _targets.append(target)
            return _curves, _targets, _budgets

        args = self.args_df.loc[exp_ids].values
        curves, targets, budgets = _get_curves()
        metafeatures = self.metafeatures[dataset]

        args = torch.tensor(args, dtype=torch.float32).squeeze()
        curves = torch.tensor(curves, dtype=torch.float32).squeeze() / 100
        targets = torch.tensor(targets, dtype=torch.float32).squeeze() / 100
        budgets = torch.tensor(budgets, dtype=torch.float32).squeeze() / 50
        metafeatures = torch.tensor(metafeatures, dtype=torch.float32).squeeze() / 10000

        return args, curves, targets, budgets, metafeatures

    def get_hyperparameters_names(self) -> List[str]:
        return list(self.args_df.columns)

    def get_num_hyperparameters(self) -> int:
        return len(self.args_df.columns)

    def get_cat_models(self) -> int:
        models = [col for col in self.args_df.columns if col.startswith("cat_model")]
        return len(models)

    def get_num_datasets(self) -> int:
        return len(self.datasets)

    def get_datasets(self) -> List[str]:
        return self.datasets

    def get_hp_candidates(self):
        return self.args_df.values
