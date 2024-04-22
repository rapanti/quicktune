import random
from typing import Optional

from sklearn.model_selection import train_test_split
import torch

from .metaset import MetaSet


class DataLoader:
    def __init__(
        self,
        dataset: MetaSet,
        batch_size: int,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

        datasets = self.dataset.get_datasets()
        self.train_split, self.val_split = train_test_split(
            datasets, test_size=0.2, random_state=seed
        )

    def get_batch(
        self, mode: str = "train", metric: str = "eval_top1"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "val"], f"Unknown mode: {mode}"

        if mode == "train":
            name = random.choice(self.train_split)
            return self.dataset.get_batch(self.batch_size, metric, name)
        else:
            name = random.choice(self.val_split)
            return self.dataset.get_batch(self.batch_size, metric, name)
