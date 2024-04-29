import os
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from quicktune.data import MetaLoader, MetaSet
# from quicktune.optimizers.surrogates import Surrogate


class MetaTrainer:
    batch_size: int = 32
    lr: float = 1e-4
    train_iter: int = 10000
    val_iter: int = 10
    val_freq: int = 20
    use_scheduler: bool = False
    use_cuda: bool = True
    cache_dir: str = ".cache/meta"
    ckpt_name: str = "model.pt"

    def __init__(self, config: Optional[dict] = None, **kwargs):
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        os.makedirs(self.cache_dir, exist_ok=True)
        self.save_path = os.path.join(self.cache_dir, self.ckpt_name)

    def train(self):
        raise NotImplementedError


class PerfMetaTrainer(MetaTrainer):
    """
    A class for training a performance meta-model.

    Args:
        MetaTrainer: The base class for meta-trainers.

    Attributes:
        device (str): The device to use for training.
        batch_size (int): The batch size for training.
        lr (float): The learning rate for the optimizer.
        use_scheduler (bool): Whether to use a learning rate scheduler.
        train_iter (int): The number of training iterations.
        val_freq (int): The frequency of validation during training.
        val_iter (int): The number of validation iterations.
        save_path (str): The path to save the trained model.

    Methods:
        train(model, dataset): Trains the performance meta-model.
        validate(model, loader): Validates the performance meta-model.

    """

    def train(self, model: nn.Module, dataset: MetaSet):
        """
        Trains the surrogate on the performance predictions.

        Args:
            model (Surrogate): The performance meta-model.
            dataset (MetaSet): The dataset for training.

        Returns:
            Surrogate: The trained performance meta-model.

        """
        device = self.device

        loader = MetaLoader(dataset, self.batch_size)

        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, self.train_iter, eta_min=1e-7)

        min_eval_val = float("inf")
        for i in range(self.train_iter):
            batch = loader.get_batch()
            config = batch["config"].to(device)
            budget = batch["budget"].to(device)
            curve = batch["curve"].to(device)
            target = batch["target"].to(device)
            metafeat = batch["metafeat"].to(device)

            loss = model.train_step(config, budget, curve, target, metafeat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step() if scheduler is not None else None

            if not i % self.val_freq:
                model.eval()
                val_error = self.validate(model, loader)
                print(f"Step {i}: Validation error: {val_error}")

                # Save model
                if val_error < min_eval_val:
                    min_eval_val = val_error
                    torch.save(model, self.save_path)
                model.train()

        # Load the model with the best validation error
        model = torch.load(self.save_path)
        return model

    @torch.no_grad()
    def validate(self, model, loader):
        """
        Validates the performance meta-model.

        Args:
            model: The performance meta-model.
            loader: The data loader for validation.

        Returns:
            float: The validation error.

        """
        device = self.device
        val_error = 0
        loss_fn = torch.nn.MSELoss()
        for _ in range(self.val_iter):
            batch_train = loader.get_batch(mode="val")
            batch_test = loader.get_batch(mode="val")

            for key, item in batch_train.items():
                batch_train[key] = item.to(device)
            for key, item in batch_test.items():
                batch_test[key] = item.to(device)
            
            target = batch_test.pop("target")

            means, _, _ = model.predict_pipeline(batch_train, batch_test)
            loss = loss_fn(means, target)

            val_error += loss.item()
        
        val_error /= self.val_iter
        return val_error


class CostMetaTrainer(MetaTrainer):
    """
    A class representing a meta-trainer for cost prediction models.

    Attributes:
        ckpt_name (str): The name of the checkpoint file to save the trained model.
    """

    ckpt_name: str = "cost_predictor.pt"

    def train(self, model: nn.Module, dataset: MetaSet):
        """
        Trains the cost prediction model using the provided dataset.

        Args:
            model (Surrogate): The cost predictor to train.
            dataset (MetaSet): The dataset containing all the training data.

        Returns:
            Surrogate: The trained cost predictor.
        """
        device = self.device
        model.to(device)
        model.train()

        loader = MetaLoader(dataset, self.batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, self.train_iter, eta_min=1e-7)

        criterion = torch.nn.MSELoss()

        min_eval_val = float("inf")
        for i in range(self.train_iter):
            batch = loader.get_batch(metric="time")
            for key, item in batch.items():
                batch[key] = item.to(device)
            target = batch.pop("target")

            logits = model(**batch)
            loss = criterion(logits.reshape(target.shape), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step() if scheduler is not None else None

            if not i % self.val_freq:
                model.eval()

                val_error = self.validate(model, device, loader)
                print(f"Step {i}: Validation error: {val_error}")

                if val_error < min_eval_val:
                    min_eval_val = val_error
                    torch.save(model, self.save_path)
                model.train()
        
        # Load the model with the best validation error
        model = torch.load(self.save_path)
        return model
    
    @torch.no_grad()
    def validate(self, model, device, loader):
        val_error = 0
        criterion = torch.nn.MSELoss()
        for _ in range(self.val_iter):
            batch = loader.get_batch(mode="val", metric="time")
            for key, item in batch.items():
                batch[key] = item.to(device)
            target = batch.pop("target")

            logits = model(**batch)
            loss = criterion(logits.reshape(target.shape), target)
            val_error += loss.item()

        val_error /= self.val_iter
        return val_error
