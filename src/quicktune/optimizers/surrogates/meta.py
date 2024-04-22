import os
from typing import Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from quicktune.data import DataLoader, MetaSet


class MetaTrainer:
    batch_size: int = 32
    lr: float = 1e-4
    train_iter: int = 1000
    val_iter: int = 50
    val_freq: int = 20
    use_scheduler: bool = False
    use_cuda: bool = True
    cache_dir: str = ".cache/meta"
    ckpt_name: str = "model.pt"

    def __init__(self, config: Optional[dict] = None):
        if config is not None:
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        os.makedirs(self.cache_dir, exist_ok=True)
        self.save_path = os.path.join(self.cache_dir, self.ckpt_name)

    def train(self):
        raise NotImplementedError


class PerfMetaTrainer(MetaTrainer):

    def train(self, model: nn.Module, dataset: MetaSet):
        device = self.device

        loader = DataLoader(dataset, self.batch_size)

        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, self.train_iter, eta_min=1e-7)

        min_eval_val = float("inf")
        for i in range(self.train_iter):
            batch = loader.get_batch()
            args = batch["args"].to(device)
            curves = batch["curves"].to(device)
            targets = batch["targets"].to(device)
            budgets = batch["budgets"].to(device)
            metafeatures = batch["metafeatures"].to(device)

            loss = model.train_step(args, targets, budgets, curves, metafeatures)

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

    def validate(self, model, loader):
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

            means, _, _ = model.predict_pipeline(batch_train, batch_test)
            loss = loss_fn(means, batch_test["targets"])
            val_error += loss.item()

        val_error /= self.val_iter
        return val_error


class CostMetaTrainer(MetaTrainer):
    ckpt_name: str = "cost_predictor.pt"

    def train(self, model: nn.Module, dataset: MetaSet):
        device = self.device
        model.to(device)
        model.train()

        loader = DataLoader(dataset, self.batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.use_scheduler:
            scheduler = CosineAnnealingLR(optimizer, self.train_iter, eta_min=1e-7)

        criterion = torch.nn.MSELoss()

        min_eval_val = float("inf")
        for i in range(self.train_iter):
            batch = loader.get_batch()
            args = batch["args"].to(device)
            curves = batch["curves"].to(device)
            targets = batch["targets"].to(device)
            budgets = batch["budgets"].to(device)
            metafeatures = batch["metafeatures"].to(device)

            logits = model(args, budgets, curves, metafeatures)
            loss = criterion(logits.reshape(targets.shape), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step() if scheduler is not None else None

            if not i % self.val_freq:
                model.eval()
                val_error = self.validate(model, loader)
                print(f"Step {i}: Validation error: {val_error}")

                if val_error < min_eval_val:
                    min_eval_val = val_error
                    torch.save(model, self.save_path)
                model.train()
        model = torch.load(self.save_path)
        return model

    def validate(self, model, loader):
        device = self.device
        val_loss = 0
        criterion = torch.nn.MSELoss()
        for _ in range(self.val_iter):
            batch = loader.get_batch(mode="val")
            args = batch["args"].to(device)
            curves = batch["curves"].to(device)
            targets = batch["targets"].to(device)
            budgets = batch["budgets"].to(device)
            metafeatures = batch["metafeatures"].to(device)

            logits = model(args, budgets, curves, metafeatures)
            loss = criterion(logits.reshape(targets.shape), targets)
            val_loss += loss.item()

        val_loss /= self.val_iter
        return val_loss
