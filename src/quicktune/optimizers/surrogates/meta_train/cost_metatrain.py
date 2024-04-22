import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from quicktune.data import DataLoader, MetaSet


def cost_meta_train(model: nn.Module, dataset: MetaSet, config: dict):
    meta = config
    lr = meta.get("lr", 1e-3)
    train_iter = meta.get("train_iter", 1000)
    val_iter = meta.get("val_iter", 50)
    val_freq = meta.get("val_freq", 20)
    batch_size = meta.get("batch_size", 32)
    with_scheduler = meta.get("with_scheduler", False)

    loader = DataLoader(dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        CosineAnnealingLR(optimizer, train_iter, eta_min=1e-7)
        if with_scheduler
        else None
    )
    criterion = torch.nn.MSELoss()
    for i in range(train_iter):
        optimizer.zero_grad()
        batch = loader.get_batch()
        args = batch["args"].to(device)
        curves = batch["curves"].to(device)
        targets = batch["targets"].to(device)
        budgets = batch["budgets"].to(device)
        metafeatures = batch["metafeatures"].to(device)

        logits = model.cost_predictor(args, budgets, curves, metafeatures)
        loss = criterion(logits.reshape(targets.shape), targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if not i % val_freq:
            model.eval()
            val_loss = validate(model, val_iter, loader, device)
            print(f"Step {i}: Validation loss: {val_loss}")
            model.train()

    return model


def validate(model, val_iter, loader, device):
    val_loss = 0
    loss_fn = torch.nn.MSELoss()
    for _ in range(val_iter):
        batch = loader.get_batch(mode="val")
        args = batch["args"].to(device)
        curves = batch["curves"].to(device)
        targets = batch["targets"].to(device)
        budgets = batch["budgets"].to(device)
        metafeatures = batch["metafeatures"].to(device)

        logits = model.cost_predictor(args, budgets, curves, metafeatures)
        loss = loss_fn(logits.reshape(targets.shape), targets)
        val_loss += loss.item()

    val_loss /= val_iter
    return val_loss
