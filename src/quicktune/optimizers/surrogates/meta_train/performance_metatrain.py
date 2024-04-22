import torch
import torch.nn as nn
from quicktune.data import DataLoader, MetaSet


def perf_meta_train(model: nn.Module, dataset: MetaSet, config: dict):
    meta = config
    lr = meta.get("lr", 1e-3)
    train_iter = meta.get("train_iter", 1000)
    val_iter = meta.get("val_iter", 50)
    val_freq = meta.get("val_freq", 20)
    batch_size = meta.get("batch_size", 32)
    with_scheduler = meta.get("with_scheduler", False)

    min_eval_val = float("inf")

    loader = DataLoader(dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if with_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, train_iter, eta_min=1e-7
        )

    for i in range(train_iter):
        optimizer.zero_grad()
        batch = loader.get_batch()
        args = batch["args"].to(device)
        curves = batch["curves"].to(device)
        targets = batch["targets"].to(device)
        budgets = batch["budgets"].to(device)
        metafeatures = batch["metafeatures"].to(device)

        loss = model.train_step(args, targets, budgets, curves, metafeatures)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if not i % val_freq:
            model.eval()
            val_error = validate(model, val_iter, loader, device)
            print(f"Step {i}: Validation error: {val_error}")

            if val_error < min_eval_val:
                min_eval_val = val_error
                torch.save(model, "best_model.pt")
            model.train()
    model = torch.load("best_model.pt")
    return model

def validate(model, val_iter, loader, device):
    val_loss = 0
    loss_fn = torch.nn.MSELoss()
    for _ in range(val_iter):
        batch_train = loader.get_batch(mode="val")
        batch_test = loader.get_batch(mode="val")

        for key, val in batch_train.items():
            batch_train[key] = val.to(device)
        for key, val in batch_test.items():
            batch_test[key] = val.to(device)

        means, _, _ = model.predict_pipeline(batch_train, batch_test)
        loss = loss_fn(means, batch_test["targets"])
        val_loss += loss.item()

    val_loss /= val_iter
    return val_loss
