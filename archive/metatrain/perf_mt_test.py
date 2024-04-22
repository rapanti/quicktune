import torch
from torch.utils.data import DataLoader, random_split
# from ..data import DataLoader


def perf_meta_train(model, dataset, config: dict):
    meta = config
    lr = meta.get("lr", 1e-3)
    train_iter = meta.get("train_iter", 1000)
    val_iter = meta.get("val_iter", 50)
    val_freq = meta.get("val_freq", 20)
    batch_size = meta.get("batch_size", 32)
    with_scheduler = meta.get("with_scheduler", False)

    train_split, val_split = random_split(dataset, (2/3, 1/3))
    train_loader = DataLoader(train_split, batch_size)
    val_loader = DataLoader(val_split, batch_size)

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
        # batch = loader.get_batch()
        # args = batch["args"].to(device)
        # curves = batch["curves"].to(device)
        # targets = batch["targets"].to(device)
        # budgets = batch["budgets"].to(device)
        # metafeatures = batch["metafeatures"].to(device)
        batch = next(iter(train_loader))
        batch = [b.to(device) for b in batch]
        args, curves, targets, budgets, metafeatures = batch
        loss = model.train_step(args, targets, budgets, curves, metafeatures)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if not i % val_freq:
            model.eval()
            val_loss = validate(model, val_iter, val_loader, device)
            print(f"Step {i}: Validation loss: {val_loss}")
            model.train()

    return model


def validate(model, val_iter, loader, device):
    val_loss = 0
    loss_fn = torch.nn.MSELoss()
    for _ in range(val_iter):
        batch_train = next(iter(loader))
        batch_test = next(iter(loader))

        batch_train = [b.to(device) for b in batch_train]
        batch_test = [b.to(device) for b in batch_test]

        batch_train = dict(zip(["args", "curves", "targets", "budgets", "metafeatures"], batch_train))
        batch_test = dict(zip(["args", "curves", "targets", "budgets", "metafeatures"], batch_test))

        means, _, _ = model.predict_pipeline(batch_train, batch_test)
        loss = loss_fn(means, batch_test["targets"])
        val_loss += loss.item()

    val_loss /= val_iter
    return val_loss
