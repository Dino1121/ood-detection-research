import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.resnet18_cifar import ResNet18CIFAR


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
SPLIT_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_per_class(targets, count, seed):
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in sorted(set(targets)):
        indices = [i for i, y in enumerate(targets) if y == label]
        chosen = torch.randperm(len(indices), generator=generator)[:count]
        selected.extend(indices[i] for i in chosen.tolist())
    return sorted(selected)


def build_loaders(batch_size, workers, seed, output_dir):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.CIFAR10(
        root="data", train=True, download=False, transform=train_transform
    )
    train_clean_dataset = datasets.CIFAR10(
        root="data", train=True, download=False, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root="data", train=False, download=False, transform=eval_transform
    )

    # Keep data partitions identical across independent training seeds.
    train_clean_indices = select_per_class(
        train_clean_dataset.targets, 500, SPLIT_SEED
    )
    val_indices = select_per_class(
        test_dataset.targets, 100, SPLIT_SEED
    )
    test_indices = sorted(set(range(len(test_dataset))) - set(val_indices))

    split_path = output_dir / "split_indices.json"
    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        if (saved["train_clean_indices"] != train_clean_indices
                or saved["val_indices"] != val_indices):
            raise ValueError("Saved split indices do not match the current seed.")
    else:
        split_path.write_text(json.dumps({
            "train_clean_indices": train_clean_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "note": "Deterministic stratified split, NOT OpenOOD's exact published image list.",
        }, indent=2), encoding="utf-8")

    def loader(dataset, shuffle):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, pin_memory=True,
            persistent_workers=workers > 0,
        )

    return (
        loader(train_dataset, True),
        loader(Subset(train_clean_dataset, train_clean_indices), False),
        loader(Subset(test_dataset, val_indices), False),
    )


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        size = labels.size(0)
        running_loss += loss.item() * size
        correct += (logits.argmax(1) == labels).sum().item()
        total += size
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        size = labels.size(0)
        running_loss += loss.item() * size
        correct += (logits.argmax(1) == labels).sum().item()
        total += size
    return running_loss / total, correct / total


def plot_history(history_path, figure_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_aug_loss"]) for r in rows]
    train_clean_loss = [float(r["train_clean_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    train_clean_acc = [float(r["train_clean_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="Train (augmented)", linestyle="--")
    axes[0].plot(epochs, train_clean_loss, label="Train (clean subset)")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, train_clean_acc, label="Train (clean subset)")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")

    set_seed(args.seed)
    device = torch.device("cuda")
    save_dir = Path("checkpoints") / f"cifar10_resnet18_seed{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, clean_loader, val_loader = build_loaders(
        args.batch_size, args.workers, args.seed, save_dir
    )
    model = ResNet18CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9,
        weight_decay=5e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6
    )

    resume_path = save_dir / "last_train.pt"
    history_path = save_dir / "history.csv"
    best_path = save_dir / "best.pth"
    final_path = save_dir / "last.pth"
    figure_path = save_dir / "learning_curves.png"
    columns = ["epoch", "train_aug_loss", "train_aug_acc", "train_clean_loss",
               "train_clean_acc", "val_loss", "val_acc", "lr", "seconds"]

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        if (state["epochs"] != args.epochs
                or state["batch_size"] != args.batch_size
                or state["seed"] != args.seed):
            raise ValueError("Resume requires matching epochs, batch size and seed.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]
    else:
        with history_path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(columns)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train: {len(train_loader.dataset)}, clean train monitor: {len(clean_loader.dataset)}")
    print(f"Validation: {len(val_loader.dataset)} (no validation samples used for training)")

    for epoch in range(start_epoch, args.epochs):
        start = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        clean_loss, clean_acc = evaluate(model, clean_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        seconds = time.perf_counter() - start
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_aug_loss={train_loss:.4f} | "
            f"train_clean_loss={clean_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.2%} | {seconds:.1f}s", flush=True,
        )

        with history_path.open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow([
                epoch + 1, train_loss, train_acc, clean_loss, clean_acc,
                val_loss, val_acc, lr, seconds,
            ])
        plot_history(history_path, figure_path)

        torch.save({
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "best_val_loss": best_val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, resume_path)

    torch.save(model.state_dict(), final_path)
    print(f"Best model: {best_path}")
    print(f"Last model: {final_path}")
    print(f"Learning curves: {figure_path}")
    print("Final ID test set is intentionally not evaluated during training.")


if __name__ == "__main__":
    main()
