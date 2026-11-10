
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.resnet18_cifar import ResNet18CIFAR
from src.ood.methods import msp_score, energy_score


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

METHODS = {
    "msp": lambda logits, features, model: msp_score(logits),
    "energy": lambda logits, features, model: energy_score(
        logits, temperature=1.0
    ),
}


def load_model(checkpoint_dir, device):
    model = ResNet18CIFAR(num_classes=10).to(device)

    weights = torch.load(
        checkpoint_dir / "best.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(weights)
    model.eval()

    return model


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transform
    )


def load_splits(checkpoint_dir):
    with open(
        checkpoint_dir / "split_indices.json",
        encoding="utf-8"
    ) as f:
        splits = json.load(f)

    val_indices = splits["val_indices"]
    test_indices = splits["test_indices"]

    assert len(val_indices) == 1000
    assert len(test_indices) == 9000
    assert not set(val_indices).intersection(test_indices)

    return val_indices, test_indices


@torch.inference_mode()
def extract_scores(model, loader, device, methods):
    results = {
        "labels": [],
        "preds": [],
    }

    for name in methods:
        results[name] = []

    model.eval()

    for images, labels in loader:
        images = images.to(device)

        logits, features = model(
            images, return_features=True
        )

        results["labels"].append(labels.numpy())
        results["preds"].append(
            logits.argmax(dim=1).cpu().numpy()
        )

        for name, score_fn in methods.items():
            scores = score_fn(logits, features, model)
            results[name].append(scores.cpu().numpy())

    return {
        key: np.concatenate(values)
        for key, values in results.items()
    }


def save_scores(path, indices, results, method_names):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image_index",
            "label",
            "prediction",
            *method_names
        ])

        for i, index in enumerate(indices):
            writer.writerow([
                index,
                results["labels"][i],
                results["preds"][i],
                *[results[name][i] for name in method_names]
            ])


def evaluate_method(val_scores, test_scores):
    threshold = float(np.percentile(val_scores, 5))

    return {
        "threshold": threshold,
        "val_acceptance": float(
            np.mean(val_scores >= threshold)
        ),
        "test_acceptance": float(
            np.mean(test_scores >= threshold)
        ),
        "test_min": float(test_scores.min()),
        "test_median": float(np.median(test_scores)),
        "test_max": float(test_scores.max()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=["msp", "energy"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    method_names = list(dict.fromkeys(args.methods))
    selected_methods = {
        name: METHODS[name] for name in method_names
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_dir = (
        Path("checkpoints")
        / f"cifar10_resnet18_seed{args.seed}"
    )

    method_tag = "_".join(sorted(method_names))
    result_dir = (
        Path("results/week1")
        / f"seed{args.seed}"
        / method_tag
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Methods: {', '.join(method_names)}")

    model = load_model(checkpoint_dir, device)
    dataset = load_dataset()
    val_indices, test_indices = load_splits(checkpoint_dir)

    def make_loader(indices):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"
        )

    val_loader = make_loader(val_indices)
    test_loader = make_loader(test_indices)

    print("Extracting ID validation scores...")
    val_results = extract_scores(
        model, val_loader, device, selected_methods
    )

    print("Extracting ID test scores...")
    test_results = extract_scores(
        model, test_loader, device, selected_methods
    )

    save_scores(
        result_dir / "id_val_scores.csv",
        val_indices,
        val_results,
        method_names
    )

    save_scores(
        result_dir / "id_test_scores.csv",
        test_indices,
        test_results,
        method_names
    )

    accuracy = np.mean(
        test_results["labels"] == test_results["preds"]
    )

    print(f"\nID Test Accuracy: {accuracy:.2%}")

    summary = {
        "seed": args.seed,
        "id_test_accuracy": float(accuracy),
        "methods": {}
    }

    for name in method_names:
        metrics = evaluate_method(
            val_results[name],
            test_results[name]
        )

        summary["methods"][name] = metrics

        print(f"\n[{name.upper()}]")
        print(f"Threshold: {metrics['threshold']:.6f}")
        print(f"Val acceptance: {metrics['val_acceptance']:.2%}")
        print(f"Test acceptance: {metrics['test_acceptance']:.2%}")
        print(f"Test median: {metrics['test_median']:.6f}")

    with open(
        result_dir / "evaluation.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {result_dir}")


if __name__ == "__main__":
    main()
