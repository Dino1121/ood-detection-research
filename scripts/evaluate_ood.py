
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models.resnet18_cifar import ResNet18CIFAR
from src.ood.methods import msp_score, energy_score


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# All training runs use the same data split as seed 42.
SPLIT_SEED = 42

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
        weights_only=True,
    )

    model.load_state_dict(weights)
    model.eval()
    return model


def load_indices(checkpoint_dir):
    split_path = checkpoint_dir / "split_indices.json"

    with split_path.open(encoding="utf-8") as f:
        splits = json.load(f)

    val_indices = splits["val_indices"]
    test_indices = splits["test_indices"]

    if len(val_indices) != 1000:
        raise ValueError(
            f"Expected 1000 validation images, got {len(val_indices)}"
        )

    if len(test_indices) != 9000:
        raise ValueError(
            f"Expected 9000 test images, got {len(test_indices)}"
        )

    val_set = set(val_indices)
    test_set = set(test_indices)

    if len(val_set) != 1000 or len(test_set) != 9000:
        raise ValueError("Duplicate indices detected.")

    if val_set & test_set:
        raise ValueError("Validation and test sets overlap.")

    if val_set | test_set != set(range(10000)):
        raise ValueError(
            "Validation and test sets do not cover CIFAR-10 test."
        )

    # Verify that all training seeds use the original seed-42 split.
    reference_path = (
        Path("checkpoints")
        / f"cifar10_resnet18_seed{SPLIT_SEED}"
        / "split_indices.json"
    )

    with reference_path.open(encoding="utf-8") as f:
        reference = json.load(f)

    keys_to_check = [
        "train_clean_indices",
        "val_indices",
        "test_indices",
    ]

    for key in keys_to_check:
        if splits.get(key) != reference.get(key):
            raise ValueError(
                f"Data split mismatch: {key}\n"
                f"Current checkpoint: {split_path}\n"
                f"Reference split: {reference_path}\n"
                "All training seeds must use the same split."
            )

    return val_indices, test_indices


def load_dataset(name, transform):
    if name == "cifar10":
        return datasets.CIFAR10(
            root="data",
            train=False,
            download=False,
            transform=transform
        )

    if name == "cifar100":
        return datasets.CIFAR100(
            root="data/cifar100_clean",
            train=False,
            download=False,
            transform=transform
        )

    if name == "svhn":
        return datasets.SVHN(
            root="data",
            split="test",
            download=False,
            transform=transform,
        )

    raise ValueError(f"Unknown dataset: {name}")


def make_loader(dataset, batch_size, workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


@torch.inference_mode()
def extract_scores(model, loader, device, methods):
    outputs = {
        "labels": [],
        "preds": [],
    }

    for name in methods:
        outputs[name] = []

    for images, labels in loader:
        images = images.to(device)

        logits, features = model(
            images, return_features=True
        )

        outputs["labels"].append(labels.numpy())
        outputs["preds"].append(
            logits.argmax(dim=1).cpu().numpy()
        )

        for name, score_fn in methods.items():
            scores = score_fn(logits, features, model)
            outputs[name].append(scores.cpu().numpy())

    return {
        key: np.concatenate(values)
        for key, values in outputs.items()
    }


def save_scores(path, indices, results, method_names):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image_index",
            "original_label",
            "predicted_id_class",
            *method_names,
        ])

        for i, index in enumerate(indices):
            writer.writerow([
                index,
                int(results["labels"][i]),
                int(results["preds"][i]),
                *[
                    float(results[name][i])
                    for name in method_names
                ],
            ])


def evaluate_ood(id_scores, ood_scores, threshold):
    # Positive class = ID.
    # Higher scores indicate a greater likelihood of ID.
    labels = np.r_[
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ]

    scores = np.r_[id_scores, ood_scores]

    fpr, tpr, _ = roc_curve(
        labels,
        scores,
        pos_label=1,
        drop_intermediate=False,
    )

    idx = np.searchsorted(tpr, 0.95, side="left")

    return {
        "auroc": float(
            roc_auc_score(labels, scores)
        ),
        "fpr95_roc": float(
            fpr[idx]
        ),
        "aupr_in": float(
            average_precision_score(labels, scores)
        ),
        "aupr_out": float(
            average_precision_score(1 - labels, -scores)
        ),
        "ood_acceptance_fixed": float(
            np.mean(ood_scores >= threshold)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=["msp", "energy"],
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifar100", "svhn"],
        default=["cifar100", "svhn"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    method_names = list(dict.fromkeys(args.methods))
    dataset_names = list(dict.fromkeys(args.datasets))

    methods = {
        name: METHODS[name]
        for name in method_names
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Check data split before running evaluation.
    val_indices, test_indices = load_indices(checkpoint_dir)

    print(f"Training seed: {args.seed}")
    print(f"Fixed split seed: {SPLIT_SEED}")
    print("Data split verification: PASSED")

    model = load_model(checkpoint_dir, device)

    id_dataset = load_dataset("cifar10", transform)

    val_loader = make_loader(
        Subset(id_dataset, val_indices),
        args.batch_size,
        args.workers,
        device,
    )

    test_loader = make_loader(
        Subset(id_dataset, test_indices),
        args.batch_size,
        args.workers,
        device,
    )

    print("\nExtracting ID validation scores...")
    val_results = extract_scores(
        model, val_loader, device, methods
    )

    print("Extracting ID test scores...")
    test_results = extract_scores(
        model, test_loader, device, methods
    )

    save_scores(
        result_dir / "id_val_scores.csv",
        val_indices,
        val_results,
        method_names,
    )

    save_scores(
        result_dir / "id_test_scores.csv",
        test_indices,
        test_results,
        method_names,
    )

    accuracy = float(np.mean(
        test_results["labels"] == test_results["preds"]
    ))

    # Calibrate separately for each trained model.
    # ID validation acceptance target: 95%.
    thresholds = {
        name: float(np.percentile(val_results[name], 5))
        for name in method_names
    }

    summary = {
        "training_seed": args.seed,
        "split_seed": SPLIT_SEED,
        "id_test_accuracy": accuracy,
        "methods": {},
        "ood_datasets": {},
    }

    print(f"\nDevice: {device}")
    print(f"ID Test Accuracy: {accuracy:.2%}")

    for name in method_names:
        threshold = thresholds[name]

        val_acceptance = float(np.mean(
            val_results[name] >= threshold
        ))

        test_acceptance = float(np.mean(
            test_results[name] >= threshold
        ))

        summary["methods"][name] = {
            "threshold": threshold,
            "id_val_acceptance": val_acceptance,
            "id_test_acceptance": test_acceptance,
        }

        print(f"\n[{name.upper()}]")
        print(f"Threshold: {threshold:.6f}")
        print(f"ID Val acceptance: {val_acceptance:.2%}")
        print(f"ID Test acceptance: {test_acceptance:.2%}")

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name, transform)

        loader = make_loader(
            dataset,
            args.batch_size,
            args.workers,
            device,
        )

        print(
            f"\nEvaluating {dataset_name}: "
            f"{len(dataset)} images"
        )

        ood_results = extract_scores(
            model, loader, device, methods
        )

        save_scores(
            result_dir / f"{dataset_name}_scores.csv",
            range(len(dataset)),
            ood_results,
            method_names,
        )

        summary["ood_datasets"][dataset_name] = {}

        for name in method_names:
            metrics = evaluate_ood(
                test_results[name],
                ood_results[name],
                thresholds[name],
            )

            summary["ood_datasets"][dataset_name][name] = metrics

            print(f"\n  {name.upper()}")
            print(f"  AUROC: {metrics['auroc']:.2%}")
            print(f"  FPR95 (ROC): {metrics['fpr95_roc']:.2%}")
            print(
                "  OOD acceptance (fixed): "
                f"{metrics['ood_acceptance_fixed']:.2%}"
            )
            print(f"  AUPR-IN: {metrics['aupr_in']:.2%}")
            print(f"  AUPR-OUT: {metrics['aupr_out']:.2%}")

    with (result_dir / "evaluation.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {result_dir}")


if __name__ == "__main__":
    main()
