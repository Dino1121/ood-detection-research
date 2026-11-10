
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

from src.models.resnet18_cifar import ResNet18CIFAR


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cifar100(transform):
    for root in ["data/cifar100_clean", "data"]:
        try:
            return datasets.CIFAR100(
                root=root,
                train=False,
                download=False,
                transform=transform
            )
        except RuntimeError:
            continue

    raise RuntimeError("CIFAR-100 dataset not found.")


@torch.inference_mode()
def inspect(model, dataset, index, device):
    image, label = dataset[index]

    logits = model(
        image.unsqueeze(0).to(device)
    ).squeeze(0).cpu()

    probabilities = torch.softmax(logits, dim=0)

    msp = probabilities.max().item()
    energy = torch.logsumexp(logits, dim=0).item()
    prediction = logits.argmax().item()

    return {
        "logits": logits.numpy(),
        "probabilities": probabilities.numpy(),
        "msp": msp,
        "energy": energy,
        "max_logit": logits.max().item(),
        "prediction": prediction,
        "label": int(label),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    seed = args.seed

    root = (
        Path("results/week1")
        / f"seed{seed}"
        / "energy_msp"
    )

    with (root / "evaluation.json").open(
        encoding="utf-8"
    ) as f:
        evaluation = json.load(f)

    msp_threshold = evaluation["methods"]["msp"]["threshold"]
    energy_threshold = evaluation["methods"]["energy"]["threshold"]

    ood_rows = read_csv(
        root / "cifar100_scores.csv"
    )

    id_rows = read_csv(
        root / "id_test_scores.csv"
    )

    # MSP fails: accepts OOD as ID.
    # Energy succeeds: rejects OOD.
# CIFAR-100 class IDs
# 24 = cockroach, 79 = spider
    excluded_labels = {24, 79}

    candidates = [
        row for row in ood_rows
        if float(row["msp"]) >= msp_threshold
        and float(row["energy"]) < energy_threshold
        and int(row["original_label"]) not in excluded_labels
    ]

    if not candidates:
        raise RuntimeError("No Energy-only success cases found.")

    # Choose an OOD example with a particularly high MSP.
    selected_ood = max(
        candidates,
        key=lambda row: (
            float(row["msp"]),
            -float(row["energy"])
        )
    )

    ood_prediction = int(
        selected_ood["predicted_id_class"]
    )

    # Find a correctly classified ID image with the same
    # predicted class and similar MSP.
    id_candidates = [
        row for row in id_rows
        if int(row["original_label"]) == ood_prediction
        and int(row["predicted_id_class"]) == ood_prediction
        and float(row["msp"]) >= msp_threshold
        and float(row["energy"]) >= energy_threshold
    ]

    if not id_candidates:
        raise RuntimeError(
            "No matching accepted ID example found."
        )

    selected_id = min(
        id_candidates,
        key=lambda row: abs(
            float(row["msp"]) - float(selected_ood["msp"])
        )
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNet18CIFAR(num_classes=10).to(device)

    model.load_state_dict(torch.load(
        Path("checkpoints")
        / f"cifar10_resnet18_seed{seed}"
        / "best.pth",
        map_location=device,
        weights_only=True
    ))

    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    id_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transform
    )

    ood_dataset = load_cifar100(transform)

    id_index = int(selected_id["image_index"])
    ood_index = int(selected_ood["image_index"])

    id_result = inspect(
        model, id_dataset, id_index, device
    )

    ood_result = inspect(
        model, ood_dataset, ood_index, device
    )

    print(f"\nTraining seed: {seed}")
    print(f"MSP threshold: {msp_threshold:.6f}")
    print(f"Energy threshold: {energy_threshold:.6f}")

    for name, result in [
        ("ID", id_result),
        ("OOD", ood_result)
    ]:
        print(f"\n========== {name} ==========")

        print("Logits:")
        for cls, value in zip(
            CLASS_NAMES, result["logits"]
        ):
            print(f"  {cls:12s}: {value:9.4f}")

        print(
            f"Prediction: "
            f"{CLASS_NAMES[result['prediction']]}"
        )
        print(f"Max Logit: {result['max_logit']:.6f}")
        print(f"MSP:       {result['msp']:.6f}")
        print(f"Energy:    {result['energy']:.6f}")

    # Plot original images and their 10 logits.
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9)
    )

    for col, (name, dataset, index, result) in enumerate([
        ("ID", id_dataset, id_index, id_result),
        ("OOD", ood_dataset, ood_index, ood_result)
    ]):
        # Display the original image without normalization.
        image, _ = dataset[index]
        image = image * torch.tensor(STD).view(3, 1, 1)
        image = image + torch.tensor(MEAN).view(3, 1, 1)
        image = image.clamp(0, 1)

        axes[0, col].imshow(
            image.permute(1, 2, 0).numpy()
        )
        axes[0, col].axis("off")

        original = (
            id_dataset.classes[result["label"]]
            if name == "ID"
            else ood_dataset.classes[result["label"]]
        )

        axes[0, col].set_title(
            f"{name} | True: {original}\n"
            f"Predicted: {CLASS_NAMES[result['prediction']]}\n"
            f"MSP={result['msp']:.6f} | "
            f"Energy={result['energy']:.3f}",
            fontsize=12
        )

        axes[1, col].bar(
            CLASS_NAMES,
            result["logits"]
        )

        axes[1, col].set_title(
            f"{name}: 10 Class Logits"
        )
        axes[1, col].set_ylabel("Logit")
        axes[1, col].tick_params(
            axis="x", labelrotation=55
        )
        axes[1, col].axhline(
            0, linewidth=0.8, color="gray"
        )

    fig.suptitle(
        f"Seed {seed} | Similar MSP, Different Energy",
        fontsize=15
    )
    fig.tight_layout()

    output_dir = Path("results/analysis/energy_msp")
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"seed{seed}_logit_comparison.png"

    fig.savefig(path, dpi=180)
    plt.close(fig)

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
