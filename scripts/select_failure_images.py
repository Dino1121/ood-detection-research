
"""Select repeated OOD success/failure images."""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import rankdata
from torchvision.datasets import CIFAR100, SVHN

from scripts.analyze_disagreements import (
    load_run,
    write_csv,
)


CIFAR10_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def load_images(dataset):
    if dataset == "cifar100":
        for root in ("data/cifar100_clean", "data"):
            try:
                return CIFAR100(
                    root=root,
                    train=False,
                    download=False,
                )
            except RuntimeError:
                continue

        raise FileNotFoundError(
            "CIFAR-100 not found in data."
        )

    if dataset == "svhn":
        return SVHN(
            root="data",
            split="test",
            download=False,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["msp", "energy"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar100", "svhn"],
    )
    parser.add_argument(
        "--run-tag",
        default=None,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--group",
        choices=["all_failure", "all_success"],
        default="all_failure",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    seeds = list(dict.fromkeys(args.seeds))
    methods = list(dict.fromkeys(args.methods))

    run_tag = (
        args.run_tag or "_".join(sorted(methods))
    )

    output_dir = (
        Path("results/analysis") / run_tag
    )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    for dataset_name in args.datasets:
        runs = {
            seed: load_run(
                seed,
                dataset_name,
                methods,
                run_tag,
            )[0]
            for seed in seeds
        }

        image_indices = sorted(runs[seeds[0]])

        for seed in seeds[1:]:
            if set(runs[seed]) != set(image_indices):
                raise ValueError(
                    f"{dataset_name}: different indices "
                    "across seeds"
                )

        # Percentile ranks within the OOD dataset.
        # Equal scores receive the same average rank.
        percentiles = []

        for seed in seeds:
            for method in methods:
                values = [
                    runs[seed][i]["scores"][method]
                    for i in image_indices
                ]

                ranks = (
                    rankdata(values, method="average")
                    / len(values)
                )

                percentiles.append(
                    dict(zip(image_indices, ranks))
                )

        candidates = [
            i for i in image_indices
            if all(
                runs[s][i]["group"] == args.group
                for s in seeds
            )
        ]

        if args.group == "all_failure":
            # Strong failures: high scores in EVERY
            # selected method and seed.
            strength = {
                i: min(p[i] for p in percentiles)
                for i in candidates
            }

            selected = sorted(
                candidates,
                key=lambda i: strength[i],
                reverse=True,
            )[:args.top_k]

            suffix = "top_common_failures"

        else:
            # Strong successes: low scores in EVERY
            # selected method and seed.
            strength = {
                i: max(p[i] for p in percentiles)
                for i in candidates
            }

            selected = sorted(
                candidates,
                key=lambda i: strength[i],
            )[:args.top_k]

            suffix = "top_common_successes"

        print(
            f"{dataset_name}: "
            f"{len(candidates):,} repeated "
            f"{args.group} images; "
            f"selected {len(selected)}"
        )

        if not selected:
            continue

        dataset = load_images(dataset_name)

        cols = min(3, len(selected))
        rows = math.ceil(len(selected) / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(5 * cols, 4.8 * rows),
            squeeze=False,
        )

        axes = axes.flat
        records = []

        for ax, index in zip(axes, selected):
            image, label = dataset[index]

            ax.imshow(image)
            ax.axis("off")

            if dataset_name == "cifar100":
                original = dataset.classes[label]
            else:
                original = f"digit {label}"

            title_lines = [
                f"Index #{index} | OOD: {original}"
            ]

            record = {
                "image_index": index,
                "original_class": original,
                "selection_percentile": float(
                    strength[index]
                ),
            }

            for seed in seeds:
                item = runs[seed][index]

                predicted = CIFAR10_NAMES[
                    item["predicted_id_class"]
                ]

                score_text = ", ".join(
                    f"{m}={item['scores'][m]:.3f}"
                    for m in methods
                )

                title_lines.append(
                    f"Seed {seed} -> {predicted}"
                    f" | {score_text}"
                )

                record[
                    f"seed{seed}_prediction"
                ] = predicted

                for method in methods:
                    record[
                        f"seed{seed}_{method}_score"
                    ] = item["scores"][method]

            ax.set_title(
                "\n".join(title_lines),
                fontsize=9,
                pad=8,
            )

            records.append(record)

        for ax in axes:
            if not ax.has_data():
                ax.axis("off")

        fig.suptitle(
            f"{dataset_name.upper()}: "
            f"repeated {args.group} "
            f"across {len(seeds)} seeds",
            fontsize=13,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        png_path = (
            output_dir
            / f"{dataset_name}_{suffix}.png"
        )

        csv_path = (
            output_dir
            / f"{dataset_name}_{suffix}.csv"
        )

        fig.savefig(png_path, dpi=180)
        plt.close(fig)

        write_csv(
            csv_path,
            list(records[0]),
            records,
        )

        print(f"  Saved: {png_path}")
        print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()
