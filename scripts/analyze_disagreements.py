import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_run(seed, dataset, methods, run_tag):
    root = Path("results/week1") / f"seed{seed}" / run_tag

    with (root / "evaluation.json").open(encoding="utf-8") as f:
        evaluation = json.load(f)

    thresholds = {}

    for method in methods:
        if method not in evaluation["methods"]:
            raise ValueError(
                f"Seed {seed}: '{method}' missing from evaluation.json"
            )
        thresholds[method] = float(
            evaluation["methods"][method]["threshold"]
        )

    result = {}

    with (root / f"{dataset}_scores.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)

        if not set(methods).issubset(reader.fieldnames or []):
            raise ValueError(
                f"Seed {seed}: requested methods missing from CSV"
            )

        for row in reader:
            index = int(row["image_index"])

            if index in result:
                raise ValueError(f"Duplicate image index: {index}")

            scores = {
                m: float(row[m]) for m in methods
            }

            # All method scores must be ID-directed:
            # higher score = more ID-like.
            accepted = {
                m: scores[m] >= thresholds[m]
                for m in methods
            }

            successes = [
                m for m in methods if not accepted[m]
            ]

            if len(successes) == len(methods):
                group = "all_success"
            elif not successes:
                group = "all_failure"
            else:
                group = "success_" + "_".join(successes)

            result[index] = {
                "original_label": int(row["original_label"]),
                "predicted_id_class": int(
                    row["predicted_id_class"]
                ),
                "scores": scores,
                "accepted": accepted,
                "group": group,
            }

    return result, thresholds


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar100", "svhn"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["msp", "energy"],
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Existing evaluation folder name",
    )

    args = parser.parse_args()

    methods = list(dict.fromkeys(args.methods))
    seeds = list(dict.fromkeys(args.seeds))

    run_tag = (
        args.run_tag or "_".join(sorted(methods))
    )

    output_dir = Path("results/analysis") / run_tag

    summary = {
        "seeds": seeds,
        "methods": methods,
        "datasets": {},
    }

    for dataset in args.datasets:
        runs = {
            seed: load_run(
                seed, dataset, methods, run_tag
            )[0]
            for seed in seeds
        }

        indices = set(runs[seeds[0]])

        for seed in seeds[1:]:
            if set(runs[seed]) != indices:
                raise ValueError(
                    f"{dataset}: indices differ across seeds"
                )

        print(
            f"\n[{dataset.upper()}] "
            f"{len(indices):,} OOD images"
        )

        dataset_summary = {"per_seed": {}}

        for seed in seeds:
            counts = Counter(
                item["group"]
                for item in runs[seed].values()
            )

            dataset_summary["per_seed"][str(seed)] = dict(
                counts
            )

            print(f"  Seed {seed}:")

            for group, count in sorted(counts.items()):
                print(
                    f"    {group}: {count:,} "
                    f"({count / len(indices):.2%})"
                )

            rows = []

            for index in sorted(indices):
                item = runs[seed][index]

                row = {
                    "image_index": index,
                    "original_label": item["original_label"],
                    "predicted_id_class": item[
                        "predicted_id_class"
                    ],
                    "group": item["group"],
                }

                for method in methods:
                    row[f"{method}_score"] = (
                        item["scores"][method]
                    )
                    row[f"{method}_accepted_as_id"] = int(
                        item["accepted"][method]
                    )

                rows.append(row)

            columns = [
                "image_index",
                "original_label",
                "predicted_id_class",
                "group",
            ]

            for method in methods:
                columns += [
                    f"{method}_score",
                    f"{method}_accepted_as_id",
                ]

            write_csv(
                output_dir / f"seed{seed}_{dataset}_groups.csv",
                columns,
                rows,
            )

        repeated_fail = [
            index
            for index in sorted(indices)
            if all(
                runs[seed][index]["group"] == "all_failure"
                for seed in seeds
            )
        ]

        repeated_success = [
            index
            for index in sorted(indices)
            if all(
                runs[seed][index]["group"] == "all_success"
                for seed in seeds
            )
        ]

        for group, selected in [
            ("failures", repeated_fail),
            ("successes", repeated_success),
        ]:
            write_csv(
                output_dir
                / f"{dataset}_repeated_{group}.csv",
                ["image_index", "original_label"],
                [
                    {
                        "image_index": index,
                        "original_label": runs[seeds[0]][
                            index
                        ]["original_label"],
                    }
                    for index in selected
                ],
            )

        dataset_summary["repeated_all_failures"] = len(
            repeated_fail
        )
        dataset_summary["repeated_all_successes"] = len(
            repeated_success
        )

        summary["datasets"][dataset] = dataset_summary

        print(
            "  Repeated all-method failures: "
            f"{len(repeated_fail):,}"
        )
        print(
            "  Repeated all-method successes: "
            f"{len(repeated_success):,}"
        )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
