"""Export observed JSONL optimization history; optional reproducible Matplotlib plots."""

import argparse
import csv
import json
from pathlib import Path
from scripts.data.file_io import new_output


def read_history(source: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(source).read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Incomplete/invalid history at line {line_number}"
            ) from error
        if type(row.get("update")) is not int or row["update"] <= (
            rows[-1]["update"] if rows else 0
        ):
            raise ValueError("History update numbers must increase strictly")
        rows.append(row)
    if not rows:
        raise ValueError("History contains no completed updates")
    return rows


def export_csv(source: Path, output: Path) -> None:
    rows = read_history(source)
    columns = ["update"] + sorted(
        set().union(*(row.keys() for row in rows)) - {"update"}
    )
    with new_output(output) as temporary:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def plot_history(source: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_history(source)
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    steps = [row["update"] for row in rows]
    for key in ("total", "invariance", "regularizer"):
        if any(key in row for row in rows):
            axes[0].plot(steps, [row.get(key, float("nan")) for row in rows], label=key)
    axes[0].legend()
    axes[0].set_ylabel("Observed loss")
    for ax, key, label in (
        (axes[1], "lr", "Learning rate"),
        (axes[2], "update_seconds", "Update seconds"),
    ):
        ax.plot(steps, [row.get(key, float("nan")) for row in rows])
        ax.set_ylabel(label)
    axes[2].set_xlabel("Completed optimizer update")
    with new_output(output) as temporary:
        figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path, help="PNG; optional matplotlib dependency")
    args = parser.parse_args()
    if not args.csv and not args.plot:
        parser.error("provide --csv and/or --plot")
    if args.csv:
        export_csv(args.history, args.csv)
    if args.plot:
        plot_history(args.history, args.plot)


if __name__ == "__main__":
    main()
