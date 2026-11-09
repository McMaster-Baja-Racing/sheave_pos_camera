import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAME_RATE_FPS = 60.0
REQUIRED_COLUMNS = {"frame_index", "primary_gap_in", "secondary_gap_in"}


def load_gaps(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing required columns: {missing}")

        rows = []
        for row in reader:
            try:
                frame_index = int(row["frame_index"])
                primary_gap = float(row["primary_gap_in"])
                secondary_gap = float(row["secondary_gap_in"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid measurement at CSV line {reader.line_num}") from error
            rows.append((frame_index, primary_gap, secondary_gap))

    if not rows:
        raise ValueError("no measurements found")

    times = [frame_index / FRAME_RATE_FPS for frame_index, _, _ in rows]
    primary_gaps = [primary_gap for _, primary_gap, _ in rows]
    secondary_gaps = [secondary_gap for _, _, secondary_gap in rows]
    return times, primary_gaps, secondary_gaps


def plot_csv(csv_path, plots_dir):
    times, primary_gaps, secondary_gaps = load_gaps(csv_path)
    figure, axis = plt.subplots(figsize=(11, 6), layout="constrained")
    axis.plot(times, primary_gaps, label="Primary gap", linewidth=1.2)
    axis.plot(times, secondary_gaps, label="Secondary gap", linewidth=1.2)
    axis.set_title(f"Sheave Gaps: {csv_path.stem}")
    axis.set_xlabel("Time (seconds at 60 fps)")
    axis.set_ylabel("Gap (inches)")
    axis.grid(True, alpha=0.3)
    axis.legend()

    output_path = plots_dir / f"{csv_path.stem}_travel.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main():
    results_dir = Path(__file__).resolve().parent
    plots_dir = results_dir / "plots"
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return 0

    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for csv_path in csv_files:
        try:
            output_path = plot_csv(csv_path, plots_dir)
        except (OSError, ValueError) as error:
            print(f"Skipping {csv_path.name}: {error}")
            continue
        print(f"Saved {output_path}")
        generated += 1

    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())