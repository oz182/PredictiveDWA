import os
import csv
import math
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def mean_and_ci95(values: List[float]) -> Tuple[float, float]:
    """Return (mean, half-width of 95% CI) for a list of floats."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    ci = 1.96 * std / math.sqrt(n)
    return m, ci


def analyze_simulation_data(data_dir: str) -> str:
    """Aggregate metrics for each CSV in simulation_data and write a summary table."""
    metrics = [
        "return",
        "steps",
        "collisions",
        "overlap_free_pct",
        "overlap_person_pct",
        "overlap_door_pct",
        "overlap_both_pct",
    ]

    # Only analyze per-episode evaluation CSVs produced by test_algorithm.py
    # (avoid including aggregated summaries or unrelated data dumps)
    files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and f.startswith("algo_eval_")
    ]
    files.sort()

    rows: List[Dict[str, object]] = []

    for filename in files:
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
            algo_name = None

            for row in reader:
                if algo_name is None and "algo" in row:
                    algo_name = row["algo"]
                for m in metrics:
                    if m in row and row[m] != "":
                        try:
                            metric_values[m].append(float(row[m]))
                        except ValueError:
                            continue

        summary: Dict[str, object] = {
            "file": filename,
        }
        if algo_name is not None:
            summary["algo"] = algo_name

        for m in metrics:
            mean, ci = mean_and_ci95(metric_values[m])
            summary[f"{m}_mean"] = mean
            summary[f"{m}_ci95"] = ci

        rows.append(summary)

    # Write summary CSV
    if not rows:
        print("No CSV files with data found in", data_dir)
        return ""

    # Collect all keys for header
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(data_dir, f"analysis_summary_{ts}.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.write_row = writer.writerow(row)

    print(f"Wrote analysis summary to: {summary_path}")

    # Create separate bar plots with 95% CI for each metric across files
    labels = [r.get("algo", r.get("file", "")) for r in rows]
    x = list(range(len(rows)))

    for m in metrics:
        means = [r.get(f"{m}_mean", float("nan")) for r in rows]
        cis = [r.get(f"{m}_ci95", 0.0) for r in rows]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, means, yerr=cis, capsize=4, alpha=0.8)
        ax.set_title(m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()

    # Show all figures interactively
    plt.show()

    return summary_path


def main() -> None:
    # Default to the simulation_data directory under src (where test_algorithm.py writes)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "src", "simulation_data")

    if not os.path.isdir(data_dir):
        print(f"simulation_data directory not found at: {data_dir}")
        return

    analyze_simulation_data(data_dir)


if __name__ == "__main__":
    main()


