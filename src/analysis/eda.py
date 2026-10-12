
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("reports/eda")

sns.set_theme(style="whitegrid")


def load_data(filename: str) -> pd.DataFrame:
    """Load a cleaned dataset."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the cleaning pipeline first."
        )

    return pd.read_csv(path)


def save_plot(filename: str) -> None:
    """Save the current chart and close the figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_quality(quality: pd.DataFrame) -> dict:
    """Analyze inspection results and defect patterns."""
    quality["quality_score"] = pd.to_numeric(
        quality["quality_score"], errors="coerce"
    )

    defect_counts = quality["defect_status"].value_counts(dropna=False)
    defect_rates = (
        quality.groupby("defect_status", dropna=False)
        .size()
        .rename("inspection_count")
        .reset_index()
    )

    total_inspections = len(quality)
    defect_rate = (
        quality["defect_status"].astype(str).str.lower()
        .isin(["1", "true", "yes", "defect", "defective"])
        .mean() * 100
        if total_inspections else 0
    )

    plt.figure(figsize=(8, 5))
    sns.histplot(
        quality["quality_score"].dropna(),
        bins=30,
        kde=True,
    )
    plt.title("Distribution of Quality Scores")
    plt.xlabel("Quality Score")
    plt.ylabel("Inspection Count")
    save_plot("quality_score_distribution.png")

    plt.figure(figsize=(8, 5))
    defect_counts.plot(kind="bar")
    plt.title("Inspection Counts by Defect Status")
    plt.xlabel("Defect Status")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    save_plot("defect_status_counts.png")

    return {
        "total_inspections": total_inspections,
        "defect_rate_percent": round(float(defect_rate), 2),
        "mean_quality_score": round(
            float(quality["quality_score"].mean()), 2
        ),
        "median_quality_score": round(
            float(quality["quality_score"].median()), 2
        ),
    }


def analyze_sensors(sensors: pd.DataFrame, quality: pd.DataFrame) -> dict:
    """Analyze sensor distributions and relationships with defects."""
    merged = sensors.merge(
        quality[["batch_id", "defect_status"]],
        on="batch_id",
        how="inner",
        validate="one_to_one",
    )

    sensor_columns = [
        column for column in [
            "temperature", "pressure", "vibration", "rpm",
            "motor_current", "humidity", "machine_runtime_hours",
        ]
        if column in merged.columns
    ]

    for column in sensor_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    numeric = merged[sensor_columns].dropna()

    if not numeric.empty:
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
        )
        plt.title("Machine Sensor Correlation Matrix")
        save_plot("sensor_correlation_matrix.png")

    if "temperature" in merged.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=merged,
            x="temperature",
            hue="defect_status",
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
        )
        plt.title("Temperature Distribution by Defect Status")
        save_plot("temperature_by_defect.png")

    if "vibration" in merged.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=merged,
            x="vibration",
            hue="defect_status",
            bins=30,
            element="step",
            stat="density",
            common_norm=False,
        )
        plt.title("Vibration Distribution by Defect Status")
        save_plot("vibration_by_defect.png")

    return {
        "matched_batches": len(merged),
        "sensor_columns_analyzed": sensor_columns,
    }


def analyze_energy(energy: pd.DataFrame) -> dict:
    """Analyze energy consumption and energy efficiency."""
    for column in [
        "production_volume",
        "energy_consumption_kwh",
        "energy_per_unit",
    ]:
        energy[column] = pd.to_numeric(
            energy[column], errors="coerce"
        )

    valid = energy.dropna(
        subset=["production_volume", "energy_consumption_kwh"]
    )

    if not valid.empty:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=valid,
            x="production_volume",
            y="energy_consumption_kwh",
            hue="machine_id",
            legend=False,
            alpha=0.6,
        )
        plt.title("Production Volume vs Energy Consumption")
        plt.xlabel("Production Volume")
        plt.ylabel("Energy Consumption (kWh)")
        save_plot("production_vs_energy.png")

    return {
        "energy_records": len(energy),
        "mean_energy_per_unit": round(
            float(energy["energy_per_unit"].mean()), 4
        ),
        "total_energy_kwh": round(
            float(energy["energy_consumption_kwh"].sum()), 2
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    production = load_data("production.csv")
    sensors = load_data("machine_sensors.csv")
    quality = load_data("quality_inspections.csv")
    energy = load_data("energy_consumption.csv")

    print("Running FactoryLens AI exploratory analysis...\n")

    results = {
        "quality": analyze_quality(quality),
        "sensors": analyze_sensors(sensors, quality),
        "energy": analyze_energy(energy),
        "production": {
            "production_records": len(production),
            "production_lines": int(
                production["production_line"].nunique()
            ),
            "machines": int(production["machine_id"].nunique()),
            "total_production_volume": float(
                pd.to_numeric(
                    production["production_volume"], errors="coerce"
                ).sum()
            ),
        },
    }

    import json

    report_path = OUTPUT_DIR / "eda_summary.json"
    report_path.write_text(json.dumps(results, indent=2))

    print("EDA SUMMARY")
    print("=" * 50)

    for section, metrics in results.items():
        print(f"\n{section.upper()}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print(f"\nCharts and report saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
