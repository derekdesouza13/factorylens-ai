
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")


def load_csv(filename: str) -> pd.DataFrame:
    """Load a cleaned dataset."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the data-cleaning pipeline first."
        )

    return pd.read_csv(path)


def build_features() -> pd.DataFrame:
    """Integrate manufacturing datasets and engineer model features."""

    production = load_csv("production.csv")
    sensors = load_csv("machine_sensors.csv")
    quality = load_csv("quality_inspections.csv")
    energy = load_csv("energy_consumption.csv")

    # Each batch should have one record in each batch-level dataset.
    for name, df in {
        "production": production,
        "machine_sensors": sensors,
        "quality_inspections": quality,
        "energy_consumption": energy,
    }.items():
        if df["batch_id"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate batch_id values. "
                "Resolve these before joining."
            )

    # Keep only features that will be useful downstream.
    production = production[
        [
            "batch_id", "timestamp", "machine_id", "production_line",
            "product_type", "shift", "production_volume", "cycle_time",
        ]
    ].copy()

    sensors = sensors[
        [
            "batch_id", "temperature", "pressure", "vibration",
            "rpm", "motor_current", "humidity", "machine_runtime_hours",
        ]
    ].copy()

    quality = quality[
        ["batch_id", "quality_score", "defect_status"]
    ].copy()

    energy = energy[
        [
            "batch_id", "energy_consumption_kwh",
            "power_factor", "energy_per_unit",
        ]
    ].copy()

    # Integrate the four datasets using batch_id.
    features = production.merge(
        sensors, on="batch_id", how="inner", validate="one_to_one"
    )

    features = features.merge(
        quality, on="batch_id", how="inner", validate="one_to_one"
    )

    features = features.merge(
        energy, on="batch_id", how="inner", validate="one_to_one"
    )

    # Convert numeric fields safely.
    numeric_columns = [
        "production_volume", "cycle_time", "temperature", "pressure",
        "vibration", "rpm", "motor_current", "humidity",
        "machine_runtime_hours", "quality_score",
        "energy_consumption_kwh", "power_factor", "energy_per_unit",
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    # Time-based features.
    features["timestamp"] = pd.to_datetime(
        features["timestamp"], errors="coerce"
    )

    features["production_hour"] = features["timestamp"].dt.hour
    features["production_day_of_week"] = (
        features["timestamp"].dt.dayofweek
    )

    # Derived operational features.
    features["energy_per_produced_unit"] = (
        features["energy_consumption_kwh"]
        / features["production_volume"].where(
            features["production_volume"] > 0
        )
    )

    features["temperature_vibration_interaction"] = (
        features["temperature"] * features["vibration"]
    )

    # Standardize the defect label.
    # This assumes the generator uses common boolean/numeric/text labels.
    label = features["defect_status"].astype(str).str.strip().str.lower()

    label_mapping = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "defect": 1,
        "defective": 1,
        "0": 0,
        "false": 0,
        "no": 0,
        "none": 0,
        "no defect": 0,
        "non-defective": 0,
        "non_defective": 0,
        "good": 0,
        "pass": 0,
    }

    features["defect_target"] = label.map(label_mapping)

    if features["defect_target"].isna().any():
        unknown_labels = sorted(
            label[features["defect_target"].isna()].unique().tolist()
        )
        raise ValueError(
            f"Unrecognized defect_status labels: {unknown_labels}. "
            "Inspect the source labels before assigning targets."
        )

    features["defect_target"] = features["defect_target"].astype("int8")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "manufacturing_features.csv"
    features.to_csv(output_path, index=False)

    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 50)
    print(f"Integrated rows: {len(features):,}")
    print(f"Total features: {len(features.columns)}")
    print(f"Defect rate: {features['defect_target'].mean() * 100:.2f}%")
    print(f"Output: {output_path}")

    return features


if __name__ == "__main__":
    build_features()
