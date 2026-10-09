"""
FactoryLens AI
Synthetic Manufacturing Data Generator

Generates interconnected datasets representing:
- Production
- Machine sensor data
- Quality inspections
- Energy consumption
- Maintenance history
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_production_data(n_records: int = 25_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    machine_ids = [f"M{i:02d}" for i in range(1, 21)]
    production_lines = [f"LINE_{i}" for i in range(1, 6)]
    product_types = ["TYPE_A", "TYPE_B", "TYPE_C"]

    timestamps = pd.date_range(
        start="2025-01-01",
        end="2025-12-31",
        periods=n_records,
    )

    production = pd.DataFrame(
        {
            "batch_id": [f"B{i:06d}" for i in range(1, n_records + 1)],
            "timestamp": timestamps,
            "machine_id": rng.choice(machine_ids, n_records),
            "production_line": rng.choice(
                production_lines,
                n_records,
            ),
            "product_type": rng.choice(
                product_types,
                n_records,
            ),
            "shift": rng.choice(
                ["Morning", "Evening", "Night"],
                n_records,
                p=[0.4, 0.4, 0.2],
            ),
            "production_volume": rng.integers(
                80,
                180,
                n_records,
            ),
            "cycle_time": rng.normal(
                42,
                5,
                n_records,
            ).clip(25, 70),
        }
    )

    return production


def generate_machine_data(
    production: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)

    machine = production[
        ["batch_id", "timestamp", "machine_id"]
    ].copy()

    machine["temperature"] = rng.normal(
        75,
        8,
        len(machine),
    ).clip(45, 110)

    machine["pressure"] = rng.normal(
        15,
        1.5,
        len(machine),
    ).clip(8, 25)

    machine["vibration"] = rng.normal(
        0.45,
        0.15,
        len(machine),
    ).clip(0.05, 1.5)

    machine["rpm"] = rng.normal(
        1500,
        120,
        len(machine),
    ).clip(900, 2100)

    machine["motor_current"] = rng.normal(
        18,
        3,
        len(machine),
    ).clip(8, 35)

    machine["humidity"] = rng.normal(
        55,
        10,
        len(machine),
    ).clip(20, 90)

    machine["machine_runtime_hours"] = rng.uniform(
        100,
        10_000,
        len(machine),
    )

    return machine


def generate_quality_data(
    production: pd.DataFrame,
    machine: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 2)

    quality = production[["batch_id"]].copy()

    temperature = machine["temperature"].to_numpy()
    vibration = machine["vibration"].to_numpy()
    pressure = machine["pressure"].to_numpy()
    cycle_time = production["cycle_time"].to_numpy()

    risk_score = (
        0.035 * (temperature - 75)
        + 2.5 * (vibration - 0.45)
        + 0.4 * np.abs(pressure - 15)
        + 0.04 * np.abs(cycle_time - 42)
    )

    probability = 1 / (
        1 + np.exp(-risk_score + 2.2)
    )

    defect = rng.binomial(
        1,
        probability,
    )

    quality["quality_score"] = (
        100
        - 12 * defect
        - rng.normal(0, 4, len(quality))
    ).clip(40, 100)

    quality["defect_status"] = defect

    quality["defect_type"] = np.where(
        defect == 1,
        rng.choice(
            [
                "Surface Defect",
                "Dimensional Defect",
                "Material Defect",
            ],
            len(quality),
        ),
        "None",
    )

    quality["inspection_result"] = np.where(
        defect == 1,
        "FAIL",
        "PASS",
    )

    return quality


def generate_energy_data(
    production: pd.DataFrame,
    machine: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 3)

    energy = production[
        [
            "batch_id",
            "timestamp",
            "machine_id",
            "production_volume",
        ]
    ].copy()

    energy["energy_consumption_kwh"] = (
        18
        + 0.07 * energy["production_volume"]
        + 0.08 * machine["motor_current"]
        + rng.normal(0, 2, len(energy))
    ).clip(5, None)

    energy["power_factor"] = rng.normal(
        0.92,
        0.03,
        len(energy),
    ).clip(0.75, 1.0)

    energy["energy_per_unit"] = (
        energy["energy_consumption_kwh"]
        / energy["production_volume"]
    )

    return energy


def generate_maintenance_data(
    production: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 4)

    machines = production["machine_id"].unique()

    n_records = 500

    maintenance = pd.DataFrame(
        {
            "maintenance_id": [
                f"MT{i:05d}"
                for i in range(1, n_records + 1)
            ],
            "machine_id": rng.choice(
                machines,
                n_records,
            ),
            "maintenance_date": pd.date_range(
                start="2025-01-01",
                end="2025-12-31",
                periods=n_records,
            ),
            "failure_type": rng.choice(
                [
                    "Bearing Failure",
                    "Motor Failure",
                    "Vibration Issue",
                    "Temperature Issue",
                    "Pressure Issue",
                    "Preventive Maintenance",
                ],
                n_records,
            ),
            "maintenance_duration_hours": rng.uniform(
                1,
                12,
                n_records,
            ).round(2),
            "parts_replaced": rng.choice(
                [
                    "Bearing",
                    "Motor",
                    "Seal",
                    "Sensor",
                    "None",
                ],
                n_records,
            ),
        }
    )

    return maintenance


def save_dataset(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    path = RAW_DATA_DIR / filename

    dataframe.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved {filename}: "
        f"{len(dataframe):,} records"
    )


def main() -> None:
    print(
        "Generating FactoryLens manufacturing datasets...\n"
    )

    production = generate_production_data()

    machine = generate_machine_data(
        production
    )

    quality = generate_quality_data(
        production,
        machine,
    )

    energy = generate_energy_data(
        production,
        machine,
    )

    maintenance = generate_maintenance_data(
        production
    )

    save_dataset(
        production,
        "production.csv",
    )

    save_dataset(
        machine,
        "machine_sensors.csv",
    )

    save_dataset(
        quality,
        "quality_inspections.csv",
    )

    save_dataset(
        energy,
        "energy_consumption.csv",
    )

    save_dataset(
        maintenance,
        "maintenance.csv",
    )

    print(
        "\nDataset generation complete."
    )


if __name__ == "__main__":
    main()