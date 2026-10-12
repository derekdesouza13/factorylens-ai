
from pathlib import Path

import pandas as pd

from src.preprocessing.validate_data import EXPECTED_FILES, load_dataset


PROCESSED_DATA_DIR = Path("data/processed")


REQUIRED_COLUMNS = {
    "production": [
        "batch_id", "timestamp", "machine_id", "production_line",
        "product_type", "shift", "production_volume", "cycle_time",
    ],
    "machine_sensors": [
        "batch_id", "timestamp", "machine_id", "temperature",
        "pressure", "vibration", "rpm", "motor_current",
        "humidity", "machine_runtime_hours",
    ],
    "quality_inspections": [
        "batch_id", "quality_score", "defect_status",
        "defect_type", "inspection_result",
    ],
    "energy_consumption": [
        "batch_id", "timestamp", "machine_id", "production_volume",
        "energy_consumption_kwh", "power_factor", "energy_per_unit",
    ],
    "maintenance": [
        "maintenance_id", "machine_id", "maintenance_date",
        "failure_type", "maintenance_duration_hours", "parts_replaced",
    ],
}


# Explicitly identify numeric fields. parts_replaced is treated as a
# categorical field because its source representation may be text.
NUMERIC_COLUMNS = {
    "production": ["production_volume", "cycle_time"],
    "machine_sensors": [
        "temperature", "pressure", "vibration", "rpm",
        "motor_current", "humidity", "machine_runtime_hours",
    ],
    "quality_inspections": ["quality_score"],
    "energy_consumption": [
        "production_volume", "energy_consumption_kwh",
        "power_factor", "energy_per_unit",
    ],
    "maintenance": ["maintenance_duration_hours"],
}


NON_NEGATIVE_COLUMNS = {
    "production": ["production_volume", "cycle_time"],
    "machine_sensors": [
        "vibration", "rpm", "motor_current", "machine_runtime_hours",
    ],
    "quality_inspections": [],
    "energy_consumption": [
        "production_volume", "energy_consumption_kwh", "energy_per_unit",
    ],
    "maintenance": ["maintenance_duration_hours"],
}


CRITICAL_COLUMNS = {
    "production": ["batch_id", "machine_id", "timestamp"],
    "machine_sensors": ["batch_id", "machine_id", "timestamp"],
    "quality_inspections": ["batch_id", "defect_status"],
    "energy_consumption": ["batch_id", "machine_id", "timestamp"],
    "maintenance": ["maintenance_id", "machine_id", "maintenance_date"],
}


def clean_dataset(name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean one manufacturing dataset."""

    if name not in REQUIRED_COLUMNS:
        raise ValueError(f"Unknown dataset: {name}")

    original_rows = len(df)
    original_missing = int(df.isna().sum().sum())

    # 1. Validate the expected schema.
    missing_columns = sorted(set(REQUIRED_COLUMNS[name]) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"{name}: missing required columns: {missing_columns}"
        )

    cleaned = df.copy()

    # 2. Remove exact duplicate rows.
    duplicate_rows_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    # 3. Parse date columns safely.
    date_columns = [
        column
        for column in ("timestamp", "maintenance_date")
        if column in cleaned.columns
    ]

    for column in date_columns:
        cleaned[column] = pd.to_datetime(
            cleaned[column], errors="coerce"
        )

    # 4. Convert known numeric fields explicitly.
    # Invalid numeric strings become NaN and are reported below.
    numeric_conversion_failures = 0

    for column in NUMERIC_COLUMNS[name]:
        original_values = cleaned[column]
        converted_values = pd.to_numeric(
            original_values, errors="coerce"
        )

        failed_conversion = (
            original_values.notna() & converted_values.isna()
        )
        numeric_conversion_failures += int(failed_conversion.sum())

        cleaned[column] = converted_values

    # 5. Remove rows missing critical identifiers or labels.
    critical_columns = CRITICAL_COLUMNS[name]
    missing_critical_mask = cleaned[critical_columns].isna().any(axis=1)
    missing_critical_rows_removed = int(missing_critical_mask.sum())

    cleaned = cleaned.loc[~missing_critical_mask].copy()

    # 6. Remove rows containing known invalid negative values.
    invalid_mask = pd.Series(False, index=cleaned.index)

    for column in NON_NEGATIVE_COLUMNS[name]:
        invalid_mask |= (
            cleaned[column].notna() & (cleaned[column] < 0)
        )

    # Quality scores must be between 0 and 100.
    if name == "quality_inspections":
        score = cleaned["quality_score"]
        invalid_mask |= score.notna() & ~score.between(0, 100)

    # Power factor must be between 0 and 1.
    if name == "energy_consumption":
        power_factor = cleaned["power_factor"]
        invalid_mask |= (
            power_factor.notna() & ~power_factor.between(0, 1)
        )

    invalid_value_rows_removed = int(invalid_mask.sum())
    cleaned = cleaned.loc[~invalid_mask].copy()

    # 7. Report remaining missing values instead of blindly imputing them.
    remaining_missing = int(cleaned.isna().sum().sum())

    report = {
        "dataset": name,
        "rows_before": original_rows,
        "rows_after": len(cleaned),
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_critical_rows_removed": missing_critical_rows_removed,
        "invalid_value_rows_removed": invalid_value_rows_removed,
        "numeric_conversion_failures": numeric_conversion_failures,
        "missing_values_before": original_missing,
        "missing_values_after": remaining_missing,
    }

    return cleaned, report


def main() -> None:
    """Clean all expected datasets and write a quality report."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    reports = []

    for name, filename in EXPECTED_FILES.items():
        print(f"\nCleaning {filename}...")

        df = load_dataset(filename)
        cleaned, report = clean_dataset(name, df)

        output_path = PROCESSED_DATA_DIR / filename
        cleaned.to_csv(output_path, index=False)

        reports.append(report)

        print(
            f"  Rows: {report['rows_before']:,} -> "
            f"{report['rows_after']:,}"
        )
        print(f"  Duplicates removed: {report['duplicate_rows_removed']:,}")
        print(
            "  Critical-field rows removed: "
            f"{report['missing_critical_rows_removed']:,}"
        )
        print(
            f"  Invalid-value rows removed: "
            f"{report['invalid_value_rows_removed']:,}"
        )
        print(
            f"  Numeric conversion failures: "
            f"{report['numeric_conversion_failures']:,}"
        )
        print(
            f"  Remaining missing values: "
            f"{report['missing_values_after']:,}"
        )

    report_df = pd.DataFrame(reports)
    report_path = PROCESSED_DATA_DIR / "data_quality_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETED")
    print("=" * 60)
    print(f"Cleaned datasets: {PROCESSED_DATA_DIR}")
    print(f"Quality report: {report_path}")


if __name__ == "__main__":
    main()
