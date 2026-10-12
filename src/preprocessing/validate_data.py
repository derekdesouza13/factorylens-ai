from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")


EXPECTED_FILES = {
    "production": "production.csv",
    "machine_sensors": "machine_sensors.csv",
    "quality_inspections": "quality_inspections.csv",
    "energy_consumption": "energy_consumption.csv",
    "maintenance": "maintenance.csv",
}


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a CSV dataset from the raw data directory."""
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    return pd.read_csv(filepath)


def validate_dataset(name: str, df: pd.DataFrame) -> dict:
    """Run basic data-quality checks on a dataset."""

    result = {
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(df.select_dtypes(include="number").shape[1]),
    }

    return result


def main():
    print("=" * 70)
    print("FACTORYLENS AI - DATA VALIDATION")
    print("=" * 70)

    validation_results = []

    for name, filename in EXPECTED_FILES.items():
        print(f"\nValidating: {filename}")

        df = load_dataset(filename)

        result = validate_dataset(name, df)
        validation_results.append(result)

        print(f"Rows:             {result['rows']:,}")
        print(f"Columns:          {result['columns']}")
        print(f"Missing values:   {result['missing_values']:,}")
        print(f"Duplicate rows:   {result['duplicate_rows']:,}")
        print(f"Numeric columns:  {result['numeric_columns']}")

    summary = pd.DataFrame(validation_results)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
    