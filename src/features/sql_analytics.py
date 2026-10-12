
from pathlib import Path
import sqlite3

import pandas as pd


DATA_PATH = Path("data/processed/manufacturing_features.csv")
DB_PATH = Path("data/processed/factorylens.db")
REPORT_DIR = Path("reports/sql")


QUERIES = {
    "machine_performance": """
        SELECT
            machine_id,
            COUNT(*) AS total_batches,
            ROUND(AVG(defect_target) * 100, 2) AS defect_rate_percent,
            ROUND(AVG(temperature), 2) AS avg_temperature,
            ROUND(AVG(vibration), 3) AS avg_vibration,
            ROUND(AVG(energy_per_produced_unit), 4) AS avg_energy_per_unit
        FROM manufacturing_features
        GROUP BY machine_id
        ORDER BY defect_rate_percent DESC
    """,
    "production_line_performance": """
        SELECT
            production_line,
            COUNT(*) AS total_batches,
            ROUND(AVG(defect_target) * 100, 2) AS defect_rate_percent,
            ROUND(AVG(quality_score), 2) AS avg_quality_score,
            ROUND(AVG(cycle_time), 2) AS avg_cycle_time
        FROM manufacturing_features
        GROUP BY production_line
        ORDER BY defect_rate_percent DESC
    """,
    "shift_performance": """
        SELECT
            shift,
            COUNT(*) AS total_batches,
            ROUND(AVG(defect_target) * 100, 2) AS defect_rate_percent,
            ROUND(AVG(energy_per_produced_unit), 4) AS avg_energy_per_unit
        FROM manufacturing_features
        GROUP BY shift
        ORDER BY defect_rate_percent DESC
    """,
    "highest_risk_batches": """
        SELECT
            batch_id,
            machine_id,
            production_line,
            temperature,
            vibration,
            quality_score,
            defect_target
        FROM manufacturing_features
        WHERE defect_target = 1
        ORDER BY temperature DESC, vibration DESC
        LIMIT 20
    """,
}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run feature engineering first."
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Persist the integrated dataset to a local relational database.
    with sqlite3.connect(DB_PATH) as connection:
        df.to_sql(
            "manufacturing_features",
            connection,
            if_exists="replace",
            index=False,
        )

        # Add indexes for common query and lookup fields.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_id "
            "ON manufacturing_features(batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_machine_id "
            "ON manufacturing_features(machine_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_defect_target "
            "ON manufacturing_features(defect_target)"
        )

        for report_name, query in QUERIES.items():
            result = pd.read_sql_query(query, connection)

            output_path = REPORT_DIR / f"{report_name}.csv"
            result.to_csv(output_path, index=False)

            print(f"\n{report_name.replace('_', ' ').title()}")
            print("=" * 60)
            print(result.head(10).to_string(index=False))
            print(f"Saved: {output_path}")

    print("\nSQL ANALYTICS COMPLETED")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
