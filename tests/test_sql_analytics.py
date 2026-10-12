
import sqlite3

import pandas as pd

from src.features.sql_analytics import QUERIES


def test_sql_queries_execute_against_sample_data():
    sample = pd.DataFrame({
        "batch_id": ["B1", "B2", "B3"],
        "machine_id": ["M1", "M1", "M2"],
        "production_line": ["L1", "L1", "L2"],
        "shift": ["Day", "Night", "Day"],
        "defect_target": [0, 1, 0],
        "temperature": [60.0, 80.0, 55.0],
        "vibration": [0.2, 0.8, 0.1],
        "quality_score": [95.0, 70.0, 98.0],
        "cycle_time": [4.0, 6.0, 3.5],
        "energy_per_produced_unit": [1.2, 1.8, 1.0],
    })

    with sqlite3.connect(":memory:") as connection:
        sample.to_sql(
            "manufacturing_features",
            connection,
            index=False,
        )

        for query in QUERIES.values():
            result = pd.read_sql_query(query, connection)
            assert not result.empty


def test_machine_query_calculates_defect_rate():
    sample = pd.DataFrame({
        "batch_id": ["B1", "B2"],
        "machine_id": ["M1", "M1"],
        "production_line": ["L1", "L1"],
        "shift": ["Day", "Night"],
        "defect_target": [0, 1],
        "temperature": [60.0, 80.0],
        "vibration": [0.2, 0.8],
        "quality_score": [95.0, 70.0],
        "cycle_time": [4.0, 6.0],
        "energy_per_produced_unit": [1.2, 1.8],
    })

    with sqlite3.connect(":memory:") as connection:
        sample.to_sql(
            "manufacturing_features",
            connection,
            index=False,
        )

        result = pd.read_sql_query(
            QUERIES["machine_performance"], connection
        )

    assert result.loc[0, "machine_id"] == "M1"
    assert result.loc[0, "defect_rate_percent"] == 50.0
