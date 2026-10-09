from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


def test_production_dataset_exists():
    path = DATA_DIR / "production.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) >= 20_000
    assert "batch_id" in df.columns
    assert "machine_id" in df.columns


def test_quality_dataset_exists():
    path = DATA_DIR / "quality_inspections.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) >= 20_000
    assert "defect_status" in df.columns