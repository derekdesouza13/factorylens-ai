from pathlib import Path
import pandas as pd

from src.preprocessing.validate_data import (
    EXPECTED_FILES,
    load_dataset,
    validate_dataset,
)


def test_all_expected_datasets_exist():
    data_dir = Path("data/raw")

    for filename in EXPECTED_FILES.values():
        assert (data_dir / filename).exists()


def test_datasets_have_rows():
    for name, filename in EXPECTED_FILES.items():
        df = load_dataset(filename)

        assert len(df) > 0
        assert len(df.columns) > 0


def test_datasets_have_no_duplicate_rows():
    for name, filename in EXPECTED_FILES.items():
        df = load_dataset(filename)

        result = validate_dataset(name, df)

        assert result["duplicate_rows"] == 0