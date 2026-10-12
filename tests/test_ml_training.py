
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from src.models.train_quality_model import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    build_models,
    build_preprocessor,
    load_ml_data,
)


def make_sample_data():
    """Create a small, balanced dataset for testing."""

    rows = 40
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "production_volume": rng.uniform(50, 200, rows),
        "cycle_time": rng.uniform(2, 10, rows),
        "temperature": rng.uniform(40, 100, rows),
        "pressure": rng.uniform(1, 10, rows),
        "vibration": rng.uniform(0.1, 2, rows),
        "rpm": rng.uniform(500, 2000, rows),
        "motor_current": rng.uniform(2, 20, rows),
        "humidity": rng.uniform(20, 80, rows),
        "machine_runtime_hours": rng.uniform(10, 1000, rows),
        "power_factor": rng.uniform(0.5, 1, rows),
        "production_hour": rng.integers(0, 24, rows),
        "production_day_of_week": rng.integers(0, 7, rows),
        "temperature_vibration_interaction": rng.uniform(5, 200, rows),
        "machine_id": rng.choice(["M1", "M2", "M3"], rows),
        "production_line": rng.choice(["L1", "L2"], rows),
        "product_type": rng.choice(["A", "B"], rows),
        "shift": rng.choice(["Day", "Night"], rows),
    })

    # Balanced labels so the test checks both classes.
    y = pd.Series([0, 1] * (rows // 2), name=TARGET)

    return X[NUMERIC_FEATURES + CATEGORICAL_FEATURES], y


def test_load_ml_data_returns_valid_features_and_target():
    X, y = load_ml_data()

    assert len(X) == len(y)
    assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert set(y.unique()) == {0, 1}


def test_preprocessor_handles_missing_values():
    X, _ = make_sample_data()

    X.loc[0, "temperature"] = np.nan
    X.loc[1, "machine_id"] = None

    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert np.isfinite(np.asarray(transformed)).all()


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ],
)
def test_model_trains_and_predicts_both_classes(model_name):
    X, y = make_sample_data()

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", build_models()[model_name]),
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == len(y)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (len(y), 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
