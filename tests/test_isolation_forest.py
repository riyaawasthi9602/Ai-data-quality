import numpy as np
import pandas as pd

from src.isolation_forest import (
    IsolationForestAnomalyDetector,
    IsolationForestConfig,
)


# ============================================================
# TEST DATA
# ============================================================

def create_reference_data():

    np.random.seed(42)

    return pd.DataFrame({
        "feature_a": np.random.normal(
            50,
            5,
            100
        ),

        "feature_b": np.random.normal(
            100,
            10,
            100
        ),

        "feature_c": np.random.normal(
            20,
            2,
            100
        ),
    })


# ============================================================
# TEST 1 — CONFIGURATION
# ============================================================

def test_isolation_forest_config():

    config = IsolationForestConfig(
        contamination=0.05,
        n_estimators=50
    )

    config.validate()

    assert config.contamination == 0.05
    assert config.n_estimators == 50


# ============================================================
# TEST 2 — TRAINING
# ============================================================

def test_isolation_forest_training():

    reference = create_reference_data()

    config = IsolationForestConfig(
        contamination=0.05,
        n_estimators=50
    )

    detector = IsolationForestAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    assert detector.model is not None

    assert detector.scaler is not None

    assert len(
        detector.feature_columns
    ) == 3


# ============================================================
# TEST 3 — ANOMALY PREDICTION
# ============================================================

def test_isolation_forest_prediction():

    reference = create_reference_data()

    current = pd.DataFrame({
        "feature_a": [
            50,
            51,
            49,
            52,
            150
        ],

        "feature_b": [
            100,
            101,
            99,
            102,
            300
        ],

        "feature_c": [
            20,
            21,
            19,
            20,
            80
        ],
    })

    config = IsolationForestConfig(
        contamination=0.05,
        n_estimators=50
    )

    detector = IsolationForestAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    result = detector.predict(
        current
    )

    assert "results" in result

    assert "anomaly_count" in result

    assert "anomaly_percentage" in result

    assert (
        len(result["results"])
        == len(current)
    )

    assert (
        "isolation_score"
        in result["results"].columns
    )

    assert (
        "is_anomaly"
        in result["results"].columns
    )

    assert (
        result["anomaly_count"] >= 0
    )


# ============================================================
# TEST 4 — MISSING + INFINITE VALUES
# ============================================================

def test_isolation_forest_handles_missing_and_infinite():

    reference = create_reference_data()

    reference.loc[
        0,
        "feature_a"
    ] = np.nan

    reference.loc[
        1,
        "feature_b"
    ] = np.inf

    reference.loc[
        2,
        "feature_c"
    ] = -np.inf

    config = IsolationForestConfig(
        contamination=0.05,
        n_estimators=50
    )

    detector = IsolationForestAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    current = create_reference_data()

    current.loc[
        0,
        "feature_a"
    ] = np.nan

    current.loc[
        1,
        "feature_b"
    ] = np.inf

    result = detector.predict(
        current
    )

    assert (
        len(result["results"])
        == len(current)
    )

    assert np.isfinite(
        result["results"][
            "isolation_score"
        ]
    ).all()


# ============================================================
# TEST 5 — MISSING FEATURES
# ============================================================

def test_isolation_forest_rejects_missing_features():

    reference = create_reference_data()

    current = pd.DataFrame({
        "feature_a": [
            50,
            51,
            52
        ],

        "feature_b": [
            100,
            101,
            102
        ],
    })

    config = IsolationForestConfig(
        contamination=0.05,
        n_estimators=50
    )

    detector = IsolationForestAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    try:

        detector.predict(
            current
        )

        assert False, (
            "Expected ValueError for "
            "missing feature"
        )

    except ValueError as error:

        assert (
            "missing expected features"
            in str(error).lower()
        )


# ============================================================
# TEST 6 — SAVE AND LOAD
# ============================================================

def test_isolation_forest_save_and_load(tmp_path):

    reference = create_reference_data()

    config = IsolationForestConfig(
        contamination=0.05,
        n_estimators=50,

        model_path=str(
            tmp_path /
            "isolation_forest.pkl"
        ),

        scaler_path=str(
            tmp_path /
            "isolation_forest_scaler.pkl"
        ),

        metadata_path=str(
            tmp_path /
            "isolation_forest_metadata.pkl"
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    detector = IsolationForestAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    original_features = (
        detector.feature_columns.copy()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    saved_paths = detector.save()

    assert len(
        saved_paths
    ) == 3

    # --------------------------------------------------------
    # Create new detector
    # --------------------------------------------------------

    loaded_detector = (
        IsolationForestAnomalyDetector(
            config=config
        )
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    loaded_detector.load()

    assert (
        loaded_detector.model
        is not None
    )

    assert (
        loaded_detector.scaler
        is not None
    )

    assert (
        loaded_detector.feature_columns
        == original_features
    )

    # --------------------------------------------------------
    # Prediction after loading
    # --------------------------------------------------------

    result = loaded_detector.predict(
        reference
    )

    assert (
        len(result["results"])
        == len(reference)
    )

    assert (
        "isolation_score"
        in result["results"].columns
    )