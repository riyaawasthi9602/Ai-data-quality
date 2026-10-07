import numpy as np
import pandas as pd

from src.anomaly import (
    AutoencoderAnomalyDetector,
    AutoencoderConfig,
    calculate_anomaly_threshold,
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
            40
        ),

        "feature_b": np.random.normal(
            100,
            10,
            40
        ),

        "feature_c": np.random.normal(
            20,
            2,
            40
        ),
    })


# ============================================================
# TEST 1 — THRESHOLD
# ============================================================

def test_anomaly_threshold():

    errors = np.array([
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ])

    threshold = calculate_anomaly_threshold(
        errors,
        percentile=95
    )

    assert threshold > 0

    assert threshold >= np.max(
        errors[:-1]
    )


# ============================================================
# TEST 2 — TRAIN AUTOENCODER
# ============================================================

def test_prepare_and_train_autoencoder():

    reference = create_reference_data()

    config = AutoencoderConfig(
        epochs=2,
        batch_size=8,
        patience=1
    )

    detector = AutoencoderAnomalyDetector(
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

    assert detector.threshold is not None

    assert (
        detector.reference_mean_error
        is not None
    )


# ============================================================
# TEST 3 — PREDICTION
# ============================================================

def test_autoencoder_prediction():

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

    config = AutoencoderConfig(
        epochs=2,
        batch_size=8,
        patience=1
    )

    detector = AutoencoderAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    result = detector.predict(
        current
    )

    assert "results" in result

    assert "threshold" in result

    assert "anomaly_count" in result

    assert "anomaly_percentage" in result

    assert len(
        result["results"]
    ) == len(current)

    assert (
        "reconstruction_error"
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

def test_autoencoder_handles_missing_and_infinite_values():

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

    config = AutoencoderConfig(
        epochs=2,
        batch_size=8,
        patience=1
    )

    detector = AutoencoderAnomalyDetector(
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

    assert len(
        result["results"]
    ) == len(current)

    assert np.isfinite(
        result["results"][
            "reconstruction_error"
        ]
    ).all()


# ============================================================
# TEST 5 — MISSING FEATURES
# ============================================================

def test_autoencoder_rejects_missing_features():

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

    config = AutoencoderConfig(
        epochs=2,
        batch_size=8,
        patience=1
    )

    detector = AutoencoderAnomalyDetector(
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

def test_autoencoder_save_and_load(tmp_path):

    reference = create_reference_data()

    config = AutoencoderConfig(
        epochs=2,
        batch_size=8,
        patience=1,

        model_path=str(
            tmp_path /
            "autoencoder.keras"
        ),

        scaler_path=str(
            tmp_path /
            "scaler.pkl"
        ),

        metadata_path=str(
            tmp_path /
            "metadata.pkl"
        )
    )

    # --------------------------------------------------------
    # Train detector
    # --------------------------------------------------------

    detector = AutoencoderAnomalyDetector(
        config=config
    )

    detector.train(
        reference
    )

    original_threshold = (
        detector.threshold
    )

    original_features = (
        detector.feature_columns.copy()
    )

    # --------------------------------------------------------
    # Save detector
    # --------------------------------------------------------

    saved_paths = detector.save()

    assert len(
        saved_paths
    ) == 3

    # --------------------------------------------------------
    # Create new detector
    # --------------------------------------------------------

    loaded_detector = (
        AutoencoderAnomalyDetector(
            config=config
        )
    )

    # --------------------------------------------------------
    # Load saved model
    # --------------------------------------------------------

    loaded_detector.load()

    # --------------------------------------------------------
    # Verify model
    # --------------------------------------------------------

    assert (
        loaded_detector.model
        is not None
    )

    # --------------------------------------------------------
    # Verify scaler
    # --------------------------------------------------------

    assert (
        loaded_detector.scaler
        is not None
    )

    # --------------------------------------------------------
    # Verify threshold
    # --------------------------------------------------------

    assert (
        loaded_detector.threshold
        == original_threshold
    )

    # --------------------------------------------------------
    # Verify feature metadata
    # --------------------------------------------------------

    assert (
        loaded_detector.feature_columns
        == original_features
    )

    # --------------------------------------------------------
    # Verify prediction
    # --------------------------------------------------------

    result = loaded_detector.predict(
        reference
    )

    assert (
        len(result["results"])
        == len(reference)
    )

    assert (
        result["threshold"]
        == original_threshold
    )