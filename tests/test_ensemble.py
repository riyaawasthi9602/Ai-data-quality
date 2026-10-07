import numpy as np
import pytest

from src.ensemble import (
    EnsembleAnomalyDetector,
    EnsembleConfig,
    normalize_scores,
)


def test_normalize_scores():
    scores = np.array([10, 20, 30, 40, 50])

    normalized = normalize_scores(scores)

    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)


def test_normalize_constant_scores():
    scores = np.array([5, 5, 5, 5])

    normalized = normalize_scores(scores)

    assert np.all(normalized == 0.0)


def test_ensemble_config():
    config = EnsembleConfig(
        autoencoder_weight=0.6,
        isolation_forest_weight=0.4,
        threshold=0.5,
    )

    config.validate()

    assert config.autoencoder_weight == 0.6
    assert config.isolation_forest_weight == 0.4
    assert config.threshold == 0.5


def test_ensemble_score_combination():
    config = EnsembleConfig(
        autoencoder_weight=0.5,
        isolation_forest_weight=0.5,
    )

    detector = EnsembleAnomalyDetector(config)

    autoencoder_scores = np.array([
        0.1,
        0.2,
        0.9,
        1.0,
    ])

    isolation_scores = np.array([
        0.1,
        0.3,
        0.8,
        1.0,
    ])

    result = detector.combine_scores(
        autoencoder_scores,
        isolation_scores,
    )

    assert "autoencoder_score" in result
    assert "isolation_forest_score" in result
    assert "ensemble_score" in result

    assert len(result["ensemble_score"]) == 4

    assert np.all(result["ensemble_score"] >= 0)
    assert np.all(result["ensemble_score"] <= 1)


def test_ensemble_prediction():
    config = EnsembleConfig(
        autoencoder_weight=0.5,
        isolation_forest_weight=0.5,
        threshold=0.5,
    )

    detector = EnsembleAnomalyDetector(config)

    autoencoder_scores = np.array([
        0.1,
        0.2,
        0.9,
        1.0,
    ])

    isolation_scores = np.array([
        0.1,
        0.2,
        0.8,
        1.0,
    ])

    autoencoder_flags = np.array([
        False,
        False,
        True,
        True,
    ])

    isolation_flags = np.array([
        False,
        False,
        True,
        True,
    ])

    result = detector.predict(
        autoencoder_scores,
        isolation_scores,
        autoencoder_flags,
        isolation_flags,
    )

    assert "results" in result
    assert "anomaly_count" in result
    assert "anomaly_percentage" in result

    assert "ensemble_score" in result["results"].columns
    assert "ensemble_anomaly" in result["results"].columns

    assert "both_models_anomaly" in result["results"].columns
    assert "model_agreement" in result["results"].columns

    assert result["anomaly_count"] >= 0
    assert result["anomaly_percentage"] >= 0


def test_model_agreement():
    config = EnsembleConfig()

    detector = EnsembleAnomalyDetector(config)

    autoencoder_scores = np.array([
        0.1,
        0.9,
        0.8,
        0.2,
    ])

    isolation_scores = np.array([
        0.1,
        0.9,
        0.2,
        0.8,
    ])

    autoencoder_flags = np.array([
        False,
        True,
        True,
        False,
    ])

    isolation_flags = np.array([
        False,
        True,
        False,
        True,
    ])

    result = detector.predict(
        autoencoder_scores,
        isolation_scores,
        autoencoder_flags,
        isolation_flags,
    )

    assert result["agreement_count"] == 2
    assert result["agreement_percentage"] == 50.0


def test_ensemble_rejects_different_lengths():
    detector = EnsembleAnomalyDetector()

    autoencoder_scores = np.array([1, 2, 3])
    isolation_scores = np.array([1, 2])

    with pytest.raises(ValueError):
        detector.combine_scores(
            autoencoder_scores,
            isolation_scores,
        )


def test_invalid_config():

    with pytest.raises(ValueError):
        config = EnsembleConfig(
            autoencoder_weight=-0.1
        )
        config.validate()

    with pytest.raises(ValueError):
        config = EnsembleConfig(
            threshold=1.5
        )
        config.validate()


def test_anomaly_categories():

    config = EnsembleConfig(
        autoencoder_weight=0.5,
        isolation_forest_weight=0.5,
        threshold=0.99,
    )

    detector = EnsembleAnomalyDetector(config)

    autoencoder_scores = np.array([
        0.1,
        0.9,
        0.8,
        0.2,
    ])

    isolation_scores = np.array([
        0.1,
        0.9,
        0.2,
        0.8,
    ])

    autoencoder_flags = np.array([
        False,
        True,
        True,
        False,
    ])

    isolation_flags = np.array([
        False,
        True,
        False,
        True,
    ])

    result = detector.predict(
        autoencoder_scores,
        isolation_scores,
        autoencoder_flags,
        isolation_flags,
    )

    categories = result[
        "results"
    ]["anomaly_category"].tolist()

    assert categories == [
        "normal",
        "strong_anomaly",
        "possible_anomaly",
        "possible_anomaly",
    ]

    assert result["strong_anomaly_count"] == 1

    assert result["possible_anomaly_count"] == 2