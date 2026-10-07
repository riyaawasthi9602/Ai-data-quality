import numpy as np
import pandas as pd
import pytest

from src.explainability import (
    AnomalyExplainer,
    ExplainabilityConfig,
)


def create_reference_data():
    return pd.DataFrame({
        "Heart Rate (bpm)": [
            70, 72, 68, 71, 69,
            73, 70, 72, 71, 69
        ],
        "SpO2 Level (%)": [
            98, 97, 99, 98, 97,
            98, 99, 97, 98, 99
        ],
        "Systolic Blood Pressure (mmHg)": [
            120, 122, 118, 121, 119,
            123, 120, 121, 119, 122
        ],
        "Body Temperature (°C)": [
            36.5, 36.6, 36.4, 36.5, 36.6,
            36.5, 36.4, 36.6, 36.5, 36.4
        ],
    })


def test_explainability_config():

    config = ExplainabilityConfig(
        z_score_threshold=3.0,
        top_features=3,
    )

    config.validate()

    assert config.z_score_threshold == 3.0
    assert config.top_features == 3


def test_invalid_explainability_config():

    with pytest.raises(ValueError):
        config = ExplainabilityConfig(
            z_score_threshold=0
        )
        config.validate()

    with pytest.raises(ValueError):
        config = ExplainabilityConfig(
            top_features=0
        )
        config.validate()


def test_feature_deviation():

    reference = create_reference_data()

    current_row = pd.Series({
        "Heart Rate (bpm)": 150,
        "SpO2 Level (%)": 98,
        "Systolic Blood Pressure (mmHg)": 120,
        "Body Temperature (°C)": 36.5,
    })

    explainer = AnomalyExplainer()

    deviations = explainer.calculate_feature_deviation(
        reference,
        current_row,
    )

    assert len(deviations) == 4

    assert deviations[0]["feature"] == "Heart Rate (bpm)"

    assert deviations[0]["z_score"] > 3

    assert deviations[0]["is_significant"] is True


def test_explain_row():

    reference = create_reference_data()

    current_row = pd.Series({
        "Heart Rate (bpm)": 150,
        "SpO2 Level (%)": 90,
        "Systolic Blood Pressure (mmHg)": 120,
        "Body Temperature (°C)": 36.5,
    })

    explainer = AnomalyExplainer()

    explanation = explainer.explain_row(
        reference_df=reference,
        current_row=current_row,
        anomaly_category="strong_anomaly",
        autoencoder_score=0.95,
        isolation_forest_score=0.90,
        ensemble_score=0.925,
    )

    assert explanation["summary"].startswith(
        "Strong anomaly"
    )

    assert explanation["anomaly_category"] == (
        "strong_anomaly"
    )

    assert explanation["autoencoder_score"] == 0.95

    assert explanation["isolation_forest_score"] == 0.90

    assert explanation["ensemble_score"] == 0.925

    assert (
        explanation["significant_feature_count"] >= 1
    )

    assert len(explanation["top_features"]) <= 3


def test_explain_normal_row():

    reference = create_reference_data()

    current_row = pd.Series({
        "Heart Rate (bpm)": 70,
        "SpO2 Level (%)": 98,
        "Systolic Blood Pressure (mmHg)": 120,
        "Body Temperature (°C)": 36.5,
    })

    explainer = AnomalyExplainer()

    explanation = explainer.explain_row(
        reference_df=reference,
        current_row=current_row,
        anomaly_category="normal",
    )

    assert explanation["anomaly_category"] == "normal"

    assert explanation["summary"].startswith(
        "No strong anomaly"
    )


def test_explain_dataset():

    reference = create_reference_data()

    current = pd.DataFrame({
        "Heart Rate (bpm)": [
            70,
            150,
            72,
        ],
        "SpO2 Level (%)": [
            98,
            90,
            97,
        ],
        "Systolic Blood Pressure (mmHg)": [
            120,
            180,
            122,
        ],
        "Body Temperature (°C)": [
            36.5,
            37.0,
            36.6,
        ],
    })

    ensemble_result = {
        "results": pd.DataFrame({
            "ensemble_anomaly": [
                False,
                True,
                False,
            ],
            "anomaly_category": [
                "normal",
                "strong_anomaly",
                "normal",
            ],
            "autoencoder_score": [
                0.1,
                0.9,
                0.2,
            ],
            "isolation_forest_score": [
                0.1,
                0.95,
                0.2,
            ],
            "ensemble_score": [
                0.1,
                0.925,
                0.2,
            ],
        })
    }

    explainer = AnomalyExplainer()

    explanations = explainer.explain_dataset(
        reference_df=reference,
        current_df=current,
        ensemble_result=ensemble_result,
        only_anomalies=True,
    )

    assert len(explanations) == 1

    assert explanations[0]["row_index"] == 1

    assert explanations[0]["anomaly_category"] == (
        "strong_anomaly"
    )

    assert (
        explanations[0]["significant_feature_count"]
        >= 1
    )