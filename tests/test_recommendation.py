from src.recommendation import (
    RecommendationConfig,
    RecommendationEngine,
)


def test_recommendation_config():
    config = RecommendationConfig()

    assert config.high_drift_threshold == 0.25
    assert config.moderate_drift_threshold == 0.05
    assert config.high_anomaly_percentage == 10.0


def test_invalid_recommendation_config():
    config = RecommendationConfig(
        high_drift_threshold=0.01,
        moderate_drift_threshold=0.05,
    )

    try:
        config.validate()
        assert False
    except ValueError:
        assert True


def test_low_quality_recommendation():
    engine = RecommendationEngine()

    quality_result = {
        "overall_score": 60.0,
        "dimensions": {
            "completeness": 55.0,
            "validity": 90.0,
            "consistency": 80.0,
            "uniqueness": 95.0,
            "outliers": 90.0,
        },
    }

    result = engine.generate_recommendations(
        quality_result=quality_result
    )

    assert result["total_recommendations"] > 0
    assert result["high_priority"] > 0


def test_high_drift_recommendation():
    engine = RecommendationEngine()

    drift_result = {
        "overall_severity": "high",
        "drift_percentage": 30.0,
        "affected_features": 3,
        "missingness_drift_percentage": 0.0,
    }

    result = engine.generate_recommendations(
        drift_result=drift_result
    )

    assert result["total_recommendations"] > 0
    assert result["high_priority"] >= 1

    titles = [
        item["title"]
        for item in result["recommendations"]
    ]

    assert "High data drift detected" in titles


def test_missingness_drift_recommendation():
    engine = RecommendationEngine()

    drift_result = {
        "overall_severity": "low",
        "drift_percentage": 0.0,
        "affected_features": 0,
        "missingness_drift_percentage": 25.0,
    }

    result = engine.generate_recommendations(
        drift_result=drift_result
    )

    assert result["total_recommendations"] == 1

    recommendation = result["recommendations"][0]

    assert recommendation["category"] == "missingness_drift"


def test_high_anomaly_recommendation():
    engine = RecommendationEngine()

    anomaly_result = {
        "anomaly_percentage": 15.0,
        "strong_anomaly_count": 100,
    }

    result = engine.generate_recommendations(
        anomaly_result=anomaly_result
    )

    assert result["total_recommendations"] == 2
    assert result["high_priority"] == 2


def test_explainability_recommendation():
    engine = RecommendationEngine()

    explanations = [
        {
            "significant_features": [
                {
                    "feature": "SpO2 Level (%)",
                    "z_score": 4.2,
                }
            ]
        },
        {
            "significant_features": [
                {
                    "feature": "SpO2 Level (%)",
                    "z_score": 3.8,
                }
            ]
        },
    ]

    result = engine.generate_recommendations(
        explanations=explanations
    )

    assert result["total_recommendations"] == 1

    recommendation = result["recommendations"][0]

    assert recommendation["category"] == "explainability"
    assert "SpO2 Level (%)" in recommendation["message"]


def test_no_recommendations_for_healthy_data():
    engine = RecommendationEngine()

    quality_result = {
        "overall_score": 98.0,
        "dimensions": {
            "completeness": 100.0,
            "validity": 100.0,
            "consistency": 100.0,
            "uniqueness": 100.0,
            "outliers": 98.0,
        },
    }

    drift_result = {
        "overall_severity": "low",
        "drift_percentage": 0.0,
        "affected_features": 0,
        "missingness_drift_percentage": 0.0,
    }

    anomaly_result = {
        "anomaly_percentage": 1.0,
        "strong_anomaly_count": 0,
    }

    result = engine.generate_recommendations(
        quality_result=quality_result,
        drift_result=drift_result,
        anomaly_result=anomaly_result,
    )

    assert result["total_recommendations"] == 0