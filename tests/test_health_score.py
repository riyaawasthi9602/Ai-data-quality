from src.health_score import (
    HealthScoreConfig,
    DataHealthScore,
)


def test_health_score_config():

    config = HealthScoreConfig()

    config.validate()

    assert config.quality_weight == 0.40
    assert config.drift_weight == 0.30
    assert config.anomaly_weight == 0.30


def test_invalid_health_score_config():

    config = HealthScoreConfig(
        quality_weight=-1.0
    )

    try:
        config.validate()
        assert False
    except ValueError:
        assert True


def test_quality_score():

    calculator = DataHealthScore()

    quality_result = {
        "overall_quality_score": 95.0
    }

    score = calculator.calculate_quality_score(
        quality_result
    )

    assert score == 95.0


def test_drift_score():

    calculator = DataHealthScore()

    drift_result = {
        "drift_percentage": 20.0
    }

    score = calculator.calculate_drift_score(
        drift_result
    )

    assert score == 80.0


def test_anomaly_score():

    calculator = DataHealthScore()

    anomaly_result = {
        "anomaly_percentage": 10.0
    }

    score = calculator.calculate_anomaly_score(
        anomaly_result
    )

    assert score == 90.0


def test_health_level():

    calculator = DataHealthScore()

    assert (
        calculator.get_health_level(95)
        == "EXCELLENT"
    )

    assert (
        calculator.get_health_level(80)
        == "GOOD"
    )

    assert (
        calculator.get_health_level(65)
        == "NEEDS ATTENTION"
    )

    assert (
        calculator.get_health_level(45)
        == "POOR"
    )

    assert (
        calculator.get_health_level(20)
        == "CRITICAL"
    )


def test_complete_health_score():

    calculator = DataHealthScore()

    quality_result = {
        "overall_quality_score": 100.0
    }

    drift_result = {
        "drift_percentage": 20.0
    }

    anomaly_result = {
        "anomaly_percentage": 10.0
    }

    result = calculator.calculate_health_score(
        quality_result,
        drift_result,
        anomaly_result,
    )

    # 100 * 0.40 + 80 * 0.30 + 90 * 0.30
    # = 91

    assert result["health_score"] == 91.0

    assert (
        result["health_level"]
        == "EXCELLENT"
    )

    assert (
        result["components"]["quality_score"]
        == 100.0
    )

    assert (
        result["components"]["drift_score"]
        == 80.0
    )

    assert (
        result["components"]["anomaly_score"]
        == 90.0
    )


def test_health_score_with_healthy_data():

    calculator = DataHealthScore()

    quality_result = {
        "overall_quality_score": 100.0
    }

    drift_result = {
        "drift_percentage": 0.0
    }

    anomaly_result = {
        "anomaly_percentage": 0.0
    }

    result = calculator.calculate_health_score(
        quality_result,
        drift_result,
        anomaly_result,
    )

    assert result["health_score"] == 100.0

    assert (
        result["health_level"]
        == "EXCELLENT"
    )