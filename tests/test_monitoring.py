import json

import pytest

from src.monitoring import HistoricalMonitor, MonitoringConfig


def sample_analysis_result():
    return {
        "quality": {
            "overall_quality_score": 92.5
        },

        "drift": {
            "drift_percentage": 15.0,
            "affected_features": 3
        },

        "ensemble": {
            "anomaly_percentage": 4.5,
            "anomaly_count": 45
        },

        "health_score": {
            "health_score": 87.2,
            "health_level": "GOOD"
        },

        "recommendations": {
            "total_recommendations": 4,
            "high_priority": 2
        }
    }


def test_monitoring_config():

    config = MonitoringConfig(
        history_path="test_history.json"
    )

    config.validate()

    assert config.history_path == "test_history.json"


def test_empty_history(tmp_path):

    history_file = tmp_path / "history.json"

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    history = monitor.get_history()

    assert history == []


def test_record_run(tmp_path):

    history_file = tmp_path / "history.json"

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    snapshot = monitor.record_run(
        sample_analysis_result(),
        reference_path="data/reference.csv",
        current_path="data/current.csv",
        run_id="test_run_001",
        timestamp="2026-10-07T10:00:00+00:00"
    )

    assert snapshot["run_id"] == "test_run_001"
    assert snapshot["quality_score"] == 92.5
    assert snapshot["drift_percentage"] == 15.0
    assert snapshot["affected_features"] == 3
    assert snapshot["anomaly_percentage"] == 4.5
    assert snapshot["ensemble_anomaly_count"] == 45
    assert snapshot["health_score"] == 87.2
    assert snapshot["health_level"] == "GOOD"


def test_load_history(tmp_path):

    history_file = tmp_path / "history.json"

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    monitor.record_run(
        sample_analysis_result(),
        run_id="run_001"
    )

    history = monitor.load_history()

    assert len(history) == 1
    assert history[0]["run_id"] == "run_001"


def test_multiple_runs(tmp_path):

    history_file = tmp_path / "history.json"

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    monitor.record_run(
        sample_analysis_result(),
        run_id="run_001"
    )

    monitor.record_run(
        sample_analysis_result(),
        run_id="run_002"
    )

    history = monitor.get_history()

    assert len(history) == 2
    assert history[0]["run_id"] == "run_001"
    assert history[1]["run_id"] == "run_002"


def test_get_latest(tmp_path):

    history_file = tmp_path / "history.json"

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    monitor.record_run(
        sample_analysis_result(),
        run_id="run_001"
    )

    monitor.record_run(
        sample_analysis_result(),
        run_id="run_002"
    )

    latest = monitor.get_latest()

    assert latest is not None
    assert latest["run_id"] == "run_002"


def test_get_trends(tmp_path):

    history_file = tmp_path / "history.json"

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    result = sample_analysis_result()

    monitor.record_run(
        result,
        run_id="run_001",
        timestamp="2026-10-07T10:00:00+00:00"
    )

    monitor.record_run(
        result,
        run_id="run_002",
        timestamp="2026-10-07T11:00:00+00:00"
    )

    trends = monitor.get_trends()

    assert len(trends["timestamps"]) == 2
    assert trends["health_scores"] == [87.2, 87.2]
    assert trends["quality_scores"] == [92.5, 92.5]
    assert trends["drift_percentages"] == [15.0, 15.0]
    assert trends["anomaly_percentages"] == [4.5, 4.5]
    assert trends["affected_features"] == [3, 3]


def test_invalid_json_history(tmp_path):

    history_file = tmp_path / "history.json"

    history_file.write_text(
        "{invalid json",
        encoding="utf-8"
    )

    monitor = HistoricalMonitor(
        MonitoringConfig(
            history_path=str(history_file)
        )
    )

    with pytest.raises(ValueError):

        monitor.load_history()