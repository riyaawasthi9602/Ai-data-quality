import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class MonitoringConfig:
    history_path: str = "data/analysis_history.json"

    def validate(self):
        if not self.history_path:
            raise ValueError("history_path cannot be empty")


class HistoricalMonitor:
    """
    Stores lightweight historical snapshots of analysis runs.

    The monitor does NOT store the complete analysis result.
    It stores only important metrics needed for trend monitoring.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.config.validate()

    def _ensure_history_file(self):
        """Create the history file if it does not exist."""

        directory = os.path.dirname(self.config.history_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.config.history_path):
            with open(self.config.history_path, "w", encoding="utf-8") as file:
                json.dump([], file, indent=4)

    def load_history(self) -> List[Dict[str, Any]]:
        """Load all historical analysis runs."""

        self._ensure_history_file()

        try:
            with open(
                self.config.history_path,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError("History file must contain a JSON list.")

            return data

        except json.JSONDecodeError as exc:
            raise ValueError("History file contains invalid JSON.") from exc

    def record_run(
        self,
        result: Dict[str, Any],
        reference_path: Optional[str] = None,
        current_path: Optional[str] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store one analysis run.

        Only important monitoring metrics are stored.
        """

        history = self.load_history()

        health = result.get("health_score", {})
        quality = result.get("quality", {})
        drift = result.get("drift", {})
        ensemble = result.get("ensemble", {})
        recommendations = result.get("recommendations", {})

        snapshot = {
            "run_id": run_id or self._generate_run_id(),
            "timestamp": timestamp or self._current_timestamp(),

            "reference_path": reference_path,
            "current_path": current_path,

            "quality_score": self._to_float(
                quality.get("overall_quality_score")
            ),

            "drift_percentage": self._to_float(
                drift.get("drift_percentage")
            ),

            "affected_features": self._to_int(
                drift.get("affected_features")
            ),

            "anomaly_percentage": self._to_float(
                ensemble.get("anomaly_percentage")
            ),

            "ensemble_anomaly_count": self._to_int(
                ensemble.get("anomaly_count")
            ),

            "health_score": self._to_float(
                health.get("health_score")
            ),

            "health_level": health.get("health_level"),

            "recommendation_count": self._to_int(
                recommendations.get("total_recommendations")
            ),

            "high_priority_recommendations": self._to_int(
                recommendations.get("high_priority")
            ),
        }

        history.append(snapshot)

        with open(
            self.config.history_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=4)

        return snapshot

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all stored analysis runs."""

        return self.load_history()

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent analysis run."""

        history = self.load_history()

        if not history:
            return None

        return history[-1]

    def get_trends(self) -> Dict[str, List[Any]]:
        """
        Return important metrics in time-series form.
        """

        history = self.load_history()

        return {
            "timestamps": [
                item.get("timestamp")
                for item in history
            ],

            "health_scores": [
                item.get("health_score")
                for item in history
            ],

            "quality_scores": [
                item.get("quality_score")
                for item in history
            ],

            "drift_percentages": [
                item.get("drift_percentage")
                for item in history
            ],

            "anomaly_percentages": [
                item.get("anomaly_percentage")
                for item in history
            ],

            "affected_features": [
                item.get("affected_features")
                for item in history
            ],
        }

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID."""

        return datetime.now(timezone.utc).strftime(
            "run_%Y%m%d_%H%M%S_%f"
        )

    @staticmethod
    def _current_timestamp() -> str:
        """Return current UTC timestamp."""

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """Safely convert a value to float."""

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        """Safely convert a value to integer."""

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None